"""Output-owned sparse contact reductions for MJWarp 4.0.1.

Experimental, CUDA sparse Newton with pyramidal contact-only constraints.
The host retains constraint laws, active sets, metric assembly and line search.
Each Hessian entry and generalized force has one writer, visiting canonical
constraint rows in increasing order with FP64 accumulation. No floating-point
atomics or global record sorting are used by these contact reductions.
"""
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import inspect
import linecache
import threading
from types import FunctionType

import numpy as np
import warp as wp

_LOCK = threading.RLock()
_GLOBAL = '_deform3d_owned_contact_assembly'
_HASHES = {
    '_solve': 'ac3c770dd9380834f4fe3f6d89c2c65418dec0ffa92cc5f98d29550ae19ae43a',
    '_update_constraint': 'bb3241b42b78170d3d8b86673e90427ca0a4f0e8f5f30a1f6d8d9910c7739eab',
    '_update_gradient': '54cea02cf18fd41dc78424d908f33516b705a3877c6cca6858f74f18b89abc82',
    '_update_gradient_incremental': 'cbc732e2fcd7f79938345d0ce356687f846950a9c4c95cb78491cbf0f8e17f59',
}


@wp.kernel(enable_backward=False)
def _unpack(nv: int, nefc: wp.array[int], ne: wp.array[int], nf: wp.array[int], nl: wp.array[int],
            kind: wp.array2d[int], nnz: wp.array2d[int], adr: wp.array2d[int],
            columns: wp.array3d[int], packed: wp.array3d[float],
            dense: wp.array3d[float], status: wp.array[int]):
    w, row = wp.tid()
    count = nefc[w]
    if count < 0 or count > dense.shape[1] or ne[w] != 0 or nf[w] != 0 or nl[w] != 0:
        wp.atomic_or(status, w, 1)
        return
    if row >= count:
        return
    # MuJoCo ConstraintType: frictionless=5, pyramidal=6, elliptic=7.
    if kind[w, row] != 5 and kind[w, row] != 6:
        wp.atomic_or(status, w, 1)
        return
    start, size = adr[w, row], nnz[w, row]
    if size < 0 or size > nv or start < 0 or start > packed.shape[2] - size:
        wp.atomic_or(status, w, 1)
        return
    for k in range(size):
        col = columns[w, 0, start+k]
        value = packed[w, 0, start+k]
        if col < 0 or col >= nv:
            wp.atomic_or(status, w, 1)
            return
        # Contact assembly does not promise sorted column indices. It does
        # promise unique support; reject duplicates without imposing an order.
        for before in range(k):
            if columns[w, 0, start+before] == col:
                wp.atomic_or(status, w, 1)
                return
        if not wp.isfinite(value):
            wp.atomic_or(status, w, 2)
            return
        dense[w, row, col] = value


@wp.kernel(enable_backward=False)
def _incidence(nefc: wp.array[int], dense: wp.array3d[float],
               rows: wp.array3d[int], counts: wp.array2d[int]):
    w, col = wp.tid()
    count = int(0)
    for row in range(wp.clamp(nefc[w], 0, dense.shape[1])):
        if dense[w, row, col] != 0.0:
            rows[w, col, count] = row
            count += 1
    counts[w, col] = count


@wp.kernel(enable_backward=False)
def _weights(incremental: bool, nefc: wp.array[int], done: wp.array[bool],
             d: wp.array2d[float], state: wp.array2d[int], previous: wp.array2d[int],
             weights: wp.array2d[wp.float64], status: wp.array[int]):
    w, row = wp.tid()
    weights[w, row] = wp.float64(0.0)
    if done[w] or row >= nefc[w]:
        return
    s, value = state[w, row], d[w, row]
    if (s != 0 and s != 1) or not wp.isfinite(value) or value < 0.0:
        wp.atomic_or(status, w, 2)
        return
    active = int(s == 1)
    sign = active
    if incremental:
        sign -= previous[w, row]
    weights[w, row] = wp.float64(sign) * wp.float64(value)
    previous[w, row] = active


@wp.kernel(enable_backward=False)
def _hessian(nv: int, done: wp.array[bool], dense: wp.array3d[float],
             rows: wp.array3d[int], counts: wp.array2d[int], weights: wp.array2d[wp.float64],
             h: wp.array3d[float], status: wp.array[int]):
    w, i, j = wp.tid()
    if done[w] or i > j:
        return
    # Choosing the shorter column list changes work, never row summation order.
    column = i
    if counts[w, j] < counts[w, i]:
        column = j
    total = wp.float64(0.0)
    for k in range(counts[w, column]):
        row = rows[w, column, k]
        coefficient = weights[w, row]
        if coefficient != wp.float64(0.0):
            total += wp.float64(dense[w, row, i]) * coefficient * wp.float64(dense[w, row, j])
    value = float(wp.float64(h[w, i, j]) + total)
    if not wp.isfinite(value):
        wp.atomic_or(status, w, 2)
    h[w, i, j] = value


@wp.kernel(enable_backward=False)
def _force(done: wp.array[bool], changed: wp.array[int], dense: wp.array3d[float],
           rows: wp.array3d[int], counts: wp.array2d[int], force: wp.array2d[float],
           out: wp.array2d[float], status: wp.array[int]):
    w, col = wp.tid()
    if done[w] or changed[w] == 0:
        return
    total = wp.float64(0.0)
    for k in range(counts[w, col]):
        row = rows[w, col, k]
        total += wp.float64(dense[w, row, col]) * wp.float64(force[w, row])
    value = float(total)
    if not wp.isfinite(value):
        wp.atomic_or(status, w, 2)
    out[w, col] = value


def _replace_region(source, start, end, replacement):
    if source.count(start) != 1 or source.count(end) != 1:
        raise RuntimeError('unexpected MJWarp assembly structure')
    a, b = source.index(start), source.index(end)
    if b <= a:
        raise RuntimeError('unexpected MJWarp assembly region order')
    return source[:a] + replacement + source[b:]


def _configured(original, name, solver_reductions=False):
    source = inspect.getsource(original)
    if hashlib.sha256(source.encode()).hexdigest() != _HASHES[name]:
        raise RuntimeError('unreviewed or already patched MJWarp ' + name)
    if name == '_update_gradient':
        source = _replace_region(source, '      max_condim = 3\n',
            '    else:\n      m_mat = d.qH', f'      {_GLOBAL}.full(d, ctx)\n')
    elif name == '_update_gradient_incremental':
        source = _replace_region(source, '    inc_kernel = _update_gradient_h_incremental_sparse(sc)\n',
            '  else:\n    # two groups per world', f'    {_GLOBAL}.incremental(d, ctx)\n')
    elif name == '_update_constraint':
        source = _replace_region(source, '    wp.launch(\n      _zero_qfrc_constraint_sparse,\n',
            '  else:\n    wp.launch(\n      _update_constraint_init_qfrc_constraint_dense',
            f'    {_GLOBAL}.force(d, ctx, changed)\n')
    if solver_reductions and name in ('_update_gradient', '_update_gradient_incremental'):
        from .mujoco_solver_ops import configure
        source = configure(source, name)
    filename = '<deform3d_owned_assembly_' + hashlib.sha256(source.encode()).hexdigest() + '>'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(original.__globals__)
    exec(compile(source, filename, 'exec'), namespace)
    # Live module globals preserve composition with the metric/Newton hooks.
    return FunctionType(namespace[name].__code__, original.__globals__, name, original.__defaults__)


class OwnedContactAssembly:
    """Prepare once, enter during warmup/capture, retain through graph replay.

    Workspace is O(worlds * nv * njmax), limited to 256 MiB and 1..512 DOFs.
    Rebuild/recapture after topology, capacity or device changes. Setup is serial;
    one workspace cannot serve overlapping solves/streams. Replay has no host
    reads or allocations. Status is sticky: 1=invalid sparse/contact layout,
    2=invalid numeric data/state/output. Any nonzero status invalidates the run.
    This policy changes floating-point summation order, not contact equations.
    It requires canonical rows for agreement between differently ordered worlds;
    other physics kernels still need their own determinism policy.
    solver_reductions=True also replaces effective-metric insertion and gradient
    norms with owned Warp-binned reductions; immutable metric CSR is required.
    """
    def __init__(self, model, data, *, solver_reductions=False):
        self.model = model
        self._validate_model()
        self.nworld, self.njmax = data.nworld, data.njmax
        self.nv = model.nv
        self.device = model.body_mass.device
        self._validate_data(data)
        self.workspace_bytes = (8*self.nworld*self.njmax*self.nv
                                + 12*self.nworld*self.njmax + 4*self.nworld*(self.nv+1))
        if self.workspace_bytes > 256*1024**2:
            raise ValueError('owned contact workspace exceeds 256 MiB')
        self.dense = wp.zeros((self.nworld,self.njmax,self.nv), dtype=float, device=self.device)
        self.rows = wp.empty((self.nworld,self.nv,self.njmax), dtype=int, device=self.device)
        self.counts = wp.zeros((self.nworld,self.nv), dtype=int, device=self.device)
        self.previous = wp.zeros((self.nworld,self.njmax), dtype=int, device=self.device)
        self.weights = wp.zeros((self.nworld,self.njmax), dtype=wp.float64, device=self.device)
        self.status = wp.zeros(self.nworld, dtype=int, device=self.device)
        self.solver_ops = None
        if solver_reductions:
            from .mujoco_solver_ops import OwnedSolverReductions
            self.solver_ops = OwnedSolverReductions(model,data,self.status)
            self.workspace_bytes += self.solver_ops.workspace_bytes
            if self.workspace_bytes > 256*1024**2:
                raise ValueError('owned contact and solver workspace exceeds 256 MiB')

    def _validate_model(self):
        from mujoco_warp._src import types
        m = self.model
        if version('mujoco-warp') != '4.0.1':
            raise RuntimeError('reviewed only for mujoco-warp 4.0.1')
        if (not m.body_mass.device.is_cuda or not m.is_sparse or not 1 <= m.nv <= 512
            or m.opt.solver != types.SolverType.NEWTON or m.opt.cone != types.ConeType.PYRAMIDAL
            or m.neq or m.ntendon or m.flg_adhesion or m.opt.noslip_iterations
            or m.opt.enableflags & types.EnableBit.SLEEP or m.jnt_limited_ball_adr.size
            or m.jnt_limited_slide_hinge_adr.size or np.any(m.dof_frictionloss.numpy() != 0)):
            raise ValueError('requires CUDA sparse pyramidal Newton, 1..512 DOFs, contact-only; '
                             'no sleeping, adhesion or noslip iterations')

    def _validate_data(self, d):
        if (d.nworld != self.nworld or d.njmax != self.njmax or d.nworld < 1 or d.njmax < 1
            or d.qpos.device != self.device or d.efc.J.shape[1] != 1
            or d.efc.J.shape != d.efc.J_colind.shape
            or d.efc.J_rownnz.shape != (self.nworld,self.njmax)
            or d.efc.J_rowadr.shape != (self.nworld,self.njmax)):
            raise ValueError('data dimensions/device changed; rebuild assembly workspace')
        for name in ('type','D','state','force','J_rownnz','J_rowadr'):
            a = getattr(d.efc,name)
            dtype = wp.float32 if name in ('D','force') else wp.int32
            if (a.shape != (self.nworld,self.njmax) or a.dtype != dtype
                or a.device != self.device or not a.is_contiguous):
                raise ValueError('invalid constraint array ' + name)
        for name,dtype in (('J',wp.float32),('J_colind',wp.int32)):
            a = getattr(d.efc,name)
            if a.shape[0] != self.nworld or a.dtype != dtype or a.device != self.device or not a.is_contiguous:
                raise ValueError('invalid packed Jacobian array ' + name)
        for a in (d.nefc,d.ne,d.nf,d.nl):
            if a.shape != (self.nworld,) or a.dtype != wp.int32 or a.device != self.device:
                raise ValueError('invalid constraint count array')
        if (d.qfrc_constraint.shape != (self.nworld,self.nv) or d.qfrc_constraint.dtype != wp.float32
            or d.qfrc_constraint.device != self.device):
            raise ValueError('invalid generalized force array')

    def _validate_context(self, ctx):
        if (ctx.done.shape != (self.nworld,) or ctx.done.dtype != wp.bool or ctx.done.device != self.device
            or ctx.h.ndim != 3 or ctx.h.shape[0] != self.nworld
            or min(ctx.h.shape[1:]) < self.nv or ctx.h.dtype != wp.float32
            or ctx.h.device != self.device or not ctx.h.is_contiguous):
            raise ValueError('invalid Newton assembly context')

    def prepare(self, d):
        """Rebuild the owned transpose after constraint assembly/order, once per solve."""
        self._validate_data(d)
        self.dense.zero_()
        wp.launch(_unpack, dim=(self.nworld,self.njmax), inputs=[self.nv,d.nefc,d.ne,d.nf,d.nl,
            d.efc.type,d.efc.J_rownnz,d.efc.J_rowadr,d.efc.J_colind,d.efc.J,self.dense,self.status], device=self.device)
        wp.launch(_incidence, dim=(self.nworld,self.nv), inputs=[d.nefc,self.dense,self.rows,self.counts], device=self.device)

    def _assemble(self, d, ctx, incremental):
        self._validate_data(d)
        self._validate_context(ctx)
        wp.launch(_weights, dim=self.weights.shape, inputs=[incremental,d.nefc,ctx.done,d.efc.D,
            d.efc.state,self.previous,self.weights,self.status], device=self.device)
        wp.launch(_hessian, dim=(self.nworld,self.nv,self.nv), inputs=[self.nv,ctx.done,self.dense,
            self.rows,self.counts,self.weights,ctx.h,self.status], device=self.device)

    def full(self, d, ctx):
        """Add J^T diag(D * quadratic) J to the host-initialized upper triangle."""
        self._assemble(d, ctx, False)

    def incremental(self, d, ctx):
        """Apply the current minus previous quadratic mask, in row order."""
        self._assemble(d, ctx, True)

    def force(self, d, ctx, changed):
        self._validate_data(d)
        self._validate_context(ctx)
        if changed.shape != (self.nworld,) or changed.dtype != wp.int32 or changed.device != self.device:
            raise ValueError('invalid state-change counts')
        wp.launch(_force, dim=(self.nworld,self.nv), inputs=[ctx.done,changed,self.dense,self.rows,
            self.counts,d.efc.force,d.qfrc_constraint,self.status], device=self.device)

    @contextmanager
    def capture_policy(self):
        from mujoco_warp._src import solver
        self._validate_model()
        if self.model.nv != self.nv or self.model.body_mass.device != self.device:
            raise ValueError('model dimensions/device changed; rebuild assembly workspace')
        with _LOCK:
            if hasattr(solver, _GLOBAL):
                raise RuntimeError('nested owned assembly policy')
            originals = {name: getattr(solver,name) for name in _HASHES}
            if self.solver_ops:
                self.solver_ops.validate_setup()
                configured = {name: _configured(fn,name,True) for name,fn in originals.items()}
            else:
                configured = {name: _configured(fn,name) for name,fn in originals.items()}
            solving = False
            def dispatch(name):
                def call(m, d, ctx, *args, **kwargs):
                    nonlocal solving
                    if m is not self.model:
                        return originals[name](m,d,ctx,*args,**kwargs)
                    if name == '_solve':
                        if solving:
                            raise RuntimeError('reentrant owned assembly solve')
                        self._validate_data(d)
                        self.prepare(d)
                        solving = True
                        try:
                            return configured[name](m,d,ctx,*args,**kwargs)
                        finally:
                            solving = False
                    # Inverse dynamics and standalone context diagnostics also
                    # call these functions; no prepared solve exists for them.
                    fn = configured[name] if solving else originals[name]
                    return fn(m,d,ctx,*args,**kwargs)
                return call
            setattr(solver, _GLOBAL, self)
            try:
                for name in originals:
                    setattr(solver, name, dispatch(name))
                yield self
            finally:
                for name, original in originals.items():
                    setattr(solver, name, original)
                delattr(solver, _GLOBAL)
