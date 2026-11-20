"""Owned effective-metric insertion and reproducible Newton gradient norm.

Internal extension of OwnedContactAssembly. The CSR topology is validated once;
each upper-triangle metric entry has exactly one writer. Gradient norms reuse
Warp's reviewed binned accumulator in one CUDA warp per world. Host active-set,
stable-gradient scaling, line search and convergence decisions are unchanged.
"""
import ast
import hashlib
import inspect

import numpy as np
import warp as wp

from ._warp_binned import Bins, add_record, rounded_sum, _DEFINITION

_HASHES = {
    '_update_gradient_grad': 'ddc637ad5ebba5c535c4462a59d2e1b10b8bb5ab3bbe001ee9fa4bd560df3d70',
    '_add_efmK_metric_dense': '79924d17c1e9b974bb6f027ca54df4dc35601387d85429290fc45fcf468fb621',
}
_GLOBAL = '_deform3d_owned_contact_assembly'


def _disjoint(reads, writes):
    for i, out in enumerate(writes):
        if not out.capacity:
            continue
        lo, hi = out.ptr, out.ptr + out.capacity
        for a in reads + writes[:i]:
            if a.capacity and lo < a.ptr + a.capacity and a.ptr < hi:
                raise ValueError('solver reduction outputs must not alias inputs or each other')


@wp.kernel(enable_backward=False)
def _metric(indices: wp.array[int], rows: wp.array[int], columns: wp.array[int],
            values: wp.array2d[float], done: wp.array[bool],
            h: wp.array3d[float], status: wp.array[int]):
    w, entry = wp.tid()
    if done[w]:
        return
    # Match the deterministic path even for a singleton record (subnormals,
    # signed zeros). The destination addition remains a separate rounding.
    total = add_record(Bins(), values[w, indices[entry]])
    row, col = rows[entry], columns[entry]
    value = h[w, row, col] + rounded_sum(total)
    if not wp.isfinite(value):
        wp.atomic_or(status, w, 2)
    h[w, row, col] = value


@wp.func_native(_DEFINITION + r'''
    #if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x;
    const float* ma = reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(mass_acceleration.data) + world*mass_acceleration.strides[0]);
    const float* smooth = reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(smooth_force.data) + world*smooth_force.strides[0]);
    const float* contact = reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(contact_force.data) + world*contact_force.strides[0]);
    float* grad = reinterpret_cast<float*>(
        reinterpret_cast<char*>(gradient.data) + world*gradient.strides[0]);
    BinnedFloatAccumulator sum;
    for (int i=lane; i<nv; i+=32) {
        const float value = (ma[i] - smooth[i]) - contact[i];
        grad[i] = value;
        BinnedFloatAccumulator record;
        record.add(value*value);
        sum.add(record);
    }
    // All 32 lanes execute every shuffle. Only valid tree edges merge.
    for (int offset=16; offset; offset>>=1) {
        BinnedFloatAccumulator other;
        #pragma unroll
        for (int i=0; i<3; ++i) {
            other.primary[i] = __shfl_down_sync(0xffffffffu, sum.primary[i], offset);
            other.carry[i] = __shfl_down_sync(0xffffffffu, sum.carry[i], offset);
        }
        if (lane < offset) sum.add(other);
    }
    if (lane==0) {
        const float value = norm.data[world] + sum.to_float();
        norm.data[world] = value;
        if (!isfinite(value)) atomicOr(status.data+world, 2);
    }
    #endif
''')
def _gradient_warp(world: int, nv: int, mass_acceleration: wp.array2d[float],
                   smooth_force: wp.array2d[float], contact_force: wp.array2d[float],
                   gradient: wp.array2d[float], norm: wp.array[float], status: wp.array[int]): ...


@wp.kernel(module='unique', enable_backward=False)
def _gradient(nv: int, stable: bool, changed: wp.array[int], done: wp.array[bool],
              ma: wp.array2d[float], smooth: wp.array2d[float], contact: wp.array2d[float],
              grad: wp.array2d[float], norm: wp.array[float], status: wp.array[int]):
    w, lane = wp.tid()
    if not done[w] and (not stable or changed[w] != 0):
        _gradient_warp(w, nv, ma, smooth, contact, grad, norm, status)


class _Launches(ast.NodeTransformer):
    def __init__(self):
        self.counts = dict.fromkeys(_HASHES, 0)

    def visit_Expr(self, node):
        node = self.generic_visit(node)
        call = node.value
        if not isinstance(call, ast.Call) or ast.unparse(call.func) != 'wp.launch' or not call.args:
            return node
        factory = call.args[0]
        if not isinstance(factory, ast.Call) or not isinstance(factory.func, ast.Name):
            return node
        name = factory.func.id
        if name not in self.counts:
            return node
        self.counts[name] += 1
        if name == '_add_efmK_metric_dense':
            replacement = f'{_GLOBAL}.solver_ops.add_metric(d, ctx)'
        else:
            stable = ast.unparse(factory.args[0])
            replacement = f'{_GLOBAL}.solver_ops.gradient(d, ctx, {stable})'
        return ast.parse(replacement).body[0]


def configure(source, name):
    """Called only after the assembly hook's exact upstream source guard."""
    tree = ast.parse(source)
    transform = _Launches()
    tree = ast.fix_missing_locations(transform.visit(tree))
    expected = {'_update_gradient_grad': 1,
                '_add_efmK_metric_dense': int(name == '_update_gradient')}
    if transform.counts != expected:
        raise RuntimeError('unexpected Newton reduction launch sites')
    return ast.unparse(tree) + '\n'


class OwnedSolverReductions:
    """Internal, same supported domain/lifetime as OwnedContactAssembly."""
    def __init__(self, model, data, status):
        self.model, self.nworld, self.status = model, data.nworld, status
        self.device, self.nv = model.body_mass.device, model.nv
        self.nefmK = model.nefmK
        if not self.device.is_cuda or not 1 <= self.nv <= 512 or self.nefmK < 0 or self.nworld < 1:
            raise ValueError('solver reductions require CUDA, 1..512 DOFs and positive world count')
        self._array(status, (self.nworld,), wp.int32)
        self.validate_source()
        rownnz, rowadr, colind = self._topology()
        if rownnz.shape != (self.nv,) or rowadr.shape != (self.nv,) or colind.shape != (self.nefmK,):
            raise ValueError('invalid effective-metric CSR dimensions')
        occupied = np.zeros(self.nefmK, bool)
        entries, rows, columns = [], [], []
        for row, (start, count) in enumerate(zip(rowadr, rownnz)):
            start, count = int(start), int(count)
            if start < 0 or count < 0 or start > self.nefmK-count or occupied[start:start+count].any():
                raise ValueError('overlapping or invalid effective-metric CSR rows')
            occupied[start:start+count] = True
            cols = colind[start:start+count]
            if np.any(cols < 0) or np.any(cols >= self.nv) or len(np.unique(cols)) != count:
                raise ValueError('duplicate or invalid effective-metric CSR columns')
            for k, col in enumerate(cols):
                if col >= row:
                    entries.append(start+k); rows.append(row); columns.append(col)
        self.signature = self._fingerprint()
        self.indices, self.rows, self.columns = [wp.array(np.asarray(a, np.int32), dtype=int, device=self.device)
                                                for a in (entries, rows, columns)]
        self.workspace_bytes = sum(a.capacity for a in (self.indices, self.rows, self.columns))

    @staticmethod
    def validate_source():
        from mujoco_warp._src import solver
        for name, digest in _HASHES.items():
            if hashlib.sha256(inspect.getsource(getattr(solver, name)).encode()).hexdigest() != digest:
                raise RuntimeError('unreviewed Newton reduction ' + name)

    def _topology(self):
        arrays = [getattr(self.model, 'efm_K_'+name) for name in ('rownnz', 'rowadr', 'colind')]
        for array, length in zip(arrays, (self.nv, self.nv, self.nefmK)):
            self._array(array, (length,), wp.int32)
        return tuple(array.numpy() for array in arrays)

    def _fingerprint(self):
        h = hashlib.sha256()
        for array in self._topology():
            h.update(str(array.shape).encode()); h.update(str(array.dtype).encode()); h.update(array.tobytes())
        return h.digest()

    def validate_setup(self):
        self.validate_source()
        if (self.model.nv != self.nv or self.model.nefmK != self.nefmK
            or self.model.body_mass.device != self.device or self._fingerprint() != self.signature):
            raise ValueError('effective-metric topology changed; rebuild solver reductions')

    def _array(self, a, shape, dtype):
        if a.shape != shape or a.dtype != dtype or a.device != self.device or not a.is_contiguous:
            raise ValueError('invalid solver reduction array')

    def add_metric(self, data, ctx):
        self._array(data.efm_K_val, (self.nworld, self.nefmK), wp.float32)
        self._array(ctx.done, (self.nworld,), wp.bool)
        if (ctx.h.ndim != 3 or ctx.h.shape[0] != self.nworld or min(ctx.h.shape[1:]) < self.nv
            or ctx.h.dtype != wp.float32 or ctx.h.device != self.device or not ctx.h.is_contiguous):
            raise ValueError('invalid metric insertion destination')
        _disjoint([data.efm_K_val,ctx.done],[ctx.h,self.status])
        if self.indices.size:
            wp.launch(_metric, (self.nworld, self.indices.size), inputs=[self.indices,self.rows,self.columns,
                data.efm_K_val,ctx.done,ctx.h,self.status], device=self.device)

    def gradient(self, data, ctx, stable=False):
        changed = ctx.state_changed_count if stable else data.nefc
        arrays = [data.efc.Ma, data.qfrc_smooth, data.qfrc_constraint, ctx.grad]
        for a in arrays:
            if a.ndim != 2 or a.shape[0] != self.nworld or a.shape[1] < self.nv:
                raise ValueError('invalid gradient dimensions')
            self._array(a, a.shape, wp.float32)
        self._array(ctx.grad_dot, (self.nworld,), wp.float32)
        self._array(changed, (self.nworld,), wp.int32)
        self._array(ctx.done, (self.nworld,), wp.bool)
        _disjoint(arrays[:3]+[changed,ctx.done],[ctx.grad,ctx.grad_dot,self.status])
        wp.launch_tiled(_gradient, dim=self.nworld, inputs=[self.nv,stable,changed,ctx.done,*arrays,
            ctx.grad_dot,self.status], block_dim=32, device=self.device)
