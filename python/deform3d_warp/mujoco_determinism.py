"""Topology-bounded scratch for MJWarp's deterministic sparse Newton assembly.

Only the number of reserved atomic records changes. The upstream kernels,
summation order and physics equations remain untouched. Unsupported constraint
families are rejected rather than guessed. Model topology must stay immutable.
"""
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import inspect
import linecache
import threading
from types import FunctionType

import numpy as np

_LOCK = threading.RLock()
_HASHES = {
    '_update_gradient': '54cea02cf18fd41dc78424d908f33516b705a3877c6cca6858f74f18b89abc82',
    '_update_gradient_incremental': 'cbc732e2fcd7f79938345d0ce356687f846950a9c4c95cb78491cbf0f8e17f59',
}
_KERNEL_HASHES = {
    '_JTDACJ_sparse': '27e3d830caede4825b808bbc16223dd3f45dffcfe387d828aef9837225bb8234',
    '_update_gradient_h_incremental_sparse': '2b8dac804a19f2a273c7a717306061a357a1010ab7b0b9bdd38da672ba6170cb',
}


def _support_bound(model):
    """Conservative row support from both sides' full ancestor DOF unions.

    A rigid contact uses one body per side. A vertex flex contact uses at most
    the bodies of one simplex per side. Each limit/friction-loss row has one
    DOF, except ball-joint limits (three). Pyramidal combinations and elliptic
    blocks share the same contact support union.
    """
    if model.nv < 1 or model.neq or model.ntendon or not model.is_sparse:
        raise ValueError('requires sparse Jacobians, no equality constraints and no tendons')
    arrays = {name: getattr(model, name).numpy() for name in (
        'body_parentid', 'body_dofadr', 'body_dofnum', 'geom_bodyid', 'flex_dim',
        'flex_interp', 'flex_vertadr', 'flex_vertnum', 'flex_vertbodyid',
        'flex_elemdataadr', 'flex_elemnum', 'flex_elem')}
    if np.any(arrays['flex_interp'] != 0) or np.any(~np.isin(arrays['flex_dim'], [2, 3])):
        raise ValueError('only vertex-based triangle and tetrahedral flexes are supported')
    bodies = []
    for body in range(model.nbody):
        parent = int(arrays['body_parentid'][body])
        if body and not 0 <= parent < body:
            raise ValueError('invalid body topology')
        start, count = int(arrays['body_dofadr'][body]), int(arrays['body_dofnum'][body])
        if count < 0 or (count and not 0 <= start <= model.nv - count):
            raise ValueError('invalid body DOFs')
        bodies.append((bodies[parent] if body else set()) | set(range(start, start + count)))
    def union_size(ids):
        ids = [int(i) for i in ids]
        if any(i < 0 or i >= len(bodies) for i in ids):
            raise ValueError('invalid flex or geometry body index')
        return len(set().union(*(bodies[i] for i in ids)))
    largest = max((union_size([b]) for b in arrays['geom_bodyid']), default=0)
    for flex in range(model.nflex):
        va, nv = int(arrays['flex_vertadr'][flex]), int(arrays['flex_vertnum'][flex])
        vertex_bodies = arrays['flex_vertbodyid'][va:va + nv]
        for body in vertex_bodies:
            largest = max(largest, union_size([body]))
        width = int(arrays['flex_dim'][flex]) + 1
        address, count = int(arrays['flex_elemdataadr'][flex]), int(arrays['flex_elemnum'][flex])
        elements = arrays['flex_elem'][address:address + count * width].reshape(-1, width)
        if np.any(elements < 0) or np.any(elements >= nv):
            raise ValueError('invalid flex element index')
        for element in elements:
            largest = max(largest, union_size(vertex_bodies[element]))
    return min(model.nv, max(3, 2 * largest))


def _configured(original, name, bound):
    source = inspect.getsource(original)
    if hashlib.sha256(source.encode()).hexdigest() != _HASHES[name]:
        raise RuntimeError('unreviewed or already patched MJWarp ' + name)
    old = '(mj.nv * (mj.nv + 1) // 2)'
    if source.count(old) != 1:
        raise RuntimeError('unexpected deterministic record budget site')
    # This changes only the conservative allocation bound. No launch dimension,
    # sparse row data, kernel or arithmetic expression is changed.
    source = source.replace(old, f'(min(mj.nv, {bound}) * (min(mj.nv, {bound}) + 1) // 2)')
    filename = '<deform3d_determinism_' + hashlib.sha256(source.encode()).hexdigest() + '>'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(original.__globals__)
    exec(compile(source, filename, 'exec'), namespace)
    # Share the real module globals so composing the Newton factorization hook
    # works in either context order; do not freeze references to old hooks.
    return FunctionType(namespace[name].__code__, original.__globals__, name, original.__defaults__)


class SparseDeterminismBudget:
    """Prepare outside capture; enable determinism separately on the model.

    Scopes both full and incremental sparse Hessian assembly to one model.
    It does not enable determinism or promise cross-device reproducibility.
    Recreate this object after any topology or constraint-family change, and
    recapture every graph. Serial setup is required; replay uses no Python hook.
    """
    def __init__(self, model):
        if version('mujoco-warp') != '4.0.1':
            raise RuntimeError('reviewed only for mujoco-warp 4.0.1')
        self.model = model
        self.max_row_dofs = _support_bound(model)

    @contextmanager
    def capture_policy(self):
        from mujoco_warp._src import solver
        with _LOCK:
            for name, digest in _KERNEL_HASHES.items():
                if hashlib.sha256(inspect.getsource(getattr(solver, name)).encode()).hexdigest() != digest:
                    raise RuntimeError('unreviewed deterministic sparse kernel ' + name)
            # Setup is allowed to inspect topology. Replay has no host readback.
            if _support_bound(self.model) > self.max_row_dofs:
                raise RuntimeError('model support grew; recreate the budget and recapture')
            originals = {name: getattr(solver, name) for name in _HASHES}
            configured = {name: _configured(fn, name, self.max_row_dofs) for name, fn in originals.items()}
            def dispatch(original, transformed):
                def call(m, *args, **kwargs):
                    return (transformed if m is self.model else original)(m, *args, **kwargs)
                return call
            try:
                for name, original in originals.items():
                    setattr(solver, name, dispatch(original, configured[name]))
                yield self
            finally:
                for name, original in originals.items():
                    setattr(solver, name, original)
