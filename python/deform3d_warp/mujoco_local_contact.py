"""Opt-in local coordinates for MJWarp 4.0.1 flex-pair GJK/EPA.

The upstream algorithm, candidate pairs, radii and tolerance are preserved.
Each pair's private scratch vertices are translated to its first vertex before
GJK/EPA. Contact normals are formed from local witnesses before translating only
the midpoint back to world coordinates. This avoids subtracting nearby world
witnesses after their float32 rounding. It cannot recover already rounded input.
"""
from contextlib import contextmanager
from functools import lru_cache
import hashlib
from importlib.metadata import version
import inspect
import linecache
import threading
from types import FunctionType

_LOCK = threading.RLock()
_KERNEL_SHA = '18b38ffe203debb815130815cd841bc0fb4278b145ca044ed1ea16fd7a011747'
_LAUNCH_SHA = '5a986f2df5e6c4839f876020d285e810eaceaa3e4025d395cd1cba45a6880317'


def _translate_source(source):
    if hashlib.sha256(source.encode()).hexdigest() != _KERNEL_SHA:
        raise RuntimeError('unreviewed MJWarp flex narrowphase source')
    replacements = (
        ('      # GJK/EPA narrowphase\n',
         '      # GJK/EPA narrowphase in coordinates local to this pair.\n'
         '      origin = workspace_verts_out[offset1]\n'
         '      for idx in range(dim1 + 1):\n'
         '        workspace_verts_out[offset1 + idx] = workspace_verts_out[offset1 + idx] - origin\n'),
        ('        workspace_verts_out[offset2 + idx] = flexvert_xpos_in[worldid, vert_adr2 + v2_indices[idx]]\n',
         '        workspace_verts_out[offset2 + idx] = flexvert_xpos_in[worldid, vert_adr2 + v2_indices[idx]] - origin\n'),
        ('        pos = 0.5 * (w1 + w2)\n', '        pos = origin + 0.5 * (w1 + w2)\n'))
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError('unexpected flex narrowphase transform site')
        source = source.replace(old, new, 1)
    return source


@lru_cache(None)
def _configured_factory(original):
    source = _translate_source(inspect.getsource(original))
    # MJWarp caches by module/name/arguments rather than function identity.
    # A transformed factory sharing the native name can silently reuse the
    # native kernel, or leak the transformed kernel into later native calls.
    name = '_flex_local_'+hashlib.sha256(source.encode()).hexdigest()[:16]
    source = source.replace('def _flex_narrowphase(', 'def '+name+'(', 1)
    # Warp needs inspectable Python source for kernel code generation. Keep the
    # transformed function in linecache without writing dependency files.
    filename = '<deform3d_local_flex_' + hashlib.sha256(source.encode()).hexdigest() + '>'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(inspect.unwrap(original).__globals__)
    exec(compile(source, filename, 'exec'), namespace)
    return namespace[name]


@contextmanager
def local_flex_contact_policy(model):
    """Use local flex-pair witnesses for this model during warmup/capture.

    Serialize setup; do not nest contexts. The context restores the original
    launcher even on error. Replay has no hook or host synchronization. This
    changes rounding, so full trajectory and force validation remains required.
    Rigid-flex and the analytic rod-pair branches remain upstream operations.
    """
    from mujoco_warp._src import collision_flex as flex
    if version('mujoco-warp') != '4.0.1':
        raise RuntimeError('reviewed only for mujoco-warp 4.0.1')
    with _LOCK:
        original = flex._run_flex_narrowphase
        if hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != _LAUNCH_SHA:
            raise RuntimeError('unreviewed flex launcher or nested local contact policy')
        namespace = dict(original.__globals__)
        namespace['_flex_narrowphase'] = _configured_factory(flex._flex_narrowphase)
        configured = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)

        def dispatch(m, d, ctx, ws):
            return (configured if m is model else original)(m, d, ctx, ws)

        flex._run_flex_narrowphase = dispatch
        try:
            yield
        finally:
            flex._run_flex_narrowphase = original
