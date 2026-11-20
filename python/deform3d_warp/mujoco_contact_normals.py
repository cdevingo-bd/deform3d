"""Opt-in preservation of the GJK core normal for inflated flex contacts.

For separated polygonal cores whose radii overlap, upstream GJK computes a
normal before inflating the witnesses. Reconstructing that same direction from
the inflated world-space witnesses divides float32 cancellation error by the
penetration depth. This policy carries the pre-inflation normal to the contact.
It preserves candidates, distances, positions, material and solver equations.
Deep core overlap and analytic rod pairs retain upstream normals.

This changes rounding and therefore requires full trajectory validation. It is
not enabled by default. No dependency file is modified and graph replay has no
Python hook, host readback or additional allocation.
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
_GJK_SHA = '219c963a5bcc6e4f423b109fcdf6ec9fdec9b8416ee451bd7a3bec04e6ddc4af'
_CCD_SHA = '220bc61d817b5590c337090ade5813491a9c641414645b7969534942b2324cd2'
_FLEX_SHA = '18b38ffe203debb815130815cd841bc0fb4278b145ca044ed1ea16fd7a011747'
_LAUNCH_SHA = '5a986f2df5e6c4839f876020d285e810eaceaa3e4025d395cd1cba45a6880317'


def _replace(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('unexpected contact normal transform site')
    return source.replace(old, new, 1)


def _ccd_source(source):
    if hashlib.sha256(source.encode()).hexdigest() != _CCD_SHA:
        raise RuntimeError('unreviewed MJWarp CCD source')
    source = _replace(source, 'def ccd(', 'def _ccd_core_normal(')
    source = _replace(source, '-> Tuple[float, int, wp.vec3, wp.vec3, int]:',
                       '-> Tuple[float, int, wp.vec3, wp.vec3, int, wp.vec3]:')
    source = _replace(source, '    return dist, ncontact, x1, x2, idx\n',
        '    normal = wp.normalize(x1 - x2)\n'
        '    if dist < 0.0 and result.dist > tolerance and result.dist < FLOAT_MAX and result.dim > 0:\n'
        '      normal = wp.normalize(result.x2 - result.x1)\n'
        '    return dist, ncontact, x1, x2, idx, normal\n')
    source = _replace(source, '  return epa_phase(\n',
                       '  dist, ncontact, x1, x2, idx = epa_phase(\n')
    return source.rstrip()+'\n  return dist, ncontact, x1, x2, idx, wp.normalize(x1 - x2)\n'


def _flex_source(source):
    if hashlib.sha256(source.encode()).hexdigest() != _FLEX_SHA:
        raise RuntimeError('unreviewed MJWarp flex narrowphase source')
    source = _replace(source, '      dist, ncontact, w1, w2, _ = ccd(\n',
                       '      dist, ncontact, w1, w2, _, core_normal = _ccd_core_normal(\n')
    source = _replace(source, '        nrm = wp.normalize(w1 - w2)\n',
                       '        nrm = core_normal\n')
    # cache_kernel keys by factory name and arguments: prevent collision with
    # the native kernel or a different policy configured in this same process.
    name = '_flex_core_normal_'+hashlib.sha256(source.encode()).hexdigest()[:16]
    return _replace(source, 'def _flex_narrowphase(', 'def '+name+'('), name


def _compile(source, namespace, name):
    filename = '<deform3d_contact_normal_'+hashlib.sha256(source.encode()).hexdigest()+'>'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, 'exec'), namespace)
    return namespace[name]


@lru_cache(None)
def _configured_factory(original):
    from mujoco_warp._src import collision_gjk as gjk
    if hashlib.sha256(inspect.getsource(gjk).encode()).hexdigest() != _GJK_SHA:
        raise RuntimeError('unreviewed MJWarp GJK dependencies')
    source = _ccd_source(inspect.getsource(gjk.ccd.func))
    ccd = _compile(source, dict(gjk.ccd.func.__globals__), '_ccd_core_normal')
    source, name = _flex_source(inspect.getsource(original))
    namespace = dict(inspect.unwrap(original).__globals__)
    namespace['_ccd_core_normal'] = ccd
    return _compile(source, namespace, name)


@contextmanager
def stable_flex_normals_policy(model):
    """Scope the reviewed normal transport to one model's warmup/capture.

    Serialize context entry/exit. Nesting and combination with the experimental
    local-coordinate flex policy are rejected. Other models retain native
    behavior. Keep the model and captured state storage alive during replay.
    Only the mesh/mesh flex path calls the modified CCD function; the function
    must not be reused for heightfields or arbitrary convex shapes.
    """
    from mujoco_warp._src import collision_flex as flex
    if version('mujoco-warp') != '4.0.1':
        raise RuntimeError('stable flex normals reviewed only for mujoco-warp 4.0.1')
    with _LOCK:
        original = flex._run_flex_narrowphase
        if hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != _LAUNCH_SHA:
            raise RuntimeError('unreviewed flex launcher or nested contact normal policy')
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
