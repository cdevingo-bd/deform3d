"""Scoped scratch sizing for isolated MJWarp 4.0.1 validation processes.

This changes only the host allocation width of EPA horizon arrays. It does not
patch collision kernels, iteration budgets, tolerances or overflow handling.
Enter before model warmup/capture. Captured graphs retain their allocated arrays.
The two upstream allocator constants are process-global: this tool is for the
single-threaded harness, not concurrent setup of unrelated models in an engine.
"""
from contextlib import contextmanager
from importlib.metadata import version
import numbers
import threading

_LOCK=threading.RLock()
_ACTIVE=False


@contextmanager
def epa_horizon_capacity(capacity):
    global _ACTIVE
    if isinstance(capacity,bool) or not isinstance(capacity,numbers.Integral) or not 24<=capacity<=4096:
        raise ValueError('EPA horizon capacity must be an integer from 24 through 4096')
    if version('mujoco-warp')!='4.0.1':
        raise RuntimeError('EPA allocation sizing reviewed only for MJWarp 4.0.1')
    from mujoco_warp._src import collision_flex,collision_convex
    with _LOCK:
        if _ACTIVE:raise RuntimeError('nested EPA capacity scope')
        original=[module.MJ_MAX_EPAHORIZON for module in (collision_flex,collision_convex)]
        if original!=[24,24]:raise RuntimeError('unreviewed or modified EPA allocator capacity')
        _ACTIVE=True
        try:
            collision_flex.MJ_MAX_EPAHORIZON=capacity
            collision_convex.MJ_MAX_EPAHORIZON=capacity
            yield
        finally:
            collision_flex.MJ_MAX_EPAHORIZON,collision_convex.MJ_MAX_EPAHORIZON=original
            _ACTIVE=False
