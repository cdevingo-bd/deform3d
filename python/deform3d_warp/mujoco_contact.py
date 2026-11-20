"""Explicit MJWarp 4.0.1 contact policy for graph construction.

MuJoCo CPU retains coincident contacts contributed by different flex elements.
MJWarp's 1 mm proximity filter removes them, changing compliant force response.
This opt-in policy preserves their multiplicity and replaces the many-launch FPS
implementation with a single kernel. It does not change materials or solvers.

Use only while warming/capturing the selected model, then replay the graph after
leaving the context. Installed files are never modified. Calls for other models
keep the original behavior. A version/source guard rejects unreviewed internals.
"""
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import inspect
import threading
from types import FunctionType

import warp as wp

_LOCK = threading.RLock()
_FILTER_SHA = 'bbde9127735ee5820aab0b7b971a5f61181815a097013a0689526df92f2e2388'
_FPS_SHA = '2f77ee0d7377c2210b60c7796e2c28d4132cf9a0acc7e7d1bcc32593a3d04df2'


@wp.func
def _earlier(a: int, b: int, elements: wp.array(dtype=wp.vec2i)):
    if b < 0:
        return True
    ea, eb = elements[a], elements[b]
    if ea[0] != eb[0]:
        return ea[0] < eb[0]
    if ea[1] != eb[1]:
        return ea[1] < eb[1]
    return a < b


@wp.kernel(enable_backward=False)
def _fused_farthest_points(
    group_starts: wp.array(dtype=wp.int32),
    group_count: wp.array(dtype=wp.int32),
    candidate_count: wp.array(dtype=wp.int32),
    active_sorted: wp.array(dtype=wp.int32),
    order: wp.array(dtype=wp.int32),
    depth: wp.array(dtype=wp.float32),
    position: wp.array(dtype=wp.vec3f),
    elements: wp.array(dtype=wp.vec2i),
    geoms: wp.array(dtype=wp.vec2i),
    limit: int,
    nearest: wp.array(dtype=wp.float32),
    active: wp.array(dtype=wp.int32),
):
    # One worker owns a disjoint group and its scratch. Groups with <= limit
    # active candidates require only counting, with no selection or output writes.
    g = wp.tid()
    if g >= group_count[0] or candidate_count[0] <= limit:
        return
    count = wp.min(candidate_count[0], active_sorted.shape[0])
    begin = group_starts[g]
    if begin < 0 or begin >= count:
        return
    if geoms[order[begin]][0] >= 0:
        return
    end = count
    if g + 1 < group_count[0] and g + 1 < group_starts.shape[0]:
        end = wp.min(end, group_starts[g + 1])
    total = int(0)
    seed = int(-1)
    seed_depth = float(1.0e10)
    for s in range(begin, end):
        if active_sorted[s] == 1:
            total += 1
            c = order[s]
            z = depth[c]
            if z < seed_depth or (z == seed_depth and _earlier(c, seed, elements)):
                seed, seed_depth = c, z
    if total <= limit or seed < 0:
        return
    for s in range(begin, end):
        if active_sorted[s] == 1:
            c = order[s]
            if c == seed:
                active[c] = 1
                nearest[c] = -1.0e10
            else:
                active[c] = 0
                nearest[c] = wp.length(position[c] - position[seed])
    for selection in range(1, limit):
        best = int(-1)
        distance = float(-1.0e10)
        for s in range(begin, end):
            if active_sorted[s] == 1:
                c = order[s]
                d = nearest[c]
                if d > distance or (d == distance and _earlier(c, best, elements)):
                    best, distance = c, d
        if best < 0 or distance <= 0.0:
            break
        active[best] = 1
        nearest[best] = -1.0e10
        for s in range(begin, end):
            if active_sorted[s] == 1:
                c = order[s]
                if nearest[c] > 0.0:
                    nearest[c] = wp.min(nearest[c], wp.length(position[c] - position[best]))


def _launch_fused(model, data, workspace, group_capacity):
    wp.launch(_fused_farthest_points, dim=group_capacity,
              inputs=[workspace.flex_group_start_indices, workspace.flex_num_groups,
                      workspace.ncand, workspace.cand_active_sorted, workspace.filter_val,
                      workspace.dist, workspace.pos, workspace.elem, workspace.geom, 50],
              outputs=[workspace.flex_fps_min_dist, workspace.cand_active])


@contextmanager
def flex_contact_policy(model, *, preserve_multiplicity=True, fused_fps=True):
    """Select contact behavior for one model during warmup or graph construction.

The flags are independent: fused_fps preserves MJWarp's greedy selection and
tie-breaking; preserve_multiplicity changes its proximity filter to match CPU
MuJoCo's retention of per-element constraints. Neither guarantees trajectory
parity for every scene. Graph replay needs no Python hook or synchronization.

Enter setup contexts serially; do not overlap context entry/exit across threads.
Uncaptured stepping requires keeping the context active. Nested policies are
rejected so the selected configuration remains unambiguous.
"""
    from mujoco_warp._src import collision_flex as flex
    if version('mujoco-warp') != '4.0.1':
        raise RuntimeError('flex_contact_policy has been reviewed only for mujoco-warp 4.0.1')
    with _LOCK:
        original = flex._filter_and_write_contacts
        for function, expected in ((original, _FILTER_SHA), (flex._run_filter_flex_fps, _FPS_SHA)):
            if hashlib.sha256(inspect.getsource(function).encode()).hexdigest() != expected:
                raise RuntimeError('unsupported MJWarp contact implementation or nested contact policy')
        if flex.MJ_MAXCONPAIR != 50:
            raise RuntimeError('unexpected MJWarp contact limit')
        namespace = dict(original.__globals__)
        if preserve_multiplicity:
            namespace['_FLEX_CONTACT_DEDUP_TOLERANCE'] = 0.0
        if fused_fps:
            namespace['_run_filter_flex_fps'] = _launch_fused
        configured = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)

        def dispatch(m, d, ws, enable_fps=False):
            return (configured if m is model else original)(m, d, ws, enable_fps)

        flex._filter_and_write_contacts = dispatch
        try:
            yield
        finally:
            flex._filter_and_write_contacts = original
