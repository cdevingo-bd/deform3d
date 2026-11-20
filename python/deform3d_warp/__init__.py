"""Optional Warp transport for deform3d's engine-independent geometry bindings.

Setup uploads immutable binding/topology arrays. Frame functions only enqueue
kernels into the supplied stream (or Warp's current stream); output and face
scratch belong to the caller. Warm up the kernels before CUDA graph capture.

Positions are ``wp.array2d(dtype=wp.vec3f)`` with shape [world, vertex]. Each
world uses the same bindings/topology. Vertex and world offsets support views
inside larger engine buffers. The module never imports a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers

import numpy as np
import warp as wp

__all__ = ["DeviceBindings", "DeviceSurfaceBindings", "SurfaceTopology", "enqueue_embedding",
           "enqueue_surface_embedding", "enqueue_normals"]
_INT32_MAX = np.iinfo(np.int32).max
_FLOAT32_MAX = np.finfo(np.float32).max


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    value = int(value)
    if value > _INT32_MAX:
        raise ValueError(f"{name} exceeds the int32 device index range")
    return value


def _indices(values, width, vertex_count, name):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != width or values.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer array of shape (count, {width})")
    if values.size and (np.min(values) < 0 or np.max(values) >= vertex_count):
        raise ValueError(f"{name} contains an out-of-range vertex index")
    _integer(len(values), f"{name} count")
    return np.ascontiguousarray(values, dtype=np.int32)


@dataclass(frozen=True, init=False)
class DeviceBindings:
    """Upload validated local tetra indices and float32 barycentric weights.

    Signed weights are allowed for intentional extrapolation. Their sum must
    be one within 2e-5, matching deform3d's exported binding format. Setup is a
    host operation; retain the object for all subsequent frames. Do not mutate
    its device arrays while it is in use.
    """

    indices: wp.array
    weights: wp.array
    source_vertex_count: int
    count: int
    device: object

    def __init__(self, indices, weights, *, source_vertex_count, device="cpu"):
        vertex_count = _integer(source_vertex_count, "source_vertex_count")
        ids = _indices(indices, 4, vertex_count, "indices")
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape != ids.shape or not np.isfinite(weights).all():
            raise ValueError("weights must be finite and match indices shape")
        if not np.allclose(weights.sum(axis=1, dtype=np.float64), 1.0, rtol=0.0, atol=2e-5):
            raise ValueError("barycentric weights must sum to one")
        device = wp.get_device(device)
        object.__setattr__(self, "indices", wp.array(ids, dtype=wp.vec4i, device=device))
        object.__setattr__(self, "weights", wp.array(np.ascontiguousarray(weights), dtype=wp.vec4f, device=device))
        object.__setattr__(self, "source_vertex_count", vertex_count)
        object.__setattr__(self, "count", len(ids))
        object.__setattr__(self, "device", device)


@dataclass(frozen=True, init=False)
class DeviceSurfaceBindings:
    """Upload triangle bindings with signed offsets in world length units.

    Matches C++ SurfaceEmbedding<float>::Binding vertices/weights/normal_offset.
    Offsets follow the current triangle normal and retain their magnitude under
    stretching; they are not affine material coordinates through thickness.
    """

    indices: wp.array
    weights: wp.array
    normal_offsets: wp.array
    source_vertex_count: int
    count: int
    device: object

    def __init__(self, indices, weights, normal_offsets=None, *, source_vertex_count, device="cpu"):
        vertex_count = _integer(source_vertex_count, "source_vertex_count")
        ids = _indices(indices, 3, vertex_count, "indices")
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape != ids.shape or not np.isfinite(weights).all():
            raise ValueError("weights must be finite and match indices shape")
        if not np.allclose(weights.sum(axis=1, dtype=np.float64), 1.0, rtol=0.0, atol=2e-5):
            raise ValueError("barycentric weights must sum to one")
        offsets = np.zeros(len(ids), np.float32) if normal_offsets is None else np.asarray(normal_offsets, np.float32)
        if offsets.shape != (len(ids),) or not np.isfinite(offsets).all():
            raise ValueError("normal_offsets must be finite with one value per binding")
        device = wp.get_device(device)
        object.__setattr__(self, "indices", wp.array(ids, dtype=wp.vec3i, device=device))
        object.__setattr__(self, "weights", wp.array(np.ascontiguousarray(weights), dtype=wp.vec3f, device=device))
        object.__setattr__(self, "normal_offsets", wp.array(np.ascontiguousarray(offsets), dtype=wp.float32, device=device))
        object.__setattr__(self, "source_vertex_count", vertex_count)
        object.__setattr__(self, "count", len(ids))
        object.__setattr__(self, "device", device)


@dataclass(frozen=True, init=False)
class SurfaceTopology:
    """Upload oriented triangles and deterministic vertex-to-face CSR links.

    Incident faces are stored in input face order. Normals use area-weighted
    face cross products and one thread per vertex, without floating-point
    atomics. Vertices on creases must be split by the caller when hard normals
    are wanted. Isolated vertices and cancelling/degenerate fans yield zero.
    """

    triangles: wp.array
    offsets: wp.array
    incident_faces: wp.array
    vertex_count: int
    face_count: int
    device: object

    def __init__(self, triangles, *, vertex_count, device="cpu"):
        vertex_count = _integer(vertex_count, "vertex_count")
        triangles = _indices(triangles, 3, vertex_count, "triangles")
        _integer(triangles.size, "triangle corner count")
        corners = triangles.reshape(-1)
        # Stable sorting preserves face order within each vertex's fan.
        order = np.argsort(corners, kind="stable")
        faces = np.repeat(np.arange(len(triangles), dtype=np.int32), 3)[order]
        counts = np.bincount(corners, minlength=vertex_count)
        offsets = np.empty(vertex_count + 1, dtype=np.int32)
        offsets[0] = 0
        offsets[1:] = np.cumsum(counts, dtype=np.int64)
        device = wp.get_device(device)
        object.__setattr__(self, "triangles", wp.array(triangles, dtype=wp.vec3i, device=device))
        object.__setattr__(self, "offsets", wp.array(offsets, dtype=wp.int32, device=device))
        object.__setattr__(self, "incident_faces", wp.array(faces, dtype=wp.int32, device=device))
        object.__setattr__(self, "vertex_count", vertex_count)
        object.__setattr__(self, "face_count", len(triangles))
        object.__setattr__(self, "device", device)


@wp.kernel
def _tetra_gather(source: wp.array2d(dtype=wp.vec3f),
                 indices: wp.array(dtype=wp.vec4i), weights: wp.array(dtype=wp.vec4f),
                 source_vertex_offset: int, source_world_offset: int,
                 output_vertex_offset: int, output_world_offset: int,
                 output: wp.array2d(dtype=wp.vec3f)):
    world, vertex = wp.tid()
    ids = indices[vertex]
    w = weights[vertex]
    sw = world + source_world_offset
    output[world + output_world_offset, vertex + output_vertex_offset] = (
        source[sw, ids[0] + source_vertex_offset] * w[0]
        + source[sw, ids[1] + source_vertex_offset] * w[1]
        + source[sw, ids[2] + source_vertex_offset] * w[2]
        + source[sw, ids[3] + source_vertex_offset] * w[3])


@wp.func
def _finite_vec3(value: wp.vec3f):
    return wp.isfinite(value[0]) and wp.isfinite(value[1]) and wp.isfinite(value[2])


@wp.func
def _max_component(value: wp.vec3f):
    return wp.max(wp.abs(value[0]), wp.max(wp.abs(value[1]), wp.abs(value[2])))


@wp.kernel
def _surface_gather(source: wp.array2d(dtype=wp.vec3f),
                    indices: wp.array(dtype=wp.vec3i), weights: wp.array(dtype=wp.vec3f),
                    normal_offsets: wp.array(dtype=wp.float32),
                    source_vertex_offset: int, source_world_offset: int,
                    output_vertex_offset: int, output_world_offset: int,
                    output: wp.array2d(dtype=wp.vec3f), validity: wp.array2d(dtype=wp.int32)):
    world, vertex = wp.tid()
    sw, ow, ov = world + source_world_offset, world + output_world_offset, vertex + output_vertex_offset
    validity[ow, ov] = 0
    ids = indices[vertex]
    a = source[sw, ids[0] + source_vertex_offset]
    b = source[sw, ids[1] + source_vertex_offset]
    c = source[sw, ids[2] + source_vertex_offset]
    if not (_finite_vec3(a) and _finite_vec3(b) and _finite_vec3(c)):
        return
    ab, ac, bc = b - a, c - a, c - b
    if not (_finite_vec3(ab) and _finite_vec3(ac) and _finite_vec3(bc)):
        return
    scale = wp.max(_max_component(ab), wp.max(_max_component(ac), _max_component(bc)))
    if scale <= 0.0:
        return
    ab_scaled, ac_scaled, bc_scaled = ab / scale, ac / scale, bc / scale
    edge = wp.max(wp.length(ab_scaled), wp.max(wp.length(ac_scaled), wp.length(bc_scaled)))
    normal = wp.cross(ab_scaled / edge, ac_scaled / edge)
    magnitude = wp.length(normal)
    # The float32 C++ binding policy uses 64 * numeric_limits<float>::epsilon().
    if magnitude <= 7.62939453125e-6:
        return
    w = weights[vertex]
    result = a * w[0] + b * w[1] + c * w[2] + (normal / magnitude) * normal_offsets[vertex]
    if not _finite_vec3(result):
        return
    output[ow, ov] = result
    validity[ow, ov] = 1


@wp.kernel
def _face_cross(positions: wp.array2d(dtype=wp.vec3f),
                triangles: wp.array(dtype=wp.vec3i),
                source_vertex_offset: int, source_world_offset: int,
                face_normals: wp.array2d(dtype=wp.vec3f)):
    world, face = wp.tid()
    ids = triangles[face]
    sw = world + source_world_offset
    a = positions[sw, ids[0] + source_vertex_offset]
    b = positions[sw, ids[1] + source_vertex_offset]
    c = positions[sw, ids[2] + source_vertex_offset]
    face_normals[world, face] = wp.cross(b - a, c - a)


@wp.kernel
def _vertex_normals(face_normals: wp.array2d(dtype=wp.vec3f),
                    offsets: wp.array(dtype=wp.int32),
                    incident_faces: wp.array(dtype=wp.int32),
                    output_vertex_offset: int, output_world_offset: int,
                    epsilon: float, output: wp.array2d(dtype=wp.vec3f)):
    world, vertex = wp.tid()
    total = wp.vec3f(0.0)
    for cursor in range(offsets[vertex], offsets[vertex + 1]):
        total = total + face_normals[world, incident_faces[cursor]]
    # Scaling first avoids overflow/underflow in the norm of a finite sum.
    scale = wp.max(wp.abs(total[0]), wp.max(wp.abs(total[1]), wp.abs(total[2])))
    normal = wp.vec3f(0.0)
    if scale > 0.0:
        scaled = total / scale
        length = wp.length(scaled)
        if scale > epsilon / length:
            normal = scaled / length
    output[world + output_world_offset, vertex + output_vertex_offset] = normal


def _buffer(array, name, device, dtype=wp.vec3f):
    if not isinstance(array, wp.array) or array.dtype != dtype or array.ndim != 2:
        raise ValueError(f"{name} must be a 2D Warp {dtype.__name__} array")
    if array.device != device:
        raise ValueError(f"{name} is on a different device")
    if array.size and (array.ptr % 4 or any(s % 4 for s, n in zip(array.strides, array.shape) if n > 1)):
        raise ValueError(f"{name} pointer and used strides must be aligned to float32 (4 bytes)")
    if array.size and any(s > _INT32_MAX for s, n in zip(array.strides, array.shape) if n > 1):
        raise ValueError(f"{name} used strides must fit Warp's int32 stride ABI")
    if array.size and any(stride <= 0 for stride in array.strides):
        raise ValueError(f"{name} must have positive strides")
    # Reject user-constructed views that alias their own elements, which
    # would otherwise race when a kernel writes different logical vertices.
    span = 12 if dtype == wp.vec3f else 4
    for stride, extent in sorted((s, n) for s, n in zip(array.strides, array.shape) if n > 1):
        if array.size and stride < span:
            raise ValueError(f"{name} has overlapping elements")
        span += (extent - 1) * stride


def _span(array):
    # Conservative bounding span also catches overlapping strided views.
    if not array.size:
        return 0, 0
    item_size = 12 if array.dtype == wp.vec3f else 4
    size = sum((n - 1) * s for n, s in zip(array.shape, array.strides)) + item_size
    return array.ptr, array.ptr + size


def _no_overlap(a, b, message):
    alo, ahi = _span(a)
    blo, bhi = _span(b)
    if alo < bhi and blo < ahi:
        raise ValueError(message)


def _window(array, name, world_offset, world_count, vertex_offset, vertex_count):
    if world_offset + world_count > array.shape[0] or vertex_offset + vertex_count > array.shape[1]:
        raise ValueError(f"{name} is too small for the requested world/vertex window")


def _launch_window(source, output, device, world_count, source_world_offset,
                   output_world_offset, source_vertex_offset, output_vertex_offset, stream):
    _buffer(source, "source", device)
    _buffer(output, "output", device)
    sw = _integer(source_world_offset, "source_world_offset")
    ow = _integer(output_world_offset, "output_world_offset")
    sv = _integer(source_vertex_offset, "source_vertex_offset")
    ov = _integer(output_vertex_offset, "output_vertex_offset")
    worlds = _integer(source.shape[0] - sw if world_count is None else world_count, "world_count")
    if stream is not None and stream.device != device:
        raise ValueError("stream is on a different device")
    _no_overlap(source, output, "source and output buffers must not overlap")
    return worlds, sw, ow, sv, ov


def enqueue_embedding(source, bindings: DeviceBindings, output, *, world_count=None,
                      source_vertex_offset=0, output_vertex_offset=0,
                      source_world_offset=0, output_world_offset=0, stream=None):
    """Enqueue dense tetra gathering; no device allocations, copies, or synchronization.

    Coordinates must be finite float32 values. Source/output memory must not
    overlap. Batched worlds share the same local tetra binding indices. Setup
    validates binding indices; frame calls check only host-visible metadata.
    """
    worlds, sw, ow, sv, ov = _launch_window(
        source, output, bindings.device, world_count, source_world_offset,
        output_world_offset, source_vertex_offset, output_vertex_offset, stream)
    _window(source, "source", sw, worlds, sv, bindings.source_vertex_count)
    _window(output, "output", ow, worlds, ov, bindings.count)
    if worlds and bindings.count:
        wp.launch(_tetra_gather, dim=(worlds, bindings.count),
                  inputs=[source, bindings.indices, bindings.weights, sv, sw, ov, ow],
                  outputs=[output], device=bindings.device, stream=stream)


def enqueue_surface_embedding(source, bindings: DeviceSurfaceBindings, output, validity, *, world_count=None,
                              source_vertex_offset=0, output_vertex_offset=0,
                              source_world_offset=0, output_world_offset=0, stream=None):
    """Enqueue triangle gathering with current-normal offsets and device validity.

    Caller-owned ``validity`` is a 2D int32 array using the output's world and
    vertex offsets. Each evaluated entry becomes 1 on success or 0 for invalid
    geometry/arithmetic. Invalid entries leave the corresponding output intact.
    Degenerate triangles fail even when normal_offset is zero. Unlike C++'s
    whole-call transaction, failure is per binding/world and remains on device.
    No device allocation, copy, or host synchronization occurs.
    """
    worlds, sw, ow, sv, ov = _launch_window(
        source, output, bindings.device, world_count, source_world_offset,
        output_world_offset, source_vertex_offset, output_vertex_offset, stream)
    _buffer(validity, "validity", bindings.device, dtype=wp.int32)
    _window(source, "source", sw, worlds, sv, bindings.source_vertex_count)
    _window(output, "output", ow, worlds, ov, bindings.count)
    _window(validity, "validity", ow, worlds, ov, bindings.count)
    _no_overlap(source, validity, "source and validity buffers must not overlap")
    _no_overlap(output, validity, "output and validity buffers must not overlap")
    if worlds and bindings.count:
        wp.launch(_surface_gather, dim=(worlds, bindings.count),
                  inputs=[source, bindings.indices, bindings.weights, bindings.normal_offsets, sv, sw, ov, ow],
                  outputs=[output, validity], device=bindings.device, stream=stream)


def enqueue_normals(positions, topology: SurfaceTopology, face_scratch, output, *,
                    world_count=None, source_vertex_offset=0, output_vertex_offset=0,
                    source_world_offset=0, output_world_offset=0,
                    epsilon=1e-20, stream=None):
    """Enqueue area-weighted smooth normals through two deterministic kernels.

    ``face_scratch`` must be a separate vec3f array with shape at least
    ``(world_count, topology.face_count)``. Scratch starts at [0, 0] regardless
    of source/output offsets. The sum of adjacent face crosses is normalized
    only when its length exceeds ``epsilon`` (units: position units squared).
    Otherwise the output is exactly zero. No host readback is performed.
    Cross products and their sums must remain finite float32 values.
    """
    worlds, sw, ow, sv, ov = _launch_window(
        positions, output, topology.device, world_count, source_world_offset,
        output_world_offset, source_vertex_offset, output_vertex_offset, stream)
    _buffer(face_scratch, "face_scratch", topology.device)
    _window(positions, "positions", sw, worlds, sv, topology.vertex_count)
    _window(output, "output", ow, worlds, ov, topology.vertex_count)
    _window(face_scratch, "face_scratch", 0, worlds, 0, topology.face_count)
    _no_overlap(face_scratch, positions, "face_scratch and positions must not overlap")
    _no_overlap(face_scratch, output, "face_scratch and output must not overlap")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon < 0 or epsilon > _FLOAT32_MAX:
        raise ValueError("epsilon must be finite, nonnegative, and fit float32")
    if worlds and topology.vertex_count:
        if topology.face_count:
            wp.launch(_face_cross, dim=(worlds, topology.face_count),
                      inputs=[positions, topology.triangles, sv, sw], outputs=[face_scratch],
                      device=topology.device, stream=stream)
        wp.launch(_vertex_normals, dim=(worlds, topology.vertex_count),
                  inputs=[face_scratch, topology.offsets, topology.incident_faces, ov, ow, epsilon],
                  outputs=[output], device=topology.device, stream=stream)
