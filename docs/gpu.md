# Optional Warp device adapter

The C++ core remains a self-contained header-only library. The optional
`python/deform3d_warp` package adds GPU execution through NVIDIA Warp and NumPy.
It has no MuJoCo dependency. A simulator or engine supplies coarse world-space
positions; prepared tetra bindings drive dense positions, and surface topology
drives area-weighted vertex normals. This adapter does not solve material
mechanics or contact.

## Install and test

Install the root package into a virtual environment:

```sh
python -m pip install -e '.[test]'
python tests/test_warp_device.py --device cpu -v
python tests/test_warp_device.py --device cuda:0 -v
```

Tests cover multiple worlds, world/vertex offsets, affine covariance, signed
barycentric weights, area weighting, positive-stride views, invalid inputs,
overlapping buffers, degenerate faces, empty topology, and CUDA graph replay.
The CPU run skips the two CUDA-only graph tests. Use a Warp wheel supported by
your GPU and driver; older Pascal devices need a compatible CUDA 12 build.

## Minimal integration

```python
import numpy as np
import warp as wp
from deform3d_warp import (
    DeviceBindings, SurfaceTopology, enqueue_embedding, enqueue_normals,
)

wp.init()
device = wp.get_device("cuda:0")

# Immutable setup data. In production these come from C++ prepared bindings
# and the render mesh; negative weights are allowed for extrapolation.
ids = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]], np.int32)
weights = np.full((3, 4), 0.25, np.float32)
bindings = DeviceBindings(ids, weights, source_vertex_count=4, device=device)
topology = SurfaceTopology([[0, 1, 2]], vertex_count=3, device=device)

# All geometry arrays have logical shape [world, vertex] and dtype vec3f.
source = wp.array(np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                           dtype=np.float32), dtype=wp.vec3f, device=device)
positions = wp.empty((1, bindings.count), dtype=wp.vec3f, device=device)
normals = wp.empty_like(positions)
face_scratch = wp.empty((1, topology.face_count), dtype=wp.vec3f, device=device)

def frame():
    enqueue_embedding(source, bindings, positions)
    enqueue_normals(positions, topology, face_scratch, normals)

with wp.ScopedDevice(device):
    frame()                         # Compile/load kernels before capture.
    wp.synchronize_device(device)   # One-time setup synchronization.
    wp.capture_begin(device=device)
    try:
        frame()
    finally:
        graph = wp.capture_end(device=device)
    # The host updates source, or the simulator writes it on this stream.
    wp.capture_launch(graph)        # Positions and normals remain on device.
```

`enqueue_embedding` and `enqueue_normals` accept `world_count`,
`source_world_offset`, `output_world_offset`, `source_vertex_offset`,
`output_vertex_offset`, and an optional `stream`. Defaults select all source
worlds and zero offsets. All worlds share the prepared bindings and topology.
Face scratch always begins at `[0, 0]`, and must hold
`(world_count, topology.face_count)` vectors.

Frame calls validate host-visible array metadata and enqueue kernels. They do
not allocate device buffers, copy positions, or synchronize with the host.
The caller owns the position, normal, and scratch buffers and must keep them
alive until execution completes; captured graphs require stable buffer storage.
Use stream events or explicit setup synchronization when a producer uses another
stream. Setup uploads and host validation occur before graph capture.

Positive-stride engine views are supported. Nonempty buffer pointers and strides
of dimensions with more than one element must align to 4-byte float32 values.
Used strides must also fit Warp's signed 32-bit stride ABI.
Arrays can be wrapped with Warp's
external-pointer/DLPack interfaces where the engine supports them; the engine
retains responsibility for actual allocation sizes, ownership, and stream
ordering. Inputs, outputs, and scratch must not overlap. Conservative span
checks can reject disjoint interleaved views within the same underlying buffer;
use separate allocations in that case. Metadata device arrays are immutable by
contract after preparation.

Normals first compute one unnormalized cross product per triangle, then one
thread per vertex gathers incident faces in fixed input order from CSR links.
There are no floating-point atomics. Repeated runs on the same device and binary
are deterministic; cross-device results may differ slightly through floating
point rounding. Split render vertices at hard edges. Isolated vertices,
degenerate triangles, and exactly cancelling fans yield zero normals. The
`epsilon` argument defaults to `1e-20` in squared position units and suppresses
smaller summed cross-product lengths. Inputs, cross products, and accumulated
cross products must remain finite float32 values.

## Thin-shell surface attachments

`DeviceSurfaceBindings(indices, weights, normal_offsets=None,
source_vertex_count=N, device=device)` accepts three indices and barycentric
weights per point, matching `SurfaceEmbedding<float>::Binding` in C++.
`normal_offsets` is a signed distance in world units, following the current
winding-oriented triangle normal. Its magnitude stays constant under stretching.
Zero offsets give ordinary affine barycentric interpolation. This is suitable
for driving a dense render surface from a coarse cloth or shell simulation.

```python
from deform3d_warp import DeviceSurfaceBindings, enqueue_surface_embedding
binding = DeviceSurfaceBindings([[0, 1, 2]], [[0.2, 0.3, 0.5]], [0.0001],
                                source_vertex_count=3, device=device)
surface_output = wp.empty((worlds, binding.count), dtype=wp.vec3f, device=device)
validity = wp.empty((worlds, binding.count), dtype=wp.int32, device=device)
enqueue_surface_embedding(source, binding, surface_output, validity)
```

The required caller-owned `validity` array uses the same world and vertex offsets
as the output: `1` means success and `0` means invalid geometry or arithmetic.
Invalid entries leave the corresponding output unchanged. Every bound triangle
is checked, even for zero offsets. Collapse uses the float32 C++ policy:
`length(cross(ab / longest_edge, ac / longest_edge)) <= 64 * float_epsilon`.
Nonfinite coordinates, edges, or outputs are rejected by the device kernel.
Failure is per point/world; C++ instead validates the whole call before writing.
The caller can consume validity entirely on the GPU, without a host readback.
Batching, positive strides, offsets, explicit streams, and CUDA graphs follow
the tetra API above. CPU and GPU tests cover affine transformations, signed
offsets, winding reversal, stretching, extreme scales, collapse, nonfinite
inputs, and graph replay with changing geometry.

## Reproduce the geometry benchmark

```sh
python bench/warp_dense.py --device cuda:0 --side 1000 --worlds 1 \
  --samples 20 --repeat 5 --warmup 5 --json results/warp_dense_1m_cuda.json
python bench/warp_dense.py --device cuda:0 --side 1000 --worlds 2 \
  --samples 20 --repeat 5 --warmup 5 --json results/warp_dense_2x1m_cuda.json
```

The fixture is a curved million-vertex surface spanning a 4,913-vertex
tetrahedral cage, with 1,996,002 surface triangles per world. Setup, compilation,
and reference checks are excluded. Each CUDA sample times five repeated frames
inside a graph, after five warmup calls. CUDA event duration and synchronized
host wall duration are reported separately. Median and p95 refer to per-frame
averages of these graph batches, not isolated-frame latency percentiles.
Transfers use reusable pinned host buffers and are timed independently.

Report your GPU, driver, Warp version, world count, and separate geometry time
from physics, transfers, and rendering when comparing results. Historical
workstation timings are not performance guarantees for this checkout.
