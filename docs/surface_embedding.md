# Triangle surface embedding for cloth and shells

`deform3d/surface_embed.hpp` attaches dense geometry to a coarse triangle mesh.
Use it for cloth bags, backpacks, thin plastic shells and other objects whose
simulation representation is a surface. Tetrahedral embedding remains the
appropriate mapping for volumetric solids. Neither representation supplies a
material or contact model; the host simulator owns those.

## Mapping

Setup builds a triangle BVH and finds the nearest point for each dense vertex.
A binding stores three simulation vertex indices, three barycentric weights,
the source triangle index and a signed normal offset in scene length units.
The triangle winding determines the positive side. Evaluation computes

```text
p = w0*x0 + w1*x1 + w2*x2 + normal_offset * normalize(cross(x1-x0, x2-x0))
```

This mapping is covariant under proper rigid transformations. With zero offset,
it also follows arbitrary affine motion of the triangle. Nonzero offsets retain
their physical magnitude: stretching a bag does not scale its visual thickness.
Consequently, nonzero-offset points do not follow an arbitrary affine transform
of the original dense mesh. The normal comes from the current triangle, never a
fixed world direction or a stored rest-normal fallback.

## Prepare and evaluate

```cpp
#include <deform3d/surface_embed.hpp>

deform3d::SurfaceEmbedding<float> embedding;
deform3d::SurfaceEmbedOptions<float> options;
options.max_distance = 0.003f; // Example: 3 mm, if scene units are metres.
options.outside = deform3d::SurfaceOutsidePolicy::PreserveNormalOffset;
auto status = embedding.prepare(
    rest_sim_vertices, sim_vertex_count,
    triangle_indices, triangle_count,
    rest_render_vertices, render_vertex_count, options);
// Check status before using a new mapping.
status = embedding.evaluate(current_sim_vertices, sim_vertex_count,
                             render_output, render_vertex_count);
```

`max_distance` is a finite, nonnegative bound on distance to the closest point
on the actual triangle, not its infinite plane. The default is zero, so callers
should provide an explicit physical bound for thickness and imported assets.
Setup rejects NaN/infinite bounds and nonfinite geometry. Failed setup leaves
an existing embedding intact. All simulation vertices and all triangles are
validated, including vertices unused by dense bindings.

`PreserveNormalOffset` keeps signed thickness. At an open edge, a closest point
can have a tangential residual that a single normal offset cannot represent.
Such a point is rejected with `OutsideSurface` unless the residual is within
`tangential_tolerance * longest_triangle_edge`. The default tolerance is
`64 * epsilon(T)`, intended for rounding rather than large lateral attachment.
`ProjectToSurface` discards both normal and tangential residuals and follows the
closest point. Both policies obey `max_distance`.

The BVH uses deterministic lowest-triangle-index selection for exactly equal
distances. At creases and seams a single triangle is selected; this is a faceted
attachment model, not a smooth normal blending method. Choose coherent triangle
winding before preparation and retain topology and vertex indexing afterward.
Geometry crossing through a fold can change which surface is physically nearest,
but prepared material attachments remain attached to their original triangle.

## Runtime contract

`evaluate_range` accepts the same arguments as `evaluate`, followed by the global
half-open dense range `[begin, end)`. `evaluate_strided` accepts raw floating-point
XYZ buffers, counts and strides measured in scalar elements, with optional
`begin` and `end` arguments. Input/output types may differ; padding is preserved.
Input count must equal `simulation_vertex_count()` and output capacity must be
at least `render_vertex_count()`. Pointers must be nonnull, even for empty ranges.

Runtime calls allocate no memory, are `const noexcept`, and do not modify input.
They validate every simulation vertex and triangle before checking selected
results for representability, then write the selected output range. Any error
leaves the entire output unchanged. Input/output memory must not overlap.
Disjoint output ranges can run concurrently against the same immutable input
and embedding. Do not mutate simulation positions during evaluation or call
`prepare`/`swap` concurrently.

A triangle is considered degenerate when its cross-product magnitude, after
normalizing both edges by its longest edge, is at most `64 * epsilon(T)`. This
relative shape criterion also applies at preparation. It rejects exact collapse
and numerically unresolved slender triangles while remaining independent of
uniform scene scaling. Uniformly shrinking a well-shaped triangle is accepted
while its coordinates remain distinct and representable. Collapse returns
`DegenerateTriangle`, even when attached offsets are zero. There is no stale
normal fallback, and a timestep that jumps through collapse is not detected
retroactively: the host must prevent inverted or self-intersecting shell motion.

Nonfinite current coordinates return `NonFiniteInput`; overflow or an output
scalar unable to represent a result returns `NumericalFailure`. Internal widened
arithmetic reduces range problems. The implementation performs a representability
pass before its write pass, favoring a reliable unchanged-output failure contract.
Cost is `O(sim_vertices + sim_triangles + selected_render_vertices)` per call;
many small range calls repeat the source validation work. Nonzero offsets also
require computing a triangle normal per point on each result pass. This CPU
implementation is a correctness reference; dense GPU transport can use the
public binding records and equivalent current-triangle normals.

## Force coupling is not provided

For nonzero offsets, the derivative includes the changing unit normal. Scattering
forces with barycentric weights alone omits this term and does not conserve
virtual work. Surface force transfer is documented in [surface_coupling.md](surface_coupling.md).
The tetrahedral `transfer_forces` API applies to its separate linear mapping;
it must not be repurposed as a derivative of this shell offset mapping.

## Optional MuJoCo and device transport

`deform3d/mujoco_surface_adapter.hpp` provides `MujocoSurfaceAdapter<T>` for
MuJoCo flexes with `dim=2`. This header requires the matching MuJoCo SDK and
library; `surface_embed.hpp` remains independent. Bind after `mj_forward` at
the reference pose. The adapter exposes local triangle indices, reference
vertices, their canonical hashes and a pointer to the selected flex's current
XYZ buffer. `prepare_embedding` and `evaluate_embedding` forward to the core
surface mapper without a per-frame position copy or allocation. The host must
keep the model alive and its topology fixed. After `mj_step`, run
`mj_kinematics` followed by `mj_flex` before consuming the integrated geometry.

The exporter accepts an XYZ rest mesh in the same world frame as that reference
pose. For example, with coordinates in metres:

```sh
deform3d_export_surface scene.xml bag dense.bind \
  --mesh dense.xyz --max-distance 0.003
```

`--project` selects zero-offset projection; the default preserves signed normal
offsets. The flex name is explicit, so several bags can be exported separately.
The binary format is portable little-endian data, with no native C++ structs:

| Bytes | Contents |
| --- | --- |
| Header 0–7 | ASCII `D3SUR001` |
| Header 8–23 | Four uint32 values: record count, flex ID, global vertex offset, vertex count |
| Header 24–39 | Two uint64 FNV-1a hashes: ordered triangle indices and float32 reference XYZ |
| Each 28-byte record | Three uint32 local vertex indices, three float32 weights, one float32 signed offset |

`examples/mujoco_surface_bindings.py::read_surface_bindings(path, model)` returns
`(flex_id, vertex_offset, records)`. It checks exact file length, model identity
through topology/reference hashes, finite fields, nonnegative unit-sum weights,
and that each triple is an oriented triangle in the selected flex. Records are
returned read-only. The hashes detect accidental model mismatches, not malicious
tampering.

The optional Warp transport accepts these fields through `DeviceSurfaceBindings`
and `enqueue_surface_embedding`. Its required caller-owned integer validity
buffer reports success/failure per point without a host synchronization. This
device failure contract differs from the C++ call-wide unchanged-output contract;
consult the device API before combining it with downstream rendering or normals.

## Verification

`tests/test_surface_embedding.cpp` exercises float and double rigid covariance,
positive/negative normal offsets under rotation, constant thickness under
stretching, zero-offset affine motion, open-edge rejection and projection,
transactional failures, degenerate/collapsed triangles, nonfinite unused source
vertices, range/alias/stride checks, mixed scalar outputs, representability,
zero runtime allocations and 8,192 random closest-point queries against an
independent rectangular-shell reference. Another 2,048 points on randomly
oriented, nonorthogonal triangles check known triangle IDs and deformation with
signed offsets. Scale tests cover float triangles at `1e-20` and double triangles
at `1e-200`, plus rejection of unresolved aspect ratios. These tests validate
geometry mapping, not a bag's physical calibration, bending or contact response.

`tests/test_mujoco_surface.cpp` builds its own MuJoCo model in memory, exercises a
dim=2 flex with a nonzero global vertex offset, checks step/refresh/evaluation and
unchanged output on collapse. The Python ABI test creates temporary XML/XYZ files,
runs the actual exporter, and checks reconstruction, simulated motion, eleven
record corruptions, two model mismatches and explicit open-edge projection:

```sh
ctest --test-dir build -R 'surface_embedding|mujoco_surface' --output-on-failure
python tests/test_surface_bindings.py --exporter build/deform3d_export_surface
```

The MuJoCo tests require configuring the optional MuJoCo build and installing
the matching Python MuJoCo package. They use no external assets or GPU.
