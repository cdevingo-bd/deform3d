# Differential surface coupling

`deform3d/surface_coupling.hpp` adds a prepared, allocation-free C++17 path for
positions, velocities and force transfer on a triangular surface. It has no
MuJoCo, Warp, GPU or other third-party dependency.

[SurfaceLinearization](surface_linearization.md) adds implicit force derivatives
with geometric stiffness and compresses dense local force laws into physical
triangle blocks. Both APIs share the same scale-conditioned normal differential.

The complete [host-independent example](../examples/surface_coupling.cpp) builds
with `c++ -std=c++17 -O3 -Iinclude examples/surface_coupling.cpp -o surface_coupling`.

```cpp
deform3d::SurfaceEmbedding<float> binding;
// binding.prepare(rest_vertices, vertex_count, triangles, triangle_count,
//                 dense_rest_vertices, dense_count, options);
deform3d::SurfaceCoupling<float> coupling;
auto status = coupling.prepare(binding);   // allocates; owns a binding snapshot
// Check each status before consuming the result.
status = coupling.update(current_vertices, vertex_count);  // no allocation
status = coupling.evaluate_positions(dense_positions, dense_count);
status = coupling.evaluate_velocities(vertex_velocities, vertex_count,
                                     dense_velocities, dense_count);
status = coupling.transfer_forces(dense_contact_forces, dense_count,
                                 vertex_forces, vertex_count);
```

The host generates contact forces and integrates motion. In MuJoCo integrations,
its native contacts remain authoritative. Do not apply a contact twice through
both native collision response and an external force path.

## One map in both directions

For a prepared binding, the surface position is
`y = w0*a + w1*b + w2*c + h*n`, where `n` is the current unit triangle normal.
The class evaluates the actual differential of this map, including `h*dn`.
Simply scattering barycentric weights omits the moment of the normal offset.

Writing `e1=b-a`, `e2=c-a`, `q=e1×e2` and `P=I-n*nᵀ`, the normal velocity is
`dn = P*((vb-va)×e2 + e1×(vc-va))/|q|`. For a dense force `f`, set
`g=h*P*f/|q|`. The extra physical vertex forces are
`Fa=-(e2×g + g×e1)`, `Fb=e2×g`, and `Fc=g×e1`, in addition to `wi*f`.
This preserves virtual work, net force and net torque, up to rounding, at the
evaluated surface position. It does not add a separate thickness strain law.

## Frame lifecycle and concurrency

`prepare` copies bindings, builds a reverse index of attachments per physical
vertex, and allocates reusable frame storage. A successful call invalidates the
previous frame. `update` copies current physical positions and caches each
referenced triangle's differential once. It rejects nonfinite positions and
degenerate triangles. A failed update retains the preceding valid frame.

After update, evaluation is const. Disjoint render ranges may evaluate positions
or velocities concurrently. Force transfer gathers through the reverse index,
so disjoint **physical vertex** ranges can also run concurrently without atomics
or a shared accumulation buffer. There is no fixed thread pool; use the host's
scheduler. Complete all readers before the next update or prepare.

`SurfaceForceMode::Overwrite` replaces only the selected physical range;
`Accumulate` adds only to that range. Unreferenced physical vertices become zero
in overwrite mode. Empty ranges write nothing. Only the declared XYZ entries
are accessed; input/output overlap is rejected for velocity and force transfer.
Invalid arguments, nonfinite inputs and unrepresentable results leave output
unchanged. This guarantee requires a representability pass before output writes.

Frames use double intermediates for float data and long double for double data.
Validation follows the original embedding's relative triangle-area threshold.
Unreferenced triangles are not needed by this differential cache and are not
checked by update; the original embedding validates all source triangles.

## Verification and measured scope

`tests/test_surface_coupling.cpp` checks positive and negative offsets, affine
deformation, rigid angular velocities, independent finite differences, virtual
work, resultant force and torque, deterministic parallel ranges, zero frame
allocations, and transactional failure handling. It runs for float and double.

The one-million-point, two-triangle position microbenchmark measured 267.47 ms
for the original checked offset evaluation and 17.44 ms for the prepared cache,
including cache update, with identical float outputs. This is a single-thread
position-only measurement on the shared development machine. It excludes
physics, normals, setup, rendering and force transfer. It does not establish
whole-scene real-time performance. Reproduce with
`build/deform3d_surface_coupling_bench`; see
a local output file.
