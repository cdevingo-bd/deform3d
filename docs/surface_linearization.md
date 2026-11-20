# Implicit surface coupling in the C++ core

`deform3d/surface_linearization.hpp` provides a C++17 force operator for
offset-surface attachments, with no engine, GPU, intrinsics or external library
dependency. It complements `SurfaceCoupling` with the force derivative needed
by an implicit solver and compresses dense-point contributions onto the physical
mesh. The [standalone backward-Euler example](../examples/implicit_surface.cpp)
uses the operator to take a nonlinear spring-loaded step, then checks force
balance through the original uncompressed coupling. It converges in three Newton
iterations with relative residual about 1e-15. This is an integration example,
not a calibrated contact or material model.

## Mathematics

For `y_i = w0*a + w1*b + w2*c + h_i*n(a,b,c)`, supplied point forces `f_i` and
local world-space derivatives `K_i = df_i/dy_i`, the snapshot represents:

```text
F = Jᵀ f
A u = Jᵀ K J u + D(Jᵀ)[u] f
```

The geometric term accounts for rotation of the offset under an existing load.
It remains nonzero for constant world-space forces at nonzero offsets. Both
normal derivatives use the same scale-conditioned normal map now shared with
`SurfaceCoupling`. The conditioning scale is held fixed while differentiating;
the nonsmooth selection of a maximum edge length is not part of the normal law.

`A` is a **force Jacobian**, not an energy Hessian. For conservative forces the
energy Hessian is `-A`. Symmetric point Jacobians produce a symmetric global
operator up to rounding. Geometric stiffness can be indefinite; the class does
not project eigenvalues or promise suitability for conjugate gradients.

Write `z=(a,b,c,n)` and `c_i=(w0,w1,w2,h_i)`. Four vector force moments
`g_a=sum_i c_ia*f_i` and ten matrix moments `Q_ab=sum_i c_ia*c_ib*K_i` suffice
per triangle. `Q_ab=Q_ba` even for nonsymmetric `K_i`; this equality does **not**
transpose the matrix. The normal differential and geometric term convert the
moments into one 9-by-9 local Jacobian. No dense global matrix is assembled.

Snapshot update costs O(dense points + referenced triangles). Each subsequent
application costs O(physical vertices + referenced triangles), independent of
dense-point count. This benefits **inner linear iterations at a fixed Newton
state**. A new nonlinear state or changed force law requires another update.
The plan retains O(dense points) bindings; preparation is not free or constant
storage. Intermediate arithmetic uses double for float inputs and long double
for double inputs. Regrouped sums can differ in rounding from direct transfer;
byte equality is not promised.

## API and lifecycle

```cpp
#include <deform3d/surface_linearization.hpp>
deform3d::SurfaceLinearization<float> linear;
auto status = linear.prepare(embedding); // allocating binding/topology snapshot
// Check every status before consuming the corresponding result.
status = linear.update(positions, physical_count,
                       point_forces, point_K, dense_count);
status = linear.evaluate_forces(physical_forces, physical_count);
// Reuse at this frozen state during the host's linear solve:
status = linear.apply(displacement, physical_count,
                      force_change, physical_count);
```

`point_K[i].m` is row-major `df_i/dy_i`. A null Jacobian pointer means fixed
world-space forces; geometric stiffness still applies. Successful prepare
invalidates the previous snapshot. Runtime update, evaluation and application
are `noexcept`, with no allocations. A failed update retains the preceding
valid snapshot. Failed output calls leave output unchanged, including narrowing
and arithmetic-overflow failures. Blocks use wider internal storage and are not
required to fit the public scalar type until an output is returned.

`evaluate_forces` and `apply` accept `SurfaceForceMode::{Overwrite,Accumulate}`
and half-open ranges of **physical vertices**. Disjoint ranges may execute
concurrently: each vertex gathers its triangles in a fixed order without
atomics. Overwrite clears unreferenced selected vertices; accumulate preserves
them. Empty ranges write nothing. Accumulate validates the existing selected
output. `apply` rejects overlapping input/output and validates every physical
input vector. Finish readers before update, prepare or swap.

Physical counts must match the plan. Point counts must cover all bindings;
extra entries are ignored. Position and force pointers must be nonnull, even for
a zero-point plan. The Jacobian pointer is optional. Invalid arguments,
nonfinite data and collapsed referenced triangles return `SurfaceEmbedStatus`.
Unreferenced triangles are not evaluated. Allocating preparation can throw
without replacing a prior plan.

## Integration boundary

The matrices describe independent local laws `f_i(y_i)` at the current state.
Cross-point derivatives, friction history, velocity derivatives and cross-object
contact blocks are not supplied automatically. Two moving contact participants
need their off-diagonal blocks as well as these local blocks. A force law that
also depends directly on a triangle normal needs those extra derivatives from
the host. This operator is not yet dispatched into the MuJoCo/MJWarp Newton loop;
their retained performance reports are unchanged.

Integrate surface traction **and its derivative** with area/quadrature weights
before transfer. Refining one sample must divide both its force and tangent
weight, or the physical load changes with resolution. The operator introduces
no extra deformation modes, contact law or material calibration.

## Verification and reproduction

The new test independently differentiates the uncompressed nonlinear force map
for float/double, signed and zero offsets, shared vertices, every coordinate
direction and nonsymmetric laws. It checks symmetry for conservative laws,
differentiated force/torque identities, geometric stiffness for fixed forces,
scales from 1e-4 to 1e4, load-preserving refinement, deterministic concurrent
ranges, no runtime allocations, and transactional overflow/failure handling.
The original surface-coupling tests remain enabled.

```sh
cmake -S . -B build-core -DCMAKE_BUILD_TYPE=Release -DDEFORM3D_BUILD_MUJOCO_TEST=OFF
cmake --build build-core -j 3
ctest --test-dir build-core --output-on-failure
build-core/deform3d_implicit_surface
build-core/deform3d_surface_linearization_bench 100000
build-core/deform3d_surface_linearization_bench 1000000
```

The single-thread benchmark fixes the physical mesh at 256 vertices / 450
triangles. It compares the checked `J*u`, point-matrix multiplication, `Jᵀ*f`
path with the compressed operator. It uses zero offsets so the existing direct
path represents the complete same derivative; separate tests cover nonzero
offsets. Both paths apply twenty changing directions per sample in alternating
order, with seven retained samples after two warmups. Snapshot construction is
measured separately and included in update-plus-20-application totals. Binding
setup, host physics and rendering are excluded. This does not establish
whole-scene real-time performance.

Use the commands above to produce measurements on your machine. The standalone
tests verify the operator against finite differences and the direct path;
benchmark speed does not establish host-solver or sim-to-real accuracy.
