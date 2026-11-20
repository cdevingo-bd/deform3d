# Force transfer through a tetrahedral embedding

`TetraEmbedding<T>` provides an engine-independent path from forces on embedded
geometry back to the coarse simulation mesh. The host still owns collision
detection, contact response, material models and time integration. Force transfer
does not make render geometry a collision surface automatically.

For fixed bindings, the position map is linear:

```text
y_i = sum_a w_ia x_a               y = J x
v_i = sum_a w_ia v_a               v_render = J v_sim
F_a = sum_i w_ia f_i               F_sim = transpose(J) f_render
```

Each row has four weights. `transfer_forces` uses exactly the weights used by
`evaluate`, including negative weights for bounded extrapolation. It does not
normalize forces, multiply by mass, or recompute bindings. It is also suitable
for impulses: use one consistent physical quantity on both sides of a call.

## Conservation properties

The transpose transfer satisfies the virtual-work identity
`dot(F_sim, delta_x) = dot(f_render, J delta_x)` up to floating-point rounding.
The same identity with velocities preserves instantaneous power across the
mapping. Weights sum to one, so total force is preserved. When the forces act at
`y = J x`, torque about any common origin is also preserved:

```text
sum_a cross(x_a - origin, F_a) = sum_i cross(y_i - origin, f_i)
```

These identities apply to each transferred contribution. In accumulate mode,
subtract the target's previous forces before checking them. They do not guarantee
that a host contact solver or numerical integrator conserves energy, is stable,
or matches a physical material.

For `ClampToBoundary`, the force acts at the **evaluated, projected point**, not
the original point outside the tetrahedral volume. Any visual displacement
applied after embedding changes the force's moment arm and requires its own
derivative for consistent coupling. Extrapolated weights preserve an external
point's affine motion, but large signed weights can amplify forces and roundoff.
Use a small, physically meaningful `max_outside_distance`.

## API and buffer ownership

```cpp
#include <deform3d/embed.hpp>

// Once: embedding.prepare(...). Host owns all arrays below.
auto status = embedding.evaluate(sim_positions, sim_count,
                                  render_positions, render_count);
// The host computes point_forces at render_positions.
status = embedding.transfer_forces(point_forces, render_count,
                                    sim_forces, sim_count);
// sim_forces now contains only this embedding's contribution.

// Add another contribution to a finite existing target array:
status = other_embedding.transfer_forces(other_point_forces, other_render_count,
    sim_forces, sim_count, deform3d::EmbedForceMode::Accumulate);
```

The second embedding in that example must use the same simulation vertex
indexing. The caller checks each status before using the corresponding output.
Once prepared, these calls are `const noexcept` and allocate no memory.

`transfer_forces_range(input, input_count, output, sim_count, begin, end, mode)`
processes the half-open range `[begin, end)`. The source pointer addresses the
full render array: range indices are global, as with `evaluate_range`.

| Mode | Behavior |
| --- | --- |
| `EmbedForceMode::Overwrite` (default) | Clear every simulation force, then add the selected source forces. Unreferenced simulation vertices become zero. An empty source range still clears the target. |
| `EmbedForceMode::Accumulate` | Add the selected forces to the existing target. An empty source range leaves it unchanged. |

The source count must be at least `render_vertex_count()` and the target count
must equal `simulation_vertex_count()`. Extra source entries are ignored. Both
pointers must be nonnull even for a prepared embedding with zero render points,
matching `evaluate`'s pointer contract. Source and target memory may not overlap.

The strided variant accepts raw floating-point XYZ arrays with padding and
different input/output scalar types:

```cpp
// Strides are scalar elements, not bytes. Output padding is preserved.
status = embedding.transfer_forces_strided(
    point_force_xyz, render_count, input_stride,
    sim_force_xyz, sim_count, output_stride,
    deform3d::EmbedForceMode::Accumulate,
    0, render_count);
```

Its final `begin` and `end` arguments default to the full range. Each contribution
uses the common precision of the embedding, input and output scalars; each sum
is stored in the output scalar type. Choose double output when summing many
opposing forces or using signed extrapolation weights. Padding is not read for
finiteness and is not modified.

The calls validate readiness, pointer/count/stride arithmetic, mode, ranges,
overlap and finite source forces before changing the target. Only source entries
in the selected range are checked for finiteness. Accumulate also validates all
existing simulation XYZ forces; overwrite does not read their previous values.
Failures return `InvalidInput`, `InvalidRange`, `Aliasing` or `NonFiniteInput`
and leave output unchanged. As with position evaluation, callers must use scales
for which products and accumulated sums fit in the calculation and output types;
finite-input arithmetic overflow is not diagnosed or rolled back.

## Parallel use and throughput

Position evaluation writes separate render vertices, but force transfer scatters
many render contributions to shared coarse vertices. Concurrent range calls must
therefore use separate simulation force buffers, followed by a host reduction.
Another valid approach is serialized ranges: overwrite the first range, then
accumulate subsequent ranges. Concurrent writes to one target buffer are a data
race even if the source ranges do not overlap. Separate targets may safely share
one prepared embedding, provided no concurrent `prepare` or `swap` occurs.

Cost is linear: a finite-input validation pass and four vector contributions per
selected point, plus a pass over coarse vertices to clear or validate the target.
The implementation has no atomics, hidden allocations, host engine types or
threading dependency. The public `bindings()` data can also feed a host's device
kernels; a device scatter needs atomics or a separate reduction strategy.

## Applying the mapping in a simulator

For a point contact between two deformable objects, the host computes the contact
force once, applies it to the first object's embedded point and applies the
opposite force to the second. Each object transfers its force through its own
embedding. Global force and torque cancel when both forces act at the same world
point. If a contact lies inside a render triangle, first distribute its force to
the triangle's vertices using the contact's triangle barycentric coordinates,
then use this tetrahedral transpose transfer. Forces should be expressed in the
same coordinate frame as the simulated positions.

For a rigid participant, route the opposite force and its moment through the
host's rigid-body force API. Simulation vertices constrained by the host still
receive their share; the host decides how to account for reaction forces. This
API returns Cartesian nodal forces, not a simulator's generalized forces. An
engine with generalized coordinates needs its own vertex Jacobian transpose to
map them into that engine's force vector.

MuJoCo or MJWarp may already solve contact on their coarse flex geometry. Do not
apply those same contact forces again through dense geometry. Use this interface
for an explicit custom interaction that the host is not already resolving, and
integrate it at the host's force evaluation phase. Host substeps, feedback delay,
contact stiffness and material calibration still control stability and realism.

Continuous pressure or traction is force per unit area. Integrate it with vertex
area or quadrature weights before transfer; assigning the same force to every
dense vertex makes total load change with mesh resolution. Dense geometry can
resolve contact locations, but cannot add deformation modes absent from the
coarse simulation mesh.

## Verification

`tests/test_coupling.cpp` checks float and double transfer on 4,096 embedded
points in adjacent tetrahedra with arbitrary nodal motion; total force, torque
about an offset origin and virtual work; a finite-difference potential-energy
gradient; paired action/reaction forces across independent objects; signed
extrapolation and projected positions; overwrite/accumulate semantics; batching
and worker-buffer reduction; striding and mixed precision; errors with unchanged
outputs; and no dynamic allocations during runtime calls. It also rejects a
boundary-search overflow that previously admitted an all-zero binding.

These are mathematical transfer tests, not validation of a contact law or a
measurement of physical sim-to-real accuracy.
