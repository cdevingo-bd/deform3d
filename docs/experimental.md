# Experimental physics integrations

These modules are opt-in research implementations. They are not loaded when
using `DeviceBindings`, `DeviceSurfaceBindings`, or `FlexEmbedding`.

The `physics` extra pins MuJoCo 4.0.1, MJWarp 4.0.1, and Warp 1.19.0. Several
modules inspect or replace private MJWarp kernel factories, so their version
and source-hash guards are intentional. Do not remove them to bypass an error.

The source includes shared simplex material evaluation, deterministic material
and contact assembly, constraint ordering, Newton factorization/substitution,
contact-normal experiments, and a device simulation clock. `bench/` contains
focused measurements. The paired native-material comparison is:

```bash
mkdir -p results
python bench/native_backends.py --host gpu --duration 3 \
  --output results/native_material_pair.json
```

Use `python -m pytest -q --experimental-gpu` before comparing trajectories.
The paired benchmark advances independent states with matching inputs and
reports shape/force differences. The imported development bundle's independent
GPU material pair **failed** the 5 mm shape gate (about 31 mm). That is an open
numerical limitation, not a packaging issue solved by this repository work.

`SharedSimplexPolicy` replaces the element constitutive kernel during warmup and
graph capture; `SimplexFlexOperators` also uses owned deterministic reductions.
Native contact and integration remain. Their temporary factory changes are
process-global: do not construct unrelated model graphs concurrently. Retain
the policy/model for replay and recapture after structural changes. Wheels
include the shared C++ header automatically.

## Direct effective-metric solves (MJWarp 3.17 and 4.0.1)

MJWarp's `derivative.eff_solve` solves the discrete flex effective metric
`M_hat = M + h*D + h^2*K` with preconditioned CG, whose cost on small systems
is mostly launch latency. Two opt-in direct replacements exist for models where
`M_hat` is only `M + diag + K` with the DISCRETE integrator:

* `mujoco_effective_dense.DenseEffectiveSolve`: dense assemble + blocked Cholesky.
* `mujoco_effective_banded.BandedEffectiveSolve`: the same, after a one-time
  reverse Cuthill-McKee reordering, visiting only tiles inside the matrix band,
  with an O(nnz) sparse residual gate. Prefer this one; `worthwhile(model)`
  reports whether the band is narrow enough to help.

`supported(model)` states why a model is excluded; the constructors raise
instead of falling back, and `capture_policy()` routes only the matching model
to the direct solver while other models keep the native path. The workspace
residual gate (`status`, bit 2) is sticky; a nonzero status invalidates the run.

Measured on the generated bag scene (nv=357, band 6 of 23 tiles), on the 3.17 stack
(MuJoCo 3.17.0, Warp 1.16.0, a patched MuJoCo Warp 3.17.0 source build), one 8 GB laptop GPU,
graph replay, 'tiled' kernels (the 'register/ordered' kernels need Warp 1.19):

| worlds | native `eff_solve` | dense | banded | full step, dense / banded | rel. error vs float64: native / dense / banded |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.30 ms | 0.83 ms | 0.58 ms | 1.38x / 1.42x | 9.9e-6 / 7.4e-7 / 1.1e-6 |
| 16 | 2.58 ms | 1.13 ms | 0.66 ms | 1.21x / 1.34x | 1.0e-5 / 2.2e-6 / 2.2e-6 |
| 64 | 4.96 ms | 2.78 ms | 1.06 ms | 1.18x / 1.47x | 1.0e-5 / 2.2e-6 / 2.4e-6 |

Through a batched rollout wrapper (JAX state conversion included, 5 steps per call)
the gains are smaller because of fixed per-call costs: banded was about 25% faster at
1 world and 21% at 64 worlds (interquartile ranges separated), inconclusive at 16
worlds (ranges overlap), and dense was not distinguishable from native at 64 worlds.

Limits: float32 only. Above a condition number of about 1e4 neither solver meets a
1e-4 residual (native PCG measured 1.4e-3 at 1.3e4 and 0.13 at 1.3e6; the direct
solvers were no worse) and the workspace reports it. Dense storage is
`nworld x nv x nv` (banded still allocates it, but only touches the band).
Trajectories after 100 steps differ from native by the same order as two native runs
differ from each other (MJWarp is not bitwise repeatable), consistent with equivalence
but a single-sample comparison, not a proof. Reproduce with `bench/dense_effective_ab.py`.

Applicability: the two solvers above only matter for flex scenes using MJWarp's DISCRETE
integrator. Flex scenes that use the Euler integrator (for example the pot scene below)
never call `eff_solve`; there the Newton contact solver dominates instead
(next section).

## Newton Cholesky for unpinned builds (MuJoCo Warp 3.17 source builds)

`mujoco_newton.ParallelNewtonSolver` replaces MJWarp's Newton factor+solve with deform3d's
tile Cholesky, but is reviewed only for PyPI mujoco-warp 4.0.1 and checks `importlib.metadata`.
A MuJoCo Warp built from a patched source archive has no distribution metadata, so
`mujoco_newton_factor.NewtonFactorPolicy` applies the same replacement and guards on the SHA-256
of the source of the hook it replaces (`REVIEWED_HOOKS`; reviewed: mujoco_warp 3.17.0 source build,
upstream commit f34c6dd5 plus local patches, Warp 1.16, 'tiled' kernels). On any other build it refuses to install; add a
hash only after re-reviewing that implementation. `capture_policy()` is used exactly like the
effective-solve policies, routes only the owning model, restores the hook on exit, and the
float64 residual gate (`status` bits 1 and 2) is sticky.

Two methods exist (`NewtonFactorPolicy(method=...)`):

* `"coupled"` (default, `mujoco_newton_coupled`): the Newton Hessian `H = M + J^T D J` of an Euler flex
  scene is extremely sparse (pot scene, nv=414: at most 33 coupled DOFs of 414 in the runs below). Per
  solve it scans the upper triangle for off-diagonal entries, factors only that dense coupled block
  with per-world tile counts, and solves every other DOF by a scalar division. This is exact, not an
  approximation. A world whose coupled set exceeds `max_coupled` (default 128) is handed back to
  MJWarp's own hook (with a forced refactor), so a dense Hessian is still correct, just not faster.
  `policy.fallbacks` counts those hand-backs and `policy.coupled` gives the coupled-set sizes.
* `"dense"`: the full blocked tile Cholesky. It wins only when the solve is latency-bound and loses
  from about 20 worlds.

Pot scene (a flex-divider pot fixture plus a floor and three free bodies; Euler,
Newton, nv=414, dt=0.0005), on the 3.17 stack, graph replay of the full step, 16 contacts per world of buffer.
Solver status was 0 and there were no fallbacks in every run. Speedup over native (the dense column
is from an earlier run of the same scene, so treat the comparison as indicative):

| worlds | dense | coupled |
| --- | --- | --- |
| 1 | 1.57x | 8.1x |
| 8 | 1.38x | 7.7x |
| 32 | 0.86x | 5.0x |
| 64 | 0.64x | 3.5x |
| 128 | not run | 2.5x |

Above 128 worlds MJWarp's own scratch buffers exhausted the 8 GB laptop GPU. The laptop GPU's clocks
drift between runs (native 128-world step time varied from 8.4 to 10.2 ms), so the ratios are more
trustworthy than the absolute times. Through a batched rollout wrapper (20 steps per call): 5.5x,
5.2x, 4.6x, 3.9x at 1, 8, 16, 32 worlds (the dense method gave 1.50x, 1.33x, 1.14x at 1, 8, 16),
qpos equal to native within the native-vs-native control.

The coupled kernels were tuned once: the factor loop, right-hand-side solve and substitution run as
one kernel per world (which loops over only the tiles that world needs), the coupled-set compaction
runs on 32 lanes per world instead of one thread, and packing writes only the lower triangle. That
took 128 worlds from 2.3x to about 2.5x and one world from 6.7x to about 8x. The Hessian scan
(`mark`) reads half of H once per solve and already runs at about the GPU's memory bandwidth, so it
is the floor of this approach.

`structure="jacobian"` (opt-in, `verify_structure=True` available) replaces the Hessian scan with a
structural coupled set: the nonzero Jacobian columns of each quadratic constraint row that has at
least two of them, accumulated over the iterations of a solve (MJWarp updates H incrementally and
can leave rounding residue on a row that changed state), plus the DOFs of every nonzero stored
off-diagonal mass entry. It is a superset of H's nonzero pattern, so the solve stays exact, but it
depends on how MJWarp assembles H: it is refused for dense Jacobians, the DISCRETE integrator and
elliptic cones, and it needs the SHA-256 of the five assembly functions to match a reviewed build
(`assembly_is_reviewed`). `verify_structure=True` also runs the scan and sets `status` bit 4 if the
structure ever misses a coupled DOF (it did not in a 300-step contact-rich run or in the tests).

The structure can be too loose: a pyramidal friction group sums to exactly zero off-diagonal in H
whenever its rows are equal, and the structure cannot see that cancellation, so in contact-rich
states its set was about twice the true one. A world whose structural set exceeds `rescan_above`
DOFs (default: one tile, 32) therefore discards it and takes the exact scan; `policy.rescans` counts
how often. Measured on the pot scene, graph level, speedup over native:

| state | worlds | scan | jacobian (with rescan) |
| --- | --- | --- | --- |
| light contact (16 rows) | 1 | 7.4x | 6.9x |
| light contact (16 rows) | 64 | 3.3x | 3.5x |
| light contact (16 rows) | 128 | 2.5x | 2.8x |
| contact-rich (about 64 rows) | 64 | 2.9x | 2.9x |
| contact-rich (about 90 rows) | 64 | 2.6x | 2.6x |

Without the rescan the contact-rich states were 4-26% slower than the scan. With it they match the
scan, and light-contact states at 64 or more worlds gain 5-10%. It is about 6% slower at one world,
where the scan is cheap and the extra launches show. Because the gain is small and depends on how
MJWarp assembles H, the default stays `structure="scan"`; consider `"jacobian"` for 32 or more
worlds on scenes you have checked with `verify_structure=True`.

The speedup still falls as worlds grow; it does not go negative. The native hook has a large fixed
latency floor that coupled removes, so the gain is biggest at one world. At high world counts both
paths are throughput-bound and the remaining per-world cost is dominated by work this policy cannot
touch (flex collision sweep, constraint Jacobian) plus its own scan/pack/finish kernels. Marginal
cost per added world was about 25 us for coupled against 34 us for native between 64 and 128 worlds;
extrapolated linearly (not measured beyond 128 worlds) that approaches about 1.3x at saturation, and
about 2x at best if the policy's own kernels were free. Do not read the trend as a promise
of a growing advantage.

Where it does not help: a scene whose Hessian is dense always falls back (the repo's bag scene,
and anything on the DISCRETE integrator), which only adds the scan's cost. Check `policy.fallbacks`
and `policy.coupled` on your scene first.

Caveats: one laptop GPU (RTX PRO 2000 Blackwell, 8 GB), one scene, start states settled for 300 CPU
steps with only 16 active constraints (the graph-level runs show up to 72 efc rows). Fidelity under
heavier contact is checked on a synthetic scene (tumbling boxes plus contact-free spheres) in
`tests/test_newton_factor.py`, and the new module's tests were mutation-checked. Reproduce with
`bench/stage_profile.py` (graph level) or a rollout-level benchmark such as `bench/rollout_ab.py`.

Fixed per-call cost: a 5-step rollout-wrapper call has about 4.5-6 ms of JAX/Warp conversion
and dispatch that deform3d cannot change (per-step native cost at 1 world fell from 3.45 to
3.03 ms going from 5 to 20 steps per call on the bag scene). Increase `n_steps` per call before
chasing smaller solver wins.

One MJWarp finding from this work: on the pot scene, capturing a step graph without ever running
it, then taking an eager MJWarp step on another `Data` in the same process, faults with an illegal
memory access in `_write_filtered_contacts`, with no deform3d code involved. Run each captured graph
once before building the next one.

## Optional CPU source hook

The CPU material replacement needs a separate, pristine MuJoCo 4.0.1 source
tree. It is not a plugin for the stock wheel. Once that source is available:

```bash
python tools/prepare_mujoco_simplex_hook.py /path/to/mujoco-4.0.1
cmake -S /path/to/mujoco-4.0.1 -B build-mujoco-hook \
  -DCMAKE_BUILD_TYPE=Release -DMUJOCO_BUILD_EXAMPLES=OFF \
  -DMUJOCO_BUILD_SIMULATE=OFF -DMUJOCO_BUILD_TESTS=OFF \
  -DMUJOCO_TEST_PYTHON_UTIL=OFF
cmake --build build-mujoco-hook --target mujoco --parallel 2
DEFORM3D_MUJOCO_LIBRARY="$PWD/build-mujoco-hook/lib/libmujoco.so.4.0.1" \
LD_PRELOAD="$PWD/build-mujoco-hook/lib/libmujoco.so.4.0.1" \
  python -m pytest -q --source-hook
```

The patcher verifies the expected engine source before modifying that separate
tree. MuJoCo's own configure/build may download its dependencies. No source
hook or upstream change is needed for the core geometry API.
