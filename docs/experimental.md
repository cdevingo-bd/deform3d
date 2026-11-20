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
