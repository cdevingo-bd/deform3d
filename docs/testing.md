# Testing

## Default checks

```bash
cmake -S . -B build-core -DCMAKE_BUILD_TYPE=Release -DDEFORM3D_BUILD_BENCH=OFF
cmake --build build-core --config Release --parallel 2
ctest --test-dir build-core -C Release --output-on-failure
cmake --install build-core --prefix build-install
python tools/check_cmake_install.py build-install
python -m pytest -q
python examples/quickstart.py --device cpu
python -m build
python tools/check_wheel.py dist/deform3d_warp-0.13.0-py3-none-any.whl
```

The wheel check unpacks the wheel into a temporary directory, verifies the shared
C++ header and notices, then imports and allocates bindings outside the checkout.
It does not download dependencies. The default pytest suite collects geometry
and packaging tests only. `--collect-only` shows exactly what will run.

## Existing MuJoCo/MJWarp environment

```bash
python -m pytest -q --integration
```

This adds generated tetra and shell bridge checks: batched state refresh,
barycentric reconstruction, no accidental time advancement, and rejection of
mismatched binding metadata. It needs installed MuJoCo and MJWarp, but not
external assets. With CUDA available:

```bash
DEFORM3D_WARP_DEVICE=cuda:0 python -m pytest -q --integration
```

## Experimental solver suite

Use a separate environment so this does not replace MJLab's dependencies:

```bash
python3 -m venv .venv-physics
source .venv-physics/bin/activate
python -m pip install -e '.[physics,test]'
DEFORM3D_WARP_DEVICE=cuda:0 python -m pytest -q --integration --experimental-gpu
```

This command requires CUDA and exact MuJoCo/MJWarp 4.0.1 and Warp 1.19.0
versions. It fails early on missing/wrong dependencies. Test collection does
not silently substitute a CPU or different backend. The optional C++ MuJoCo
suite and exported-binding format checks run separately:

```bash
python tools/configure_mujoco.py
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
python tests/test_surface_bindings.py --exporter build/deform3d_export_surface \
  --tetra-exporter build/deform3d_export_bindings
```

The source-hook test is opt-in with `--source-hook`; see
[experimental.md](experimental.md). `--experimental-gpu` does not enable it.

## Validation for this repository preparation

Local checks used Linux, GCC 13.3, Python 3.12, cached Warp 1.16.0 and
MuJoCo/MJWarp 3.17.0. The standalone C++ suite, geometry tests, bridge regression,
and wheel installation were exercised locally. CUDA graph tests require a
separate GPU run; no GPU performance or policy-training result is claimed here.
The bundle's 4.0.1 solver suite and patched CPU hook were not rerun in this
environment. Historical results in the input archive are not new test results.
