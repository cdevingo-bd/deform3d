# deform3d

Deform dense meshes from a smaller simulated mesh, on the CPU or in batched
NVIDIA Warp kernels. The C++17 core provides tetrahedral and triangle bindings,
surface normals, skinning, corrective shapes, and force-transfer operators.
The Python package can read flex positions from MuJoCo Warp or MJLab.

The host simulator owns physics and contacts. Updating an embedded mesh does
not change collision geometry or automatically replace the mesh in camera
images. Applications connect the output to their renderer. The optional
material and solver experiments are separate from this geometry API.

## Quick start

```bash
git clone https://github.com/cdevingo-bd/deform3d.git
cd deform3d
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python examples/quickstart.py --device cpu
python -m pytest -q
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. Python 3.11+ is required;
Python 3.12 is the primary test environment. For CUDA, use a Warp wheel supported
by your GPU and driver, then run:

```bash
python examples/quickstart.py --device cuda:0
DEFORM3D_WARP_DEVICE=cuda:0 python -m pytest -q
```

The default Python tests exercise geometry and packaging. On CPU they skip the
two CUDA graph tests; they do not claim to validate the experimental solvers.

## C++ without Python or a GPU

```bash
cmake -S . -B build-core -DCMAKE_BUILD_TYPE=Release -DDEFORM3D_BUILD_BENCH=OFF
cmake --build build-core --config Release --parallel 2
ctest --test-dir build-core -C Release --output-on-failure
```

The 11 standalone tests need only CMake 3.16+ and a C++17 compiler. Configure and
build do not download dependencies. Use `add_subdirectory` and link
`deform3d::deform3d`, or install for use with `find_package`:

```bash
cmake --install build-core --prefix "$PWD/build-install"
```

```cmake
find_package(deform3d CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE deform3d::deform3d)
```

## Choose an integration

| Use | Installation | Scope |
| --- | --- | --- |
| C++ core | CMake above | No simulator dependency |
| Warp geometry | `pip install -e '.[test]'` | CPU or CUDA, caller-owned simulation |
| Existing MJLab environment | `pip install -e .` inside that environment | Uses `sim.mj_model`, `sim.wp_model`, `sim.wp_data`; leaves host dependency pins in place |
| MuJoCo/MJWarp experiments | `pip install -e '.[physics,test]'` in a separate environment | MuJoCo 4.0.1, MJWarp 4.0.1, Warp 1.19.0 |

The local MJLab 1.6.0 source uses MuJoCo/MJWarp 3.17. Do **not** install the
`physics` extra over it. The bridge does not import MJLab or patch its solver;
the version-specific experimental policies do patch MJWarp internals and must
use their pinned environment.

See [MuJoCo/MJWarp setup](docs/integration.md), [MJLab usage](docs/mjlab.md), and
[experimental physics](docs/experimental.md). There is no bundled trained
policy, ready-made MJLab task, or automatic camera-renderer integration.

## API

```python
from deform3d_warp import DeviceBindings, enqueue_embedding

# Prepare once: local tetra indices and barycentric weights for each dense point.
bindings = DeviceBindings(indices, weights, source_vertex_count=n_vertices,
                          device="cuda:0")
# source/output: Warp vec3f arrays shaped [world, vertex], on the same device.
enqueue_embedding(source, bindings, output)
```

For a compiled flex and an exported binding file:

```python
from deform3d_warp.mujoco_bridge import FlexEmbedding

mesh = FlexEmbedding(cpu_model, warp_data, "surface.bind")
# After the host steps, refresh derived flex positions and evaluate the mesh.
mesh.update(warp_model, warp_data)
# mesh.positions stays on the device; mesh.validity is also available for shells.
```

Binding files are checked against the model's topology and reference pose.
Rebuild bindings when either changes. Keep the model, buffers, and bindings
alive through graph replay and synchronize any separate simulation/render streams.

Detailed references: [geometry](docs/algorithms.md), [Warp arrays](docs/gpu.md),
[surface bindings](docs/surface_embedding.md), [force transfer](docs/coupling.md),
[surface forces](docs/surface_coupling.md), and
[surface linearization](docs/surface_linearization.md).

## Testing

[Testing instructions](docs/testing.md) distinguish the default suite, host
integration checks, CUDA tests, and the separately compiled CPU material hook.
CI builds the core and Python package and checks installation from a wheel.
GPU tests are run explicitly on a suitable machine.

## License

The library retains the bundle's [MIT license](LICENSE). Additional notices are
in [licenses](licenses) and [third_party](third_party). Third-party showcase
meshes, textures, videos, local-machine inventories, and generated experiment
results are excluded from this repository. Examples use generated geometry.
