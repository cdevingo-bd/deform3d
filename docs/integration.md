# MuJoCo and MuJoCo Warp

For a quick Python-only check in an existing MuJoCo/MJWarp environment:

```bash
python examples/simulator_smoke.py --backend mjwarp --device cpu
```

Use `--device cuda:0` to exercise the same bridge on the GPU. This test requires
no C++ exporter or external assets; the full exporter workflow follows below.

## Compile the C++ adapters

In a Python environment with MuJoCo 4.0.1 installed:

```bash
python tools/configure_mujoco.py
cmake --build build --config Release --parallel 2
ctest --test-dir build -C Release --output-on-failure
```

The helper locates headers and the matching library in the active Python
environment. Linux is the primary host-integration platform. For another
MuJoCo installation, set `MUJOCO_INCLUDE_DIR` and `MUJOCO_LIBRARY` directly in
CMake. Never mix headers from one version with a library from another.

The tetra adapter is `deform3d::MujocoFlexAdapter<T>` in
`deform3d/mujoco_adapter.hpp`; the shell adapter is in
`deform3d/mujoco_surface_adapter.hpp`. Bind once after `mj_forward` at the
reference pose. After `mj_step`, refresh derived geometry with
`refresh_mujoco_flex_positions`, then evaluate the embedding. Do not read or
write the same `mjData` concurrently.

## Export and run a binding

The cavity fixture is generated geometry and requires no external meshes.
On Linux, after the C++ build:

```bash
build/deform3d_export_bindings examples/mujoco_cavity.xml build/cavity.bind 2
python examples/mjwarp_embedding.py examples/mujoco_cavity.xml build/cavity.bind \
  --device cpu --steps 2 --validate
```

Use `--device cuda:0` for GPU execution. To check contact and CUDA graph replay:

```bash
python tests/optional_mjwarp.py examples/mujoco_cavity.xml build/cavity.bind \
  --device cuda:0 --graph
```

For an existing host loop, use the installed package:

```python
from deform3d_warp.mujoco_bridge import FlexEmbedding

dense = FlexEmbedding(cpu_model, warp_data, "surface.bind")
# Your loop owns the physics step.
mjw.step(warp_model, warp_data)
dense.update(warp_model, warp_data)
```

`update` refreshes kinematics/flex geometry from integrated qpos, then enqueues
the dense gather. It does not advance time or change solver behavior. Geometry
refresh does not recompute contact forces or sensors. Use `refresh=False` only
when the host already refreshed these positions. The output remains on the
Warp device, shaped `[world, dense_vertex]` with `vec3f` elements.

The same bridge accepts tetra (`D3EMB003`) and shell (`D3SUR001`) files. Shell
`validity` is 1 for success and 0 for an invalid triangle or numeric result;
invalid entries preserve the previous output. Check validity before consuming
the mesh. Zero validity on the first update means the output is not initialized.

## Verified host versions

The geometry bridge (`FlexEmbedding`, the C++ adapters and the binding exporters)
was also checked against a MuJoCo 3.17 stack, in addition to 4.0.1:

| Host | MuJoCo | Warp | mujoco_warp | Result |
| --- | --- | --- | --- | --- |
| Experimental physics | 4.0.1 | 1.19.0 | 4.0.1 | Full suite |
| 3.17 stack | 3.17.0 | 1.16.0 | 3.17.0 source build (patched) | 16 Python integration tests pass on CPU and CUDA; C++ adapters and exporters build and run |

On MuJoCo 3.17.0, two C++ checks fail: `mujoco_simplex_rest` and
`mujoco_contact_audit`. Both exercise the experimental contact/CCD-tolerance
behavior, not the embedding path. Do not use the experimental contact policies
on 3.17; they need the 4.0.1 pins above. The direct effective-metric solves and
`NewtonFactorPolicy` in `docs/experimental.md` carry their own version or
source-hash guards and were measured on 3.17.

`bench/rollout_ab.py` measures the geometry bridge's cost against a native
`mjwarp.step` CUDA graph, and checks that physics results are unchanged relative
to a native-vs-native control. Measured on one laptop GPU (8 GB) with a generated
bag scene and the 3.17.0 source build: about 0.6-1% added step time at
6k dense points and about 4% at 100k points, at 1-64 worlds. The cost at large
point counts is the embedding kernel's memory writes, not the kinematics refresh.
Treat these as indicative for that hardware only.

## Boundaries

Dense meshes do not automatically become colliders. Native MuJoCo flexes remain
the collision geometry. The force-transfer APIs can support custom two-way
coupling, but the application must calculate those forces and integrate them.
Likewise, a device output array is not automatically visible to the MuJoCo or
MJLab camera renderer; connecting a dynamic render mesh is application work.

Rebuild bindings after topology/rest-pose changes. Rebuild captured graphs after
replacing model/data/output arrays. Keep simulation and mesh updates on the same
stream, or explicitly synchronize them. Measure host staging separately if a
renderer requires CPU data.
