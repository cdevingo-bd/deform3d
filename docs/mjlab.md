# Using deform3d in MJLab

Install into the environment that already runs your MJLab task:

```bash
python -m pip install -e /path/to/deform3d
python /path/to/deform3d/examples/simulator_smoke.py --backend mjlab --device cuda:0
```

The base package depends only on NumPy and Warp. Do not add the `physics` extra:
the experimental policies require MuJoCo/MJWarp 4.0.1, whereas the locally
inspected MJLab 1.6.0 package requires 3.17. The geometry bridge uses public host
arrays and leaves those pins and the simulation solver unchanged.

The smoke example creates a tiny two-world `Simulation` with generated geometry
and checks five steps of dense position updates. It has no downloaded assets or
training service dependencies. Run it on the target MJLab/GPU environment; it
was not run on CUDA during repository preparation.

Prepare a binding against the task's **compiled model**, after entity assembly.
Exporting against an isolated object can produce different flex IDs, offsets,
or reference coordinates. The loader rejects mismatches. See
[the exporter instructions](integration.md).

```python
from deform3d_warp.mujoco_bridge import FlexEmbedding

# Run once after task/scene construction. `sim` is the task's Simulation.
dense = FlexEmbedding(sim.mj_model, sim.wp_data, "task_surface.bind")

# The task performs its usual action, stepping and reset logic.
# Run this at the point where updated geometry is needed:
dense.update(sim.wp_model, sim.wp_data)
positions = dense.positions
```

This uses `Simulation.mj_model`, `wp_model`, and `wp_data`, as exposed by the
inspected 1.6.0 source. The CPU bridge regression exercises the corresponding
MuJoCo/MJWarp 3.17 arrays. A complete MJLab training task and its camera renderer
are not validated by that regression.

If a consumer needs Torch tensors, `warp.to_torch(positions)` exposes the device
buffer without a CPU copy. Respect the producer/consumer CUDA stream ordering.
Reusing the view is fine while the output allocation remains unchanged.

Resets that preserve topology can reuse the bindings; refresh positions after
the reset. Model variants with different flex topology or reference geometry
need their own bindings. Warm up before capturing geometry kernels in a CUDA
graph. Recapture if a referenced buffer is replaced.

For camera observations, wire the dense positions and updated normals into the
renderer that produces those observations. Updating a viewer or a tensor alone
does not update policy camera images. This repository supplies the deformation
and validation pieces, not a replacement MJLab rendering backend or RL task.
