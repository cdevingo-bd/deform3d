"""Optional MJWarp device-side tetra embedding using C++ prepared bindings.

Run: python examples/mjwarp_embedding.py examples/mujoco_cavity.xml cavity.bind --device cpu
For a CUDA device, use --device cuda:0. No per-frame GPU-to-host position readback
occurs; only --validate reads the output for verification after stepping.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp


from mujoco_bindings import MAGIC, RECORD, topology_hash, rest_hash, read_bindings

@wp.kernel
def tetra_embed(
    flex_xyz: wp.array2d(dtype=wp.vec3f),
    local_ids: wp.array(dtype=wp.vec4i),
    weights: wp.array(dtype=wp.vec4f),
    flex_vertex_offset: int,
    output: wp.array2d(dtype=wp.vec3f),
):
    world, vertex = wp.tid()
    ids = local_ids[vertex]
    w = weights[vertex]
    output[world, vertex] = (
        flex_xyz[world, flex_vertex_offset + ids[0]] * w[0]
        + flex_xyz[world, flex_vertex_offset + ids[1]] * w[1]
        + flex_xyz[world, flex_vertex_offset + ids[2]] * w[2]
        + flex_xyz[world, flex_vertex_offset + ids[3]] * w[3]
    )


def prepare_device_bindings(records, device):
    ids = np.ascontiguousarray(records["indices"], dtype=np.int32)
    weights = np.ascontiguousarray(records["weights"], dtype=np.float32)
    return (wp.array(ids, dtype=wp.vec4i, device=device),
            wp.array(weights, dtype=wp.vec4f, device=device))


def refresh_flex_positions(model, data):
    """Refresh geometry from integrated qpos, without another dynamics solve.

    Like MuJoCo, mjw.step leaves derived positions at the last pre-integration
    state. These device launches are stream ordered and CUDA graph compatible.
    Contacts and sensors are not refreshed by this geometry-only operation.
    """
    mjw.kinematics(model, data)
    mjw.flex(model, data)


def run(xml: Path, binding_file: Path, *, device: str, steps: int,
        validate: bool = False):
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    model = mujoco.MjModel.from_xml_path(str(xml))
    flex_id, offset, bindings = read_bindings(binding_file, model)
    with wp.ScopedDevice(device):
        m = mjw.put_model(model)
        d = mjw.make_data(model, nworld=1)
        ids, weights = prepare_device_bindings(bindings, device)
        output = wp.empty((d.nworld, len(bindings)), dtype=wp.vec3f, device=device)
        for _ in range(steps):
            mjw.step(m, d)
            refresh_flex_positions(m, d)
            wp.launch(tetra_embed, dim=(d.nworld, len(bindings)),
                      inputs=[d.flexvert_xpos, ids, weights, offset],
                      outputs=[output], device=device)
        if steps == 0:
            # mjw.forward updates derived flex positions without advancing time.
            mjw.forward(m, d)
            wp.launch(tetra_embed, dim=(d.nworld, len(bindings)),
                      inputs=[d.flexvert_xpos, ids, weights, offset],
                      outputs=[output], device=device)
        wp.synchronize_device(device)
        if validate:
            xyz = d.flexvert_xpos.numpy()[0]
            visual = output.numpy()[0]
            ref = np.sum(xyz[offset + bindings["indices"]] *
                         bindings["weights"][..., None], axis=1)
            error = np.max(np.linalg.norm(visual - ref, axis=1))
            if not np.isfinite(error) or error > 3e-5:
                raise AssertionError(f"Warp embedding differs from CPU: {error}")
            print(f"device={device} flex={flex_id} vertices={len(bindings)} "
                  f"steps={steps} max_error={error:.3g}")
        else:
            print(f"device={device} flex={flex_id} vertices={len(bindings)} steps={steps}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path)
    parser.add_argument("bindings", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    run(args.xml, args.bindings, device=args.device, steps=args.steps,
        validate=args.validate)
