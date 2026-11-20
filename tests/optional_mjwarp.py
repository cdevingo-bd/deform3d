"""Opt-in MJWarp CPU parity test. Requires matching mujoco/mujoco-warp/warp.

Run after exporting bindings:
  python tests/optional_mjwarp.py examples/mujoco_cavity.xml /tmp/cavity.bind
"""

import argparse
from pathlib import Path
import sys
import tempfile
import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from mjwarp_embedding import (read_bindings, prepare_device_bindings, tetra_embed,
                              refresh_flex_positions)


def test(xml: Path, bindings_path: Path, device="cpu", graph=False):
    cpu_model = mujoco.MjModel.from_xml_path(str(xml))
    cpu_data = mujoco.MjData(cpu_model)
    mujoco.mj_forward(cpu_model, cpu_data)
    outer = mujoco.mj_name2id(cpu_model, mujoco.mjtObj.mjOBJ_FLEX, "cavity")
    inner = mujoco.mj_name2id(cpu_model, mujoco.mjtObj.mjOBJ_FLEX, "inner")
    probe = mujoco.mj_name2id(cpu_model, mujoco.mjtObj.mjOBJ_GEOM, "probe_geom")
    flex_id, vertex_offset, bindings = read_bindings(bindings_path, cpu_model)
    assert flex_id == outer
    # A stale or modified topology/rest fingerprint must be rejected before
    # any device upload. This also checks the versioned disk header roundtrip.
    original = bindings_path.read_bytes()
    for hash_byte in (24, 32):
        modified = bytearray(original)
        modified[hash_byte] ^= 1
        with tempfile.TemporaryDirectory() as directory:
            damaged = Path(directory) / "stale.bind"
            damaged.write_bytes(modified)
            try:
                read_bindings(damaged, cpu_model)
            except ValueError:
                pass
            else:
                raise AssertionError("stale binding fingerprint was accepted")
    with wp.ScopedDevice(device):
        model = mjw.put_model(cpu_model)
        data = mjw.make_data(cpu_model, nworld=2)
        ids, weights = prepare_device_bindings(bindings, device)
        render = wp.empty((2, len(bindings)), dtype=wp.vec3f, device=device)
        if graph:
            if not wp.get_device(device).is_cuda:
                raise ValueError("CUDA graph requires a CUDA device")
            # Compile kernels on disposable state; capture itself must not step.
            warm = mjw.make_data(cpu_model, nworld=2)
            mjw.step(model, warm)
            refresh_flex_positions(model, warm)
            wp.synchronize_device(device)
            with wp.ScopedCapture(device=device) as capture:
                mjw.step(model, data)
                refresh_flex_positions(model, data)
        flex_geom_contacts = 0
        flex_flex_contacts = 0
        for _ in range(120):
            mujoco.mj_step(cpu_model, cpu_data)
            mujoco.mj_kinematics(cpu_model, cpu_data)
            mujoco.mj_flex(cpu_model, cpu_data)
            if graph:
                wp.capture_launch(capture.graph)
            else:
                mjw.step(model, data)
                refresh_flex_positions(model, data)
            # This host inspection is test-only; the render kernel below keeps
            # simulation positions on the device in a real frame loop.
            count = int(data.nacon.numpy()[0])
            world = data.contact.worldid.numpy()[:count] == 0
            flex = data.contact.flex.numpy()[:count][world]
            geom = data.contact.geom.numpy()[:count][world]
            flex_geom_contacts += int(np.sum(
                ((flex[:, 0] == outer) & (geom[:, 1] == probe)) |
                ((flex[:, 1] == outer) & (geom[:, 0] == probe))))
            flex_flex_contacts += int(np.sum(
                ((flex[:, 0] == outer) & (flex[:, 1] == inner)) |
                ((flex[:, 1] == outer) & (flex[:, 0] == inner))))
        wp.launch(tetra_embed, dim=(2, len(bindings)),
                  inputs=[data.flexvert_xpos, ids, weights, vertex_offset],
                  outputs=[render], device=device)
        sim = data.flexvert_xpos.numpy()
        dense = render.numpy()
        reference = np.sum(sim[:, vertex_offset + bindings["indices"]] *
                           bindings["weights"][None, :, :, None], axis=2)
        kernel_error = np.max(np.linalg.norm(dense - reference, axis=2))
        flex_error = np.max(np.linalg.norm(sim[0] - cpu_data.flexvert_xpos, axis=1))
        qpos_error = np.max(np.abs(data.qpos.numpy()[0] - cpu_data.qpos))
        print(f"MJWarp device={device} graph={graph} flex-geom contacts={flex_geom_contacts} "
              f"flex-flex contacts={flex_flex_contacts} "
              f"dense max error={kernel_error:.3g} "
              f"MuJoCo flex position delta={flex_error:.3g} "
              f"qpos delta={qpos_error:.3g}")
        assert np.isfinite(sim).all() and np.isfinite(dense).all()
        assert flex_geom_contacts > 0 and flex_flex_contacts > 0
        assert kernel_error < 3e-5
        # Broad smoke thresholds, not sim-to-real calibration or bitwise parity.
        assert flex_error < 5e-3 and qpos_error < 2e-2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path)
    parser.add_argument("bindings", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    test(args.xml, args.bindings, args.device, args.graph)
