"""Geometry-only MJWarp bridge parity, including graph replay on Pascal GPUs.

This does not invoke MJWarp's contact solver, whose current kernels exceed the
Pascal kernel parameter limit. Positions are compared with current CPU qpos.
"""
from pathlib import Path
import sys
import argparse
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from mjwarp_embedding import refresh_flex_positions


def run(device):
    model = mujoco.MjModel.from_xml_path(str(ROOT / "examples/mujoco_cavity.xml"))
    cpu = mujoco.MjData(model)
    mujoco.mj_forward(model, cpu)
    with wp.ScopedDevice(device):
        gpu_model = mjw.put_model(model)
        gpu_data = mjw.put_data(model, cpu)
        refresh_flex_positions(gpu_model, gpu_data)
        wp.synchronize_device(device)
        initial = gpu_data.flexvert_xpos.numpy().copy()
        capture = None
        if wp.get_device(device).is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                refresh_flex_positions(gpu_model, gpu_data)
        worst = 0.0
        for _ in range(3):
            cpu.qvel[:] = 0.25
            mujoco.mj_step(model, cpu)
            mujoco.mj_kinematics(model, cpu)
            mujoco.mj_flex(model, cpu)
            # Test fixture updates qpos only: flex positions must come from
            # kinematics on the selected device, not a copied CPU flex array.
            gpu_data.qpos.assign(np.asarray(cpu.qpos[None], dtype=np.float32))
            if capture:
                wp.capture_launch(capture.graph)
            else:
                refresh_flex_positions(gpu_model, gpu_data)
            actual = gpu_data.flexvert_xpos.numpy()[0]
            error = float(np.max(np.abs(actual - cpu.flexvert_xpos)))
            worst = max(worst, error)
            if not np.isfinite(actual).all() or error > 2e-6:
                raise AssertionError(f"stale or incorrect flex positions: {error}")
        movement = float(np.max(np.abs(actual - initial[0])))
        if movement < 1e-5:
            raise AssertionError("geometry did not move with integrated qpos")
        print(f"MJWarp geometry device={device} graph={bool(capture)} "
              f"max_error={worst:.8g} movement={movement:.8g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args().device)
