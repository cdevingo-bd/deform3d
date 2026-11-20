"""Exercise the geometry bridge with MJWarp or an installed MJLab Simulation.

A free tetrahedron falls for a few steps. This checks array integration and
binding reconstruction, not material/contact accuracy or a trained policy.
"""

import argparse
from pathlib import Path
import struct
import tempfile

import mujoco
import numpy as np
import warp as wp

from deform3d_warp.bindings import RECORD, rest_hash, topology_hash
from deform3d_warp.mujoco_bridge import FlexEmbedding

XML = '''<mujoco><option timestep="0.001"/>
  <worldbody><flexcomp name="tet" dim="3" type="direct" mass="1"
    point="0 0 1  .1 0 1  0 .1 1  0 0 1.1" element="0 1 2 3">
    <contact contype="0" conaffinity="0"/>
  </flexcomp></worldbody></mujoco>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mjwarp", "mjlab"], default="mjwarp")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.backend == "mjlab" and not args.device.startswith("cuda"):
        parser.error("the MJLab smoke example requires --device cuda:0")
    with wp.ScopedDevice(args.device), tempfile.TemporaryDirectory() as directory:
        cm = mujoco.MjModel.from_xml_string(XML)
        if args.backend == "mjlab":
            from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
            sim = Simulation(
                num_envs=2, model=cm, device=args.device,
                cfg=SimulationCfg(mujoco=MujocoCfg(timestep=.001, integrator="euler")),
            )
            cm, model, data = sim.mj_model, sim.wp_model, sim.wp_data
            step = sim.step
        else:
            import mujoco_warp as mjw
            model = mjw.put_model(cm)
            cd = mujoco.MjData(cm)
            mujoco.mj_forward(cm, cd)
            data = mjw.put_data(cm, cd, nworld=2)
            step = lambda: mjw.step(model, data)
        cd = mujoco.MjData(cm)
        mujoco.mj_forward(cm, cd)
        # A real application exports dense surface bindings with the C++ tools.
        # Here one analytically known centroid makes the smoke test asset-free.
        records = np.zeros(1, dtype=RECORD)
        records["indices"] = [[0, 1, 2, 3]]
        records["weights"] = .25
        header = b"D3EMB003" + struct.pack(
            "<IIIIQQ", 1, 0, 0, 4, topology_hash(cm.flex_elem), rest_hash(cd.flexvert_xpos)
        )
        path = Path(directory) / "tet.bind"
        path.write_bytes(header + records.tobytes())
        mesh = FlexEmbedding(cm, data, path)
        for _ in range(5):
            step()
            mesh.update(model, data)
        points = mesh.positions.numpy()
        expected = data.flexvert_xpos.numpy().mean(axis=1)
        np.testing.assert_allclose(points[:, 0], expected, atol=1e-6)
        assert np.isfinite(points).all()
        assert points[0, 0, 2] < cd.flexvert_xpos[:, 2].mean()
        print(f"PASS: {args.backend} / {args.device}, two worlds, five steps, finite dense positions")


if __name__ == "__main__":
    main()
