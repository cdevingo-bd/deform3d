"""Reproducible C++ export / Python reader surface ABI and corruption checks.

Run: python tests/test_surface_bindings.py --exporter build/deform3d_export_surface
Requires NumPy and MuJoCo. Creates all scene/mesh/record fixtures in a temporary
directory; does not require an asset download or a GPU.
"""
from pathlib import Path
import argparse
import struct
import subprocess
import sys
import tempfile

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from mujoco_surface_bindings import RECORD, read_surface_bindings

XML = """<mujoco model="surface_binding_test">
  <option timestep="0.001" gravity="0 0 0"/>
  <worldbody>
    <flexcomp name="solid" type="direct" dim="3" mass="1"
              point="10 0 0 11 0 0 10 1 0 10 0 1" element="0 1 2 3">
      <edge equality="true"/>
    </flexcomp>
    <flexcomp name="cloth" type="direct" dim="2" mass="1" radius="0.001"
              point="0 0 0 1 0 0 0 1 0 1 1 0" element="0 1 2 1 3 2">
      <edge equality="true"/>
    </flexcomp>
  </worldbody>
</mujoco>"""


def run(exporter, tetra_exporter=None):
    exporter = str(Path(exporter).resolve())
    with tempfile.TemporaryDirectory(prefix="deform3d-surface-test-") as directory:
        directory = Path(directory)
        xml = directory / "scene.xml"
        mesh = directory / "surface.xyz"
        path = directory / "surface.bind"
        xml.write_text(XML)
        rest = np.array([[.2, .3, .02], [.7, .8, -.03], [.4, .4, 0.]])
        np.savetxt(mesh, rest)
        command = [exporter, str(xml), "cloth", str(path), "--mesh", str(mesh), "--max-distance", "0.04"]
        subprocess.run(command, check=True)
        model = mujoco.MjModel.from_xml_path(str(xml))
        flex_id, offset, records = read_surface_bindings(path, model)
        assert flex_id == 1 and offset == 4 and RECORD.itemsize == 28
        assert len(records) == 3 and not records.flags.writeable
        np.testing.assert_allclose(records["normal_offset"], [.02, -.03, 0], atol=1e-7)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        def evaluate():
            triangles = data.flexvert_xpos[offset:offset + 4][records["indices"]]
            normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            normal /= np.linalg.norm(normal, axis=1)[:, None]
            return (triangles * records["weights"][:, :, None]).sum(axis=1) + normal * records["normal_offset"][:, None]

        np.testing.assert_allclose(evaluate(), rest, atol=1e-7)
        data.qvel[:] = .25
        mujoco.mj_step(model, data)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_flex(model, data)
        np.testing.assert_allclose(evaluate(), rest + .00025, atol=1e-7)

        raw = bytearray(path.read_bytes())
        corruptions = [
            ("magic", 0, b"BADMAGIC"),
            ("topology hash", 24, b"\0" * 8),
            ("reference hash", 32, b"\0" * 8),
            ("count", 8, struct.pack("<I", 4)),
            ("negative weight", 52, struct.pack("<f", -1.)),
            ("wrong weight sum", 52, struct.pack("<f", .9)),
            ("invalid triangle", 40, struct.pack("<III", 0, 0, 0)),
            ("reversed triangle", 40, struct.pack("<III", 0, 2, 1)),
            ("out of range vertex", 40, struct.pack("<I", 999)),
            ("NaN offset", 64, struct.pack("<f", float("nan"))),
            ("truncation", len(raw) - 1, None),
        ]
        invalid = directory / "invalid.bind"
        for label, position, replacement in corruptions:
            modified = bytearray(raw)
            if replacement is None:
                del modified[position:]
            else:
                modified[position:position + len(replacement)] = replacement
            invalid.write_bytes(modified)
            try:
                read_surface_bindings(invalid, model)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted corruption: {label}")

        # A different reference geometry and a differently oriented topology
        # must both be rejected even when vertex/element counts are unchanged.
        changed_models = [
            mujoco.MjModel.from_xml_string(XML.replace('point="0 0 0 1 0 0', 'point="0 0 0 1.1 0 0')),
            mujoco.MjModel.from_xml_string(XML.replace('element="0 1 2 1 3 2"', 'element="0 2 1 1 3 2"')),
        ]
        for changed in changed_models:
            try:
                read_surface_bindings(path, changed)
            except ValueError:
                pass
            else:
                raise AssertionError("accepted mismatched model")

        # At an open edge, preserve-offset mode cannot encode a tangential
        # residual. Projection is an explicit, separately tested alternative.
        np.savetxt(mesh, [[1.02, .5, .01]])
        rejected = subprocess.run(command, capture_output=True, text=True)
        assert rejected.returncode != 0 and "outside permitted surface" in rejected.stderr
        subprocess.run(command + ["--project"], check=True)
        _, _, projected = read_surface_bindings(path, model)
        assert projected["normal_offset"][0] == 0
        projection = (data.flexvert_xpos[offset:offset + 4][projected["indices"]]
                      * projected["weights"][:, :, None]).sum(axis=1)
        np.testing.assert_allclose(projection[0], [1.00025, .50025, .00025], atol=1e-7)

        # A tiny export fits stdio's buffer, so fwrite alone misses ENOSPC.
        # Linux /dev/full deterministically exercises the final flush error.
        if sys.platform.startswith("linux") and Path("/dev/full").is_char_device():
            failure_command = command.copy()
            failure_command[3] = "/dev/full"
            rejected = subprocess.run(failure_command + ["--project"], capture_output=True, text=True)
            assert rejected.returncode != 0 and "exported" not in rejected.stdout
            if tetra_exporter is not None:
                rejected = subprocess.run([str(Path(tetra_exporter).resolve()), str(xml),
                                           "/dev/full", "1", "--flex", "solid"],
                                          capture_output=True, text=True)
                assert rejected.returncode != 0 and "exported" not in rejected.stdout
        print(f"surface ABI: reconstruction, stepping, {len(corruptions)} corruptions, 2 model mismatches, and edge projection passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exporter", type=Path, default=ROOT / "build/deform3d_export_surface")
    parser.add_argument("--tetra-exporter", type=Path,
                        help="also verify this tetra exporter rejects a buffered write error")
    args = parser.parse_args()
    run(args.exporter, args.tetra_exporter)
