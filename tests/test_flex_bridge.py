"""Exercise the installed bridge on a tiny generated flex, with no visual assets."""

import os
import struct

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

from deform3d_warp.bindings import RECORD, rest_hash, topology_hash
from deform3d_warp.mujoco_bridge import FlexEmbedding
from deform3d_warp.surface_bindings import RECORD as SURFACE_RECORD


@pytest.mark.integration
@pytest.mark.parametrize("dimension", [2, 3])
def test_batched_step_refresh_and_binding_validation(tmp_path, dimension):
    points = "0 0 1  .1 0 1  0 .1 1" + (" 0 0 1.1" if dimension == 3 else "")
    elements = "0 1 2 3" if dimension == 3 else "0 1 2"
    count = dimension + 1
    cm = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep="0.001"/>
      <worldbody><flexcomp name="patch" dim="{dimension}" type="direct" mass="1"
        point="{points}" element="{elements}">
        <contact contype="0" conaffinity="0"/>
      </flexcomp></worldbody></mujoco>''')
    cd = mujoco.MjData(cm)
    mujoco.mj_forward(cm, cd)
    path = tmp_path / "tet.bind"
    records = np.zeros(1, dtype=RECORD if dimension == 3 else SURFACE_RECORD)
    records["indices"] = [list(range(count))]
    records["weights"] = [[1 / count] * count]
    if dimension == 2:
        records["normal_offset"] = .02
    magic = b"D3EMB003" if dimension == 3 else b"D3SUR001"
    header = magic + struct.pack(
        "<IIIIQQ", 1, 0, 0, count, topology_hash(cm.flex_elem), rest_hash(cd.flexvert_xpos)
    )
    path.write_bytes(header + records.tobytes())
    with wp.ScopedDevice(os.environ.get("DEFORM3D_WARP_DEVICE", "cpu")):
        model = mjw.put_model(cm)
        data = mjw.put_data(cm, cd, nworld=2)
        bridge = FlexEmbedding(cm, data, path)
        for _ in range(3):
            mjw.step(model, data)
        before = data.time.numpy().copy()
        result = bridge.update(model, data).numpy()
        np.testing.assert_array_equal(data.time.numpy(), before)
        expected = data.flexvert_xpos.numpy().mean(axis=1)
        offset = .02 if dimension == 2 else 0
        expected[:, 2] += offset
        np.testing.assert_allclose(result[:, 0], expected, atol=1e-6)
        assert result[0, 0, 2] < cd.flexvert_xpos[:, 2].mean() + offset
        if dimension == 2:
            np.testing.assert_array_equal(bridge.validity.numpy(), 1)
        np.testing.assert_allclose(result[0], result[1], atol=1e-6)
        changed = bytearray(path.read_bytes())
        changed[24] ^= 1
        path.write_bytes(changed)
        with pytest.raises(ValueError, match="topology"):
            FlexEmbedding(cm, data, path)
