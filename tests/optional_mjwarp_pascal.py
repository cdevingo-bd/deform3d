"""Run the original nested contact parity test through an explicit patched copy.

python tools/mjwarp_pascal_compat.py --run tests/optional_mjwarp_pascal.py \
    examples/mujoco_cavity.xml build/cavity.bind --device cuda:0 --graph
"""

import argparse
from pathlib import Path
import sys

import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import collision_flex
import warp as wp

from optional_mjwarp import test


def run(xml, bindings, device, graph):
    if not getattr(collision_flex, "DEFORM3D_PASCAL_COMPAT", False):
        raise RuntimeError("select the opt-in copy using tools/mjwarp_pascal_compat.py --run")
    print(f"MJWarp explicit compatibility source: {collision_flex.__file__}", flush=True)
    with wp.ScopedDevice(device):
        # The persistent descriptor storage is reused and owns references to
        # its arrays; replacement requires rebuilding the model/graphs.
        model = mjw.put_model(mujoco.MjModel.from_xml_path(str(xml)))
        descriptor = collision_flex._deform3d_mesh_descriptors(model)
        assert descriptor.device == wp.get_device(device)
        assert collision_flex._deform3d_mesh_descriptors(model) is descriptor
        original = model.mesh_vert
        model.mesh_vert = wp.empty_like(original)
        try:
            try:
                collision_flex._deform3d_mesh_descriptors(model)
            except RuntimeError as error:
                assert "storage changed" in str(error)
            else:
                raise AssertionError("replaced mesh storage was silently accepted")
        finally:
            model.mesh_vert = original
        if wp.get_device(device).is_cuda:
            fresh = mjw.put_model(mujoco.MjModel.from_xml_path(str(xml)))
            wp.synchronize_device(device)
            wp.capture_begin(device=device)
            try:
                try:
                    collision_flex._deform3d_mesh_descriptors(fresh)
                except RuntimeError as error:
                    assert "warmup" in str(error)
                else:
                    raise AssertionError("first descriptor upload inside capture was accepted")
            finally:
                wp.capture_end(device=device)
    test(xml, bindings, device=device, graph=graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path)
    parser.add_argument("bindings", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    run(args.xml, args.bindings, args.device, args.graph)
