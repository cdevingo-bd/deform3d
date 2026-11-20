"""Process-scoped allocation controls must restore globals, including errors."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
import warp as wp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from mjwarp_capacity import epa_horizon_capacity
from mujoco_warp._src import collision_flex,collision_convex


class CapacityTests(unittest.TestCase):
    def test_restore_and_nested(self):
        self.assertEqual(collision_flex.MJ_MAX_EPAHORIZON,24)
        with self.assertRaisesRegex(ValueError,'sentinel'):
            with epa_horizon_capacity(128):
                self.assertEqual(collision_flex.MJ_MAX_EPAHORIZON,128)
                self.assertEqual(collision_convex.MJ_MAX_EPAHORIZON,128)
                model=NS(opt=NS(ccd_iterations=8),nmesh=1,has_ellipsoid_geom=True,
                         has_flex_selfcollide=False,nflex=1,has_1d_flex=False,has_3d_flex=True)
                data=NS(naccdmax=2,naconmax=4,nworld=1)
                # Exercise the actual upstream allocator, rather than checking
                # only that its configuration constant changed.
                with wp.ScopedDevice('cpu'):
                    workspace=collision_flex._allocate_flex_workspace(model,data)
                self.assertEqual(workspace.epa_horizon.shape,(2,128))
                with self.assertRaisesRegex(RuntimeError,'nested'):
                    with epa_horizon_capacity(64):pass
                raise ValueError('sentinel')
        self.assertEqual(collision_flex.MJ_MAX_EPAHORIZON,24)
        self.assertEqual(collision_convex.MJ_MAX_EPAHORIZON,24)

    def test_invalid_sizes(self):
        for value in (True,0,23,4097,128.,'128'):
            with self.assertRaises(ValueError):
                with epa_horizon_capacity(value):pass
        self.assertEqual(collision_flex.MJ_MAX_EPAHORIZON,24)

    def test_version_and_existing_override(self):
        with patch('mjwarp_capacity.version',return_value='unknown'):
            with self.assertRaisesRegex(RuntimeError,'reviewed'):
                with epa_horizon_capacity(128):pass
        with patch.object(collision_convex,'MJ_MAX_EPAHORIZON',32):
            with self.assertRaisesRegex(RuntimeError,'modified'):
                with epa_horizon_capacity(128):pass
            self.assertEqual(collision_flex.MJ_MAX_EPAHORIZON,24)
            self.assertEqual(collision_convex.MJ_MAX_EPAHORIZON,32)


if __name__=='__main__':unittest.main(verbosity=2)
