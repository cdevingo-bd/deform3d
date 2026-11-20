"""Audit matching must expose feature changes and retain the original trajectory."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from audit_contact_checkpoints import compare_contacts
from check_contact_convergence import trajectory


def record(element=7):
    return dict(geom=np.array([0, -1]), flex=np.array([-1, 0]), elem=np.array([-1, element]),
                vert=np.array([-1, -1]), dist=-.001, pos=np.array([1., 2., 3.]),
                normal=np.array([1., 0., 0.]))


class ContactCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(geom_type=[mujoco.mjtGeom.mjGEOM_ELLIPSOID])

    def test_reversed_sides_compare_without_sign_error(self):
        a, b = record(), record()
        for field in ('geom', 'flex', 'elem', 'vert'):
            b[field] = b[field][::-1]
        b['normal'] = -b['normal']
        r = compare_contacts(self.model, [a], [b])
        self.assertEqual(r['matched_unique_features'], 1)
        for name in ('distance_delta_m', 'position_delta_m', 'unit_normal_delta'):
            self.assertEqual(r['matched_features'][0][name], 0.)

    def test_multiplicity_and_missing_features_are_not_arbitrarily_paired(self):
        r = compare_contacts(self.model, [record(), record(), record(8)], [record(), record(), record(9)])
        self.assertEqual(r['matched_unique_features'], 0)
        self.assertEqual(len(r['ambiguous_or_unmatched']), 3)
        self.assertEqual(sorted((x['reference_count'], x['candidate_count']) for x in r['ambiguous_or_unmatched']),
                         [(0, 1), (1, 0), (2, 2)])

    def test_reports_actual_geometry_differences_without_alignment(self):
        a, b = record(), record()
        b['dist'] -= .0001
        b['pos'][0] += .002
        b['normal'] = np.array([0., 1., 0.])
        r = compare_contacts(self.model, [a], [b])['by_category']['ellipsoid_flex']
        self.assertAlmostEqual(r['max_distance_delta_m'], .0001)
        self.assertAlmostEqual(r['max_position_delta_m'], .002)
        self.assertAlmostEqual(r['max_unit_normal_delta'], np.sqrt(2))

    def test_snapshot_observer_preserves_cpu_history_and_pre_step_timing(self):
        scene = ROOT/'examples/mujoco_bag_strong_ref.xml'
        observed = []
        def snapshot(model, data, step):
            specification = mujoco.mjtState.mjSTATE_INTEGRATION
            state = np.empty(mujoco.mj_stateSize(model, specification))
            mujoco.mj_getState(model, data, state, specification)
            observed.append((step, data.time, state.copy()))
        a = trajectory(scene, 1e-14, .004, .2)
        b = trajectory(scene, 1e-14, .004, .2, snapshot=snapshot)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_allclose([x[1] for x in observed], np.arange(4)*.001, atol=1e-15)
        self.assertEqual([x[0] for x in observed], list(range(4)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
