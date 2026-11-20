"""Local-coordinate flex contact checks against analytic face geometry."""
import inspect
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import collision_flex as flex
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from deform3d_warp.mujoco_local_contact import _configured_factory, _translate_source, local_flex_contact_policy


class LocalContactTests(unittest.TestCase):
    def test_native_and_transformed_factories_have_independent_cache_entries(self):
        # Populate native first to reproduce the former name-based collision.
        native = flex._flex_narrowphase(0)
        configured = _configured_factory(flex._flex_narrowphase)
        changed = configured(0)
        self.assertIsNot(changed, native)
        self.assertIs(configured(0), changed)
        self.assertIs(flex._flex_narrowphase(0), native)
        # Also check the opposite initialization order with another key.
        changed_second = configured(1)
        native_second = flex._flex_narrowphase(1)
        self.assertIsNot(changed_second, native_second)
        self.assertIs(flex._flex_narrowphase(1), native_second)

    def test_source_guard(self):
        source = inspect.getsource(flex._flex_narrowphase)
        changed = _translate_source(source)
        self.assertIn('origin + 0.5 * (w1 + w2)', changed)
        for invalid in (source+'\n', changed):
            with self.assertRaisesRegex(RuntimeError, 'unreviewed'):
                _translate_source(invalid)

    def test_context_restoration(self):
        original = flex._run_flex_narrowphase
        with self.assertRaisesRegex(ValueError, 'sentinel'):
            with local_flex_contact_policy(object()):
                with self.assertRaisesRegex(RuntimeError, 'nested'):
                    with local_flex_contact_policy(object()):
                        pass
                raise ValueError('sentinel')
        self.assertIs(original, flex._run_flex_narrowphase)
        with patch('deform3d_warp.mujoco_local_contact.version', return_value='unreviewed'):
            with self.assertRaisesRegex(RuntimeError, '4.0.1'):
                with local_flex_contact_policy(object()):
                    pass

    def test_analytic_face_and_translation_graph(self):
        with wp.ScopedDevice('cuda:0'):
            for position in ('0 0 0', '4 -2 8', '-128 0 32'):
                with self.subTest(position=position):
                    cpu_model = mujoco.MjModel.from_xml_string(f'''<mujoco>
                      <option gravity="0 0 0" ccd_tolerance="1e-9"/>
                      <worldbody>
                        <flexcomp name="sheet" type="direct" dim="2" mass=".1" radius=".001"
                          pos="{position}" point="-.1 -.1 0 .1 -.1 0 0 .1 0" element="0 1 2"/>
                        <flexcomp name="tet" type="direct" dim="3" mass=".1" radius=".001"
                          pos="{position}" point="0 0 .0005 .02 0 .02 0 .02 .02 0 0 .04" element="0 1 2 3"/>
                      </worldbody></mujoco>''')
                    model = mjw.put_model(cpu_model)
                    data = mjw.make_data(cpu_model)
                    with local_flex_contact_policy(model):
                        mjw.fwd_position(model, data, factorize=False)
                        wp.synchronize_device('cuda:0')
                        with wp.ScopedCapture() as capture:
                            mjw.fwd_position(model, data, factorize=False)
                    # Change the state after capture, so replay cannot pass with
                    # stale contact positions. The tetra's last 12 DOFs translate.
                    for displacement in (0., .0002, -.0001):
                        qpos = cpu_model.qpos0.astype(np.float32)[None].copy()
                        qpos[0, -12+2::3] += displacement
                        data.qpos.assign(qpos)
                        wp.capture_launch(capture.graph)
                        count = int(data.nacon.numpy()[0])
                        self.assertEqual(count, 1)
                        xyz = data.flexvert_xpos.numpy()[0].astype(np.float64)
                        expected_gap = xyz[3, 2]-xyz[0, 2]-.002
                        expected_midpoint = (xyz[3]+np.array([xyz[3, 0], xyz[3, 1], xyz[0, 2]]))*.5
                        frame = data.contact.frame.numpy()[0]
                        np.testing.assert_allclose(frame[0], [0, 0, 1], atol=2e-4)
                        self.assertAlmostEqual(float(data.contact.dist.numpy()[0]), expected_gap, delta=2e-6)
                        np.testing.assert_allclose(data.contact.pos.numpy()[0], expected_midpoint, atol=2e-5)
                        np.testing.assert_array_equal(data.overflow.numpy(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
