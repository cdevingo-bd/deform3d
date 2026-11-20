"""Geometric, differential and capture tests for pre-inflation normal transport."""
import inspect
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import collision_flex as flex
from mujoco_warp._src import collision_gjk as gjk
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from deform3d_warp.mujoco_contact_normals import (
    _ccd_source, _flex_source, stable_flex_normals_policy)


def fixture(position='0 0 0', quaternion='1 0 0 0', height=.001999):
    return mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option gravity="0 0 0" ccd_tolerance="1e-8"/>
      <worldbody>
        <flexcomp name="sheet" type="direct" dim="2" mass=".1" radius=".001"
          pos="{position}" quat="{quaternion}" point="-.1 -.1 0 .1 -.1 0 0 .1 0" element="0 1 2"/>
        <flexcomp name="tet" type="direct" dim="3" mass=".1" radius=".001"
          pos="{position}" quat="{quaternion}"
          point="0 0 {height} .02 0 .02 0 .02 .02 0 0 .04" element="0 1 2 3"/>
      </worldbody></mujoco>''')


def state(model, cpu_model):
    data = mjw.make_data(cpu_model, nworld=2)
    mjw.fwd_position(model, data, factorize=False)
    return data


def records(data):
    count = int(data.nacon.numpy()[0])
    out = {k: getattr(data.contact, k).numpy()[:count] for k in (
        'dist', 'pos', 'frame', 'geom', 'flex', 'elem', 'vert', 'worldid')}
    order = np.lexsort((out['elem'][:, 1], out['elem'][:, 0], out['worldid']))
    return {k: v[order] for k, v in out.items()}


class ContactNormalTests(unittest.TestCase):
    def test_source_guards_and_private_factory(self):
        ccd, source = inspect.getsource(gjk.ccd.func), inspect.getsource(flex._flex_narrowphase)
        self.assertIn('result.x2 - result.x1', _ccd_source(ccd))
        changed, name = _flex_source(source)
        self.assertIn('def '+name+'(', changed)
        self.assertNotEqual(name, '_flex_narrowphase')
        for transform, text in ((_ccd_source, ccd), (_flex_source, source)):
            with self.assertRaisesRegex(RuntimeError, 'unreviewed'):
                transform(text+'\n')

    def test_scope_restoration_and_version_guard(self):
        original = flex._run_flex_narrowphase
        with self.assertRaisesRegex(ValueError, 'sentinel'):
            with stable_flex_normals_policy(object()):
                with self.assertRaisesRegex(RuntimeError, 'nested'):
                    with stable_flex_normals_policy(object()):
                        pass
                raise ValueError('sentinel')
        self.assertIs(original, flex._run_flex_narrowphase)
        with patch('deform3d_warp.mujoco_contact_normals.version', return_value='unreviewed'):
            with self.assertRaisesRegex(RuntimeError, '4.0.1'):
                with stable_flex_normals_policy(object()):
                    pass

    def test_local_coordinate_experiment_cannot_overlap(self):
        from deform3d_warp.mujoco_local_contact import local_flex_contact_policy
        original = flex._run_flex_narrowphase
        for first, second in ((stable_flex_normals_policy, local_flex_contact_policy),
                              (local_flex_contact_policy, stable_flex_normals_policy)):
            with first(object()):
                with self.assertRaises(RuntimeError):
                    with second(object()):
                        pass
            self.assertIs(original, flex._run_flex_narrowphase)

    def test_rotated_shallow_contact_geometry_and_unchanged_witnesses(self):
        with wp.ScopedDevice('cuda:0'):
            # Expected face normals come from realized GPU input vertices in
            # double precision, not from the intended pose before quantization.
            for position in ('0 0 0', '.35 -.21 .73', '4 -2 8'):
                for quaternion in ('1 0 0 0', '.866025403784 .288675134595 .288675134595 .288675134595'):
                    for height in (.0019999, .00199, .0015):
                        with self.subTest(position=position, quaternion=quaternion, height=height):
                            cm = fixture(position, quaternion, height)
                            m = mjw.put_model(cm)
                            native = state(m, cm)
                            with stable_flex_normals_policy(m):
                                owned = state(m, cm)
                            a, b = records(native), records(owned)
                            for name in a:
                                if name != 'frame':
                                    np.testing.assert_array_equal(a[name], b[name], err_msg=name)
                            # Some tiny penetrations round to separation after
                            # large translations; contact presence stays native.
                            if not len(b['dist']):
                                continue
                            xyz = owned.flexvert_xpos.numpy().astype(np.float64)
                            for i, w in enumerate(b['worldid']):
                                n = np.cross(xyz[w, 1]-xyz[w, 0], xyz[w, 2]-xyz[w, 0])
                                n /= np.linalg.norm(n)
                                if np.dot(n, xyz[w, 3]-xyz[w, 0]) < 0:
                                    n = -n
                                np.testing.assert_allclose(b['frame'][i, 0], n, atol=5e-4)
                            np.testing.assert_array_equal(owned.overflow.numpy(), 0)

    def test_deep_core_overlap_keeps_native_result(self):
        with wp.ScopedDevice('cuda:0'):
            cm = fixture(height=-.004)
            m = mjw.put_model(cm)
            a = records(state(m, cm))
            with stable_flex_normals_policy(m):
                b = records(state(m, cm))
            self.assertGreater(len(a['dist']), 0)
            for name in a:
                np.testing.assert_array_equal(a[name], b[name])

    def test_changing_graph_inputs_and_other_model_scope(self):
        with wp.ScopedDevice('cuda:0'):
            cm = fixture(position='.35 -.21 .73', height=.001999)
            m, other = mjw.put_model(cm), mjw.put_model(cm)
            native = records(state(other, cm))
            with stable_flex_normals_policy(m):
                scoped_other = records(state(other, cm))
                for name in native:
                    np.testing.assert_array_equal(native[name], scoped_other[name])
                data = state(m, cm)
                with wp.ScopedCapture() as capture:
                    mjw.fwd_position(m, data, factorize=False)
            previous = None
            for displacement in (0., -.0001, .0002, -.0002):
                qpos = np.tile(cm.qpos0.astype(np.float32), (2, 1))
                qpos[:, -12+2::3] += displacement
                data.qpos.assign(qpos)
                wp.capture_launch(capture.graph)
                got = records(data)
                fresh = mjw.make_data(cm, nworld=2)
                fresh.qpos.assign(qpos)
                with stable_flex_normals_policy(m):
                    mjw.fwd_position(m, fresh, factorize=False)
                expected = records(fresh)
                for name in expected:
                    np.testing.assert_array_equal(got[name], expected[name])
                if previous is not None:
                    self.assertFalse(np.array_equal(previous, got['dist']))
                previous = got['dist'].copy()
                np.testing.assert_array_equal(data.overflow.numpy(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
