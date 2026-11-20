"""Topology proof cases and differential CUDA Hessian allocation checks."""
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'tools')]
from deform3d_warp.mujoco_determinism import SparseDeterminismBudget, _support_bound


def host_model(m):
    class Array:
        def __init__(self, value): self.value = value
        def numpy(self): return self.value
    names = ('body_parentid', 'body_dofadr', 'body_dofnum', 'geom_bodyid', 'flex_dim',
             'flex_interp', 'flex_vertadr', 'flex_vertnum', 'flex_vertbodyid',
             'flex_elemdataadr', 'flex_elemnum', 'flex_elem')
    return SimpleNamespace(**{n: Array(getattr(m, n).copy()) for n in names},
                           **{n: getattr(m, n) for n in ('nv', 'neq', 'ntendon', 'nbody', 'nflex')},
                           is_sparse=True)


class BoundTests(unittest.TestCase):
    def test_contact_support_and_ball_limits(self):
        m = mujoco.MjModel.from_xml_path(str(ROOT/'examples/mujoco_bag_strong_ref.xml'))
        self.assertEqual(_support_bound(host_model(m)), 24)
        # A ball limit can use three coordinates without any colliding geometry.
        m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
          <joint type="ball" limited="true" range="0 30"/>
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          </body></worldbody></mujoco>''')
        self.assertEqual(_support_bound(host_model(m)), 3)
        # Ancestor DOFs must count, including fixed descendants with geometry.
        m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body><freejoint/><geom size=".1"/>
            <body pos="0 0 .3"><joint/><geom size=".1"/>
              <body pos="0 0 .3"><geom size=".1"/></body></body></body>
          <body pos="1 0 0"><freejoint/><geom size=".1"/></body>
          </worldbody></mujoco>''')
        # Two sides are conservatively bounded by 2*7, clipped to all 13 DOFs.
        self.assertEqual(_support_bound(host_model(m)), 13)

    def test_unsupported_and_invalid_topology(self):
        m = host_model(mujoco.MjModel.from_xml_path(str(ROOT/'examples/mujoco_bag_strong_ref.xml')))
        for name, value in [('neq', 1), ('ntendon', 1), ('is_sparse', False), ('nv', 0)]:
            with self.subTest(name=name), patch.object(m, name, value), self.assertRaises(ValueError):
                _support_bound(m)
        for name, value in [('flex_interp', 1), ('flex_dim', 1), ('flex_vertbodyid', -1),
                            ('flex_elem', -1), ('body_parentid', m.nbody)]:
            original = getattr(m, name).value.copy()
            try:
                getattr(m, name).value.fill(value)
                with self.subTest(name=name), self.assertRaises(ValueError): _support_bound(m)
            finally:
                getattr(m, name).value[:] = original

    def test_source_guards_and_exception_restoration(self):
        from mujoco_warp._src import solver
        m = host_model(mujoco.MjModel.from_xml_path(str(ROOT/'examples/mujoco_bag_strong_ref.xml')))
        policy = SparseDeterminismBudget(m)
        original = solver._update_gradient
        with self.assertRaisesRegex(ValueError, 'sentinel'):
            with policy.capture_policy():
                with self.assertRaises(RuntimeError):
                    with policy.capture_policy(): pass
                raise ValueError('sentinel')
        self.assertIs(original, solver._update_gradient)
        with patch('deform3d_warp.mujoco_determinism.version', return_value='changed'):
            with self.assertRaises(RuntimeError): SparseDeterminismBudget(m)
        with patch('deform3d_warp.mujoco_determinism.inspect.getsource', return_value='unreviewed'):
            with self.assertRaises(RuntimeError):
                with policy.capture_policy(): pass
        self.assertIs(original, solver._update_gradient)


class CudaTests(unittest.TestCase):
    def test_identical_hessian_with_changed_record_budget(self):
        import mujoco_warp as mjw
        from mujoco_warp._src import solver
        from deform3d_warp.mujoco_contact import flex_contact_policy
        from deform3d_warp.mujoco_metric import CooperativeMetricSolver
        from record_bag import drive
        cm = mujoco.MjModel.from_xml_path(str(ROOT/'examples/mujoco_bag_fast.xml'))
        cpu = mujoco.MjData(cm)
        for _ in range(400): drive(cm, cpu); mujoco.mj_step(cm, cpu)
        with wp.ScopedDevice('cuda:0'):
            model = mjw.put_model(cm)
            model.opt.deterministic = mjw.DeterminismType.ALL
            metric = CooperativeMetricSolver(model, 1)
            policy = SparseDeterminismBudget(model)
            snapshots = []
            class Captured(Exception): pass
            def take(m, d, ctx, *args, **kwargs):
                active = int(d.nefc.numpy()[0])
                self.assertLessEqual(int(d.efc.J_rownnz.numpy()[0, :active].max(initial=0)), policy.max_row_dofs)
                self.assertGreater(int(d.nefc.numpy()[0]), 0)
                snapshots.append((np.triu(ctx.h.numpy()), ctx.grad.numpy()))
                raise Captured()
            for bounded in (False, True):
                data = mjw.put_data(cm, cpu, nworld=1)
                with ExitStack() as stack:
                    stack.enter_context(flex_contact_policy(model))
                    stack.enter_context(metric.capture_policy())
                    if bounded: stack.enter_context(policy.capture_policy())
                    # Hook installed after the policy checks composition through
                    # live module globals, not a frozen copy of the old solver.
                    stack.enter_context(patch.object(solver, '_cholesky_factorize_solve', take))
                    with self.assertRaises(Captured): mjw.forward(model, data)
            for reference, bounded in zip(*snapshots):
                np.testing.assert_array_equal(reference, bounded)


if __name__ == '__main__':
    wp.init()
    unittest.main()
