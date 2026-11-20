"""Independent linear-system and CUDA graph checks for cooperative PCG."""
import argparse
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from deform3d_warp.mujoco_metric import _kernel, CooperativeMetricSolver

DEVICE = 'cuda:0'


def solve_system(matrix, rhs, iterations=1000, tolerance=1e-5, graph=False):
    # The fixture splits a known matrix into mass diagonal, additional diagonal,
    # and a dense CSR stiffness. Each world has a different RHS.
    n = len(matrix)
    block = max(32, 1 << (n - 1).bit_length())
    worlds = len(rhs)
    array = lambda x, dtype: wp.array(x, dtype=dtype, device=DEVICE)
    m_row = array(np.arange(n + 1, dtype=np.int32), wp.int32)
    m_col = array(np.arange(n, dtype=np.int32), wp.int32)
    mass = array(np.full((worlds, n), .125, np.float32), wp.float32)
    diagonal = array(np.full((worlds, n), .25, np.float32), wp.float32)
    k = matrix.astype(np.float32).copy()
    k[np.diag_indices(n)] -= .375
    k_count = array(np.full(n, n, np.int32), wp.int32)
    k_row = array(np.arange(n, dtype=np.int32) * n, wp.int32)
    k_col = array(np.tile(np.arange(n, dtype=np.int32), n), wp.int32)
    stiffness = array(np.tile(k.ravel(), (worlds, 1)), wp.float32)
    force = array(np.asarray(rhs, np.float32), wp.float32)
    zero = wp.zeros((worlds, n), dtype=wp.float32, device=DEVICE)
    tol = array([tolerance], wp.float32)
    output = wp.zeros_like(force)
    count = wp.zeros(worlds, dtype=wp.int32, device=DEVICE)
    residual = wp.zeros(worlds, dtype=wp.float32, device=DEVICE)
    status = wp.zeros_like(count)
    def launch():
        wp.launch_tiled(_kernel(block), dim=worlds,
            inputs=[n, iterations, True, m_row, m_col, m_col, mass,
                    k_count, k_row, k_col, stiffness, diagonal, force, zero, zero, tol],
            outputs=[output, count, residual, status], block_dim=block, device=DEVICE)
    launch()
    if graph:
        wp.synchronize_device(DEVICE)
        with wp.ScopedCapture(device=DEVICE) as capture:
            launch()
        output.zero_()
        wp.capture_launch(capture.graph)
    effective = k.astype(np.float64) + np.eye(n) * .375
    return output.numpy(), count.numpy(), residual.numpy(), status.numpy(), effective


class MetricTests(unittest.TestCase):
    def test_spd_multiple_worlds_and_graph(self):
        rng = np.random.default_rng(71)
        with wp.ScopedDevice(DEVICE):
            for n in (3, 31, 65, 117, 288, 512):
                with self.subTest(n=n):
                    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
                    a = q @ np.diag(np.geomspace(1, 500, n)) @ q.T
                    b = rng.normal(size=(3, n))
                    x, count, residual, status, effective = solve_system(a, b, graph=True)
                    np.testing.assert_array_equal(status, 0)
                    reference = np.linalg.solve(effective, b.astype(np.float32).T).T
                    np.testing.assert_allclose(x, reference, atol=1e-4, rtol=3e-3)
                    actual = np.linalg.norm(x @ effective.T - b.astype(np.float32), axis=1) / np.linalg.norm(b.astype(np.float32), axis=1)
                    np.testing.assert_allclose(residual, actual, rtol=2e-5, atol=2e-8)
                    self.assertTrue(np.all(count > 0))
                    self.assertTrue(np.all(residual < 5e-5))

    def test_zero_rhs_and_budget_failure(self):
        with wp.ScopedDevice(DEVICE):
            n = 65
            a = np.diag(np.full(n, 2.0)) + np.diag(np.full(n - 1, -.9), 1) + np.diag(np.full(n - 1, -.9), -1)
            x, count, residual, status, _ = solve_system(a, np.zeros((2, n)))
            np.testing.assert_array_equal(x, 0)
            np.testing.assert_array_equal(count, 0)
            np.testing.assert_array_equal(status, 0)
            np.testing.assert_array_equal(residual, 0)
            _, _, _, status, _ = solve_system(a, np.ones((1, n)), iterations=1)
            self.assertEqual(status[0], 1)

    def test_non_spd_and_nonfinite_rejected(self):
        with wp.ScopedDevice(DEVICE):
            for a, b in ((-np.eye(3), np.ones((1, 3))), (np.eye(3), np.array([[np.nan, 1, 1]]))):
                _, _, _, status, _ = solve_system(a, b)
                self.assertEqual(status[0], 2)

    def test_float32_output_residual_is_checked(self):
        with wp.ScopedDevice(DEVICE):
            a = np.array([[1e6, 1e6-1], [1e6-1, 1e6]])
            x, _, residual, status, effective = solve_system(a, np.array([[1., 0.]]))
            actual = np.linalg.norm(effective @ x[0] - [1., 0.])
            self.assertGreater(actual, 5e-5)
            self.assertEqual(status[0], 3)
            self.assertAlmostEqual(float(residual[0]), actual, places=6)


class MetricPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco
        import mujoco_warp as mjw
        cls.cpu_model = mujoco.MjModel.from_xml_path(str(ROOT / 'examples/mujoco_bag_fast.xml'))
        with wp.ScopedDevice(DEVICE):
            cls.model = mjw.put_model(cls.cpu_model)
            cls.data = mjw.make_data(cls.cpu_model, nworld=2)

    def test_unsupported_terms_rejected(self):
        m = self.model
        for name, value in [('ntendon', 1), ('nactuator', 1), ('has_non_simple_flex', True),
                            ('nflexrodedge', 1), ('efm0_active', True), ('has_flex_passive', True),
                            ('has_fluid', True), ('flex_interp_assemblable', False)]:
            with self.subTest(name=name):
                saved = getattr(m, name)
                try:
                    setattr(m, name, value)
                    with self.assertRaisesRegex(ValueError, 'unsupported terms'):
                        CooperativeMetricSolver(m, 2)
                finally:
                    setattr(m, name, saved)
        for value in (True, 0, 1.5, 2**31):
            with self.assertRaises(ValueError):
                CooperativeMetricSolver(m, nworld=value)
            with self.assertRaises(ValueError):
                CooperativeMetricSolver(m, max_iterations=value)

    def test_external_buffers_rejected_before_launch(self):
        m, d = self.model, self.data
        solver = CooperativeMetricSolver(m, 2)
        with patch('deform3d_warp.mujoco_metric.wp.launch_tiled',
                   side_effect=AssertionError('unsafe launch')):
            with self.assertRaisesRegex(ValueError, 'overlap'):
                solver.enqueue(d, d.qfrc_smooth)
            for ptr, strides in ((d.qacc_smooth.ptr + 1, d.qacc_smooth.strides),
                                 (d.qacc_smooth.ptr, (4, 4))):
                invalid = wp.array(ptr=ptr, dtype=wp.float32, shape=(2, m.nv),
                                   strides=strides, device=DEVICE, copy=False)
                with self.assertRaises(ValueError):
                    solver.enqueue(d, invalid)
            with self.assertRaisesRegex(ValueError, 'world count'):
                CooperativeMetricSolver(m, 1).enqueue(d, d.qacc_smooth)

    def test_context_restoration_and_model_dispatch(self):
        from mujoco_warp._src import derivative
        solver = CooperativeMetricSolver(self.model, 2)
        original = derivative.eff_solve
        with patch.object(solver, 'enqueue') as enqueue:
            with self.assertRaisesRegex(ValueError, 'sentinel'):
                with solver.capture_policy():
                    derivative.eff_solve(self.model, self.data, self.data.qacc_smooth)
                    enqueue.assert_called_once_with(self.data, self.data.qacc_smooth, None)
                    with self.assertRaisesRegex(RuntimeError, 'nested'):
                        with solver.capture_policy():
                            pass
                    raise ValueError('sentinel')
        self.assertIs(derivative.eff_solve, original)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    DEVICE = args.device
    if not wp.get_device(DEVICE).is_cuda:
        parser.error('CUDA required')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    raise SystemExit(0 if result.wasSuccessful() else 1)
