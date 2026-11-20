"""Long-horizon clock, invalid-step, capture, and independent control checks."""
from pathlib import Path
import sys
import unittest

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from deform3d_warp.clock import DeviceSimulationClock


class ClockTests(unittest.TestCase):
    def test_long_horizon_and_changed_graph_timestep(self):
        with wp.ScopedDevice('cuda:0'):
            initial = np.array([0., 1.e8, -.5])
            c = DeviceSimulationClock(3, initial_time=initial)
            out = wp.zeros(3)
            dt = wp.array(np.array([.001, .003, 0.], np.float32))
            c.enqueue_write(out)
            c.enqueue_advance(dt, out)
            c.reset(initial)
            with wp.ScopedCapture() as capture:
                for _ in range(100):
                    c.enqueue_advance(dt, out)
            expected = initial.copy()
            for steps in ([.001, .003, 0.], [.002, 0., .004]):
                values = np.array(steps, np.float32)
                dt.assign(values)
                for _ in range(100):
                    wp.capture_launch(capture.graph)
                expected += 10000*values.astype(np.float64)
                np.testing.assert_allclose(c.elapsed.numpy(), expected, rtol=0., atol=1.5e-8)
                np.testing.assert_array_equal(out.numpy(), expected.astype(np.float32))
                np.testing.assert_array_equal(c.status.numpy(), 0)

    def test_scalar_timestep_broadcast_reset_and_cpu(self):
        for device in ('cpu', 'cuda:0'):
            with wp.ScopedDevice(device):
                c = DeviceSimulationClock(3, initial_time=[-1., 2., 4.], device=device)
                dt, out = wp.array([.25], dtype=float, device=device), wp.zeros(3, device=device)
                c.enqueue_advance(dt, out)
                np.testing.assert_array_equal(c.elapsed.numpy(), [-.75, 2.25, 4.25])
                c.reset(3.)
                c.enqueue_write(out)
                np.testing.assert_array_equal(out.numpy(), 3.)
                np.testing.assert_array_equal(c.compensation.numpy(), 0.)

    def test_pause_preserves_pending_compensation(self):
        with wp.ScopedDevice('cuda:0'):
            c = DeviceSimulationClock(initial_time=1.e8)
            dt, out = wp.array([.001], dtype=float), wp.zeros(1)
            c.enqueue_advance(dt, out)
            before = c.elapsed.numpy(), c.compensation.numpy()
            self.assertNotEqual(float(before[1][0]), 0.)
            dt.zero_()
            for _ in range(5):
                c.enqueue_advance(dt, out)
            np.testing.assert_array_equal(c.elapsed.numpy(), before[0])
            np.testing.assert_array_equal(c.compensation.numpy(), before[1])

    def test_invalid_steps_are_sticky_and_transactional_per_world(self):
        with wp.ScopedDevice('cuda:0'):
            c = DeviceSimulationClock(4, initial_time=[1., 2., 3., 4.])
            out, dt = wp.full(4, -99.), wp.array(np.array([-.1, np.nan, np.inf, .25], np.float32))
            c.enqueue_advance(dt, out)
            np.testing.assert_array_equal(c.elapsed.numpy(), [1., 2., 3., 4.25])
            np.testing.assert_array_equal(out.numpy(), [-99., -99., -99., 4.25])
            np.testing.assert_array_equal(c.status.numpy(), [1, 1, 1, 0])
            dt.fill_(.25)
            c.enqueue_advance(dt, out)
            np.testing.assert_array_equal(c.elapsed.numpy(), [1., 2., 3., 4.5])
            c.reset([1., 2., 3., 4.])
            c.enqueue_advance(dt, out)
            np.testing.assert_array_equal(out.numpy(), [1.25, 2.25, 3.25, 4.25])

    def test_bad_layout_and_alias_rejected_before_writes(self):
        with wp.ScopedDevice('cuda:0'):
            c = DeviceSimulationClock(2)
            out, dt = wp.full(2, 9.), wp.full(2, .1)
            for bad in (wp.zeros(3), wp.zeros((1, 2)), wp.zeros(2, dtype=wp.float64)):
                with self.assertRaises(ValueError):
                    c.enqueue_advance(dt, bad)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                c.enqueue_advance(out, out)
            with self.assertRaises(ValueError):
                c.enqueue_advance(wp.zeros(3), out)
            np.testing.assert_array_equal(out.numpy(), 9.)
            np.testing.assert_array_equal(c.elapsed.numpy(), 0.)
            for count in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    DeviceSimulationClock(count)
            for initial in (np.nan, np.inf, 1.e40, [1, 2, 3]):
                with self.assertRaises(ValueError):
                    DeviceSimulationClock(2, initial_time=initial)

    def test_float32_time_overflow_preserves_state(self):
        with wp.ScopedDevice('cuda:0'):
            maximum = float(np.finfo(np.float32).max)
            c = DeviceSimulationClock(initial_time=maximum)
            out, dt = wp.full(1, 3.), wp.array([maximum], dtype=float)
            c.enqueue_advance(dt, out)
            np.testing.assert_array_equal(c.status.numpy(), 1)
            np.testing.assert_array_equal(c.elapsed.numpy(), maximum)
            np.testing.assert_array_equal(out.numpy(), 3.)

    def test_precise_controls_against_independent_double_curves(self):
        from optional_mjwarp_bag import drive_precise
        with wp.ScopedDevice('cuda:0'):
            # Include exact event boundaries and values immediately either side.
            times = np.unique(np.r_[np.arange(3001)*.001,
                np.nextafter([.7, .9, 1.2, 1.4, 1.6, 1.7, 2.2, 2.5], -np.inf),
                np.nextafter([.7, .9, 1.2, 1.4, 1.6, 1.7, 2.2, 2.5], np.inf)])
            n = len(times)
            mocap = wp.zeros((n, 2), dtype=wp.vec3)
            loads = wp.zeros((n, 4), dtype=wp.spatial_vector)
            t = wp.array(times, dtype=wp.float64)
            bodies = wp.array([2, 3], dtype=int)
            wp.launch(drive_precise, n, inputs=[t, 0, 1, .09, .064, 1, bodies, .2, mocap, loads])
            smooth = lambda x: x*x*(3.-2.*x)
            closure = smooth(np.clip((times-.7)/.7, 0, 1))-smooth(np.clip((times-1.7)/.8, 0, 1))
            center = .09-.064*closure
            amplitude = .2*(smooth(np.clip((times-.9)/.3, 0, 1))-smooth(np.clip((times-1.6)/.6, 0, 1)))
            expected = np.zeros((n, 4, 6))
            expected[:, 1, 0] = amplitude
            expected[:, 1, 5] = .015*amplitude
            expected[:, 2:, 0] = -amplitude[:, None]/2
            np.testing.assert_allclose(mocap.numpy()[:, 0, 1], -center, atol=4e-9, rtol=0.)
            np.testing.assert_allclose(mocap.numpy()[:, 1, 1], center, atol=4e-9, rtol=0.)
            np.testing.assert_allclose(loads.numpy(), expected, atol=8e-9, rtol=0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
