"""CUDA differential checks against MJWarp's original FPS implementation."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np
import warp as wp
from mujoco_warp._src import collision_flex as original

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from deform3d_warp.mujoco_contact import _launch_fused, flex_contact_policy

DEVICE = 'cuda:0'


def workspace(lengths, seed=7, duplicates=False, masked=False, geoms=False):
    rng = np.random.default_rng(seed)
    count, groups = sum(lengths), len(lengths)
    capacity = max(1, count + 11)
    position = rng.normal(size=(capacity, 3)).astype(np.float32)
    if duplicates:
        position = np.floor(position * 2) * .5
        position[:min(65, count)] = 0
    depth = rng.uniform(-.01, 0, capacity).astype(np.float32)
    # Ties exercise the lexicographic element/index ordering.
    depth[:min(65, count)] = -.01
    elements = rng.integers(0, 30, (capacity, 2), dtype=np.int32)
    order = np.arange(capacity, dtype=np.int32)
    start = 0
    for n in lengths:
        rng.shuffle(order[start:start+n])
        start += n
    active = np.ones(capacity, np.int32)
    active[count:] = 0
    if masked:
        active[rng.choice(count, count // 3, replace=False)] = 0
    geom = np.full((capacity, 2), -1, np.int32)
    if geoms:
        geom[order[:lengths[0]], 0] = 2
    array = lambda x, dtype: wp.array(x, dtype=dtype, device=DEVICE)
    empty = lambda shape, dtype: wp.zeros(shape, dtype=dtype, device=DEVICE)
    ws = NS(flex_group_start_indices=array(np.cumsum([0] + lengths[:-1]), wp.int32),
            flex_num_groups=array([groups], wp.int32), ncand=array([count], wp.int32),
            cand_active_sorted=array(active[order], wp.int32), filter_val=array(order, wp.int32),
            dist=array(depth, wp.float32), pos=array(position, wp.vec3f), elem=array(elements, wp.vec2i),
            geom=array(geom, wp.vec2i), flex_fps_min_dist=empty(capacity, wp.float32),
            cand_active=array(active, wp.int32), fps_scratch_dist=empty((groups, 64), wp.float32),
            fps_scratch_cidx=empty((groups, 64), wp.int32), fps_scratch_count=empty((groups, 64), wp.int32),
            fps_selected_cidx=empty(groups, wp.int32), fps_selected_pos=empty(groups, wp.vec3f),
            fps_groups_active=empty(1, wp.int32), fps_condition=empty(1, wp.int32), fps_iter=empty(1, wp.int32))
    return ws, groups, active


class ContactPolicyTests(unittest.TestCase):
    def test_fps_matches_original(self):
        model = NS(opt=NS(graph_conditional=False))
        with wp.ScopedDevice(DEVICE):
            cases = ([1], [49], [50], [51], [75, 51, 3], [110, 99], [250, 51, 50, 1])
            for lengths in cases:
                for duplicates, masked, geoms in ((False, False, False), (True, False, False),
                                                  (True, True, False), (False, True, True)):
                    with self.subTest(lengths=lengths, duplicates=duplicates, masked=masked, geoms=geoms):
                        a, groups, _ = workspace(lengths, duplicates=duplicates, masked=masked, geoms=geoms)
                        b, _, _ = workspace(lengths, duplicates=duplicates, masked=masked, geoms=geoms)
                        original._run_filter_flex_fps(model, None, a, groups)
                        _launch_fused(model, None, b, groups)
                        np.testing.assert_array_equal(a.cand_active.numpy(), b.cand_active.numpy())

    def test_empty_and_graph_replay(self):
        with wp.ScopedDevice(DEVICE):
            ws, groups, active = workspace([75, 60], duplicates=True)
            _launch_fused(None, None, ws, groups)
            expected = ws.cand_active.numpy().copy()
            ws.cand_active.assign(active)
            with wp.ScopedCapture(device=DEVICE) as capture:
                _launch_fused(None, None, ws, groups)
            for _ in range(3):
                ws.cand_active.assign(active)
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(ws.cand_active.numpy(), expected)
            ws.ncand.assign([0])
            ws.flex_num_groups.assign([0])
            ws.cand_active.assign(active)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(ws.cand_active.numpy(), active)

    def test_policy_restores_on_error_and_rejects_nesting(self):
        saved = original._filter_and_write_contacts
        with self.assertRaisesRegex(ValueError, 'sentinel'):
            with flex_contact_policy(object()):
                self.assertIsNot(original._filter_and_write_contacts, saved)
                with self.assertRaisesRegex(RuntimeError, 'nested'):
                    with flex_contact_policy(object()):
                        pass
                raise ValueError('sentinel')
        self.assertIs(original._filter_and_write_contacts, saved)


def benchmark():
    model = NS(opt=NS(graph_conditional=False))
    result = {}
    with wp.ScopedDevice(DEVICE):
        for label, function in [('upstream', original._run_filter_flex_fps), ('fused', _launch_fused)]:
            # Common small-contact case: upstream still launches its unrolled
            # selection kernels on Pascal even though every group is <= 50.
            ws, groups, _ = workspace([30, 20, 5])
            function(model, None, ws, groups)
            wp.synchronize_device(DEVICE)
            with wp.ScopedCapture(device=DEVICE) as capture:
                function(model, None, ws, groups)
            start, end = (wp.Event(enable_timing=True) for _ in range(2))
            samples = []
            for _ in range(20):
                wp.record_event(start)
                for _ in range(10):
                    wp.capture_launch(capture.graph)
                wp.record_event(end)
                samples.append(wp.get_event_elapsed_time(start, end) / 10)
            result[label] = {'median_ms': float(np.median(samples)), 'p95_ms': float(np.percentile(samples, 95))}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/contact_policy_validation.json')
    args = parser.parse_args()
    DEVICE = args.device
    if not wp.get_device(DEVICE).is_cuda:
        parser.error('CUDA is required to test graph capture and replay')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContactPolicyTests))
    report = {'device': DEVICE, 'tests_passed': result.wasSuccessful(), 'differential_cases': 28,
              'fps_graph_microbenchmark': benchmark() if result.wasSuccessful() else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
