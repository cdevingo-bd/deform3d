"""Check a GPU recording against its report and independent CPU kinematics.

This verifies state provenance/temporal consistency, not trajectory parity or
material realism. A failed simulation report remains failed after this check.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def verify(report_path, cpu=False):
    report = json.loads(report_path.read_text())
    record = report['cpu_recording' if cpu else 'recording']
    path = report_path.parent/Path(record['path']).name
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('recording checksum mismatch')
    with np.load(path, allow_pickle=False) as history:
        count = record['samples']
        if count != report['steps'] or count != report['diagnostic_samples']:
            raise ValueError('recording must contain every integrated step')
        if any(len(history[name]) != count or not np.isfinite(history[name]).all() for name in history.files):
            raise ValueError('invalid recording sample count or values')
        times = history['diagnostics'][:, 0]
        if np.any(np.diff(times) <= 0): raise ValueError('nonmonotonic recording times')
        np.testing.assert_allclose(history['force_time'], times-report['timestep_s'], atol=1e-12, rtol=0)
        expected_forces = np.abs(np.array([p['cpu_pad_forces_N'] if cpu else p['gpu_pad_forces_N'][0]
                                          for p in report['checkpoints']]))
        np.testing.assert_array_equal(history['diagnostics'][:, 5:7], expected_forces)
        m = mujoco.MjModel.from_xml_path(str(ROOT/report['scene']))
        d = mujoco.MjData(m)
        fruit = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'fruit')
        maximum = 0.
        orientation = 0.
        for qpos, mocap, points, pose in zip(history['qpos'], history['mocap'], history['points'], history['fruit_pose']):
            d.qpos[:] = qpos
            d.mocap_pos[:] = mocap
            mujoco.mj_kinematics(m, d)
            mujoco.mj_flex(m, d)
            maximum = max(maximum, float(np.linalg.norm(d.flexvert_xpos-points, axis=1).max()),
                          float(np.linalg.norm(d.xpos[fruit]-pose[:3])))
            orientation = max(orientation, float(min(np.linalg.norm(d.xquat[fruit]-pose[3:]),
                                                    np.linalg.norm(d.xquat[fruit]+pose[3:]))))
        if maximum > 5e-6 or orientation > 1e-5:
            raise ValueError('recorded geometry does not match the recorded integrated state')
    return {'report': str(report_path), 'samples': count, 'state_recording_verified': True,
            'backend': 'native MuJoCo CPU' if cpu else 'MJWarp GPU',
            'max_kinematics_position_delta_m': maximum, 'max_quaternion_delta': orientation,
            'simulation_validation_passed': report['validation_passed'],
            'note': 'recording integrity is separate from full simulation acceptance'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cpu', action='store_true', help='verify the independently recorded native CPU reference instead')
    args = parser.parse_args()
    result = verify(args.report, args.cpu)
    text = json.dumps(result, indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
