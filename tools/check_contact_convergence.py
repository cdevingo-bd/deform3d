"""Check CPU contact-tolerance convergence before using a trajectory as truth.

Only ccd_tolerance changes. The timestep, material, solver, control and gates
are held fixed. Every candidate is compared to the tightest requested tolerance
over the full trajectory, including signed gripper reaction forces.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from record_bag import drive

ROOT = Path(__file__).resolve().parents[1]


def trajectory(scene, tolerance, duration, contents_shear_force=0., *, snapshot=None):
    """Record the unmodified CPU trajectory; optional read-only pre-step observer.

    ``snapshot(model, data, step)`` runs after controls/loads have been applied.
    Observers must copy the state, never forward or modify this live data.
    """
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.ccd_tolerance = tolerance
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pad_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name+'_geom')
               for name in ('front_pad', 'back_pad')]
    if min(pad_ids) < 0:
        raise ValueError('scene requires front_pad_geom and back_pad_geom')
    if contents_shear_force:
        fruit=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'fruit')
        package=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_FLEX,'package')
        if min(fruit,package)<0:raise ValueError('contents load requires fruit and package')
        start=int(model.flex_vertadr[package]);count=int(model.flex_vertnum[package])
        bodies=np.unique(model.flex_vertbodyid[start:start+count])
        if not bodies.size or fruit in bodies or np.any(bodies==0):raise ValueError('invalid load bodies')
    steps = round(duration/model.opt.timestep)
    points, forces, costs, contact_counts = [], [], [], []
    force = np.zeros(6)
    for step in range(steps):
        drive(model, data)
        if contents_shear_force:
            # Independent CPU-only controller: clamped smoothstep ramp/release
            # reproduces the harness's piecewise load without importing Warp.
            rise=np.clip((data.time-.9)/.3,0.,1.)
            fall=np.clip((data.time-1.6)/.6,0.,1.)
            amplitude=contents_shear_force*(rise*rise*(3.-2.*rise)-fall*fall*(3.-2.*fall))
            data.xfrc_applied[fruit]=[amplitude,0.,0.,0.,0.,.015*amplitude]
            data.xfrc_applied[bodies]=[-amplitude/len(bodies),0.,0.,0.,0.,0.]
        if snapshot is not None:
            snapshot(model, data, step)
        began = time.perf_counter()
        mujoco.mj_step(model, data)
        costs.append(1000*(time.perf_counter()-began))
        net = np.zeros(2)
        for index, contact in enumerate(data.contact):
            for j, pad in enumerate(pad_ids):
                if pad in contact.geom:
                    mujoco.mj_contactForce(model, data, index, force)
                    net[j] += (-1 if contact.geom[0] == pad else 1)*(contact.frame.reshape(3, 3).T@force[:3])[1]
        forces.append(net)
        contact_counts.append(data.ncon)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_flex(model, data)
        points.append(data.flexvert_xpos.copy())
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError('nonfinite state in CPU reference convergence run')
    if np.any(data.warning.number):
        raise RuntimeError('CPU warning in reference convergence run')
    return np.asarray(points), np.asarray(forces), {
        'ccd_tolerance': tolerance, 'timestep_s': float(model.opt.timestep),
        'steps': steps, 'warning_counts': data.warning.number.tolist(),
        'physics_step_median_ms': float(np.median(costs)),
        'contact_observations': int(sum(contact_counts))}


def run(args):
    tolerances = sorted(set(args.tolerances), reverse=True)
    histories, reports = [], []
    for tolerance in tolerances:
        points, forces, report = trajectory(args.scene, tolerance, args.duration,args.contents_shear_force)
        histories.append((points, forces)); reports.append(report)
        print('Recorded CPU tolerance', tolerance, flush=True)
    ref_points, ref_forces = histories[-1]
    ref_rms = float(np.sqrt(np.mean(ref_forces**2)))
    for (points, forces), report in zip(histories, reports):
        report['point_history_sha256']=hashlib.sha256(points.tobytes()).hexdigest()
        report['signed_pad_force_history_sha256']=hashlib.sha256(forces.tobytes()).hexdigest()
        delta = np.linalg.norm(points-ref_points, axis=2)
        report['max_flex_delta_m'] = float(delta.max())
        report['max_flex_delta_time_s'] = float((np.unravel_index(delta.argmax(), delta.shape)[0]+1)*report['timestep_s'])
        force_rms = float(np.sqrt(np.mean((forces-ref_forces)**2)))
        report['force_rms_delta_N'] = force_rms
        report['force_reference_rms_N'] = ref_rms
        report['force_relative_rms_delta'] = force_rms/max(ref_rms, 1e-12)
        report['within_gates'] = (report['max_flex_delta_m'] <= args.max_shape_delta
                                  and report['force_relative_rms_delta'] <= args.max_force_delta)
    # Tightest vs itself is not a convergence test. Require a distinct second
    # tightest tolerance to agree; retain all failures at coarser tolerances.
    result = {'scene': str(args.scene), 'mujoco_version': mujoco.__version__,
              'duration_s': args.duration, 'reference_tolerance': tolerances[-1],
              'contents_shear_force_N':args.contents_shear_force,
              'shape_gate_m': args.max_shape_delta, 'relative_force_gate': args.max_force_delta,
              'scope': 'full CPU trajectories; only CCD termination tolerance changes; all steps sampled; '
                       'force timestamps are preintegration, positions are postintegration',
              'timing_note': 'diagnostic run on shared desktop; timing is not an isolated performance benchmark',
              'tightest_pair_converged': reports[-2]['within_gates'], 'results': reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if not result['tightest_pair_converged']:
        raise RuntimeError('CPU contact reference has not converged at the declared gates')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=ROOT/'examples/mujoco_bag_strong_ref.xml')
    parser.add_argument('--duration', type=float, default=3.)
    parser.add_argument('--contents-shear-force',type=float,default=0.)
    parser.add_argument('--tolerances', type=float, nargs='+', default=[1e-6, 1e-8, 1e-9, 1e-10])
    parser.add_argument('--max-shape-delta', type=float, default=.005)
    parser.add_argument('--max-force-delta', type=float, default=.1)
    parser.add_argument('--output', type=Path, default=ROOT/'results/cpu_contact_convergence_0.6.json')
    args = parser.parse_args()
    if (len(set(args.tolerances)) < 2 or not np.isfinite(args.tolerances).all()
        or min(args.tolerances) <= 0 or not np.isfinite(args.duration) or args.duration <= 0
        or not np.isfinite([args.max_shape_delta, args.max_force_delta]).all()
        or min(args.max_shape_delta, args.max_force_delta) < 0):
        parser.error('requires two distinct positive tolerances, positive duration and finite nonnegative gates')
    if not np.isfinite(args.contents_shear_force) or args.contents_shear_force<0:
        parser.error('contents shear force must be finite and nonnegative')
    run(args)
