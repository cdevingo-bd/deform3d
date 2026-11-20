"""Separate contact geometry errors from accumulated trajectory divergence.

Reproduce a converged CPU trajectory, retaining its complete history hashes.
At selected pre-integration states run CPU and GPU collision independently.
Both the original CPU state and its float32-quantized input are evaluated, so
state transport error is not silently charged to the collision algorithm.
No alignment, correction, time stepping, or GPU performance claim is made.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from contextlib import ExitStack

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from check_contact_convergence import trajectory
from deform3d_warp.mujoco_contact import flex_contact_policy


def contacts_cpu(data):
    return [dict(geom=c.geom.copy(), flex=c.flex.copy(), elem=c.elem.copy(),
                 vert=c.vert.copy(), dist=float(c.dist), pos=c.pos.copy(),
                 normal=c.frame[:3].copy()) for c in data.contact]


def contacts_gpu(data):
    count = int(data.nacon.numpy()[0])
    if not 0 <= count <= data.naconmax:
        raise RuntimeError('contact buffer overflow')
    arrays = {key: getattr(data.contact, key).numpy()[:count]
              for key in ('geom', 'flex', 'elem', 'vert', 'dist', 'pos', 'frame')}
    return [dict(**{key: arrays[key][i] for key in ('geom', 'flex', 'elem', 'vert', 'pos')},
                 dist=float(arrays['dist'][i]), normal=arrays['frame'][i, 0]) for i in range(count)]


def grouped(records):
    groups = defaultdict(list)
    for c in records:
        sides = [tuple(int(c[key][side]) for key in ('geom', 'flex', 'elem', 'vert'))
                 for side in range(2)]
        normal = np.asarray(c['normal'], np.float64)
        if sides[1] < sides[0]:
            sides.reverse()
            normal = -normal
        groups[tuple(sides)].append(dict(dist=c['dist'], pos=np.asarray(c['pos'], np.float64), normal=normal))
    return groups


def compare_contacts(model, reference, candidate):
    """Match only unambiguous element/vertex identities; expose all unmatched rows.

    Multiple contacts on a single feature are retained as multiplicities, not
    arbitrarily paired by distance, which could disguise a feature change.
    """
    a, b = grouped(reference), grouped(candidate)
    deltas, unmatched = [], []
    for key in sorted(a.keys() | b.keys()):
        left, right = a.get(key, []), b.get(key, [])
        if len(left) != 1 or len(right) != 1:
            unmatched.append(dict(key=key, reference_count=len(left), candidate_count=len(right)))
            continue
        x, y = left[0], right[0]
        geoms = [s[0] for s in key if s[0] >= 0]
        category = 'flex_flex' if not geoms else (
            'ellipsoid_flex' if any(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_ELLIPSOID for g in geoms)
            else 'other_geom_flex')
        deltas.append(dict(key=key, category=category,
                           distance_delta_m=abs(x['dist']-y['dist']),
                           position_delta_m=float(np.linalg.norm(x['pos']-y['pos'])),
                           unit_normal_delta=float(np.linalg.norm(x['normal']-y['normal'])),
                           reference_distance_m=x['dist'], candidate_distance_m=y['dist']))
    summary = {}
    for category in sorted({d['category'] for d in deltas}):
        rows = [d for d in deltas if d['category'] == category]
        summary[category] = dict(matched_unique_features=len(rows), **{
            'max_'+name: max(d[name] for d in rows)
            for name in ('distance_delta_m', 'position_delta_m', 'unit_normal_delta')})
    return dict(reference_contacts=len(reference), candidate_contacts=len(candidate),
                matched_unique_features=len(deltas), ambiguous_or_unmatched=unmatched,
                by_category=summary, matched_features=deltas)


def run(args):
    convergence = json.loads(args.convergence.read_text())
    if not convergence['tightest_pair_converged']:
        raise ValueError('an independently converged CPU tolerance pair is required')
    scene = Path(convergence['scene'])
    if not scene.is_absolute():
        scene = ROOT/scene
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.ccd_tolerance = convergence['reference_tolerance']
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    size = mujoco.mj_stateSize(model, specification)
    selected = sorted(set(round(t/model.opt.timestep) for t in args.times))
    steps = round(convergence['duration_s']/model.opt.timestep)
    if not selected or min(selected) < 0 or max(selected) >= steps:
        raise ValueError('checkpoint times must lie within the recorded trajectory')
    snapshots = {}

    def observe(m, d, step):
        if step in selected:
            state = np.empty(size)
            mujoco.mj_getState(m, d, state, specification)
            snapshots[step] = state

    points, forces, cpu_report = trajectory(scene, model.opt.ccd_tolerance,
        convergence['duration_s'], convergence['contents_shear_force_N'], snapshot=observe)
    hashes = dict(point_history_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
                  signed_pad_force_history_sha256=hashlib.sha256(forces.tobytes()).hexdigest())
    for name, digest in hashes.items():
        if digest != convergence['results'][-1][name]:
            raise RuntimeError('CPU reference did not independently reproduce: '+name)
    print('CPU reference reproduced, checkpoints', selected, flush=True)
    results = []
    with wp.ScopedDevice('cuda:0'):
        gpu_model = mjw.put_model(model)
        gpu_model.opt.ccd_tolerance.fill_(args.gpu_ccd_tolerance)
        gpu_model.opt.deterministic = mjw.DeterminismType.ALL
        for step, state in snapshots.items():
            cpu = mujoco.MjData(model)
            mujoco.mj_setState(model, cpu, state, specification)
            mujoco.mj_fwdPosition(model, cpu)
            quantized = mujoco.MjData(model)
            mujoco.mj_setState(model, quantized, state.astype(np.float32).astype(np.float64), specification)
            mujoco.mj_fwdPosition(model, quantized)
            gpu = mjw.put_data(model, quantized)
            with ExitStack() as policies:
                policies.enter_context(flex_contact_policy(gpu_model))
                if args.stable_contact_normals:
                    from deform3d_warp.mujoco_contact_normals import stable_flex_normals_policy
                    policies.enter_context(stable_flex_normals_policy(gpu_model))
                mjw.fwd_position(gpu_model, gpu, factorize=False)
            wp.synchronize_device('cuda:0')
            overflow = gpu.overflow.numpy().tolist()
            if any(overflow):
                raise RuntimeError('GPU collision overflow: '+str(overflow))
            cc, qc, gc = contacts_cpu(cpu), contacts_cpu(quantized), contacts_gpu(gpu)
            item = dict(step=step, time_s=step*float(model.opt.timestep),
                state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                quantization=compare_contacts(model, cc, qc),
                gpu_vs_quantized_cpu=compare_contacts(model, qc, gc),
                gpu_vs_original_cpu=compare_contacts(model, cc, gc),
                max_flex_position_input_delta_m=float(np.linalg.norm(
                    gpu.flexvert_xpos.numpy()[0].astype(np.float64)-quantized.flexvert_xpos,axis=1).max()),
                max_geom_position_input_delta_m=float(np.linalg.norm(
                    gpu.geom_xpos.numpy()[0].astype(np.float64)-quantized.geom_xpos,axis=1).max()),
                overflow=overflow)
            results.append(item)
            print(step, json.dumps(item['gpu_vs_quantized_cpu']['by_category']), flush=True)
    report = dict(scene=str(scene.resolve()), convergence_report=str(args.convergence.resolve()),
        convergence_report_sha256=hashlib.sha256(args.convergence.read_bytes()).hexdigest(),
        cpu_ccd_tolerance=model.opt.ccd_tolerance, gpu_ccd_tolerance=args.gpu_ccd_tolerance,
        contents_shear_force_N=convergence['contents_shear_force_N'], reference_hashes=hashes,
        stable_contact_normals=args.stable_contact_normals,
        scope='Same preintegration inputs; independently evaluated collision geometry; no trajectories are corrected; '
              'GPU float32 kinematics may differ and their input geometry deltas are reported; '
              'multi-contact features remain explicitly unpaired; not a solver, timing or trajectory acceptance test',
        results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'), steps=selected,
                        integration_states=np.asarray([snapshots[s] for s in selected]),
                        specification=int(specification), **hashes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('convergence', type=Path)
    parser.add_argument('--times', type=float, nargs='+', default=[.12, .9, 1.2, 1.4, 1.7, 2.1, 2.3, 2.349, 2.45])
    parser.add_argument('--gpu-ccd-tolerance', type=float, default=1e-8)
    parser.add_argument('--stable-contact-normals', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.times).all() or not np.isfinite(args.gpu_ccd_tolerance) or args.gpu_ccd_tolerance <= 0:
        parser.error('finite times and a positive finite GPU tolerance are required')
    run(args)
