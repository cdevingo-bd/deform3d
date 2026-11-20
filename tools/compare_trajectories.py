#!/usr/bin/env python3
"""Compare uninterrupted contact trajectories at common physical timestamps.

This is a numerical convergence gate, not validation against a real specimen.
Inputs are result prefixes from record_bag.py or build_backpack_scene.py.
Forces use their recorded timestamps; geometry uses integrated-state timestamps.
No rigid alignment is applied: drift and recovery error remain visible.
"""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(prefix, kind):
    suffix = '_validation' if kind == 'backpack' else ''
    report = json.loads(Path(str(prefix)+suffix+'.json').read_text())
    trajectory = np.load(Path(str(prefix)+('_trajectory' if kind == 'backpack' else '')+'.npz'))
    if kind == 'backpack':
        times = trajectory['time']
        points = trajectory['flex_positions']
        force_times = trajectory['force_time']
        forces = trajectory['pad_forces']
    else:
        times = trajectory['diagnostics'][:, 0]
        points = trajectory['points']
        force_times = times
        forces = trajectory['diagnostics'][:, 5:7]
    for t, values in [(times, points), (force_times, forces)]:
        if len(t)<2 or np.any(np.diff(t)<=0) or not np.isfinite(t).all() or not np.isfinite(values).all():
            raise ValueError('trajectory must be finite with strictly increasing timestamps')
    scene = Path(report['scene'])
    if not scene.is_absolute():scene=ROOT/scene
    model=mujoco.MjModel.from_xml_path(str(scene))
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    return report, times, points, force_times, forces, model.flex_elem.copy(), data.flexvert_xpos.copy()


def interpolate(times, values, query):
    shape=values.shape[1:];flat=values.reshape(len(values),-1)
    return np.stack([np.interp(query,times,flat[:,i]) for i in range(flat.shape[1])],axis=1).reshape((len(query),)+shape)


def compare(reference, candidate, kind, position_limit=.005, force_limit=.10):
    r,rt,rp,rft,rf,re,rest=load(reference,kind)
    c,ct,cp,cft,cf,ce,cest=load(candidate,kind)
    if rp.shape[1:]!=cp.shape[1:] or not np.array_equal(re,ce) or not np.allclose(rest,cest,rtol=0,atol=1e-10):
        raise ValueError('reference and candidate must share the same ordered physical mesh and rest pose')
    qt=rt[(rt>=max(rt[0],ct[0]))&(rt<=min(rt[-1],ct[-1]))]
    qf=rft[(rft>=max(rft[0],cft[0]))&(rft<=min(rft[-1],cft[-1]))]
    if len(qt)<2 or len(qf)<2:raise ValueError('insufficient overlapping physical time')
    delta=np.linalg.norm(interpolate(rt,rp,qt)-interpolate(ct,cp,qt),axis=-1)
    fref=interpolate(rft,rf,qf);fcand=interpolate(cft,cf,qf)
    force_rms=float(np.sqrt(np.mean((fref-fcand)**2)))
    reference_rms=float(np.sqrt(np.mean(fref**2)))
    relative=force_rms/max(reference_rms,1e-12)
    passed=bool(delta.max()<=position_limit and relative<=force_limit)
    return {'kind':kind,'reference':str(reference),'candidate':str(candidate),
            'comparison_scope':'Numerical trajectory convergence only; no measured physical data.',
            'same_ordered_mesh_and_rest':True,'geometry_samples':len(qt),'force_samples':len(qf),
            'maximum_position_delta_m':float(delta.max()),'rms_position_delta_m':float(np.sqrt(np.mean(delta**2))),
            'force_rms_delta_N':force_rms,'reference_force_rms_N':reference_rms,'relative_force_rms_delta':relative,
            'reference_peak_pad_force_N':r['peak_pad_force_N'],'candidate_peak_pad_force_N':c['peak_pad_force_N'],
            'reference_max_penetration_m':r['max_penetration_m'],'candidate_max_penetration_m':c['max_penetration_m'],
            'candidate_physics_only_realtime_factor':c.get('physics_only_real_time_factor'),
            'limits':{'maximum_position_delta_m':position_limit,'relative_force_rms_delta':force_limit},
            'numerical_convergence_passed':passed,
            'timing_scope':'Recorded mj_step only, excludes validation, refresh, geometry, transfers and rendering.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=['bag','backpack'],required=True)
    parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--position-limit',type=float,default=.005);parser.add_argument('--force-limit',type=float,default=.10)
    args=parser.parse_args()
    if not np.isfinite([args.position_limit,args.force_limit]).all() or min(args.position_limit,args.force_limit)<0:
        parser.error('limits must be finite and nonnegative')
    report=compare(args.reference,args.candidate,args.kind,args.position_limit,args.force_limit)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['numerical_convergence_passed'] else 1)
