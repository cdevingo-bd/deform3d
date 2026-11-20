"""Compare an unchanged GPU recording to a separately converged CPU reference.

This writes a separate audit. It never rewrites the original failed report,
retimes the GPU, aligns states, or changes numerical acceptance thresholds.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from check_contact_convergence import trajectory

ROOT=Path(__file__).resolve().parents[1]


def audit(report_path,convergence_path):
    report=json.loads(report_path.read_text())
    convergence=json.loads(convergence_path.read_text())
    if not convergence['tightest_pair_converged']:
        raise ValueError('a separately converged CPU tolerance pair is required')
    scene=ROOT/report['scene']
    if scene.resolve()!=Path(convergence['scene']).resolve():raise ValueError('different scenes')
    load=report.get('contents_load')
    if load and load['case']!='opposed_contents_shear_v1':raise ValueError('unknown load program')
    peak=load['peak_force_N'] if load else 0.
    if convergence['contents_shear_force_N']!=peak or convergence['duration_s']!=report['duration_s']:
        raise ValueError('different loading/duration')
    record=report['recording'];path=report_path.parent/Path(record['path']).name
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('recording checksum mismatch')
    points,forces,reference=trajectory(scene,convergence['reference_tolerance'],report['duration_s'],peak)
    point_hash=hashlib.sha256(points.tobytes()).hexdigest()
    force_hash=hashlib.sha256(forces.tobytes()).hexdigest()
    previous=convergence['results'][-1]
    if point_hash!=previous['point_history_sha256'] or force_hash!=previous['signed_pad_force_history_sha256']:
        raise ValueError('CPU reference did not reproduce independently')
    with np.load(path,allow_pickle=False) as recording:
        gpu=recording['points']
        if gpu.shape!=points.shape or len(gpu)!=report['steps']:raise ValueError('different sample layout')
        delta=np.linalg.norm(gpu.astype(np.float64)-points,axis=2)
    gpu_force=np.asarray([c['gpu_pad_forces_N'] for c in report['checkpoints']],np.float64)
    force_delta=float(np.sqrt(np.mean((gpu_force-forces[:,None,:])**2)))
    force_rms=float(np.sqrt(np.mean(forces**2)))
    relative=force_delta/max(force_rms,1e-12)
    max_shape=float(delta.max());shape_gate=report['cpu_parity_tolerance_m']
    force_gate=report['pad_force_relative_tolerance']
    return dict(gpu_report=str(report_path),convergence_report=str(convergence_path),
        original_report_unchanged=True,gpu_recording_sha256=record['sha256'],
        cpu_reference_point_sha256=point_hash,cpu_reference_force_sha256=force_hash,
        cpu_reference_independently_repeated=True,cpu_ccd_tolerance=convergence['reference_tolerance'],
        max_flex_delta_m=max_shape,max_flex_delta_time_s=float((np.unravel_index(delta.argmax(),delta.shape)[0]+1)*report['timestep_s']),
        shape_gate_m=shape_gate,force_relative_rms_delta=relative,force_relative_rms_gate=force_gate,
        shape_passed=max_shape<=shape_gate,force_passed=relative<=force_gate,
        shape_force_comparison_passed=bool(max_shape<=shape_gate and relative<=force_gate),
        original_simulation_validation_passed=report['validation_passed'],cpu_reference=reference,
        scope='Same recorded GPU trajectory and every-world pad forces; independently repeated CPU points/forces at tighter converged CCD tolerance; same step indices, no corrections; not a replacement full simulation acceptance report')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('report',type=Path);p.add_argument('convergence',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.report,args.convergence)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if not result['shape_force_comparison_passed']:raise RuntimeError('recording fails tighter-reference comparison')
