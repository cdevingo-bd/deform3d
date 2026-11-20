"""Compare actual recorded states and every-world force histories across runs.

Byte equality here is limited to the arrays recorded by the harness on the same
device/software. It is not cross-device determinism or physical calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(first, second, allow_material_change=False, allow_solver_reductions_change=False,
            allow_effective_change=False, allow_wrench_change=False, allow_contact_normals_change=False,
            allow_controls_change=False, allow_substitution_change=False, allow_capacity_change=False,
            allow_factor_schedule_change=False):
    reports=[json.loads(path.read_text()) for path in (first,second)]
    for key in ('scene','steps','timestep_s','worlds','newton_precision','newton_tile_size',
                'newton_diagonal','gpu_determinism','cpu_ccd_tolerance','gpu_ccd_tolerance',
                'canonical_constraints','owned_contact_assembly'):
        if reports[0][key] != reports[1][key]:
            raise ValueError('different run configuration: '+key)
    changes = {}
    for key,default in (('owned_material',False),('owned_material_reduction',None),
                        ('shared_simplex',False),('shared_simplex_core_sha256',None)):
        if reports[0].get(key,default) != reports[1].get(key,default):
            changes[key] = [reports[0].get(key,default),reports[1].get(key,default)]
    if changes and not allow_material_change:
        raise ValueError('different material policy; explicitly allow this comparison')
    solver_change = [r.get('owned_solver_reductions',False) for r in reports]
    if solver_change[0] != solver_change[1] and not allow_solver_reductions_change:
        raise ValueError('different solver reductions policy; explicitly allow this comparison')
    operator_changes={}
    for key,allowed in (('owned_effective',allow_effective_change),('owned_wrench',allow_wrench_change),
                        ('stable_contact_normals',allow_contact_normals_change)):
        values=[r.get(key,False) for r in reports]
        if values[0]!=values[1]:
            if not allowed:raise ValueError('different '+key+' policy; explicitly allow this comparison')
            operator_changes[key]=values
    controls_change = [r.get('precise_controls', False) for r in reports]
    if controls_change[0] != controls_change[1] and not allow_controls_change:
        raise ValueError('different control timing/precision; explicitly allow this comparison')
    substitution_change = [r.get('newton_substitution', 'tiled' if r.get('parallel_newton') else None)
                           for r in reports]
    if substitution_change[0] != substitution_change[1] and not allow_substitution_change:
        raise ValueError('different Newton substitution; explicitly allow this comparison')
    factor_changes={}
    for key,default in (('newton_panel','tiled'),('newton_trailing','tiled'),('newton_update_threads',64),('conditional_newton_factor',False)):
        values=[r.get(key,default if key=='conditional_newton_factor' or r.get('parallel_newton') else None) for r in reports]
        if values[0]!=values[1]:
            if not allow_factor_schedule_change:
                raise ValueError('different Newton factor schedule; explicitly allow this comparison')
            factor_changes[key]=values
    capacity_change=[r.get('gpu_epa_horizon_capacity',24) for r in reports]
    if capacity_change[0]!=capacity_change[1] and not allow_capacity_change:
        raise ValueError('different EPA scratch capacity; explicitly allow this comparison')
    if reports[0].get('contents_load')!=reports[1].get('contents_load'):
        raise ValueError('different contents load configuration')
    recordings=[]
    for path,r in zip((first,second),reports):
        record_path=path.parent/Path(r['recording']['path']).name
        if hashlib.sha256(record_path.read_bytes()).hexdigest()!=r['recording']['sha256']:
            raise ValueError('recording checksum mismatch')
        recordings.append(record_path)
    arrays={}
    with np.load(recordings[0],allow_pickle=False) as a, np.load(recordings[1],allow_pickle=False) as b:
        if set(a.files)!=set(b.files):raise ValueError('different recording fields')
        for name in a.files:
            x,y=np.ascontiguousarray(a[name]),np.ascontiguousarray(b[name])
            same_layout=x.shape==y.shape and x.dtype==y.dtype
            same_bytes=same_layout and np.array_equal(x.view(np.uint8),y.view(np.uint8))
            arrays[name]=dict(shape=list(x.shape),dtype=str(x.dtype),identical_bytes=bool(same_bytes),
                max_absolute_difference=float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)),initial=0)) if same_layout else None)
    forces=[np.asarray([p['gpu_pad_forces_N'] for p in r['checkpoints']],dtype=np.float64) for r in reports]
    same_force=forces[0].shape==forces[1].shape and np.array_equal(forces[0].view(np.uint8),forces[1].view(np.uint8))
    world_spreads=[float(np.max(np.abs(f-f[:,0:1,:]),initial=0)) for f in forces]
    result=dict(first=str(first),second=str(second),arrays=arrays,
        allowed_material_configuration_differences=changes,
        allowed_solver_reductions_difference=solver_change if solver_change[0]!=solver_change[1] else None,
        allowed_operator_configuration_differences=operator_changes,
        allowed_control_configuration_difference=controls_change if controls_change[0]!=controls_change[1] else None,
        allowed_substitution_configuration_difference=substitution_change if substitution_change[0]!=substitution_change[1] else None,
        allowed_capacity_configuration_difference=capacity_change if capacity_change[0]!=capacity_change[1] else None,
        allowed_factor_schedule_differences=factor_changes,
        recorded_arrays_identical=all(a['identical_bytes'] for a in arrays.values()),
        all_world_signed_pad_forces_identical=bool(same_force),
        max_interworld_force_spread_N=world_spreads,
        max_interworld_flex_spread_m=[r['max_identical_world_flex_spread_m'] for r in reports],
        simulation_validation_passed=[r['validation_passed'] for r in reports],
        scope='All stored world-zero state/geometry/diagnostic arrays and all-world signed pad-force histories; same GPU/software only')
    result['repeatability_passed']=bool(result['recorded_arrays_identical'] and same_force
        and not any(world_spreads) and not any(result['max_interworld_flex_spread_m']))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first',type=Path);parser.add_argument('second',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-material-change',action='store_true',help='explicitly compare retained states across a material assembly policy change')
    parser.add_argument('--allow-solver-reductions-change',action='store_true',help='explicitly compare retained states across a solver reductions policy change')
    parser.add_argument('--allow-effective-change',action='store_true')
    parser.add_argument('--allow-wrench-change',action='store_true')
    parser.add_argument('--allow-contact-normals-change',action='store_true')
    parser.add_argument('--allow-controls-change',action='store_true')
    parser.add_argument('--allow-substitution-change',action='store_true')
    parser.add_argument('--allow-capacity-change',action='store_true')
    parser.add_argument('--allow-factor-schedule-change',action='store_true')
    args=parser.parse_args()
    result=compare(args.first,args.second,args.allow_material_change,args.allow_solver_reductions_change,
                   args.allow_effective_change,args.allow_wrench_change,args.allow_contact_normals_change,
                   args.allow_controls_change,args.allow_substitution_change,args.allow_capacity_change,args.allow_factor_schedule_change)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['repeatability_passed']:raise RuntimeError('recorded runs differ; inspect report')
