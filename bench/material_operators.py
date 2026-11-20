"""Paired frozen-state elastic force, stiffness-build and force-shift timings.

Every case starts from the same compressed CPU state. Each phase has its own
captured graph; this does not measure full simulation or advance physical time.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import passive, derivative
import numpy as np
import warp as wp

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'tools')]
from deform3d_warp.mujoco_material import OwnedFlexOperators
from record_bag import drive


def processes():
    return subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()


def run(args):
    before=processes()
    cm=mujoco.MjModel.from_xml_path(str(ROOT/'examples/mujoco_bag_strong_ref.xml'))
    cm.opt.ccd_tolerance=1e-9
    cpu=mujoco.MjData(cm)
    for _ in range(round(args.time/cm.opt.timestep)): drive(cm,cpu); mujoco.mj_step(cm,cpu)
    drive(cm,cpu); mujoco.mj_forward(cm,cpu)
    fields={'stiffness_build':('flexedge_hessian','flexvert_hessian'),
            'passive_forces':('qfrc_spring','qfrc_damper','qfrc_passive'),
            'force_shift':('efm_c','efm_diag')}
    cases=[]; references={}
    with wp.ScopedDevice('cuda:0'):
        for mode in ('sorted_deterministic','atomic','owned_deterministic'):
            m=mjw.put_model(cm)
            m.opt.deterministic=mjw.DeterminismType.ALL if mode!='atomic' else mjw.DeterminismType.NONE
            d=mjw.put_data(cm,cpu,nworld=args.worlds)
            p=OwnedFlexOperators(m,d) if mode=='owned_deterministic' else None
            with p.capture_policy() if p else nullcontext():
                d.flex_hessian_valid.zero_(); passive.flex_hessian(m,d)
                for phase,names in fields.items():
                    def enqueue():
                        if phase=='stiffness_build':
                            d.flex_hessian_valid.zero_(); passive.flex_hessian(m,d)
                        elif phase=='passive_forces': passive.passive(m,d)
                        else: derivative.eff_shift(m,d)
                    enqueue(); wp.synchronize()
                    arrays={name:getattr(d,name).numpy().astype(np.float64) for name in names}
                    if mode=='sorted_deterministic': references[phase]=arrays
                    errors={}
                    for name,x in arrays.items():
                        y=references[phase][name]
                        rel=float(np.linalg.norm(x-y)/max(1e-30,np.linalg.norm(y)))
                        if rel>3e-6 or not np.all(np.isfinite(x)):
                            raise AssertionError(f'{mode}/{phase}/{name}: relative error {rel}')
                        if mode=='owned_deterministic': np.testing.assert_array_equal(x,y)
                        errors[name]=dict(relative_to_sorted=rel,max_abs=float(np.max(np.abs(x-y),initial=0)),
                            worlds_equal=all(np.array_equal(x[0],v) for v in x))
                    with wp.ScopedCapture() as cap:
                        for _ in range(args.repeat): enqueue()
                    cases.append(dict(mode=mode,phase=phase,graph=cap.graph,model=m,data=d,policy=p,
                                      errors=errors,times=[]))
        # Alternate all mode/phase orders; keep graphs, data and workspaces alive.
        start,end=wp.Event(enable_timing=True),wp.Event(enable_timing=True)
        for sample in range(args.samples):
            for case in cases[::1 if sample%2==0 else -1]:
                wp.record_event(start); wp.capture_launch(case['graph']); wp.record_event(end)
                case['times'].append(wp.get_event_elapsed_time(start,end)/args.repeat)
        result=dict(device=wp.get_device().name,worlds=args.worlds,nv=cm.nv,
                    physical_flex_vertices=cm.nflexvert,physical_elements=cm.nflexelem,
                    checkpoint_s=float(cpu.time),samples=args.samples,executions_per_sample=args.repeat,
                    scope='Frozen operator phases; no collision, constrained solve, integration, dense mesh or rendering; stiffness build invalidates cache each execution',
                    gpu_processes_before=before,gpu_processes_after=processes(),results={})
        for c in cases:
            p=c['policy']
            result['results'][c['mode']+'/'+c['phase']]=dict(cuda_median_ms=float(np.median(c['times'])),
                cuda_p95_ms=float(np.percentile(c['times'],95)),cuda_samples_ms=c['times'],
                errors=c['errors'],status=p.status.numpy().tolist() if p else None,
                workspace_bytes=p.workspace_bytes if p else None)
            if p and np.any(p.status.numpy()): raise AssertionError('owned material numeric status')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--time',type=float,default=1.4)
    parser.add_argument('--worlds',type=int,default=2)
    parser.add_argument('--samples',type=int,default=30)
    parser.add_argument('--repeat',type=int,default=10)
    parser.add_argument('--output',type=Path,default=ROOT/'results/material_operators_0.9.json')
    args=parser.parse_args()
    if min(args.worlds,args.samples,args.repeat)<1: parser.error('counts must be positive')
    run(args)
