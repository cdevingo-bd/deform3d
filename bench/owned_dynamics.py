"""Paired frozen-checkpoint effective-metric and sparse-wrench measurements."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import derivative,support
import numpy as np
import warp as wp

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'tools')]
from deform3d_warp.mujoco_effective import OwnedEffectiveMetric
from deform3d_warp.mujoco_material import OwnedFlexOperators
from deform3d_warp.mujoco_wrench import OwnedWrenchProjection
from record_bag import drive


def processes():
    return subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()


def run(args):
    before=processes();cm=mujoco.MjModel.from_xml_path(str(ROOT/'examples/mujoco_bag_strong_ref.xml'))
    cm.opt.ccd_tolerance=1e-9;cd=mujoco.MjData(cm)
    for _ in range(round(args.time/cm.opt.timestep)):drive(cm,cd);mujoco.mj_step(cm,cd)
    drive(cm,cd);mujoco.mj_forward(cm,cd)
    with wp.ScopedDevice('cuda:0'):
        m=mjw.put_model(cm);m.opt.deterministic=mjw.DeterminismType.ALL
        d=mjw.put_data(cm,cd,nworld=args.worlds)
        effective=OwnedEffectiveMetric(m,d);wrench=OwnedWrenchProjection(m,d);material=OwnedFlexOperators(m,d)
        d.flex_hessian_valid.zero_()
        cases=[];references={};rng=np.random.default_rng(2026)
        # Nonzero loads on every fourth body, with all six wrench components.
        loads=np.zeros((d.nworld,m.nbody,6),np.float32)
        loads[:,::4]=rng.normal(scale=.1,size=loads[:,::4].shape)
        ft_zero=wp.zeros_like(d.xfrc_applied);ft_loaded=wp.array(loads,dtype=wp.spatial_vector)
        out=wp.zeros((d.nworld,m.nv))
        with material.capture_policy():
            for phase in ('effective','wrench_zero','wrench_loaded'):
                for owned in (False,True):
                    def enqueue():
                        if phase=='effective':derivative.eff_build(m,d)
                        else:
                            ft=ft_zero if phase=='wrench_zero' else ft_loaded
                            if owned:wrench.project(d,ft,out,add=False)
                            else:support.apply_ft(m,d,ft,out,False,sparse=True)
                    with effective.capture_policy() if owned and phase=='effective' else nullcontext():
                        enqueue();wp.synchronize()
                        values=(d.efm_K_val if phase=='effective' else out).numpy()
                        if not owned:references[phase]=values
                        else:np.testing.assert_array_equal(values.view(np.uint8),references[phase].view(np.uint8))
                        with wp.ScopedCapture() as capture:
                            for _ in range(args.repeat):enqueue()
                    cases.append(dict(name=('owned/' if owned else 'sorted/')+phase,graph=capture.graph,times=[]))
        start,end=wp.Event(enable_timing=True),wp.Event(enable_timing=True)
        for sample in range(args.samples):
            for case in cases[::1 if sample%2==0 else -1]:
                wp.record_event(start);wp.capture_launch(case['graph']);wp.record_event(end)
                case['times'].append(wp.get_event_elapsed_time(start,end)/args.repeat)
        for p in (effective,wrench,material):np.testing.assert_array_equal(p.status.numpy(),0)
        report=dict(gpu=wp.get_device().name,nv=m.nv,worlds=d.nworld,nbody=m.nbody,
            metric_csr_entries=m.nefmK,effective_workspace_bytes=effective.workspace_bytes,
            wrench_workspace_bytes=wrench.workspace_bytes,samples=args.samples,repeat=args.repeat,
            checkpoint_s=float(cd.time),bit_identical_outputs=True,
            effective_status=effective.status.numpy().tolist(),wrench_status=wrench.status.numpy().tolist(),
            gpu_processes_before=before,gpu_processes_after=processes(),
            scope='Frozen compressed checkpoint; alternating cases; output resets included; effective build includes common owned material Hessian path; other physics and rendering excluded',
            results={c['name']:dict(cuda_median_ms=float(np.median(c['times'])),
                cuda_p95_ms=float(np.percentile(c['times'],95)),cuda_samples_ms=c['times']) for c in cases})
        args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--time',type=float,default=1.4);p.add_argument('--worlds',type=int,default=2)
    p.add_argument('--samples',type=int,default=30);p.add_argument('--repeat',type=int,default=10)
    p.add_argument('--output',type=Path,default=ROOT/'results/owned_dynamics_0.11.json')
    args=p.parse_args()
    if min(args.worlds,args.samples,args.repeat)<1 or not np.isfinite(args.time) or args.time<0:
        p.error('positive counts and finite nonnegative time required')
    run(args)
