"""Paired native/owned reductions at one compressed MJWarp Newton checkpoint."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import solver
import numpy as np
import warp as wp

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'tools')]
from deform3d_warp.mujoco_assembly import OwnedContactAssembly
from deform3d_warp.mujoco_contact import flex_contact_policy
from deform3d_warp.mujoco_material import OwnedFlexOperators
from deform3d_warp.mujoco_metric import CooperativeMetricSolver
from deform3d_warp.mujoco_ordering import CanonicalContactConstraints
from record_bag import drive


def processes():
    return subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()


def run(args):
    before=processes()
    cm=mujoco.MjModel.from_xml_path(str(ROOT/'examples/mujoco_bag_strong_ref.xml'));cm.opt.ccd_tolerance=1e-9
    cd=mujoco.MjData(cm)
    for _ in range(round(args.time/cm.opt.timestep)):drive(cm,cd);mujoco.mj_step(cm,cd)
    with wp.ScopedDevice('cuda:0'):
        m=mjw.put_model(cm);m.opt.deterministic=mjw.DeterminismType.ALL;m.opt.ccd_tolerance.fill_(1e-8)
        data=mjw.put_data(cm,cd,nworld=args.worlds)
        metric=CooperativeMetricSolver(m,args.worlds);order=CanonicalContactConstraints(m,data)
        assembly=OwnedContactAssembly(m,data,solver_reductions=True);material=OwnedFlexOperators(m,data)
        saved={}
        class Stop(Exception):pass
        def take(m,d,c,*a,**kw):saved.update(data=d,ctx=c);raise Stop()
        with ExitStack() as stack:
            for policy in (flex_contact_policy(m),metric.capture_policy(),order.capture_policy(),assembly.capture_policy(),material.capture_policy()):stack.enter_context(policy)
            stack.enter_context(patch.object(solver,'_cholesky_factorize_solve',take))
            try:mjw.forward(m,data)
            except Stop:pass
        d,c=saved['data'],saved['ctx'];c.done.zero_();ops=assembly.solver_ops
        cases=[];reference={}
        for name in ('sorted/metric','owned/metric','sorted/gradient','owned/gradient'):
            phase=name.split('/')[1]
            def enqueue():
                if phase=='metric':
                    c.h.zero_()
                    if name.startswith('owned'):ops.add_metric(d,c)
                    else:wp.launch(solver._add_efmK_metric_dense(m.nv,True),(d.nworld,m.nv),
                        inputs=[m.efm_K_rownnz,m.efm_K_rowadr,m.efm_K_colind,d.efm_K_val,c.done],outputs=[c.h])
                else:
                    c.grad_dot.zero_()
                    if name.startswith('owned'):ops.gradient(d,c)
                    else:wp.launch(solver._update_gradient_grad(False,True),(d.nworld,m.nv),
                        inputs=[d.qfrc_smooth,d.qfrc_constraint,d.efc.Ma,d.nefc,c.done],outputs=[c.grad,c.grad_dot])
            enqueue();wp.synchronize()
            values=[c.h.numpy()] if phase=='metric' else [c.grad.numpy()[:,:m.nv],c.grad_dot.numpy()]
            if name.startswith('sorted'):reference[phase]=values
            else:
                for actual,expected in zip(values,reference[phase]):
                    np.testing.assert_array_equal(np.ascontiguousarray(actual).view(np.uint8),
                                                  np.ascontiguousarray(expected).view(np.uint8))
            with wp.ScopedCapture() as capture:
                for _ in range(args.repeat):enqueue()
            cases.append(dict(name=name,graph=capture.graph,times=[]))
        start,end=wp.Event(enable_timing=True),wp.Event(enable_timing=True)
        for sample in range(args.samples):
            for case in cases[::1 if sample%2==0 else -1]:
                wp.record_event(start);wp.capture_launch(case['graph']);wp.record_event(end)
                case['times'].append(wp.get_event_elapsed_time(start,end)/args.repeat)
        np.testing.assert_array_equal(assembly.status.numpy(),0)
        report=dict(gpu=wp.get_device().name,nv=m.nv,worlds=d.nworld,metric_csr_entries=m.nefmK,
            upper_entries=ops.indices.size,workspace_bytes=ops.workspace_bytes,samples=args.samples,repeat=args.repeat,
            checkpoint_s=float(cd.time),status=assembly.status.numpy().tolist(),bit_identical_outputs=True,
            gpu_processes_before=before,gpu_processes_after=processes(),
            scope='Fixed evolving-scene checkpoint, alternating cases; includes output reset, excludes remaining physics, solve and rendering',
            results={case['name']:dict(cuda_median_ms=float(np.median(case['times'])),
                cuda_p95_ms=float(np.percentile(case['times'],95)),cuda_samples_ms=case['times']) for case in cases})
        args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--time',type=float,default=1.4);p.add_argument('--worlds',type=int,default=2)
    p.add_argument('--samples',type=int,default=30);p.add_argument('--repeat',type=int,default=10)
    p.add_argument('--output',type=Path,default=ROOT/'results/solver_reductions_0.10.json')
    a=p.parse_args()
    if min(a.worlds,a.samples,a.repeat)<1 or not np.isfinite(a.time) or a.time<0:p.error('positive counts and finite nonnegative time required')
    run(a)
