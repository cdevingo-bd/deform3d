"""Paired frozen-state contact Hessian/force reductions, including owned setup.

All cases receive exactly the same actual canonical MJWarp constraint data.
This is an assembly microbenchmark, not end-to-end simulation throughput.
"""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import solver
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'),str(ROOT/'tools')]
from deform3d_warp.mujoco_assembly import OwnedContactAssembly
from deform3d_warp.mujoco_contact import flex_contact_policy
from deform3d_warp.mujoco_determinism import SparseDeterminismBudget
from deform3d_warp.mujoco_metric import CooperativeMetricSolver
from deform3d_warp.mujoco_ordering import CanonicalContactConstraints
from record_bag import drive


def processes():
    return subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()


def run(args):
    initial_processes = processes()
    cm = mujoco.MjModel.from_xml_path(str(ROOT/'examples/mujoco_bag_strong_ref.xml'))
    cm.opt.ccd_tolerance = 1e-9
    cpu = mujoco.MjData(cm)
    for _ in range(round(args.time/cm.opt.timestep)): drive(cm,cpu); mujoco.mj_step(cm,cpu)
    with wp.ScopedDevice('cuda:0'):
        m = mjw.put_model(cm); m.opt.deterministic = mjw.DeterminismType.ALL
        m.opt.ccd_tolerance.fill_(1e-8)
        d = mjw.put_data(cm,cpu,nworld=args.worlds)
        metric = CooperativeMetricSolver(m,args.worlds)
        ordering = CanonicalContactConstraints(m,d)
        bound = SparseDeterminismBudget(m)
        p = OwnedContactAssembly(m,d)
        saved = {}
        class Captured(Exception): pass
        def take(m,d,ctx,*args,**kwargs):
            saved['ctx'] = ctx
            raise Captured()
        with ExitStack() as stack:
            stack.enter_context(flex_contact_policy(m)); stack.enter_context(metric.capture_policy())
            stack.enter_context(ordering.capture_policy()); stack.enter_context(p.capture_policy())
            stack.enter_context(patch.object(solver,'_cholesky_factorize_solve',take))
            try: mjw.forward(m,d)
            except Captured: pass
        ctx = saved['ctx']; ctx.done.zero_()
        np.testing.assert_array_equal(p.status.numpy(),0)
        np.testing.assert_array_equal(metric.status.numpy(),0)
        np.testing.assert_array_equal(ordering.status.numpy(),0)
        J = p.dense.numpy().astype(np.float64)
        D = d.efc.D.numpy().astype(np.float64)
        state = d.efc.state.numpy()
        active = np.arange(d.njmax)[None,:] < d.nefc.numpy()[:,None]
        weight = np.where(active & (state == 1),D,0)
        force = np.where(active,d.efc.force.numpy(),0).astype(np.float64)
        reference_h = np.einsum('wri,wr,wrj->wij',J,weight,J)
        reference_f = np.einsum('wri,wr->wi',J,force)
        if not np.linalg.norm(reference_h) or not np.linalg.norm(reference_f):
            raise ValueError('choose a checkpoint with nonzero contact stiffness and force')
        baseline = solver._JTDACJ_sparse(False,m.opt.cone,3,m.block_dim.update_gradient_JTDAJ_sparse)
        groups = solver._jtdaj_groups_per_world(d.nworld,d.njmax,baseline)
        records = math.ceil(d.njmax/groups)*math.ceil(bound.max_row_dofs*(bound.max_row_dofs+1)/2/32)
        inputs = [m.opt.impratio_invsqrt,d.contact.friction,d.contact.dim,d.efc.id,
            d.efc.jtdaj_adr,d.efc.jtdaj_nrow,d.efc.jtdaj_nblock,d.efc.J_rownnz,
            d.efc.J_rowadr,d.efc.J_colind,d.efc.J,d.efc.D,d.efc.state,d.dof_cdof,
            ctx.Jaref,ctx.done,groups]
        cases = []
        for name in ('atomic','sorted_deterministic','owned_with_prepare','owned_reuse'):
            deterministic = name == 'sorted_deterministic'
            hkernel = solver._JTDACJ_sparse(False,m.opt.cone,3,m.block_dim.update_gradient_JTDAJ_sparse,
                                            records if deterministic else 0,deterministic)
            fkernel = solver._update_constraint_init_qfrc_constraint_sparse(False,m.nv if deterministic else 0,deterministic)
            def enqueue():
                ctx.h.zero_(); d.qfrc_constraint.zero_()
                if name.startswith('owned'):
                    if name == 'owned_with_prepare': p.prepare(d)
                    p.full(d,ctx); p.force(d,ctx,d.nefc)
                else:
                    wp.launch(hkernel,dim=(d.nworld,groups,32),inputs=inputs,outputs=[ctx.h],
                              block_dim=m.block_dim.update_gradient_JTDAJ_sparse)
                    wp.launch(fkernel,dim=(d.nworld,d.njmax),inputs=[d.nefc,d.efc.J_rownnz,d.efc.J_rowadr,
                        d.efc.J_colind,d.efc.J,d.efc.force,d.dof_cdof,d.nefc,ctx.done],outputs=[d.qfrc_constraint])
            enqueue(); wp.synchronize()
            h,f = ctx.h.numpy()[:,:m.nv,:m.nv],d.qfrc_constraint.numpy()
            np.testing.assert_allclose(np.triu(h),np.triu(reference_h),rtol=5e-6,atol=2e-5)
            np.testing.assert_allclose(f,reference_f,rtol=5e-6,atol=2e-5)
            with wp.ScopedCapture() as capture:
                for _ in range(args.repeat): enqueue()
            cases.append(dict(name=name,graph=capture.graph,times=[],
                relative_h_error=float(np.linalg.norm(np.triu(h-reference_h))/np.linalg.norm(np.triu(reference_h))),
                relative_force_error=float(np.linalg.norm(f-reference_f)/np.linalg.norm(reference_f)),
                world_h_equal=bool(all(np.array_equal(h[0],a) for a in h)),
                world_force_equal=bool(all(np.array_equal(f[0],a) for a in f))))
        start,end = wp.Event(enable_timing=True),wp.Event(enable_timing=True)
        for sample in range(args.samples):
            for case in cases[::1 if sample%2==0 else -1]:
                wp.record_event(start); wp.capture_launch(case['graph']); wp.record_event(end)
                case['times'].append(wp.get_event_elapsed_time(start,end)/args.repeat)
        report = dict(gpu=wp.get_device().name,nv=m.nv,nworld=d.nworld,njmax=d.njmax,
            nefc=d.nefc.numpy().tolist(),checkpoint_s=float(cpu.time),workspace_bytes=p.workspace_bytes,
            cpu_ccd_tolerance=1e-9,gpu_ccd_tolerance=1e-8,samples=args.samples,repeat=args.repeat,
            status=p.status.numpy().tolist(),gpu_processes_before=initial_processes,gpu_processes_after=processes(),
            scope='Fixed contact Hessian plus generalized contact force; excludes metric/physics/linear solve/rendering; '
                  'owned_with_prepare includes transpose setup, owned_reuse amortizes it across Newton iterations',results={})
        for case in cases:
            report['results'][case['name']]={k:v for k,v in case.items() if k not in ('name','graph','times')}
            report['results'][case['name']].update(cuda_median_ms=float(np.median(case['times'])),
                cuda_p95_ms=float(np.percentile(case['times'],95)),cuda_samples_ms=case['times'])
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--time',type=float,default=1.4)
    parser.add_argument('--worlds',type=int,default=2)
    parser.add_argument('--samples',type=int,default=30)
    parser.add_argument('--repeat',type=int,default=10)
    parser.add_argument('--output',type=Path,default=ROOT/'results/contact_assembly_reproduction.json')
    args=parser.parse_args()
    if min(args.worlds,args.samples,args.repeat)<1 or not np.isfinite(args.time) or args.time<0:
        parser.error('counts must be positive; time finite and nonnegative')
    run(args)
