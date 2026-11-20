"""Paired factor/solve timings on an identical retained Newton system."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import numpy as np
import warp as wp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from deform3d_warp.mujoco_newton import _Workspace


def run(args):
    with np.load(args.system,allow_pickle=False) as z:
        a,b=z['matrix'].astype(np.float32),z['rhs'].astype(np.float32)
    if a.shape[0]!=1 or b.shape[0]!=1:raise ValueError('expected one retained system')
    a,b=np.repeat(a,args.worlds,axis=0),np.repeat(b,args.worlds,axis=0)
    worlds,n=b.shape
    with wp.ScopedDevice('cuda:0'):
        records={}
        for panel_mode,threads,conditional in (('tiled',64,False),('register',64,False),('register',128,False),('register',128,True)):
            ctx=NS(h=wp.array(a),grad=wp.array(b),done=wp.zeros(worlds,dtype=wp.bool),
                   search=wp.zeros(b.shape),search_dot=wp.zeros(worlds),newton_decrement=wp.zeros(worlds))
            ws=_Workspace(n,worlds,'cuda:0',args.precision,args.tile,'register','ordered_warp',panel_mode,threads,conditional)
            count=wp.ones(worlds,dtype=wp.int32)
            ws.enqueue(ctx,count,count)
            if args.factor_state!='changed':count.zero_()
            if args.factor_state=='inactive':ctx.done.fill_(True)
            scopes=[('full',lambda:ws.enqueue(ctx,count,count,True,False))]
            if args.factor_state=='changed':scopes.append(('factor',lambda:ws._enqueue_factor(ctx)))
            for scope,fn in scopes:
                fn();wp.synchronize()
                with wp.ScopedCapture() as capture:
                    for _ in range(args.repeat):fn()
                # Graphs do not keep all device arrays alive: retain context/count too.
                records[f'{panel_mode}/{threads}/'+('conditional/' if conditional else '')+scope]=dict(graph=capture.graph,ws=ws,ctx=ctx,count=count,ms=[])
        start,end=wp.Event(enable_timing=True),wp.Event(enable_timing=True)
        for sample in range(args.samples):
            for name in list(records)[::1 if sample%2 else -1]:
                r=records[name]
                wp.record_event(start);wp.capture_launch(r['graph']);wp.record_event(end)
                r['ms'].append(wp.get_event_elapsed_time(start,end)/args.repeat)
        report=dict(system=str(args.system),system_sha256=hashlib.sha256(args.system.read_bytes()).hexdigest(),
                    nv=n,worlds=worlds,factor_state=args.factor_state,precision=args.precision,tile=args.tile,gpu=wp.get_device().name,
                    warp_version=wp.__version__,samples=args.samples,solves_per_sample=args.repeat,
                    scope='Identical frozen Hessian/RHS; alternating CUDA graphs. Full includes original-Hessian actual-output residual/status. Excludes physics, assembly and rendering.',
                    results={},bitwise_equal={})
        baseline=records['tiled/64/full']
        for name,r in records.items():
            ctx,ws=r['ctx'],r['ws']
            x=-ctx.search.numpy().astype(np.float64)
            residual=np.linalg.norm((a.astype(np.float64)@x[...,None])[...,0]-b,axis=1)/np.linalg.norm(b.astype(np.float64),axis=1)
            np.testing.assert_array_equal(ws.status.numpy(),0)
            np.testing.assert_array_equal(ctx.h.numpy(),a);np.testing.assert_array_equal(ctx.grad.numpy(),b)
            if not np.isfinite(residual).all() or np.any(residual>1e-4):raise RuntimeError('actual output failed residual gate')
            np.testing.assert_allclose(ws.max_relative_residual.numpy(),residual,rtol=1e-4,atol=1e-8)
            equality={}
            for owner,fields in (('ctx',('search','search_dot','newton_decrement')),('ws',('work','x','status','max_relative_residual'))):
                for field in fields:
                    x,y=getattr(baseline[owner],field).numpy(),getattr(r[owner],field).numpy()
                    equality[field]=bool(np.array_equal(x.view(np.uint8),y.view(np.uint8)))
            report['bitwise_equal'][name]=equality
            report['results'][name]=dict(cuda_ms_median=float(np.median(r['ms'])),cuda_ms_p95=float(np.percentile(r['ms'],95)),
                                         status=ws.status.numpy().tolist(),relative_residual=residual.tolist())
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        if not all(all(v.values()) for v in report['bitwise_equal'].values()):raise RuntimeError('arithmetic equivalence failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system',type=Path,default=ROOT/'tests/fixtures/newton_system.npz')
    p.add_argument('--worlds',type=int,default=2)
    p.add_argument('--factor-state',choices=['changed','reuse','inactive'],default='changed')
    p.add_argument('--precision',choices=['float32','float64'],default='float32')
    p.add_argument('--tile',type=int,choices=[16,32],default=32)
    p.add_argument('--samples',type=int,default=30)
    p.add_argument('--repeat',type=int,default=10)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if min(args.worlds,args.samples,args.repeat)<1:p.error('counts must be positive')
    run(args)
