#!/usr/bin/env python3
"""Paired native vs shared C++ material backend INSIDE each host.

One world, identical settings, alternating execution order, complete evolving
trajectories. Timing includes controls, simulation and integrated geometry
refresh; excludes compilation, readback/diagnostics and offline rendering.
"""
import argparse
from contextlib import nullcontext, ExitStack
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'tools'),str(ROOT/'examples'),str(ROOT/'tests')]
from record_bag import drive

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def setup(args):
    m=mujoco.MjModel.from_xml_path(str(args.scene))
    m.opt.ccd_tolerance=1e-14 if args.host=='cpu' else 1e-8
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    names=lambda typ,n:int(mujoco.mj_name2id(m,typ,n))
    bag=names(mujoco.mjtObj.mjOBJ_FLEX,'bag');package=names(mujoco.mjtObj.mjOBJ_FLEX,'package')
    fruit=names(mujoco.mjtObj.mjOBJ_BODY,'fruit')
    pads=[names(mujoco.mjtObj.mjOBJ_GEOM,n+'_geom') for n in ('front_pad','back_pad')]
    if min(bag,package,fruit,*pads)<0:raise ValueError('requires the named bag/contents fixture')
    ids=np.arange(m.flex_vertadr[package],m.flex_vertadr[package]+m.flex_vertnum[package])
    bodies=np.unique(m.flex_vertbodyid[ids]).astype(np.int32)
    def elements(f):
        a=int(m.flex_elemdataadr[f]);n=int(m.flex_elemnum[f]);dim=int(m.flex_dim[f])+1
        return m.flex_elem[a:a+dim*n].reshape(-1,dim)+m.flex_vertadr[f]
    faces,tets=elements(bag),elements(package)
    def measures(points):
        p=points[faces];area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
        p=points[tets];volume=np.einsum('ij,ij->i',np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),p[:,3]-p[:,0])
        return area,volume
    rest_area,rest_volume=measures(d.flexvert_xpos)
    p=d.flexvert_xpos[ids];front=np.isclose(p[:,1],p[:,1].min());back=np.isclose(p[:,1],p[:,1].max())
    rest_width=float(p[back,1].mean()-p[front,1].mean())
    def quality(points):
        a,v=measures(points);p=points[ids]
        return [float((a/rest_area).min()),float((v/rest_volume).min()),
                float(1-(p[back,1].mean()-p[front,1].mean())/rest_width)]
    return m,d,fruit,pads,bodies,quality

def cpu(args,m,initial,fruit,pads,bodies):
    lib=ctypes.CDLL(str(args.library.resolve()))
    enable=lib.deform3d_simplex_enable;enable.argtypes=[ctypes.c_void_p];enable.restype=ctypes.c_int
    count=lib.deform3d_simplex_calls;count.restype=ctypes.c_ulonglong
    address=ctypes.c_void_p(m._address)
    states=[mujoco.MjData(m),mujoco.MjData(m)]
    for d in states:mujoco.mj_forward(m,d)
    if not enable(address):raise RuntimeError('model rejected by CPU core adapter')
    # Actual hook execution proves that Python calls the intended library.
    mujoco.mj_forward(m,states[1])
    if not count():raise RuntimeError('hook not executing; launch with LD_PRELOAD=--library')
    # Same-state force equivalence before independent trajectories.
    errors={}
    rng=np.random.default_rng(32)
    probe=mujoco.MjData(m);delta=rng.normal(scale=.001,size=m.nv)
    mujoco.mj_integratePos(m,probe.qpos,delta,1.);probe.qvel[:]=rng.normal(scale=.02,size=m.nv)
    enable(None);mujoco.mj_forward(m,probe)
    reference={f:getattr(probe,f).copy() for f in ('qfrc_spring','qfrc_damper','qfrc_passive')}
    enable(address);mujoco.mj_forward(m,probe)
    for field,ref in reference.items():
        errors[field]=float(np.linalg.norm(getattr(probe,field)-ref)/max(1e-20,np.linalg.norm(ref)))
    if max(errors.values())>1e-11:raise RuntimeError(('CPU same-state force mismatch',errors))
    enable(None)
    from optional_mjwarp_bag import drive_contents_cpu
    def step(i):
        enable(address if i else None)
        d=states[i];began=time.perf_counter_ns()
        drive(m,d);drive_contents_cpu(d,fruit,bodies,args.shear)
        mujoco.mj_step(m,d);mujoco.mj_kinematics(m,d);mujoco.mj_flex(m,d)
        elapsed=(time.perf_counter_ns()-began)*1e-6
        calls=int(count()) if i else 0
        return elapsed,elapsed,calls
    def read(i):
        d=states[i];net=np.zeros(2);force=np.zeros(6)
        for j,c in enumerate(d.contact):
            if any(pad in c.geom for pad in pads):
                mujoco.mj_contactForce(m,d,j,force);world=c.frame.reshape(3,3).T@force[:3]
                for p,pad in enumerate(pads):net[p]+=world[1]*(int(c.geom[1]==pad)-int(c.geom[0]==pad))
        return dict(points=d.flexvert_xpos.copy(),qpos=d.qpos.copy(),qvel=d.qvel.copy(),
            fruit_pose=np.r_[d.xpos[fruit],d.xquat[fruit]],mocap=d.mocap_pos.copy(),force=net,
            status=int(sum(w.number for w in d.warning)),iterations=int(max(d.solver_niter)),contacts=int(d.ncon))
    return step,read,dict(library=str(args.library.resolve()),library_sha256=digest(args.library),
                         same_state_force_relative_error=errors),lambda:enable(None)

def gpu(args,m,initial,fruit,pads,bodies):
    import warp as wp
    import mujoco_warp as mjw
    from deform3d_warp.mujoco_simplex import SharedSimplexPolicy
    from deform3d_warp.clock import DeviceSimulationClock
    from mjwarp_embedding import refresh_flex_positions
    from optional_mjwarp_bag import drive_precise
    wp.set_device('cuda:0')
    model=mjw.put_model(m)
    # Identical native reductions, solver and collision in both branches.
    model.opt.deterministic=mjw.DeterminismType.ALL if args.deterministic else mjw.DeterminismType.NONE
    states=[mjw.make_data(m,nworld=1),mjw.make_data(m,nworld=1)]
    clocks=[DeviceSimulationClock(1),DeviceSimulationClock(1)]
    policy=SharedSimplexPolicy(model)
    ordering=None
    if args.canonical_rows:
        from deform3d_warp.mujoco_ordering import CanonicalContactConstraints
        ordering=CanonicalContactConstraints(model,states[0])
    budget=None
    if args.deterministic:
        from deform3d_warp.mujoco_determinism import SparseDeterminismBudget
        budget=SparseDeterminismBudget(model)
    load=wp.array(bodies,dtype=int)
    mocaps=[int(m.body_mocapid[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)]) for n in ('front_pad','back_pad')]
    motion=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_NUMERIC,'deform3d_pad_motion')
    start,travel=map(float,m.numeric_data[m.numeric_adr[motion]:m.numeric_adr[motion]+2])
    def frame(state,clock):
        clock.enqueue_write(state.time)
        wp.launch(drive_precise,1,inputs=[clock.elapsed,*mocaps,start,travel,fruit,load,args.shear,state.mocap_pos,state.xfrc_applied])
        mjw.step(model,state)
        clock.enqueue_advance(model.opt.timestep,state.time)
        refresh_flex_positions(model,state)
    graphs=[]
    for i in range(2):
        with ExitStack() as context:
            if budget:context.enter_context(budget.capture_policy())
            if ordering:context.enter_context(ordering.capture_policy())
            if i and not args.native_control:context.enter_context(policy.capture_policy())
            warm=mjw.make_data(m,nworld=1);warmclock=DeviceSimulationClock(1)
            frame(warm,warmclock);wp.synchronize()
            with wp.ScopedCapture() as cap:frame(states[i],clocks[i])
        graphs.append(cap.graph)
    begin,end=wp.Event(enable_timing=True),wp.Event(enable_timing=True)
    contactids=[wp.array(np.arange(d.naconmax,dtype=np.int32),dtype=int) for d in states]
    contactforce=[wp.empty(d.naconmax,dtype=wp.spatial_vector) for d in states]
    def step(i):
        began=time.perf_counter_ns();wp.record_event(begin);wp.capture_launch(graphs[i]);wp.record_event(end)
        device_ms=wp.get_event_elapsed_time(begin,end)
        return (time.perf_counter_ns()-began)*1e-6,device_ms,0
    def read(i):
        d=states[i];n=int(d.nacon.numpy()[0]);geoms=d.contact.geom.numpy()[:n]
        mjw.contact_force(model,d,contactids[i],True,contactforce[i])
        forces=contactforce[i].numpy()[:n];net=np.zeros(2)
        for p,pad in enumerate(pads):net[p]=np.sum(forces[:,1]*((geoms[:,1]==pad).astype(int)-(geoms[:,0]==pad).astype(int)))
        return dict(points=d.flexvert_xpos.numpy()[0],qpos=d.qpos.numpy()[0],qvel=d.qvel.numpy()[0],
            fruit_pose=np.r_[d.xpos.numpy()[0,fruit],d.xquat.numpy()[0,fruit]],mocap=d.mocap_pos.numpy()[0],
            force=net,status=int(d.overflow.numpy()[0])|int(clocks[i].status.numpy()[0])|(int(ordering.status.numpy()[0]) if ordering else 0),
            iterations=int(d.solver_niter.numpy()[0]),contacts=n)
    return step,read,dict(gpu=wp.get_device().name,warp=wp.__version__,mjwarp=mjw.__version__ if hasattr(mjw,'__version__') else '4.0.1',
        determinism='native ALL in both' if args.deterministic else 'native NONE in both',
        common_deterministic_scratch_row_bound=budget.max_row_dofs if budget else None,
        common_canonical_constraint_order=bool(ordering),native_vs_native_control=bool(args.native_control),
        epa_horizon=128,core_sha256=policy.core_sha256),lambda:None

def run(args):
    m,d,fruit,pads,bodies,quality=setup(args)
    step,read,details,cleanup=(cpu if args.host=='cpu' else gpu)(args,m,d,fruit,pads,bodies)
    steps=round(args.duration/m.opt.timestep)
    recorded=[{k:[] for k in ('points','qpos','qvel','fruit_pose','mocap','force','quality','time','wall_ms','device_ms','status','iterations','contacts')} for _ in range(2)]
    calls=0
    try:
        for s in range(steps):
            # Alternate order at every step to avoid a persistent first-run bias.
            for i in ((0,1) if s%2==0 else (1,0)):
                wall,device,count=step(i);calls+=count
                state=read(i);state['quality']=quality(state['points'])
                state.update(time=(s+1)*m.opt.timestep,wall_ms=wall,device_ms=device)
                for key,value in state.items():recorded[i][key].append(value)
            if (s+1)%500==0:print(f'{args.host}: {s+1}/{steps} paired steps',flush=True)
    finally:cleanup()
    arrays=[{k:np.asarray(v) for k,v in rec.items()} for rec in recorded]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for i,(label,a) in enumerate(zip(('native','deform3d'),arrays)):
        path=args.output.with_name(args.output.stem+'_'+label+'.npz')
        np.savez_compressed(path,**a)
        summaries.append(dict(backend=label,recording=str(path.resolve()),sha256=digest(path),
            rt_factor=float(steps*m.opt.timestep/(a['wall_ms'].sum()*.001)),
            wall_p50_ms=float(np.median(a['wall_ms'])),wall_p95_ms=float(np.percentile(a['wall_ms'],95)),
            device_p50_ms=float(np.median(a['device_ms'])),status_bits=int(np.bitwise_or.reduce(a['status'])),
            max_solver_iterations=int(a['iterations'].max()),contacts_observed=int(a['contacts'].sum()),
            min_area_ratio=float(a['quality'][:,0].min()),min_volume_ratio=float(a['quality'][:,1].min()),
            max_compression_fraction=float(a['quality'][:,2].max())))
    a,b=arrays
    shape=float(np.max(np.linalg.norm(a['points'].astype(float)-b['points'],axis=2)))
    rigid=float(np.max(np.linalg.norm(a['fruit_pose'][:,:3]-b['fruit_pose'][:,:3],axis=1)))
    force=float(np.linalg.norm(a['force']-b['force'])/max(1e-12,np.linalg.norm(a['force'])))
    finite=all(all(np.isfinite(v).all() for v in data.values()) for data in arrays)
    valid=finite and shape<=.005 and rigid<=.005 and force<=.1 and all(
        r['status_bits']==0 and r['min_area_ratio']>0 and r['min_volume_ratio']>0 and r['max_solver_iterations']<m.opt.iterations for r in summaries)
    report=dict(host=args.host,scene=str(args.scene.resolve()),scene_sha256=digest(args.scene),
        steps=steps,timestep_s=float(m.opt.timestep),worlds=1,shear_force_N=args.shear,
        solver_iterations=int(m.opt.iterations),solver_tolerance=float(m.opt.tolerance),ccd_tolerance=float(m.opt.ccd_tolerance),
        timing_scope='controller + host physics step + integrated kinematics/flex refresh; synchronized wall time; excludes setup, diagnostics/readback and rendering',
        replaced=('none: native-vs-native diagnostic control' if args.native_control else
                  'triangle/tetra stretch spring and discrete Rayleigh force arithmetic only; shared C++ header'),
        retained='host collision, bending, projected material stiffness, effective metric, contact solver, integration and reductions',
        core_sha256=digest(ROOT/'include/deform3d/simplex_material.hpp'),core_element_calls=calls,details=details,
        native=summaries[0],deform3d=summaries[1],finite=finite,max_flex_delta_m=shape,
        max_fruit_translation_delta_m=rigid,pad_force_relative_rms=force,
        limits=dict(shape_m=.005,force_relative_rms=.1),validation_passed=bool(valid),
        limitation='same-host numerical parity of uncalibrated proxy materials; not calibrated sim-to-real evidence')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not valid:raise SystemExit('same-host parity gates failed; report retained')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',choices=['cpu','gpu'],required=True)
    p.add_argument('--scene',type=Path,default=ROOT/'examples/mujoco_bag_strong_ref.xml')
    p.add_argument('--library',type=Path,default=ROOT/'build/mujoco-simplex/lib/libmujoco.so.4.0.1')
    p.add_argument('--duration',type=float,default=3.);p.add_argument('--shear',type=float,default=.2)
    p.add_argument('--deterministic',action='store_true',help='enable the same native reproducible reductions in both GPU branches')
    p.add_argument('--canonical-rows',action='store_true',help='apply identical explicit constraint row sorting to both GPU branches')
    p.add_argument('--native-control',action='store_true',help='diagnostic GPU native-versus-native repeatability check')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.host=='cpu' and (args.native_control or args.canonical_rows):p.error('GPU diagnostics only')
    if args.canonical_rows and not args.deterministic:p.error('--canonical-rows requires --deterministic')
    if not np.isfinite(args.duration) or args.duration<=0 or not np.isfinite(args.shear) or args.shear<0:p.error('invalid duration/load')
    if args.host=='gpu':
        from mjwarp_capacity import epa_horizon_capacity
        with epa_horizon_capacity(128):run(args)
    else:run(args)
