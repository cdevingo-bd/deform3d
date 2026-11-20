#!/usr/bin/env python3
"""Record a full bag loading/squeeze cycle and synchronized contact diagnostics."""
import argparse
import json
from pathlib import Path
import time
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def closure(t):
    smooth=lambda x:x*x*(3-2*x)
    if t<.7:return 0.
    if t<1.4:return smooth((t-.7)/.7)
    if t<1.7:return 1.
    if t<2.5:return 1-smooth((t-1.7)/.8)
    return 0.

def drive(m,d):
    motion=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_NUMERIC,'deform3d_pad_motion')
    start,travel=m.numeric_data[m.numeric_adr[motion]:m.numeric_adr[motion]+2] if motion>=0 else (.12,.073)
    center=start-travel*closure(d.time)
    for name,sign in [('front_pad',-1),('back_pad',1)]:
        b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
        d.mocap_pos[m.body_mocapid[b],1]=sign*center

def record(scene,output,duration=3.):
    m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m)
    mujoco.mj_forward(m,d)
    bag=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_FLEX,'bag')
    package=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_FLEX,'package')
    fruit=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'fruit')
    fruit_geom=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'fruit_collision')
    pads=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n+'_geom') for n in ['front_pad','back_pad']]
    a=int(m.flex_elemdataadr[package]);n=int(m.flex_elemnum[package])
    tet=m.flex_elem[a:a+4*n].reshape(-1,4)+int(m.flex_vertadr[package])
    a=int(m.flex_elemdataadr[bag]);n=int(m.flex_elemnum[bag])
    faces=m.flex_elem[a:a+3*n].reshape(-1,3)+int(m.flex_vertadr[bag])
    def volume(p):
        q=p[tet];return np.einsum('ij,ij->i',np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),q[:,3]-q[:,0])
    def area(p):
        q=p[faces];return np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1)
    restvol=volume(d.flexvert_xpos);restarea=area(d.flexvert_xpos)
    package_ids=np.arange(m.flex_vertadr[package],m.flex_vertadr[package]+m.flex_vertnum[package])
    initial_package=d.flexvert_xpos[package_ids].copy()
    front=np.isclose(initial_package[:,1],initial_package[:,1].min())
    back=np.isclose(initial_package[:,1],initial_package[:,1].max())
    rest_thickness=initial_package[back,1].mean()-initial_package[front,1].mean()
    widths=[]
    steps=int(round(duration/m.opt.timestep))
    qpos=[];points=[];mocap=[];fruitpose=[];rows=[];timings=[];finite_velocity=True
    force=np.zeros(6);worst_contact={};worst_depth=0.
    for step in range(steps+1):
        drive(m,d)
        # forward obtains contacts and force at this exact recorded state.
        # It is outside the step-only timing and included in recording cost.
        mujoco.mj_forward(m,d)
        qpos.append(d.qpos.copy());points.append(d.flexvert_xpos.copy())
        finite_velocity=finite_velocity and bool(np.isfinite(d.qvel).all())
        mocap.append(d.mocap_pos.copy());fruitpose.append(np.r_[d.xpos[fruit],d.xquat[fruit]])
        package_points=d.flexvert_xpos[package_ids]
        widths.append(package_points[back,1].mean()-package_points[front,1].mean())
        counters=[0,0,0,0];f=[0.,0.];penetration=0.
        for i,c in enumerate(d.contact):
            penetration=max(penetration,-float(c.dist))
            if -float(c.dist)>worst_depth:
                worst_depth=-float(c.dist)
                worst_contact={'time_s':float(d.time),'depth_m':worst_depth,'flex_ids':c.flex.tolist(),
                               'geom_ids':c.geom.tolist(),'position_m':c.pos.tolist()}
            if bag in c.flex and package in c.flex:counters[0]+=1
            if bag in c.flex and fruit_geom in c.geom:counters[1]+=1
            if c.flex[0]==bag and c.flex[1]==bag:counters[3]+=1
            for j,p in enumerate(pads):
                if p in c.geom:
                    if bag in c.flex:counters[2]+=1
                    mujoco.mj_contactForce(m,d,i,force)
                    world=c.frame.reshape(3,3).T@force[:3]
                    f[j]+=(-1. if c.geom[0]==p else 1.)*world[1]
        f=np.abs(f).tolist()
        rows.append([d.time,*counters,*f,penetration,float(np.min(volume(d.flexvert_xpos)/restvol)),
                     float(np.min(area(d.flexvert_xpos)/restarea)),float(np.max(area(d.flexvert_xpos)/restarea))])
        if step<steps:
            start=time.perf_counter_ns();mujoco.mj_step(m,d);timings.append((time.perf_counter_ns()-start)*1e-6)
    out=output.with_suffix('.npz');out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,qpos=qpos,points=points,mocap=mocap,fruit_pose=fruitpose,diagnostics=rows,package_thickness=widths)
    columns='time_s,bag_package_contacts,bag_fruit_contacts,pad_bag_contacts,bag_self_contacts,front_force_N,back_force_N,max_penetration_m,package_min_volume_ratio,bag_min_area_ratio,bag_max_area_ratio'
    np.savetxt(output.with_suffix('.csv'),rows,delimiter=',',header=columns,comments='')
    r=np.array(rows)
    report={'scene':str(scene),'duration_s':duration,'steps':steps,'timestep_s':float(m.opt.timestep),
            'initial_contacts':int(sum(rows[0][1:5])),
            'contact_observations':dict(zip(columns.split(',')[1:5],r[:,1:5].sum(axis=0).astype(int).tolist())),
            'peak_pad_force_N':r[:,5:7].max(axis=0).tolist(),
            'force_scope':'Magnitude of net world-Y contact force per pad; all contact types, including tangential friction, at the recorded state.',
            'package_rest_thickness_m':float(rest_thickness),'package_min_thickness_m':float(min(widths)),
            'package_final_thickness_m':float(widths[-1]),
            'package_max_compression_fraction':float(1-min(widths)/rest_thickness),
            'max_penetration_m':float(r[:,7].max()),'package_min_volume_ratio':float(r[:,8].min()),
            'worst_contact':worst_contact,
            'bag_area_ratio_range':[float(r[:,9].min()),float(r[:,10].max())],
            'warning_counts':d.warning.number.tolist(),
            'physics_step_ms':{'p50':float(np.median(timings)),'p95':float(np.percentile(timings,95))},
            'physics_only_real_time_factor':duration/(sum(timings)/1000),
            'timing_scope':'MuJoCo step only, complete uninterrupted contact cycle; excludes recording forward, GPU geometry and rendering.',
            'finite':bool(np.isfinite(qpos).all() and np.isfinite(points).all() and finite_velocity)}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if any(report['warning_counts']) or not report['finite'] or report['package_min_volume_ratio']<=0 or report['bag_area_ratio_range'][0]<=1e-8:
        raise RuntimeError('bag cycle failed solver/finite/inversion validation')
    if not all(report['contact_observations'][k]>0 for k in columns.split(',')[1:4]):
        raise RuntimeError('expected bag/contents/pad contact was not exercised')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,default=ROOT/'examples/mujoco_bag.xml')
    p.add_argument('--output',type=Path,default=ROOT/'results/bag_cycle');p.add_argument('--duration',type=float,default=3.)
    a=p.parse_args();record(a.scene,a.output,a.duration)
