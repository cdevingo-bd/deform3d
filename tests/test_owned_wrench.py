"""Independent COM Jacobian/virtual-work references and exact sparse parity."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import support
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from deform3d_warp.mujoco_wrench import OwnedWrenchProjection

XML = '''<mujoco><option gravity="0 0 0"/><worldbody>
  <geom size=".01"/>
  <body name="root" pos=".1 .2 .4"><freejoint/><geom size=".04" mass="1"/>
    <body name="arm" pos=".13 0 .09"><joint axis="0 1 0"/><geom size=".03" mass=".2"/>
      <body name="tip" pos=".07 .03 .11"><geom pos=".01 -.02 0" size=".02" mass=".1"/></body>
      <body name="slider" pos="-.1 .1 0"><joint type="slide" axis="1 1 0"/><geom size=".02" mass=".1"/></body>
    </body>
  </body>
  <body pos="-.2 0 .3"><joint axis="1 0 0"/><geom size=".03" mass=".2"/></body>
</worldbody></mujoco>'''


def fixture(worlds=3):
    cm=mujoco.MjModel.from_xml_string(XML);cd=mujoco.MjData(cm)
    cd.qpos[3:7]=[.9,.1,.2,.3];cd.qpos[7:]=[.31,.02,-.21]
    mujoco.mj_normalizeQuat(cm,cd.qpos);mujoco.mj_forward(cm,cd)
    m=mjw.put_model(cm);m.opt.deterministic=mjw.DeterminismType.ALL
    d=mjw.put_data(cm,cd,nworld=worlds)
    return cm,cd,m,d,OwnedWrenchProjection(m,d)


def bits(x,y):
    np.testing.assert_array_equal(np.ascontiguousarray(x).view(np.uint8),np.ascontiguousarray(y).view(np.uint8))


class WrenchTests(unittest.TestCase):
    def test_independent_cpu_com_jacobian_and_virtual_work(self):
        cm,cd,m,d,p=fixture();rng=np.random.default_rng(98)
        f=rng.normal(size=(d.nworld,m.nbody,6)).astype(np.float32)
        q=rng.normal(size=(d.nworld,m.nv)).astype(np.float32);out=wp.array(q)
        J=np.zeros((m.nbody,6,m.nv))
        for b in range(m.nbody):mujoco.mj_jacBodyCom(cm,cd,J[b,:3],J[b,3:],b)
        p.project(d,wp.array(f,dtype=wp.spatial_vector),out)
        expected=np.einsum('bsv,wbs->wv',J,f)
        np.testing.assert_allclose(out.numpy(),q+expected,rtol=8e-7,atol=4e-7)
        v=rng.normal(size=(d.nworld,m.nv))
        power=np.sum((out.numpy()-q)*v,axis=1)
        body_power=np.sum(f*np.einsum('bsv,wv->wbs',J,v),axis=(1,2))
        np.testing.assert_allclose(power,body_power,rtol=5e-7,atol=2e-6)
        np.testing.assert_array_equal(p.status.numpy(),0)

    def test_native_bit_parity_worlds_exponents_zero_and_add_modes(self):
        cm,cd,m,d,p=fixture();rng=np.random.default_rng(18)
        for add in (True,False):
            for mode in ('random','zero','wide','tiny'):
                with self.subTest(add=add,mode=mode):
                    f=rng.normal(size=(d.nworld,m.nbody,6)).astype(np.float32)
                    if mode=='zero':f.fill(-0.)
                    if mode=='wide':f=np.ldexp(f,rng.integers(-60,60,f.shape)).astype(np.float32)
                    if mode=='tiny':f*=np.finfo(np.float32).tiny
                    f[:,0]=100.;f[1,2]=0. # fixed body ignored; different active records/world
                    ft=wp.array(f,dtype=wp.spatial_vector)
                    base=rng.normal(size=(d.nworld,m.nv)).astype(np.float32)
                    if mode=='zero':base.fill(-0.)
                    a,b=wp.array(base),wp.array(base)
                    support.apply_ft(m,d,ft,a,add,sparse=True);p.project(d,ft,b,add=add)
                    bits(a.numpy(),b.numpy())
        np.testing.assert_array_equal(p.status.numpy(),0)

    def test_graph_replay_updates_wrenches_and_kinematics(self):
        cm,cd,m,d,p=fixture();ft=wp.zeros_like(d.xfrc_applied);out=wp.zeros((d.nworld,m.nv))
        p.project(d,ft,out,add=False);wp.synchronize()
        with wp.ScopedCapture() as cap:p.project(d,ft,out,add=False)
        rng=np.random.default_rng(27)
        for angle in (.0,.7,-.4):
            cd.qpos[7]=angle;mujoco.mj_forward(cm,cd)
            other=mjw.put_data(cm,cd,nworld=d.nworld)
            for name in ('xipos','subtree_com','cdof'):wp.copy(getattr(d,name),getattr(other,name))
            f=rng.normal(size=(d.nworld,m.nbody,6)).astype(np.float32)
            if angle<0:f.fill(0.)
            ft.assign(f);wp.capture_launch(cap.graph)
            expected=wp.empty_like(out);support.apply_ft(m,d,ft,expected,False,sparse=True)
            bits(out.numpy(),expected.numpy())

    def test_policy_sparse_scope_dense_preservation_and_restoration(self):
        cm,cd,m,d,p=fixture();rng=np.random.default_rng(31)
        ft=wp.array(rng.normal(size=(d.nworld,m.nbody,6)).astype(np.float32),dtype=wp.spatial_vector)
        expected=wp.zeros((d.nworld,m.nv));actual=wp.zeros_like(expected)
        support.apply_ft(m,d,ft,expected,False,sparse=False)
        original=support.apply_ft
        with p.capture_policy():
            support.apply_ft(m,d,ft,actual,False,sparse=False)
            bits(actual.numpy(),expected.numpy())
            with patch.object(p,'project',wraps=p.project) as called:
                support.apply_ft(m,d,ft,actual,False,sparse=True)
                self.assertEqual(called.call_count,1)
                other=mjw.put_model(cm)
                support.apply_ft(other,d,ft,actual,False,sparse=True)
                self.assertEqual(called.call_count,1)
            with self.assertRaisesRegex(RuntimeError,'nested'):
                with p.capture_policy():pass
        self.assertIs(support.apply_ft,original)
        with self.assertRaisesRegex(RuntimeError,'test exit'):
            with p.capture_policy():raise RuntimeError('test exit')
        self.assertIs(support.apply_ft,original)

    def test_zero_dofs_and_fixed_body_loads(self):
        cm=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=".1"/></worldbody></mujoco>')
        cd=mujoco.MjData(cm);mujoco.mj_forward(cm,cd)
        m=mjw.put_model(cm);d=mjw.put_data(cm,cd,nworld=2);p=OwnedWrenchProjection(m,d)
        out=wp.empty((2,0));p.project(d,d.xfrc_applied,out,add=False)
        self.assertEqual(out.size,0);np.testing.assert_array_equal(p.status.numpy(),0)

    def test_type_alias_topology_memory_source_and_policy_guards(self):
        cm,cd,m,d,p=fixture();out=wp.zeros((d.nworld,m.nv))
        with self.assertRaises(ValueError):p.project(d,wp.zeros(d.xfrc_applied.shape),out)
        with self.assertRaises(ValueError):p.project(d,d.xfrc_applied,wp.zeros((1,m.nv)))
        alias=wp.array(ptr=d.cdof.ptr,dtype=float,shape=(d.nworld,m.nv),device=p.device)
        with self.assertRaisesRegex(ValueError,'alias'):p.project(d,d.xfrc_applied,alias)
        with self.assertRaisesRegex(ValueError,'memory bound'):OwnedWrenchProjection(m,d,max_workspace_bytes=1)
        parents=m.dof_parentid.numpy();m.dof_parentid.fill_(m.nv)
        with self.assertRaisesRegex(ValueError,'parents'):OwnedWrenchProjection(m,d)
        with self.assertRaisesRegex(ValueError,'topology changed'):p.validate_setup()
        m.dof_parentid.assign(parents)
        with patch.object(support,'apply_ft',lambda:None):
            with self.assertRaisesRegex(RuntimeError,'unreviewed'):p.validate_setup()
        m.opt.deterministic=mjw.DeterminismType.NONE
        with self.assertRaisesRegex(ValueError,'deterministic atomics'):p.validate_setup()

    def test_nonfinite_status_is_sticky(self):
        cm,cd,m,d,p=fixture();ft=wp.full_like(d.xfrc_applied,wp.spatial_vector(np.inf))
        out=wp.zeros((d.nworld,m.nv));p.project(d,ft,out)
        np.testing.assert_array_equal(p.status.numpy(),1)
        ft.zero_();p.project(d,ft,out,add=False)
        np.testing.assert_array_equal(out.numpy(),0);np.testing.assert_array_equal(p.status.numpy(),1)

    def test_contents_load_driver_cpu_gpu_balance_and_release(self):
        from optional_mjwarp_bag import drive_contents,drive_contents_cpu
        times=np.array([0.,.9,.90001,1.,1.19999,1.2,1.4,1.6,1.60001,2.,2.19999,2.2,3.],np.float32)
        ids=np.array([2,3,5],np.int32);out=wp.zeros((len(times),7),dtype=wp.spatial_vector)
        wp.launch(drive_contents,len(times),inputs=[wp.array(times),1,wp.array(ids),.2,out])
        actual=out.numpy();expected=[]
        for t in times:
            data=SimpleNamespace(time=float(t),xfrc_applied=np.zeros((7,6)))
            drive_contents_cpu(data,1,ids,.2);expected.append(data.xfrc_applied)
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=7e-8)
        np.testing.assert_allclose(actual[:,:,:3].sum(axis=1),0,rtol=0,atol=2e-8)
        np.testing.assert_array_equal(actual[[0,1,-2,-1]],0)
        self.assertGreater(float(actual[6,1,5]),0)


if __name__=='__main__':
    with wp.ScopedDevice('cuda:0'):unittest.main(verbosity=2)
