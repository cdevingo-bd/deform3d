"""Effective-metric assembly: exact native parity and independent J' H J."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import derivative,passive,types
import numpy as np
import warp as wp

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'tools'),str(ROOT/'tests')]
from deform3d_warp.mujoco_effective import OwnedEffectiveMetric,_VERT,_EDGE,_BEND
from deform3d_warp.mujoco_material import OwnedFlexOperators
from test_owned_material import jacobian,cartesian_hessian,bending_hessian
from record_bag import drive


def bits(x,y):
    np.testing.assert_array_equal(np.ascontiguousarray(x).view(np.uint8),np.ascontiguousarray(y).view(np.uint8))


def dense(m,d):
    out=np.zeros((d.nworld,m.nv,m.nv))
    values=d.efm_K_val.numpy();cols=m.efm_K_colind.numpy()
    for r,(a,n) in enumerate(zip(m.efm_K_rowadr.numpy(),m.efm_K_rownnz.numpy())):
        out[:,r,cols[a:a+n]]=values[:,a:a+n]
    return out


class EffectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cm=mujoco.MjModel.from_xml_path(str(ROOT/'examples/mujoco_bag_strong_ref.xml'));cls.cm.opt.ccd_tolerance=1e-9
        cls.cd=mujoco.MjData(cls.cm)
        for _ in range(1400):drive(cls.cm,cls.cd);mujoco.mj_step(cls.cm,cls.cd)
        drive(cls.cm,cls.cd);mujoco.mj_forward(cls.cm,cls.cd)
        cls.J=jacobian(cls.cm,cls.cd)

    def fixture(self):
        m=mjw.put_model(self.cm);m.opt.deterministic=mjw.DeterminismType.ALL
        d=mjw.put_data(self.cm,self.cd,nworld=2)
        return m,d,OwnedEffectiveMetric(m,d)

    def test_actual_compressed_metric_and_material_composition(self):
        m,d,p=self.fixture();d.flex_hessian_valid.zero_();derivative.eff_build(m,d);expected=d.efm_K_val.numpy()
        with p.capture_policy():derivative.eff_build(m,d)
        bits(d.efm_K_val.numpy(),expected)
        material=OwnedFlexOperators(m,d)
        with material.capture_policy(),p.capture_policy():
            d.flex_hessian_valid.zero_();derivative.eff_build(m,d)
        bits(d.efm_K_val.numpy(),expected)
        np.testing.assert_array_equal(p.status.numpy(),0)

    def test_independent_generalized_hessian_symmetry_energy_and_scaling(self):
        m,d,p=self.fixture();rng=np.random.default_rng(27)
        for flags in (0,types.DisableBit.SPRING,types.DisableBit.DAMPER,types.DisableBit.SPRING|types.DisableBit.DAMPER):
            with self.subTest(flags=flags):
                m.opt.disableflags=flags
                with p.capture_policy():derivative.eff_build(m,d)
                if flags==(types.DisableBit.SPRING|types.DisableBit.DAMPER):
                    np.testing.assert_array_equal(d.efm_K_val.numpy(),0);continue
                H=cartesian_hessian(self.cm,d)+bending_hessian(self.cm)
                dt=float(m.opt.timestep.numpy()[0])
                scales=(0 if flags & types.DisableBit.SPRING else dt*dt)+(
                    0 if flags & types.DisableBit.DAMPER else dt)*m.flex_damping.numpy()
                per=np.repeat(scales[m.flex_vertflexid.numpy()],3)
                K=self.J.T@(per[:,None]*H)@self.J;actual=dense(m,d)[0]
                self.assertLess(np.linalg.norm(actual-K)/np.linalg.norm(K),7e-7)
                np.testing.assert_allclose(actual,actual.T,rtol=0,atol=2e-7)
                v=rng.normal(size=m.nv)
                self.assertGreater(v@actual@v,0)
                self.assertAlmostEqual(v@actual@v,(self.J@v)@(per*(H@(self.J@v))),delta=2e-6)

    def test_dynamic_rotations_equal_quaternions_and_zero_transitions(self):
        m,d,p=self.fixture();rng=np.random.default_rng(61)
        # Operator test: vary all frame inputs; retain their matrix/quaternion consistency.
        for mode in ('different','equal','zero'):
            quat=rng.normal(size=(2,m.nbody,4));quat/=np.linalg.norm(quat,axis=-1,keepdims=True)
            if mode=='equal':quat[:]=[1,0,0,0]
            matrices=np.empty((2,m.nbody,3,3),np.float32)
            for w in range(2):
                for b in range(m.nbody):
                    mat=np.empty(9);mujoco.mju_quat2Mat(mat,quat[w,b]);matrices[w,b]=mat.reshape(3,3)
            # MJWarp's quaternions retain MuJoCo wxyz storage in the reviewed kernel.
            d.xquat.assign(quat.astype(np.float32));d.xmat.assign(matrices)
            d.flexvert_hessian.assign(rng.normal(size=(2,m.nflexvert,6)).astype(np.float32))
            d.flexedge_hessian.assign(rng.normal(size=(2,m.nflexedge,3,3)).astype(np.float32))
            if mode=='zero':m.opt.timestep.zero_()
            with patch.object(passive,'flex_hessian',lambda m,d:None):
                derivative.eff_build(m,d);expected=d.efm_K_val.numpy()
                with p.capture_policy():derivative.eff_build(m,d)
            bits(d.efm_K_val.numpy(),expected)
        np.testing.assert_array_equal(p.status.numpy(),0)

    def test_graph_replay_timestep_bending_and_hessian_changes(self):
        m,d,p=self.fixture();derivative.eff_build(m,d);wp.synchronize()
        with patch.object(passive,'flex_hessian',lambda m,d:None):
            with p.capture_policy():
                derivative.eff_build(m,d);wp.synchronize()
                with wp.ScopedCapture() as cap:derivative.eff_build(m,d)
            for value in (.003,0.,.001):
                m.opt.timestep.fill_(value);d.flexvert_hessian.fill_(types.vec6(value))
                m.flex_bending.fill_(value);wp.capture_launch(cap.graph)
                actual=d.efm_K_val.numpy();derivative.eff_build(m,d)
                bits(actual,d.efm_K_val.numpy())

    def test_cache_isolation_and_restoration(self):
        m,d,p=self.fixture();derivative.eff_build(m,d);expected=d.efm_K_val.numpy()
        original=derivative.eff_build
        for name in (_VERT,_EDGE,_BEND):
            self.assertIsNot(p.factories[name](False),getattr(derivative,name)(False))
        with p.capture_policy():
            derivative.eff_build(m,d)
            with self.assertRaisesRegex(RuntimeError,'nested'):
                with p.capture_policy():pass
        bits(d.efm_K_val.numpy(),expected);self.assertIs(derivative.eff_build,original)
        with self.assertRaisesRegex(RuntimeError,'test exit'):
            with p.capture_policy():raise RuntimeError('test exit')
        self.assertIs(derivative.eff_build,original)
        derivative.eff_build(m,d);bits(d.efm_K_val.numpy(),expected)

    def test_source_topology_sorted_csr_and_memory_guards(self):
        m,d,p=self.fixture()
        with self.assertRaisesRegex(ValueError,'memory bound'):OwnedEffectiveMetric(m,d,max_workspace_bytes=1)
        with patch.object(derivative,'eff_build',lambda m,d:None):
            with self.assertRaisesRegex(RuntimeError,'unreviewed'):p.validate_setup()
        cols=m.efm_K_colind.numpy();m.efm_K_colind.fill_(0)
        with self.assertRaisesRegex(ValueError,'sorted and unique'):OwnedEffectiveMetric(m,d)
        with self.assertRaisesRegex(ValueError,'topology changed'):p.validate_setup()
        m.efm_K_colind.assign(cols)
        m.flex_interp.fill_(1)
        with self.assertRaisesRegex(ValueError,'vertex flexes'):OwnedEffectiveMetric(m,d)
        m.flex_interp.zero_()
        with patch.object(d,'efm_K_val',wp.empty((1,m.nefmK))):
            with p.capture_policy():
                with self.assertRaises(ValueError):derivative.eff_build(m,d)

    def test_nonfinite_status_is_sticky(self):
        m,d,p=self.fixture();d.flexvert_hessian.fill_(types.vec6(np.nan))
        with patch.object(passive,'flex_hessian',lambda m,d:None),p.capture_policy():
            derivative.eff_build(m,d);np.testing.assert_array_equal(p.status.numpy(),1)
            d.flexvert_hessian.zero_();d.flexedge_hessian.zero_();derivative.eff_build(m,d)
        np.testing.assert_array_equal(p.status.numpy(),1)

    def test_empty_metric_and_output_alias_rejection(self):
        cm=mujoco.MjModel.from_xml_string('<mujoco><option integrator="discrete"/><worldbody><body><joint/><geom size=".1"/></body></worldbody></mujoco>')
        cd=mujoco.MjData(cm);mujoco.mj_forward(cm,cd)
        m=mjw.put_model(cm);m.opt.deterministic=mjw.DeterminismType.ALL
        d=mjw.put_data(cm,cd,nworld=2);p=OwnedEffectiveMetric(m,d)
        self.assertEqual(m.nefmK,0)
        with p.capture_policy():derivative.eff_build(m,d)
        self.assertEqual(d.efm_K_val.size,0);np.testing.assert_array_equal(p.status.numpy(),0)
        m,d,p=self.fixture()
        # A forged view points into a larger live allocation; rejection occurs
        # before the host builder clears the destination or reads any input.
        storage=wp.empty((2,m.nefmK),dtype=float)
        alias=wp.array(ptr=storage.ptr,dtype=wp.mat33,shape=d.xmat.shape,device=p.device)
        with patch.object(d,'xmat',alias),patch.object(d,'efm_K_val',storage):
            with p.capture_policy():
                with self.assertRaisesRegex(ValueError,'alias'):derivative.eff_build(m,d)

    def test_incomplete_block_and_multiple_world_timesteps(self):
        m,d,p=self.fixture();cols=m.efm_K_colind.numpy();changed=False
        for a,n in zip(m.efm_K_rowadr.numpy(),m.efm_K_rownnz.numpy()):
            row=cols[a:a+n]
            for k in range(2,n-1):
                if row[k]==row[k-2]+2 and row[k+1]>row[k]+1:
                    cols[a+k]+=1;changed=True;break
            if changed:break
        self.assertTrue(changed);m.efm_K_colind.assign(cols)
        with self.assertRaisesRegex(ValueError,'3-column blocks'):OwnedEffectiveMetric(m,d)
        m,d,p=self.fixture();m.opt.timestep=wp.array(np.array([.001,.002],np.float32))
        derivative.eff_build(m,d);expected=d.efm_K_val.numpy()
        with p.capture_policy():derivative.eff_build(m,d)
        bits(d.efm_K_val.numpy(),expected)
        self.assertGreater(np.linalg.norm(expected[1]-expected[0]),0)


if __name__=='__main__':
    with wp.ScopedDevice('cuda:0'):unittest.main(verbosity=2)
