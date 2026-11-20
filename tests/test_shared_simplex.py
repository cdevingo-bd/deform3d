"""Same-state native MJWarp comparison for the shared C++ material law."""
from pathlib import Path
import sys
import unittest
import tempfile
import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import passive
import numpy as np
import warp as wp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from deform3d_warp.mujoco_simplex import SimplexFlexOperators, SharedSimplexPolicy, HEADER
from deform3d_warp.mujoco_material import OwnedFlexOperators

class SharedSimplexTests(unittest.TestCase):
    def test_different_installed_headers_have_isolated_kernel_caches(self):
        cm=mujoco.MjModel.from_xml_string('''<mujoco><option integrator="discrete"/>
          <worldbody><flexcomp name="patch" type="grid" dim="2" count="2 2 1" spacing=".05 .05 .05" mass=".1">
          <elasticity young="1000" poisson=".3" thickness=".001"/></flexcomp></worldbody></mujoco>''')
        cd=mujoco.MjData(cm);mujoco.mj_forward(cm,cd)
        m=mjw.put_model(cm);d=mjw.put_data(cm,cd)
        original=SimplexFlexOperators(m,d)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'simplex_material.hpp'
            path.write_text(HEADER.read_text()+'\n// independent installation\n')
            alternate=SimplexFlexOperators(m,d,header_path=path)
            self.assertNotEqual(original.core_sha256,alternate.core_sha256)
            self.assertIsNot(original.factories['_flex_elasticity'](False),alternate.factories['_flex_elasticity'](False))

    def test_native_force_under_strain_velocity_and_disable_flags(self):
        cm=mujoco.MjModel.from_xml_path(str(ROOT/'examples/mujoco_bag_strong_ref.xml'))
        cd=mujoco.MjData(cm)
        # Nonzero strain and velocity exercise membrane, volume and dampers.
        rng=np.random.default_rng(21)
        delta=rng.normal(scale=.001,size=cm.nv)
        mujoco.mj_integratePos(cm,cd.qpos,delta,1.)
        cd.qvel[:]=rng.normal(scale=.05,size=cm.nv)
        mujoco.mj_forward(cm,cd)
        m=mjw.put_model(cm);m.opt.deterministic=mjw.DeterminismType.ALL
        a=mjw.put_data(cm,cd,nworld=2);b=mjw.put_data(cm,cd,nworld=2)
        policy=SimplexFlexOperators(m,b)
        from mujoco_warp._src.types import DisableBit
        for flags in (0,DisableBit.SPRING,DisableBit.DAMPER,DisableBit.SPRING|DisableBit.DAMPER,0):
            m.opt.disableflags=flags
            passive.passive(m,a)
            with policy.capture_policy():passive.passive(m,b)
            for field in ('qfrc_spring','qfrc_damper','qfrc_passive'):
                x,y=getattr(b,field).numpy(),getattr(a,field).numpy()
                error=np.linalg.norm(x-y)/max(1e-10,np.linalg.norm(y))
                self.assertLess(error,2e-6,(flags,field,error))
                np.testing.assert_array_equal(x[0],x[1])
        np.testing.assert_array_equal(policy.status.numpy(),0)
        # Compare pre-reduction records as well: different atomic scheduling
        # must not hide an error in the actual element constitutive arithmetic.
        reference_policy=OwnedFlexOperators(m,a)
        with reference_policy.capture_policy():passive.passive(m,a)
        with policy.capture_policy():passive.passive(m,b)
        for original,shared in zip(reference_policy.buffers['_flex_elasticity'],policy.buffers['_flex_elasticity']):
            np.testing.assert_array_equal(shared.numpy(),original.numpy())
        with policy.capture_policy():
            with wp.ScopedCapture() as cap:passive.passive(m,b)
        b.flexedge_velocity.zero_();a.flexedge_velocity.zero_()
        wp.capture_launch(cap.graph);passive.passive(m,a)
        np.testing.assert_allclose(b.qfrc_passive.numpy(),a.qfrc_passive.numpy(),rtol=3e-6,atol=2e-5)
        direct=SharedSimplexPolicy(m)
        with direct.capture_policy():passive.passive(m,b)
        np.testing.assert_allclose(b.qfrc_passive.numpy(),a.qfrc_passive.numpy(),rtol=3e-6,atol=2e-5)

if __name__=='__main__':unittest.main()
