"""Optional CPU integration tests; run with LD_PRELOAD of the hook build."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import os
from pathlib import Path
import unittest
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LIB=ctypes.CDLL(os.environ.get('DEFORM3D_MUJOCO_LIBRARY',str(ROOT/'build/mujoco-simplex/lib/libmujoco.so.4.0.1')))
LIB.deform3d_simplex_enable.argtypes=[ctypes.c_void_p]
LIB.deform3d_simplex_enable.restype=ctypes.c_int
LIB.deform3d_simplex_calls.restype=ctypes.c_ulonglong

def enable(model=None):return LIB.deform3d_simplex_enable(None if model is None else ctypes.c_void_p(model._address))

class CpuSimplexTests(unittest.TestCase):
    def setUp(self):
        self.m=mujoco.MjModel.from_xml_path(str(ROOT/'examples/mujoco_bag_strong_ref.xml'))
        self.d=mujoco.MjData(self.m)
        rng=np.random.default_rng(19)
        mujoco.mj_integratePos(self.m,self.d.qpos,rng.normal(scale=.002,size=self.m.nv),1.)
        self.d.qvel[:]=rng.normal(scale=.1,size=self.m.nv)
    def tearDown(self):enable()
    def test_force_equivalence_and_disable_transitions(self):
        m,d=self.m,self.d
        for flags in (0,mujoco.mjtDisableBit.mjDSBL_SPRING,mujoco.mjtDisableBit.mjDSBL_DAMPER,
                      mujoco.mjtDisableBit.mjDSBL_SPRING|mujoco.mjtDisableBit.mjDSBL_DAMPER,0):
            m.opt.disableflags=int(flags);enable();mujoco.mj_forward(m,d)
            reference={name:getattr(d,name).copy() for name in ('qfrc_spring','qfrc_damper','qfrc_passive')}
            self.assertTrue(enable(m));mujoco.mj_forward(m,d)
            if flags==0:self.assertGreater(LIB.deform3d_simplex_calls(),0)
            for name,value in reference.items():
                np.testing.assert_allclose(getattr(d,name),value,rtol=2e-9,atol=1e-8,err_msg=name)
    def test_reject_different_constitutive_law(self):
        m=self.m;f=next(f for f in range(m.nflex) if m.flex_dim[f]==3)
        adr=m.flex_stiffnessadr[f]+21;m.flex_stiffness[adr]=1.
        self.assertFalse(enable(m))
    def test_thread_local_enable_does_not_change_other_threads(self):
        m=self.m
        def worker(active):
            d=mujoco.MjData(m)
            self.assertTrue(enable(m if active else None))
            mujoco.mj_forward(m,d)
            result=int(LIB.deform3d_simplex_calls());enable()
            return result
        with ThreadPoolExecutor(2) as pool:
            a,b=list(pool.map(worker,(True,False)))
        self.assertGreater(a,0);self.assertEqual(b,0)

if __name__=='__main__':unittest.main()
