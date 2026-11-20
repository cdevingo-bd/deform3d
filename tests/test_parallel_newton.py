"""Independent SPD systems for the parallel Newton factorization."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
import warp as wp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from deform3d_warp.mujoco_newton import _Workspace, ParallelNewtonSolver

DEVICE='cuda:0'
_DIAGONAL='tiled'
_SUBSTITUTION='tiled'
_PANEL='tiled'
_UPDATE_THREADS=64
_CONDITIONAL_FACTOR=False
_TRAILING='tiled'


def fixture(matrices,rhs,precision='float64',tile_size=16):
    worlds,nv=rhs.shape
    size=(nv+15)//16*16
    h=np.zeros((worlds,size,size),np.float32)
    h[:,:nv,:nv]=matrices
    # The contract is upper-only; lower entries are deliberately poisoned.
    rows,cols=np.tril_indices(size,-1)
    h[:,rows,cols]=np.nan
    g=np.zeros((worlds,size),np.float32);g[:,:nv]=rhs
    ctx=NS(h=wp.array(h,device=DEVICE),grad=wp.array(g,device=DEVICE),
           search=wp.full((worlds,nv),-123.,device=DEVICE),
           search_dot=wp.zeros(worlds,device=DEVICE),newton_decrement=wp.zeros(worlds,device=DEVICE),
           done=wp.zeros(worlds,dtype=wp.bool,device=DEVICE))
    ws=_Workspace(nv,worlds,DEVICE,precision,tile_size,_DIAGONAL,_SUBSTITUTION,_PANEL,_UPDATE_THREADS,_CONDITIONAL_FACTOR,_TRAILING)
    counts=wp.ones(worlds,dtype=wp.int32,device=DEVICE)
    return ws,ctx,counts


class ParallelNewtonTests(unittest.TestCase):
    def test_larger_tiles(self):
        rng=np.random.default_rng(31)
        with wp.ScopedDevice(DEVICE):
            for precision in ('float32','float64'):
                for n in (17,117,456,512):
                    with self.subTest(n=n,precision=precision):
                        r=rng.normal(size=(n,n));a=(r@r.T+np.eye(n)*n).astype(np.float32)
                        b=rng.normal(size=(2,n)).astype(np.float32)
                        ws,ctx,count=fixture(np.tile(a,(2,1,1)),b,precision,32)
                        ws.enqueue(ctx,count,count)
                        np.testing.assert_array_equal(ws.status.numpy(),0)
                        ref=np.linalg.solve(a.astype(np.float64),b.T).T
                        np.testing.assert_allclose(-ctx.search.numpy(),ref,atol=1e-6,rtol=1e-4)

    def test_float32_factorization(self):
        rng=np.random.default_rng(129)
        with wp.ScopedDevice(DEVICE):
            for n in (17,117,456,512):
                with self.subTest(n=n):
                    q,_=np.linalg.qr(rng.normal(size=(n,n)))
                    a=(q@np.diag(np.geomspace(1,500,n))@q.T).astype(np.float32)
                    b=rng.normal(size=(3,n)).astype(np.float32)
                    ws,ctx,count=fixture(np.tile(a,(3,1,1)),b,'float32')
                    ws.enqueue(ctx,count,count)
                    wp.synchronize_device(DEVICE)
                    with wp.ScopedCapture(device=DEVICE) as capture:
                        ws.enqueue(ctx,count,count)
                    wp.capture_launch(capture.graph)
                    x=-ctx.search.numpy()
                    np.testing.assert_array_equal(ws.status.numpy(),0)
                    reference=np.linalg.solve(a.astype(np.float64),b.T).T
                    np.testing.assert_allclose(x,reference,atol=1e-5,rtol=1e-3)
                    actual=np.linalg.norm(x.astype(np.float64)@a.T-b,axis=1)/np.linalg.norm(b,axis=1)
                    np.testing.assert_allclose(ws.max_relative_residual.numpy(),actual,atol=2e-8,rtol=1e-4)

    def test_numpy_systems_and_graph(self):
        rng=np.random.default_rng(91)
        with wp.ScopedDevice(DEVICE):
            for n in (1,3,16,17,65,117,456,512):
                with self.subTest(n=n):
                    q,_=np.linalg.qr(rng.normal(size=(n,n)))
                    a=(q@np.diag(np.geomspace(1,1000,n))@q.T).astype(np.float32)
                    b=rng.normal(size=(3,n)).astype(np.float32)
                    ws,ctx,count=fixture(np.tile(a,(3,1,1)),b)
                    ws.enqueue(ctx,count,count)
                    wp.synchronize_device(DEVICE)
                    with wp.ScopedCapture(device=DEVICE) as capture:
                        ws.enqueue(ctx,count,count)
                    wp.capture_launch(capture.graph)
                    x=-ctx.search.numpy()
                    np.testing.assert_array_equal(ws.status.numpy(),0)
                    expected=np.linalg.solve(a.astype(np.float64),b.T).T
                    np.testing.assert_allclose(x,expected,rtol=2e-5,atol=2e-6)
                    actual=np.linalg.norm(x.astype(np.float64)@a.T-b,axis=1)/np.linalg.norm(b,axis=1)
                    np.testing.assert_allclose(ws.max_relative_residual.numpy(),actual,atol=2e-8,rtol=1e-4)
                    np.testing.assert_allclose(ctx.search_dot.numpy(),np.sum(x.astype(np.float64)**2,axis=1),rtol=1e-6)
                    np.testing.assert_allclose(ctx.newton_decrement.numpy(),np.sum(x.astype(np.float64)*b,axis=1),rtol=1e-6)

    def test_reuse_and_inactive_worlds(self):
        with wp.ScopedDevice(DEVICE):
            n=33
            a=np.tile(np.eye(n,dtype=np.float32)*2,(3,1,1))
            ws,ctx,count=fixture(a,np.ones((3,n),np.float32))
            ws.enqueue(ctx,count,count)
            np.testing.assert_allclose(ctx.search.numpy(),-.5)
            ctx.grad.fill_(3.)
            ctx.done.assign(np.array([False,True,False]))
            count.assign(np.array([0,0,1],np.int32))
            ws.enqueue(ctx,count,count,True,False)
            np.testing.assert_allclose(ctx.search.numpy(),np.array([[-1.5]*n,[-.5]*n,[-1.5]*n]))
            # No-flip preserves direction and its summaries even with a new RHS.
            before=ctx.search.numpy().copy()
            ctx.grad.fill_(4.)
            count.zero_()
            ws.enqueue(ctx,count,count,True,True)
            np.testing.assert_array_equal(ctx.search.numpy(),before)

    def test_changed_matrix_graph_and_first_reuse(self):
        with wp.ScopedDevice(DEVICE):
            n=32
            a=np.tile(np.eye(n,dtype=np.float32)*2,(3,1,1))
            ws,ctx,count=fixture(a,np.ones((3,n),np.float32))
            count.zero_()
            # A newly active world has no factor to reuse, even if no constraint
            # reports a change. This must factor once instead of reading garbage.
            ws.enqueue(ctx,count,count,True,False)
            np.testing.assert_allclose(ctx.search.numpy(),-.5)
            with wp.ScopedCapture() as capture:
                ws.enqueue(ctx,count,count,True,False)
            a[0]*=2
            a[2]*=3
            ctx.h.assign(a)
            count.assign(np.array([1,0,1],np.int32))
            ctx.grad.fill_(6.)
            wp.capture_launch(capture.graph)
            np.testing.assert_allclose(ctx.search.numpy(),np.array([[-1.5]*n,[-3.]*n,[-1.]*n]))
            np.testing.assert_array_equal(ws.status.numpy(),0)

    def test_unsafe_context_rejected_before_writes(self):
        with wp.ScopedDevice(DEVICE):
            n=16
            ws,ctx,count=fixture(np.eye(n,dtype=np.float32)[None],np.ones((1,n),np.float32))
            original=ctx.search
            for bad in (ctx.grad, wp.zeros((1,n-1),device=DEVICE),
                        wp.zeros((1,n),dtype=wp.float64,device=DEVICE),
                        wp.zeros((1,n),device='cpu'), wp.zeros((1,2*n),device=DEVICE)[:,::2]):
                ctx.search=bad
                with self.assertRaises(ValueError):
                    ws.enqueue(ctx,count,count)
                np.testing.assert_array_equal(original.numpy(),-123.)
            ctx.search=original
            ctx.newton_decrement=ctx.search_dot
            with self.assertRaisesRegex(ValueError,'overlap'):
                ws.enqueue(ctx,count,count)
            np.testing.assert_array_equal(original.numpy(),-123.)

    def test_invalid_factor_and_sticky_status(self):
        with wp.ScopedDevice(DEVICE):
            ws,ctx,count=fixture(np.array([-np.eye(3),np.eye(3)],np.float32),np.ones((2,3),np.float32))
            ws.enqueue(ctx,count,count)
            np.testing.assert_array_equal(ws.status.numpy(),[1,0])
            ctx.h.assign(np.tile(np.eye(16),(2,1,1)).astype(np.float32))
            ws.enqueue(ctx,count,count)
            np.testing.assert_array_equal(ws.status.numpy(),[1,0])
            np.testing.assert_allclose(ctx.search.numpy(),-1.)

    def test_nonfinite_rhs_and_zero_rhs(self):
        with wp.ScopedDevice(DEVICE):
            ws,ctx,count=fixture(np.tile(np.eye(17),(2,1,1)),np.zeros((2,17),np.float32))
            ws.enqueue(ctx,count,count)
            np.testing.assert_array_equal(ctx.search.numpy(),0)
            np.testing.assert_array_equal(ws.status.numpy(),0)
            ctx.grad.fill_(float('nan'))
            ws.enqueue(ctx,count,count)
            np.testing.assert_array_equal(ws.status.numpy(),1)

    def test_float_conversion_residual(self):
        with wp.ScopedDevice(DEVICE):
            a=np.array([[[1e6,1e6-1],[1e6-1,1e6]]],np.float32)
            ws,ctx,count=fixture(a,np.array([[1.,0.]],np.float32))
            ws.enqueue(ctx,count,count)
            self.assertEqual(ws.status.numpy()[0],2)
            self.assertGreater(ws.max_relative_residual.numpy()[0],1e-4)


class RegisterNewtonTests(ParallelNewtonTests):
    def setUp(self):
        override=patch(__name__+'._DIAGONAL','register')
        override.start()
        self.addCleanup(override.stop)


class OrderedNewtonTests(ParallelNewtonTests):
    def setUp(self):
        for name, value in (('_SUBSTITUTION','ordered_warp'),('_DIAGONAL','register')):
            override=patch(__name__+'.'+name,value)
            override.start()
            self.addCleanup(override.stop)


class RegisterPanelNewtonTests(ParallelNewtonTests):
    def setUp(self):
        for name,value in (('_SUBSTITUTION','ordered_warp'),('_DIAGONAL','register'),
                           ('_PANEL','register'),('_UPDATE_THREADS',128)):
            override=patch(__name__+'.'+name,value)
            override.start()
            self.addCleanup(override.stop)


class ConditionalPanelNewtonTests(RegisterPanelNewtonTests):
    def setUp(self):
        super().setUp()
        override=patch(__name__+'._CONDITIONAL_FACTOR',True)
        override.start()
        self.addCleanup(override.stop)


class RegisterTrailingNewtonTests(ConditionalPanelNewtonTests):
    def setUp(self):
        super().setUp()
        override=patch(__name__+'._TRAILING','register')
        override.start()
        self.addCleanup(override.stop)


class ParallelNewtonPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco
        import mujoco_warp as mjw
        with wp.ScopedDevice(DEVICE):
            cls.model=mjw.put_model(mujoco.MjModel.from_xml_path(str(ROOT/'examples/mujoco_bag_fast.xml')))

    def test_configuration_guards(self):
        for kw in ({'nworld':True},{'nworld':0},{'nworld':1.5},{'precision':'half'},{'tile_size':64},{'diagonal':'unknown'},{'substitution':'unknown'},{'panel':'unknown'},{'trailing':'unknown'},
                   {'update_threads':True},{'update_threads':64.0},{'update_threads':256},{'conditional_factor':1}):
            with self.assertRaises(ValueError):
                ParallelNewtonSolver(self.model,**kw)
        from mujoco_warp._src import types
        original=self.model.opt.solver
        try:
            self.model.opt.solver=types.SolverType.CG
            with self.assertRaises(ValueError):
                ParallelNewtonSolver(self.model)
        finally:
            self.model.opt.solver=original

    def test_context_restoration_and_dispatch(self):
        from mujoco_warp._src import solver
        policy=ParallelNewtonSolver(self.model)
        original=solver._cholesky_factorize_solve
        data=NS(nworld=1,qpos=NS(device=wp.get_device(DEVICE)),nefc=object())
        ctx=NS(quad_changed_count=object(),state_changed_count=object())
        with patch.object(policy,'enqueue') as enqueue:
            with self.assertRaisesRegex(ValueError,'sentinel'):
                with policy.capture_policy():
                    solver._cholesky_factorize_solve(self.model,data,ctx,True,True)
                    enqueue.assert_called_once_with(ctx,ctx.quad_changed_count,ctx.state_changed_count,True,True)
                    with self.assertRaisesRegex(RuntimeError,'nested'):
                        with policy.capture_policy():
                            pass
                    raise ValueError('sentinel')
        self.assertIs(solver._cholesky_factorize_solve,original)


if __name__=='__main__':
    unittest.main(verbosity=2)
