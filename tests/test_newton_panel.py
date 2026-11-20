"""Independent panel arithmetic, transport and captured-factor equivalence."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import warp as wp

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'tests')]
from deform3d_warp._newton_panel import ordered_panel
from deform3d_warp.mujoco_newton import _Workspace,_kernels
from test_parallel_newton import fixture


def exact(a,b):
    np.testing.assert_array_equal(a.view(np.uint8),b.view(np.uint8))


class RegisterPanelTests(unittest.TestCase):
    def test_panel_order_masks_and_zero_pivots(self):
        rng=np.random.default_rng(1938)
        with wp.ScopedDevice('cuda:0'):
            for precision in ('float32','float64'):
                real=wp.float32 if precision=='float32' else wp.float64
                dtype=np.float32 if precision=='float32' else np.float64
                for tile in (16,32):
                    for remaining in (1,3,14):
                        with self.subTest(precision=precision,tile=tile,remaining=remaining):
                            size=(remaining+2)*tile
                            # Nonzero k also checks panel address/stride handling.
                            k=tile
                            a=(rng.normal(size=(3,size,size))*.1).astype(dtype)
                            for w in range(3):
                                lower=np.tril(rng.normal(size=(tile,tile))*.05,-1)
                                lower+=np.diag(np.exp(rng.uniform(-1,1,tile)))
                                a[w,k:k+tile,k:k+tile]=lower
                            a[2,k+3,k+3]=0  # Match the reference zero-pivot fallback.
                            rows,cols=np.triu_indices(tile,1)
                            a[:,k+rows,k+cols]=np.nan
                            a[:,k+tile:,k:k+tile]*=np.exp(rng.uniform(-8,8,(3,remaining*tile,tile)))
                            old,new=wp.array(a,dtype=real),wp.array(a,dtype=real)
                            active=wp.array(np.array([1,0,1],np.int32))
                            wp.launch_tiled(_kernels(size,precision,tile)[1],dim=(3,remaining),
                                            inputs=[k,active,old],block_dim=64)
                            wp.launch_tiled(ordered_panel(precision,tile),dim=(3,remaining),
                                            inputs=[k,active,new],block_dim=32)
                            x,y=old.numpy(),new.numpy()
                            self.assertTrue(np.isfinite(y[:,k+tile:,k:k+tile]).all())
                            exact(x,y);exact(y[1],a[1])

    def test_complete_factors_and_actual_residual(self):
        rng=np.random.default_rng(413)
        with wp.ScopedDevice('cuda:0'):
            for precision in ('float32','float64'):
                for tile in (16,32):
                    for n in (1,17,65,117,456,512):
                        with self.subTest(precision=precision,tile=tile,n=n):
                            r=rng.normal(size=(2,n,n))
                            a=(r@r.transpose(0,2,1)+np.eye(n)*n*.1).astype(np.float32)
                            b=rng.normal(size=(2,n)).astype(np.float32)
                            old,oc,oq=fixture(a,b,precision,tile)
                            new,nc,nq=fixture(a,b,precision,tile)
                            for ws in (old,new):
                                ws.diagonal='register';ws.substitution='ordered_warp'
                            new.panel='register';new.update_threads=128
                            old.enqueue(oc,oq,oq);new.enqueue(nc,nq,nq)
                            for name in ('work','x','status','max_relative_residual'):
                                exact(getattr(old,name).numpy(),getattr(new,name).numpy())
                            for name in ('search','search_dot','newton_decrement'):
                                exact(getattr(oc,name).numpy(),getattr(nc,name).numpy())
                            x=-nc.search.numpy().astype(np.float64)
                            residual=np.linalg.norm((a.astype(np.float64)@x[...,None])[...,0]-b,axis=1)/np.linalg.norm(b.astype(np.float64),axis=1)
                            np.testing.assert_array_equal(new.status.numpy(),0)
                            self.assertLess(float(residual.max()),1e-4)
                            np.testing.assert_allclose(new.max_relative_residual.numpy(),residual,atol=1e-8,rtol=1e-4)

    def test_captured_masks_and_factor_reuse(self):
        rng=np.random.default_rng(714)
        with wp.ScopedDevice('cuda:0'):
            n=65
            r=rng.normal(size=(3,n,n))
            a=(r@r.transpose(0,2,1)+np.eye(n)*n).astype(np.float32)
            b=rng.normal(size=(3,n)).astype(np.float32)
            old,oc,oq=fixture(a,b,'float32',32)
            with patch('test_parallel_newton._CONDITIONAL_FACTOR',True):
                new,nc,nq=fixture(a,b,'float32',32)
            for ws in (old,new):
                ws.diagonal='register';ws.substitution='ordered_warp'
            new.panel='register';new.update_threads=128
            records=((old,oc,oq),(new,nc,nq));graphs=[]
            for ws,ctx,count in records:
                count.zero_()  # First active solve still has to build its factor.
                ws.enqueue(ctx,count,count,True,False)
                wp.synchronize()
                with wp.ScopedCapture() as capture:ws.enqueue(ctx,count,count,True,True)
                graphs.append(capture.graph)
                ws.initialized.assign(np.array([0,0,1],np.int32))
            for step in range(8):
                counts=np.array([step%2,1,0] if step%3 else [0,0,0],np.int32)
                done=np.array([False,step%3==0,True])
                h=oc.h.numpy()
                if counts[0]:h[0,range(n),range(n)]*=1.1
                if counts[1]:h[1,range(n),range(n)]*=1.1
                rhs=rng.normal(size=oc.grad.shape).astype(np.float32)
                for ws,ctx,count in records:
                    ctx.h.assign(h);ctx.grad.assign(rhs);ctx.done.assign(done);count.assign(counts)
                for graph in graphs:wp.capture_launch(graph)
                for name in ('work','x','status','max_relative_residual'):
                    exact(getattr(old,name).numpy(),getattr(new,name).numpy())
                for name in ('search','search_dot','newton_decrement'):
                    exact(getattr(oc,name).numpy(),getattr(nc,name).numpy())
                np.testing.assert_array_equal(new.status.numpy(),0)

    def test_guards_before_workspace_allocation(self):
        with patch('deform3d_warp.mujoco_newton.wp.empty') as allocate:
            for kwargs in ({'panel':'unknown'},{'update_threads':True},{'update_threads':64.0},{'update_threads':256},{'conditional_factor':1}):
                with self.assertRaises(ValueError):_Workspace(16,1,'cuda:0',**kwargs)
            with self.assertRaisesRegex(ValueError,'CUDA'):_Workspace(16,1,'cpu',panel='register')
            with patch('deform3d_warp._newton_substitution._TILE_SOURCES',{'tile_solve.h':'0'*64}):
                with self.assertRaisesRegex(RuntimeError,'reviewed Warp'):_Workspace(16,1,'cuda:0',panel='register')
            allocate.assert_not_called()

    def test_captured_reuse_still_solves_changing_rhs(self):
        rng=np.random.default_rng(717)
        with wp.ScopedDevice('cuda:0'):
            n=33
            r=rng.normal(size=(2,n,n))
            a=(r@r.transpose(0,2,1)+np.eye(n)*n).astype(np.float32)
            b=rng.normal(size=(2,n)).astype(np.float32)
            old,oc,oq=fixture(a,b,'float32',32)
            with patch('test_parallel_newton._CONDITIONAL_FACTOR',True):
                new,nc,nq=fixture(a,b,'float32',32)
            for ws in (old,new):
                ws.diagonal='register';ws.substitution='ordered_warp'
            new.panel='register';new.update_threads=128
            records=((old,oc,oq),(new,nc,nq));graphs=[]
            for ws,ctx,count in records:
                ws.enqueue(ctx,count,count)
                count.zero_()
                wp.synchronize()
                with wp.ScopedCapture() as capture:ws.enqueue(ctx,count,count,True,False)
                graphs.append(capture.graph)
                # Test first-use factorization inside the conditional graph,
                # followed by reuse while all worlds remain active.
                ws.initialized.zero_();ws.work.fill_(float('nan'))
            factor=None
            for step in range(5):
                rhs=rng.normal(size=oc.grad.shape).astype(np.float32)
                oc.grad.assign(rhs);nc.grad.assign(rhs)
                for graph in graphs:wp.capture_launch(graph)
                for name in ('search','search_dot','newton_decrement'):
                    exact(getattr(oc,name).numpy(),getattr(nc,name).numpy())
                for name in ('work','x','status','max_relative_residual'):
                    exact(getattr(old,name).numpy(),getattr(new,name).numpy())
                reference=np.linalg.solve(a.astype(np.float64),rhs[:,:n].astype(np.float64)[...,None])[...,0]
                np.testing.assert_allclose(-nc.search.numpy(),reference,rtol=1e-4,atol=1e-6)
                np.testing.assert_array_equal(new.status.numpy(),0)
                self.assertEqual(int(new.factor_condition.numpy()[0]),int(step==0))
                if factor is None:factor=new.work.numpy()
                else:exact(new.work.numpy(),factor)
            before=nc.search.numpy()
            with patch.object(new,'factor_condition',new.active):
                with self.assertRaisesRegex(ValueError,'overlap'):new.enqueue(nc,nq,nq)
            exact(nc.search.numpy(),before)

    def test_other_stream_capture_does_not_trigger_host_readback(self):
        with wp.ScopedDevice('cuda:0'):
            with patch('test_parallel_newton._CONDITIONAL_FACTOR',True):
                ws,ctx,count=fixture(np.eye(16,dtype=np.float32)[None],np.ones((1,16),np.float32))
            current=wp.get_stream('cuda:0')
            other=wp.Stream('cuda:0')
            ws.enqueue(ctx,count,count);wp.synchronize()
            # Device.is_capturing also sees captures on non-current streams.
            # Calling capture_if outside its own capture would read back and
            # synchronize. The policy must inspect the actual current stream.
            with wp.ScopedCapture(stream=other):
                with wp.ScopedStream(current,sync_enter=False):
                    self.assertFalse(current.is_capturing)
                    with patch('deform3d_warp.mujoco_newton.wp.capture_if') as branch:
                        ws.enqueue(ctx,count,count)
                        branch.assert_not_called()
            np.testing.assert_allclose(ctx.search.numpy(),-1.)


if __name__=='__main__':unittest.main(verbosity=2)
