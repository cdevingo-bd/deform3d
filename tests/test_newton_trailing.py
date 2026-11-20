"""Ordered Schur arithmetic, output ownership and captured factor integration."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import warp as wp

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'tests')]
from deform3d_warp._newton_trailing import ordered_trailing
from deform3d_warp.mujoco_newton import _Workspace,_kernels
from test_parallel_newton import fixture


def exact(x,y):
    np.testing.assert_array_equal(x.view(np.uint8),y.view(np.uint8))


class RegisterTrailingTests(unittest.TestCase):
    def test_update_order_masks_and_addressing(self):
        rng=np.random.default_rng(4711)
        with wp.ScopedDevice('cuda:0'):
            for precision in ('float32','float64'):
                real=wp.float32 if precision=='float32' else wp.float64
                dtype=np.float32 if precision=='float32' else np.float64
                for tile in (16,32):
                    for remaining in (1,3,14):
                        for threads in (64,128):
                            with self.subTest(precision=precision,tile=tile,remaining=remaining,threads=threads):
                                k=tile;size=(remaining+2)*tile
                                a=(rng.normal(size=(3,size,size))*np.exp(rng.uniform(-9,9,(3,size,size)))).astype(dtype)
                                # Nonzero offset and an inactive middle world expose
                                # accidental writes outside the owned lower tiles.
                                old,new=wp.array(a,dtype=real),wp.array(a,dtype=real)
                                active=wp.array(np.array([1,0,1],np.int32))
                                wp.launch_tiled(_kernels(size,precision,tile)[2],dim=(3,remaining,remaining),
                                                inputs=[k,active,old],block_dim=threads)
                                wp.launch_tiled(ordered_trailing(precision,tile,threads),dim=(3,remaining,remaining),
                                                inputs=[k,active,new],block_dim=threads)
                                x,y=old.numpy(),new.numpy()
                                exact(x,y);exact(y[1],a[1]);exact(y[:,:k+tile],a[:,:k+tile])
                                self.assertTrue(np.isfinite(y).all())

    def test_complete_factors_and_independent_solutions(self):
        rng=np.random.default_rng(526)
        with wp.ScopedDevice('cuda:0'):
            for precision in ('float32','float64'):
                for tile in (16,32):
                    for n in (1,17,65,117,456,512):
                        for threads in (64,128):
                            with self.subTest(precision=precision,tile=tile,n=n,threads=threads):
                                r=rng.normal(size=(2,n,n))
                                a=(r@r.transpose(0,2,1)+np.eye(n)*n*.1).astype(np.float32)
                                b=rng.normal(size=(2,n)).astype(np.float32)
                                old,oc,oq=fixture(a,b,precision,tile)
                                with patch('test_parallel_newton._TRAILING','register'):
                                    new,nc,nq=fixture(a,b,precision,tile)
                                for ws in (old,new):
                                    ws.diagonal='register';ws.substitution='ordered_warp';ws.panel='register';ws.update_threads=threads
                                old.enqueue(oc,oq,oq);new.enqueue(nc,nq,nq)
                                for name in ('work','x','status','max_relative_residual'):
                                    exact(getattr(old,name).numpy(),getattr(new,name).numpy())
                                for name in ('search','search_dot','newton_decrement'):
                                    exact(getattr(oc,name).numpy(),getattr(nc,name).numpy())
                                x=-nc.search.numpy().astype(np.float64)
                                reference=np.linalg.solve(a.astype(np.float64),b.astype(np.float64)[...,None])[...,0]
                                np.testing.assert_allclose(x,reference,atol=1e-6,rtol=1e-4)
                                residual=np.linalg.norm((a.astype(np.float64)@x[...,None])[...,0]-b,axis=1)/np.linalg.norm(b.astype(np.float64),axis=1)
                                self.assertLess(float(residual.max()),1e-4)
                                np.testing.assert_array_equal(new.status.numpy(),0)
                                np.testing.assert_allclose(new.max_relative_residual.numpy(),residual,atol=1e-8,rtol=1e-4)

    def test_captured_changes_reuse_and_inactive_worlds(self):
        rng=np.random.default_rng(42)
        with wp.ScopedDevice('cuda:0'):
            n=65;r=rng.normal(size=(3,n,n))
            a=(r@r.transpose(0,2,1)+np.eye(n)*n).astype(np.float32)
            b=rng.normal(size=(3,n)).astype(np.float32)
            with patch('test_parallel_newton._CONDITIONAL_FACTOR',True):
                old,oc,oq=fixture(a,b,'float32',32)
                with patch('test_parallel_newton._TRAILING','register'):
                    new,nc,nq=fixture(a,b,'float32',32)
            records=((old,oc,oq),(new,nc,nq));graphs=[]
            for ws,ctx,count in records:
                ws.diagonal='register';ws.substitution='ordered_warp';ws.panel='register';ws.update_threads=128
                ws.enqueue(ctx,count,count);wp.synchronize()
                with wp.ScopedCapture() as capture:ws.enqueue(ctx,count,count,True,False)
                graphs.append(capture.graph)
                ws.initialized.zero_();ws.work.fill_(float('nan'))
            for step in range(8):
                counts=np.array([step%2,step%3==2,0],np.int32)
                done=np.array([False,step%3==1,step>1])
                h=oc.h.numpy();h[0,range(n),range(n)]+=counts[0];h[1,range(n),range(n)]+=counts[1]
                rhs=rng.normal(size=oc.grad.shape).astype(np.float32)
                for ws,ctx,count in records:
                    ctx.h.assign(h);ctx.grad.assign(rhs);ctx.done.assign(done);count.assign(counts)
                for graph in graphs:wp.capture_launch(graph)
                for name in ('work','x','status','max_relative_residual'):
                    exact(getattr(old,name).numpy(),getattr(new,name).numpy())
                for name in ('search','search_dot','newton_decrement'):
                    exact(getattr(oc,name).numpy(),getattr(nc,name).numpy())
                np.testing.assert_array_equal(new.status.numpy(),0)

    def test_guards_before_allocation(self):
        with patch('deform3d_warp.mujoco_newton.wp.empty') as allocate:
            with self.assertRaises(ValueError):_Workspace(16,1,'cuda:0',trailing='unknown')
            with self.assertRaisesRegex(ValueError,'CUDA'):_Workspace(16,1,'cpu',trailing='register')
            with patch('deform3d_warp._newton_substitution._TILE_SOURCES',{'tile_matmul.h':'0'*64}):
                with self.assertRaisesRegex(RuntimeError,'reviewed Warp'):_Workspace(16,1,'cuda:0',trailing='register')
            allocate.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
