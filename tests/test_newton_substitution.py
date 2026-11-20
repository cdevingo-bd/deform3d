"""Bitwise operator comparisons, including order-sensitive triangular systems."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'tests')]
from deform3d_warp._newton_substitution import ordered_substitution, _check_tile_sources
from deform3d_warp.mujoco_newton import _kernels
from test_parallel_newton import fixture


def exact(a, b):
    np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))


class OrderedSubstitutionTests(unittest.TestCase):
    def test_triangular_order_and_padding(self):
        rng = np.random.default_rng(9183)
        with wp.ScopedDevice('cuda:0'):
            for precision in ('float32','float64'):
                real = wp.float32 if precision=='float32' else wp.float64
                dtype = np.float32 if precision=='float32' else np.float64
                for tile in (16,32):
                    for nv in (1,3,15,16,17,31,32,33,65,117,456,512):
                        with self.subTest(precision=precision,tile=tile,nv=nv):
                            size = (nv+tile-1)//tile*tile
                            a = np.tile(np.eye(size,dtype=dtype),(3,1,1))
                            for w in range(3):
                                lower = np.tril(rng.normal(size=(nv,nv))*.07,-1)
                                lower += np.diag(np.exp(rng.uniform(-1,1,nv)))
                                a[w,:nv,:nv] = lower
                            # Upper entries must never influence substitution.
                            rows,cols=np.triu_indices(size,1)
                            a[:,rows,cols]=np.nan
                            b=np.zeros((3,size),dtype)
                            b[:,:nv]=rng.normal(size=(3,nv))*np.exp(rng.uniform(-9,9,(3,nv)))
                            active=wp.array(np.array([1,0,1],np.int32))
                            factor=wp.array(a,dtype=real)
                            old,new=wp.array(b,dtype=real),wp.array(b,dtype=real)
                            wp.launch_tiled(_kernels(nv,precision,tile)[3],dim=3,
                                inputs=[active,factor,old.reshape((3,size,1))],block_dim=64)
                            wp.launch_tiled(ordered_substitution(nv,precision,tile),dim=3,
                                inputs=[active,factor,new],block_dim=32)
                            x,y=old.numpy(),new.numpy()
                            self.assertTrue(np.isfinite(x).all())
                            exact(x,y)
                            exact(y[1],b[1])
                            exact(factor.numpy(),a)

    def test_full_solve_graph_reuse_and_changes(self):
        rng=np.random.default_rng(823)
        with wp.ScopedDevice('cuda:0'):
            for precision in ('float32','float64'):
                for tile in (16,32):
                    n=65
                    r=rng.normal(size=(3,n,n))
                    a=(r@r.transpose(0,2,1)+np.eye(n)*n).astype(np.float32)
                    b=rng.normal(size=(3,n)).astype(np.float32)
                    old,oc,oq=fixture(a,b,precision,tile)
                    new,nc,nq=fixture(a,b,precision,tile)
                    new.substitution='ordered_warp'
                    graphs=[]
                    for ws,ctx,count in ((old,oc,oq),(new,nc,nq)):
                        ws.enqueue(ctx,count,count,True,True)
                        wp.synchronize()
                        with wp.ScopedCapture() as capture:
                            ws.enqueue(ctx,count,count,True,True)
                        graphs.append(capture.graph)
                    for iteration in range(6):
                        grad=rng.normal(size=oc.grad.shape).astype(np.float32)
                        # The same captured graph must see changing RHS, factors,
                        # active worlds, and cache reuse on each replay.
                        counts=np.array([1,0,1] if iteration%2 else [0,1,1],np.int32)
                        done=np.array([False,iteration%3==0,False])
                        h=oc.h.numpy()
                        if iteration%2:h[:,range(n),range(n)]*=1.01
                        for ctx,count in ((oc,oq),(nc,nq)):
                            ctx.h.assign(h);ctx.grad.assign(grad);ctx.done.assign(done);count.assign(counts)
                        for graph in graphs:wp.capture_launch(graph)
                        for name in ('search','search_dot','newton_decrement'):
                            exact(getattr(oc,name).numpy(),getattr(nc,name).numpy())
                        for name in ('work','x','status','max_relative_residual'):
                            exact(getattr(old,name).numpy(),getattr(new,name).numpy())
                        np.testing.assert_array_equal(new.status.numpy(),0)

    def test_reviewed_sources(self):
        _check_tile_sources()
        with patch('deform3d_warp._newton_substitution._TILE_SOURCES',{'tile_solve.h':'0'*64}):
            with self.assertRaisesRegex(RuntimeError,'reviewed Warp'):
                _check_tile_sources()

    def test_option_and_device_guards(self):
        from deform3d_warp.mujoco_newton import _Workspace
        with self.assertRaisesRegex(ValueError,'CUDA'):
            _Workspace(16,1,'cpu',substitution='ordered_warp')
        with self.assertRaisesRegex(ValueError,'substitution'):
            _Workspace(16,1,'cuda:0',substitution='unordered')


if __name__=='__main__':
    unittest.main(verbosity=2)
