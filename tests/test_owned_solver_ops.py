"""Unique-CSR insertion, exact binned norms, masks and integrated hook checks."""
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import solver
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'tools')]
from deform3d_warp.mujoco_solver_ops import OwnedSolverReductions
from deform3d_warp.mujoco_assembly import OwnedContactAssembly
from deform3d_warp.mujoco_contact import flex_contact_policy
from deform3d_warp.mujoco_metric import CooperativeMetricSolver
from deform3d_warp.mujoco_material import OwnedFlexOperators
from deform3d_warp.mujoco_ordering import CanonicalContactConstraints
from record_bag import drive


def assert_bits(actual, expected):
    """Also distinguish signed zero; NumPy value equality alone does not."""
    actual, expected = np.ascontiguousarray(actual), np.ascontiguousarray(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError('different array layout')
    np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))


def fixture(nv=33, worlds=3, seed=71):
    rng = np.random.default_rng(seed)
    # Unique but deliberately unsorted columns and nonmonotone row addresses.
    counts = np.minimum(np.arange(nv)%5+1, nv).astype(np.int32)
    addresses = np.zeros(nv, np.int32); columns = []
    for row in rng.permutation(nv):
        addresses[row] = len(columns)
        columns.extend(rng.choice(nv, size=counts[row], replace=False))
    m = SimpleNamespace(nv=nv, nefmK=len(columns), body_mass=wp.ones(1),
        efm_K_rownnz=wp.array(counts), efm_K_rowadr=wp.array(addresses),
        efm_K_colind=wp.array(np.asarray(columns, np.int32)))
    matrix = rng.normal(size=(worlds,nv+3,nv+3)).astype(np.float32)
    d = SimpleNamespace(nworld=worlds, efm_K_val=wp.array(rng.normal(size=(worlds,len(columns))).astype(np.float32)),
        efc=SimpleNamespace(Ma=wp.array(rng.normal(size=(worlds,nv+3)).astype(np.float32))),
        qfrc_smooth=wp.array(rng.normal(size=(worlds,nv+1)).astype(np.float32)),
        qfrc_constraint=wp.array(rng.normal(size=(worlds,nv+2)).astype(np.float32)), nefc=wp.ones(worlds,dtype=int))
    ctx = SimpleNamespace(h=wp.array(matrix), done=wp.zeros(worlds,dtype=bool),
        grad=wp.full((worlds,nv+4),99.), grad_dot=wp.full(worlds,.75), state_changed_count=wp.ones(worlds,dtype=int))
    status = wp.zeros(worlds,dtype=int)
    return m,d,ctx,OwnedSolverReductions(m,d,status)


def upstream_metric(m,d,ctx):
    wp.launch(solver._add_efmK_metric_dense(m.nv,True), dim=(d.nworld,m.nv),
        inputs=[m.efm_K_rownnz,m.efm_K_rowadr,m.efm_K_colind,d.efm_K_val,ctx.done],outputs=[ctx.h])


def upstream_gradient(m,d,ctx,stable=False):
    wp.launch(solver._update_gradient_grad(stable,True),dim=(d.nworld,m.nv),
        inputs=[d.qfrc_smooth,d.qfrc_constraint,d.efc.Ma,ctx.state_changed_count if stable else d.nefc,
                ctx.done],outputs=[ctx.grad,ctx.grad_dot])


class SolverReductionTests(unittest.TestCase):
    def test_metric_exact_and_numpy_unique_csr_reference(self):
        m,d,c,p = fixture(); original=c.h.numpy(); upstream_metric(m,d,c); expected=c.h.numpy()
        c.h.assign(original); p.add_metric(d,c)
        assert_bits(c.h.numpy(),expected)
        reference=original.copy(); values=d.efm_K_val.numpy()
        for row,(adr,nnz) in enumerate(zip(m.efm_K_rowadr.numpy(),m.efm_K_rownnz.numpy())):
            for k in range(adr,adr+nnz):
                col=m.efm_K_colind.numpy()[k]
                if col>=row:reference[:,row,col]+=values[:,k]
        np.testing.assert_array_equal(c.h.numpy(),reference)
        np.testing.assert_array_equal(p.status.numpy(),0)

    def test_gradient_exact_across_warp_boundaries_and_exponents(self):
        rng=np.random.default_rng(12)
        for n in (1,17,32,33,127,456,512):
            with self.subTest(n=n):
                m,d,c,p=fixture(n)
                v=np.ldexp(rng.uniform(-1,1,(d.nworld,n+3)),rng.integers(-70,60,(d.nworld,n+3))).astype(np.float32)
                d.efc.Ma.assign(v); d.qfrc_smooth.zero_(); d.qfrc_constraint.zero_()
                original=c.grad.numpy(); base=c.grad_dot.numpy()
                upstream_gradient(m,d,c); grad,norm=c.grad.numpy(),c.grad_dot.numpy()
                c.grad.assign(original); c.grad_dot.assign(base); p.gradient(d,c)
                assert_bits(c.grad.numpy(),grad)
                assert_bits(c.grad_dot.numpy(),norm)
                reference=np.sum(v[:,:n].astype(np.float64)**2,axis=1)+base
                np.testing.assert_allclose(norm,reference,rtol=9e-8,atol=1e-30)
                np.testing.assert_array_equal(p.status.numpy(),0)

    def test_signed_zero_and_subnormal_native_rounding(self):
        m,d,c,p=fixture(33)
        tiny=np.nextafter(np.float32(0),np.float32(1))
        values=np.array([0.,-0.,tiny,-tiny,np.finfo(np.float32).tiny],np.float32)
        d.efm_K_val.assign(np.resize(values,d.efm_K_val.shape))
        d.efc.Ma.assign(np.resize(values,d.efc.Ma.shape))
        d.qfrc_smooth.zero_();d.qfrc_constraint.zero_()
        c.h.fill_(-0.);c.grad_dot.fill_(-0.)
        upstream_metric(m,d,c);upstream_gradient(m,d,c)
        expected=c.h.numpy(),c.grad.numpy(),c.grad_dot.numpy()
        c.h.fill_(-0.);c.grad_dot.fill_(-0.)
        p.add_metric(d,c);p.gradient(d,c)
        for actual,reference in zip((c.h.numpy(),c.grad.numpy(),c.grad_dot.numpy()),expected):
            assert_bits(actual,reference)

    def test_done_stable_and_scale_transitions(self):
        m,d,c,p=fixture(); c.done.assign(np.array([True,False,False]));c.state_changed_count.assign(np.array([1,0,1],np.int32))
        c.alpha=wp.full(d.nworld,.2);c.newton_decrement=wp.full(d.nworld,3.)
        c.grad_scale=wp.full(d.nworld,.7);c.search_unchanged=wp.zeros(d.nworld,dtype=bool)
        wp.launch(solver._update_gradient_zero_grad_dot(True),d.nworld,
            inputs=[c.state_changed_count,c.alpha,c.done],outputs=[c.grad_dot,c.newton_decrement,c.grad_scale,c.search_unchanged])
        original=c.grad.numpy();base=c.grad_dot.numpy();h=c.h.numpy()
        upstream_gradient(m,d,c,True);expected=c.grad.numpy(),c.grad_dot.numpy()
        c.grad.assign(original);c.grad_dot.assign(base);p.gradient(d,c,True)
        np.testing.assert_array_equal(c.grad.numpy(),expected[0]);np.testing.assert_array_equal(c.grad_dot.numpy(),expected[1])
        np.testing.assert_array_equal(c.grad.numpy()[:2],original[:2]);np.testing.assert_array_equal(c.grad_dot.numpy()[:2],base[:2])
        p.add_metric(d,c);np.testing.assert_array_equal(c.h.numpy()[0],h[0])
        c.done.zero_();c.state_changed_count.zero_();p.gradient(d,c,False)
        np.testing.assert_array_equal(c.grad.numpy()[:,:m.nv],d.efc.Ma.numpy()[:,:m.nv]-d.qfrc_smooth.numpy()[:,:m.nv]-d.qfrc_constraint.numpy()[:,:m.nv])

    def test_graph_replay_observes_new_values_and_zero_transitions(self):
        m,d,c,p=fixture()
        def run():
            c.grad_dot.zero_();c.h.zero_();p.add_metric(d,c);p.gradient(d,c,True)
        run();wp.synchronize()
        with wp.ScopedCapture() as capture:run()
        for value in (1.2,0.,-2.1):
            d.efm_K_val.fill_(value);d.efc.Ma.fill_(value);d.qfrc_smooth.zero_();d.qfrc_constraint.zero_()
            wp.capture_launch(capture.graph);actual=c.h.numpy(),c.grad.numpy(),c.grad_dot.numpy()
            c.h.zero_();c.grad_dot.zero_();upstream_metric(m,d,c);upstream_gradient(m,d,c,True)
            for got,ref in zip(actual,(c.h.numpy(),c.grad.numpy(),c.grad_dot.numpy())):np.testing.assert_array_equal(got,ref)
        np.testing.assert_array_equal(p.status.numpy(),0)

    def test_empty_metric_and_nonfinite_status(self):
        m,d,c,p=fixture(1)
        d.efm_K_val.fill_(np.inf);p.add_metric(d,c)
        np.testing.assert_array_equal(p.status.numpy(),2)
        p.status.zero_()
        m.nefmK=0;m.efm_K_colind=wp.empty(0,dtype=int);m.efm_K_rownnz.zero_();m.efm_K_rowadr.zero_()
        d.efm_K_val=wp.empty((d.nworld,0));p=OwnedSolverReductions(m,d,p.status)
        before=c.h.numpy();p.add_metric(d,c);np.testing.assert_array_equal(c.h.numpy(),before)
        d.efc.Ma.fill_(np.nan);p.gradient(d,c);np.testing.assert_array_equal(p.status.numpy(),2)
        c.grad_dot.zero_();d.efc.Ma.zero_();p.gradient(d,c);np.testing.assert_array_equal(p.status.numpy(),2)

    def test_topology_and_source_guards(self):
        m,d,c,p=fixture();cols=m.efm_K_colind.numpy()
        m.efm_K_colind.fill_(0)
        with self.assertRaisesRegex(ValueError,'duplicate'):OwnedSolverReductions(m,d,p.status)
        with self.assertRaisesRegex(ValueError,'topology changed'):p.validate_setup()
        m.efm_K_colind.assign(cols)
        adr=m.efm_K_rowadr.numpy();m.efm_K_rowadr.zero_()
        with self.assertRaisesRegex(ValueError,'overlapping'):OwnedSolverReductions(m,d,p.status)
        m.efm_K_rowadr.assign(adr)
        with patch.object(solver,'_update_gradient_grad',lambda:None):
            with self.assertRaisesRegex(RuntimeError,'unreviewed'):p.validate_setup()
        with patch.object(c,'grad',wp.zeros((d.nworld,m.nv-1))):
            with self.assertRaises(ValueError):p.gradient(d,c)
        with patch.object(c,'grad',d.efc.Ma):
            with self.assertRaisesRegex(ValueError,'alias'):p.gradient(d,c)
        with patch.object(m,'efm_K_rowadr',wp.zeros(m.nv,dtype=float)):
            with self.assertRaises(ValueError):OwnedSolverReductions(m,d,p.status)
        with self.assertRaises(ValueError):OwnedSolverReductions(m,d,wp.zeros(1,dtype=int))

    def test_actual_composed_initial_newton_system_is_identical(self):
        cm=mujoco.MjModel.from_xml_path(str(ROOT/'examples/mujoco_bag_strong_ref.xml'));cm.opt.ccd_tolerance=1e-9
        cd=mujoco.MjData(cm)
        for _ in range(1400):drive(cm,cd);mujoco.mj_step(cm,cd)
        model=mjw.put_model(cm);model.opt.deterministic=mjw.DeterminismType.ALL;model.opt.ccd_tolerance.fill_(1e-8)
        outputs=[]
        class Stop(Exception):pass
        for enabled in (False,True):
            d=mjw.put_data(cm,cd,nworld=2)
            metric=CooperativeMetricSolver(model,2);order=CanonicalContactConstraints(model,d)
            assembly=OwnedContactAssembly(model,d,solver_reductions=enabled);material=OwnedFlexOperators(model,d)
            def take(m,d,c,*args,**kwargs):
                # MJWarp intentionally leaves the lower triangle and padded
                # gradient tail uninitialized; neither is part of the operator.
                outputs.append((np.triu(c.h.numpy()[:,:m.nv,:m.nv]),c.grad.numpy()[:,:m.nv],
                                c.grad_dot.numpy(),assembly.status.numpy()))
                raise Stop()
            with ExitStack() as stack:
                for policy in (flex_contact_policy(model),metric.capture_policy(),order.capture_policy(),assembly.capture_policy(),material.capture_policy()):stack.enter_context(policy)
                stack.enter_context(patch.object(solver,'_cholesky_factorize_solve',take))
                with self.assertRaises(Stop):mjw.forward(model,d)
        for a,b in zip(*outputs):assert_bits(a,b)
        np.testing.assert_array_equal(outputs[1][-1],0)


if __name__=='__main__':
    with wp.ScopedDevice('cuda:0'):unittest.main(verbosity=2)
