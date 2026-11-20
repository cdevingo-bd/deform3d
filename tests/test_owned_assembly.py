"""Independent operator, active-set, graph and MJWarp integration checks."""
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import solver, types
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'tools')]
from deform3d_warp.mujoco_assembly import OwnedContactAssembly


class OwnedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cpu = mujoco.MjModel.from_xml_string('''<mujoco>
          <option jacobian="sparse" solver="Newton" cone="pyramidal"/>
          <worldbody><geom type="plane" size="1 1 .01"/>
          <body pos="0 0 .1"><freejoint/><geom size=".1"/></body>
          <body pos=".3 0 .1"><freejoint/><geom size=".1"/></body>
          </worldbody></mujoco>''')
        with wp.ScopedDevice('cuda:0'):
            cls.model = mjw.put_model(cls.cpu)

    def fixture(self, seed=10):
        m = self.model
        d = mjw.make_data(self.cpu, nworld=2, njmax=32, njmax_nnz=384)
        p = OwnedContactAssembly(m,d)
        rng = np.random.default_rng(seed)
        # Both packed row addresses and column indices have arbitrary order.
        J = rng.normal(size=(2,32,m.nv)).astype(np.float32)
        J[rng.random(J.shape) < .7] = 0
        J[:, 20:] = 0
        J[:, 0] = 0  # legal empty row
        J[:, :, -1] = 0  # an unconstrained DOF
        J[1] = J[0]
        nnz = np.count_nonzero(J,axis=2).astype(np.int32)
        adr = np.zeros((2,32),np.int32)
        packed, cols = d.efc.J.numpy(), d.efc.J_colind.numpy()
        for w in range(2):
            start = 0
            for r in rng.permutation(32):
                inds = np.flatnonzero(J[w,r])
                rng.shuffle(inds)
                adr[w,r] = start
                cols[w,0,start:start+len(inds)] = inds
                packed[w,0,start:start+len(inds)] = J[w,r,inds]
                start += len(inds)
        d.efc.J.assign(packed); d.efc.J_colind.assign(cols)
        d.efc.J_rownnz.assign(nnz); d.efc.J_rowadr.assign(adr)
        d.efc.type.fill_(int(types.ConstraintType.CONTACT_PYRAMIDAL))
        D = rng.uniform(.1,100,(2,32)).astype(np.float32); D[1] = D[0]
        state = rng.integers(0,2,(2,32),dtype=np.int32); state[1] = state[0]
        forces = rng.normal(size=(2,32)).astype(np.float32); forces[1] = forces[0]
        d.efc.D.assign(D); d.efc.state.assign(state); d.efc.force.assign(forces)
        d.nefc.fill_(20); d.ne.zero_(); d.nf.zero_(); d.nl.zero_()
        ctx = SimpleNamespace(done=wp.zeros(2,dtype=bool), h=wp.zeros((2,m.nv_pad,m.nv_pad),dtype=float))
        base = rng.normal(size=ctx.h.shape).astype(np.float32); base[1] = base[0]
        ctx.h.assign(base)
        return d,p,ctx,J,D,state,forces,base

    def check_h(self, actual, base, J, weight):
        reference = base.astype(np.float64).copy()
        n = self.model.nv
        for w in range(2):
            reference[w,:n,:n] += J[w].astype(np.float64).T @ (weight[w,:,None]*J[w])
        upper = np.triu_indices(n)
        np.testing.assert_allclose(actual[:,:n,:n][:,upper[0],upper[1]],
            reference[:,:n,:n][:,upper[0],upper[1]],rtol=8e-8,atol=2e-6)
        lower = np.tril_indices(n,-1)
        np.testing.assert_array_equal(actual[:,:n,:n][:,lower[0],lower[1]],base[:,:n,:n][:,lower[0],lower[1]])
        return reference

    def test_operators_energy_force_and_world_identity(self):
        d,p,ctx,J,D,s,f,base = self.fixture()
        p.prepare(d); p.full(d,ctx); p.force(d,ctx,d.nefc)
        actual = ctx.h.numpy()
        self.check_h(actual,base,J,D.astype(np.float64)*s)
        np.testing.assert_array_equal(actual[0],actual[1])
        reference_force = np.einsum('wrc,wr->wc',J.astype(np.float64),f)
        np.testing.assert_allclose(d.qfrc_constraint.numpy(),reference_force,rtol=8e-8,atol=1e-7)
        # Independent energy/work identities, beyond matrix entry comparisons.
        v = np.linspace(-1,1,self.model.nv)
        for w in range(2):
            upper = np.triu(actual[w,:self.model.nv,:self.model.nv]-base[w,:self.model.nv,:self.model.nv])
            H = upper + np.triu(upper,1).T
            self.assertAlmostEqual(float(v@H@v),float(np.sum(D[w]*s[w]*(J[w]@v)**2)),delta=2e-4)
            self.assertAlmostEqual(float(v@d.qfrc_constraint.numpy()[w]),float(f[w]@(J[w]@v)),delta=2e-6)
        np.testing.assert_array_equal(p.status.numpy(),0)

    def test_incremental_activation_deactivation_and_no_change(self):
        d,p,ctx,J,D,s,f,base = self.fixture()
        p.prepare(d); p.full(d,ctx)
        old = ctx.h.numpy()
        p.incremental(d,ctx)
        np.testing.assert_array_equal(ctx.h.numpy(),old)
        changed = 1-s
        d.efc.state.assign(changed)
        p.incremental(d,ctx)
        # Float32 storage of prior H entails one rounding per incremental update.
        expected = self.check_h(old,base,J,D.astype(np.float64)*s)
        target = base.astype(np.float64)
        for w in range(2): target[w] += J[w].astype(np.float64).T @ ((D[w]*changed[w])[:,None]*J[w])
        np.testing.assert_allclose(np.triu(ctx.h.numpy()),np.triu(target),rtol=3e-6,atol=7e-5)
        d.efc.state.assign(s); p.incremental(d,ctx)
        np.testing.assert_allclose(np.triu(ctx.h.numpy()),np.triu(expected),rtol=3e-6,atol=7e-5)
        np.testing.assert_array_equal(p.status.numpy(),0)

    def test_graph_replay_rebuilds_changed_geometry_and_packed_layout(self):
        d,p,ctx,J,D,s,f,base = self.fixture()
        def enqueue():
            p.prepare(d); p.full(d,ctx); p.force(d,ctx,d.nefc)
        enqueue(); wp.synchronize()
        with wp.ScopedCapture() as capture: enqueue()
        other,_,_,J,D,s,f,base = self.fixture(seed=54)
        for name in ('J','J_colind','J_rownnz','J_rowadr','D','state','force'):
            wp.copy(getattr(d.efc,name),getattr(other.efc,name))
        ctx.h.assign(base)
        wp.capture_launch(capture.graph)
        self.check_h(ctx.h.numpy(),base,J,D.astype(np.float64)*s)
        np.testing.assert_allclose(d.qfrc_constraint.numpy(),np.einsum('wrc,wr->wc',J.astype(np.float64),f),rtol=8e-8,atol=1e-7)
        # A later zero-contact frame must not reuse old transpose entries.
        d.nefc.zero_(); ctx.h.assign(base); wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(ctx.h.numpy(),base)
        np.testing.assert_array_equal(p.counts.numpy(),0)
        np.testing.assert_array_equal(p.status.numpy(),0)

    def test_done_and_unchanged_worlds_preserve_outputs(self):
        d,p,ctx,J,D,s,f,base = self.fixture()
        p.prepare(d); ctx.done.fill_(True); d.qfrc_constraint.fill_(123)
        p.full(d,ctx); p.force(d,ctx,d.nefc)
        np.testing.assert_array_equal(ctx.h.numpy(),base)
        np.testing.assert_array_equal(d.qfrc_constraint.numpy(),123)
        ctx.done.zero_(); changed=wp.zeros(2,dtype=int); p.force(d,ctx,changed)
        np.testing.assert_array_equal(d.qfrc_constraint.numpy(),123)

    def test_dof_boundaries_and_wide_coefficient_range(self):
        for n in (1,17,512):
            with self.subTest(nv=n):
                bodies=''.join('<body><joint type="slide"/><geom size=".01" contype="0" conaffinity="0"/></body>' for _ in range(n))
                cm=mujoco.MjModel.from_xml_string('<mujoco><option jacobian="sparse" solver="Newton"/>'
                    '<worldbody>'+bodies+'</worldbody></mujoco>')
                model=mjw.put_model(cm)
                d=mjw.make_data(cm,nworld=2,njmax=32,njmax_nnz=32*n)
                p=OwnedContactAssembly(model,d)
                J=np.zeros((32,n),np.float32)
                packed=d.efc.J.numpy(); cols=d.efc.J_colind.numpy()
                counts=np.zeros((2,32),np.int32); addresses=counts.copy()
                k=0
                for r in range(32):
                    inds=sorted({r%n,(r*7+3)%n,n-1},reverse=True)
                    values=np.array([(1 if (r+i)%2 else -1)*(i+1)/3 for i in range(len(inds))],np.float32)
                    J[r,inds]=values
                    addresses[:,r]=k; counts[:,r]=len(inds)
                    packed[:,0,k:k+len(inds)]=values; cols[:,0,k:k+len(inds)]=inds
                    k+=len(inds)
                d.efc.J.assign(packed); d.efc.J_colind.assign(cols)
                d.efc.J_rownnz.assign(counts); d.efc.J_rowadr.assign(addresses)
                d.nefc.fill_(32); d.ne.zero_(); d.nf.zero_(); d.nl.zero_()
                d.efc.type.fill_(5); d.efc.state.fill_(1)
                D=np.logspace(-8,8,32,dtype=np.float32)
                d.efc.D.assign(np.tile(D,(2,1))); d.efc.force.fill_(.5)
                ctx=SimpleNamespace(done=wp.zeros(2,dtype=bool),h=wp.full((2,n+3,n+3),9.25,dtype=float))
                p.prepare(d); p.full(d,ctx); p.force(d,ctx,d.nefc)
                result=ctx.h.numpy()
                reference=J.astype(np.float64).T @ (D[:,None].astype(np.float64)*J)+9.25
                np.testing.assert_allclose(np.triu(result[:,:n,:n]),np.tile(np.triu(reference),(2,1,1)),rtol=8e-8,atol=1e-6)
                np.testing.assert_array_equal(result[:,n:],9.25)
                np.testing.assert_array_equal(result[:,:,n:],9.25)
                np.testing.assert_allclose(d.qfrc_constraint.numpy(),np.tile(J.astype(np.float64).sum(axis=0)*.5,(2,1)),rtol=8e-8,atol=1e-7)
                np.testing.assert_array_equal(d.efc.J.numpy(),packed)
                np.testing.assert_array_equal(p.status.numpy(),0)

    def test_malformed_inputs_are_sticky_and_bounded(self):
        for kind in ('overflow','negative','rowadr','column','duplicate','nan','state','D','type'):
            with self.subTest(kind=kind):
                d,p,ctx,J,D,s,f,base = self.fixture()
                if kind == 'overflow': d.nefc.fill_(33)
                elif kind == 'negative': d.nefc.fill_(-1)
                elif kind == 'rowadr': d.efc.J_rowadr.fill_(d.efc.J.shape[2])
                elif kind == 'column': d.efc.J_colind.fill_(self.model.nv)
                elif kind == 'duplicate': d.efc.J_colind.fill_(0)
                elif kind == 'nan': d.efc.J.fill_(float('nan'))
                elif kind == 'state': d.efc.state.fill_(4)
                elif kind == 'D': d.efc.D.fill_(-1)
                elif kind == 'type': d.efc.type.fill_(0)
                p.prepare(d); p.full(d,ctx)
                self.assertTrue(np.all(p.status.numpy() != 0))
                d.nefc.zero_(); p.prepare(d)
                self.assertTrue(np.all(p.status.numpy() != 0))

    def test_policy_guards_restoration_and_model_scope(self):
        d,p,ctx,*_ = self.fixture()
        original = {n:getattr(solver,n) for n in ('_solve','_update_constraint','_update_gradient','_update_gradient_incremental')}
        with self.assertRaisesRegex(ValueError,'sentinel'):
            with p.capture_policy():
                with self.assertRaises(RuntimeError):
                    with p.capture_policy(): pass
                with patch.object(d,'nworld',1), self.assertRaises(ValueError): solver._solve(self.model,d,ctx)
                raise ValueError('sentinel')
        for n,fn in original.items(): self.assertIs(getattr(solver,n),fn)
        with patch('deform3d_warp.mujoco_assembly.inspect.getsource',return_value='unreviewed'):
            with self.assertRaises(RuntimeError):
                with p.capture_policy(): pass
        with patch('deform3d_warp.mujoco_assembly.version',return_value='changed'):
            with self.assertRaises(RuntimeError): OwnedContactAssembly(self.model,d)
        for name,value in [('cone',types.ConeType.ELLIPTIC),('solver',types.SolverType.CG),('noslip_iterations',1)]:
            with patch.object(self.model.opt,name,value),self.assertRaises(ValueError): OwnedContactAssembly(self.model,d)
        with patch.object(self.model,'nv',513),self.assertRaises(ValueError): OwnedContactAssembly(self.model,d)

    def test_shape_guards_and_calls_outside_solve(self):
        d,p,ctx,*_ = self.fixture()
        with patch.object(ctx,'h',wp.zeros((2,1,1),dtype=float)),self.assertRaises(ValueError): p.full(d,ctx)
        with self.assertRaises(ValueError): p.force(d,ctx,wp.zeros(1,dtype=int))
        with patch.object(d.efc,'D',wp.zeros((2,1),dtype=float)),self.assertRaises(ValueError): p.prepare(d)
        # An inverse/context operation must not use a previous solve's transpose.
        # Spy on configuration so the original branch can be checked without
        # running an unsupported inverse solver on synthetic constraint data.
        import deform3d_warp.mujoco_assembly as assembly
        configured = assembly._configured
        def configure(original,name):
            fn = configured(original,name)
            if name == '_update_constraint':
                def fail(*args,**kwargs): raise AssertionError('owned reduction outside solve')
                return fail
            return fn
        empty = mjw.make_data(self.cpu,nworld=2,njmax=32,njmax_nnz=384)
        context = solver._create_solver_context(self.model,empty,inplace_cholesky=False)
        context.done.zero_(); context.Jaref.zero_()
        with patch.object(assembly,'_configured',configure),p.capture_policy():
            solver._update_constraint(self.model,empty,context)
        np.testing.assert_array_equal(p.status.numpy(),0)

    def test_actual_contact_operator_against_upstream(self):
        from deform3d_warp.mujoco_contact import flex_contact_policy
        from deform3d_warp.mujoco_metric import CooperativeMetricSolver
        from deform3d_warp.mujoco_ordering import CanonicalContactConstraints
        from deform3d_warp.mujoco_determinism import SparseDeterminismBudget
        from record_bag import drive
        cm = mujoco.MjModel.from_xml_path(str(ROOT/'examples/mujoco_bag_strong_ref.xml'))
        cm.opt.ccd_tolerance = 1e-9
        cpu = mujoco.MjData(cm)
        for _ in range(1400): drive(cm,cpu); mujoco.mj_step(cm,cpu)
        model = mjw.put_model(cm); model.opt.ccd_tolerance.fill_(1e-8)
        model.opt.deterministic = mjw.DeterminismType.ALL
        metric = CooperativeMetricSolver(model,2)
        outputs = []
        class Captured(Exception): pass
        def take(m,d,ctx,*args,**kwargs):
            outputs.append((np.triu(ctx.h.numpy()),ctx.grad.numpy(),d.qfrc_constraint.numpy()))
            self.assertGreater(int(d.nefc.numpy().min()),0)
            raise Captured()
        for owned in (False,True):
            data = mjw.put_data(cm,cpu,nworld=2)
            ordering = CanonicalContactConstraints(model,data)
            p = OwnedContactAssembly(model,data) if owned else SparseDeterminismBudget(model)
            with ExitStack() as stack:
                stack.enter_context(flex_contact_policy(model))
                stack.enter_context(metric.capture_policy())
                stack.enter_context(ordering.capture_policy())
                stack.enter_context(p.capture_policy())
                stack.enter_context(patch.object(solver,'_cholesky_factorize_solve',take))
                with self.assertRaises(Captured): mjw.forward(model,data)
            if owned: np.testing.assert_array_equal(p.status.numpy(),0)
            np.testing.assert_array_equal(ordering.status.numpy(),0)
            np.testing.assert_array_equal(metric.status.numpy(),0)
        for a,b in zip(*outputs):
            np.testing.assert_allclose(a,b,rtol=4e-6,atol=2e-5)
            np.testing.assert_array_equal(b[0],b[1])


if __name__ == '__main__':
    wp.init()
    with wp.ScopedDevice('cuda:0'): unittest.main()
