"""Material operator references, work/energy identities, cache and graph guards."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import passive, derivative, types
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'),str(ROOT/'tools')]
from deform3d_warp.mujoco_material import OwnedFlexOperators, _reduce_components, _components
from record_bag import drive


def jacobian(cm, cd):
    """Independent CPU MuJoCo point Jacobians, including articulated ancestors."""
    J = np.zeros((3*cm.nflexvert,cm.nv))
    for v,b in enumerate(cm.flex_vertbodyid):
        jp,jr = np.zeros((3,cm.nv)),np.zeros((3,cm.nv))
        mujoco.mj_jac(cm,cd,jp,jr,cd.flexvert_xpos[v],int(b))
        J[3*v:3*v+3] = jp
    return J


def cartesian_hessian(cm, d, world=0):
    H = np.zeros((3*cm.nflexvert,3*cm.nflexvert))
    for v,a in enumerate(d.flexvert_hessian.numpy()[world]):
        H[3*v:3*v+3,3*v:3*v+3] = [[a[0],a[1],a[2]],[a[1],a[3],a[4]],[a[2],a[4],a[5]]]
    for e,f in enumerate(np.repeat(np.arange(cm.nflex),cm.flex_edgenum)):
        i,j = cm.flex_edge[e]+cm.flex_vertadr[f]
        b = d.flexedge_hessian.numpy()[world,e]
        H[3*i:3*i+3,3*j:3*j+3] += b
        H[3*j:3*j+3,3*i:3*i+3] += b.T
    return H


def bending_hessian(cm):
    H = np.zeros((3*cm.nflexvert,3*cm.nflexvert))
    for e,f in enumerate(np.repeat(np.arange(cm.nflex),cm.flex_edgenum)):
        if cm.flex_dim[f]!=2 or cm.flex_bendingadr[f]<0 or cm.flex_edgeflap[e,1]<0: continue
        ids = np.r_[cm.flex_edge[e],cm.flex_edgeflap[e]]+cm.flex_vertadr[f]
        start = cm.flex_bendingadr[f]+17*(e-cm.flex_edgeadr[f])
        Q = cm.flex_bending[start:start+16].reshape(4,4)
        for i in range(4):
            for j in range(4): H[3*ids[i]:3*ids[i]+3,3*ids[j]:3*ids[j]+3] += Q[i,j]*np.eye(3)
    return H


@wp.kernel(module='unique',module_options={'enable_backward':False,'deterministic':wp.DeterministicMode.RUN_TO_RUN})
def upstream_binned_vec(records: wp.array2d[wp.vec3], groups: wp.array[int], out: wp.array2d[wp.vec3]):
    w,i=wp.tid()
    wp.atomic_add(out,w,groups[i],records[w,i])


class MaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cm=mujoco.MjModel.from_xml_path(str(ROOT/'examples/mujoco_bag_strong_ref.xml'))
        cls.cm.opt.ccd_tolerance=1e-9
        cls.cd=mujoco.MjData(cls.cm)
        for _ in range(1400): drive(cls.cm,cls.cd); mujoco.mj_step(cls.cm,cls.cd)
        drive(cls.cm,cls.cd); mujoco.mj_forward(cls.cm,cls.cd)
        with wp.ScopedDevice('cuda:0'):
            cls.m=mjw.put_model(cls.cm); cls.m.opt.deterministic=mjw.DeterminismType.ALL
        cls.J=jacobian(cls.cm,cls.cd)

    def fixture(self):
        d=mjw.put_data(self.cm,self.cd,nworld=2)
        return d,OwnedFlexOperators(self.m,d)

    def assert_relative(self, x, y, limit=2e-6):
        x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
        self.assertLess(np.linalg.norm(x-y)/max(1e-20,np.linalg.norm(y)),limit)

    def test_actual_compressed_material_force_and_stiffness_against_upstream(self):
        a,p=self.fixture(); b=mjw.put_data(self.cm,self.cd,nworld=2)
        def run(d):
            d.flex_hessian_valid.zero_(); passive.flex_hessian(self.m,d)
            passive.passive(self.m,d); derivative.eff_shift(self.m,d)
        run(a)
        with p.capture_policy(): run(b)
        for field in ('flexedge_hessian','flexvert_hessian','qfrc_spring','qfrc_damper','qfrc_passive','efm_c'):
            with self.subTest(field=field):
                x,y=getattr(b,field).numpy(),getattr(a,field).numpy()
                np.testing.assert_array_equal(x,y)
                np.testing.assert_array_equal(x[0],x[1])
        np.testing.assert_array_equal(p.status.numpy(),0)

    def test_cpu_point_jacobian_adjoint_and_virtual_work(self):
        d,p=self.fixture(); rng=np.random.default_rng(41)
        force=rng.normal(size=(2,self.m.nflexvert,3)).astype(np.float32)
        base=rng.normal(size=(2,self.m.nv)).astype(np.float32)
        vec=wp.array(force,dtype=wp.vec3); res=wp.array(base)
        p.scatter(self.m,d,res,vec,scale=-.7)
        expected=base.astype(float)-.7*np.einsum('ij,wi->wj',self.J,force.reshape(2,-1))
        self.assert_relative(res.numpy(),expected,2e-7)
        velocity=rng.normal(size=(2,self.m.nv)).astype(np.float32)
        gathered=wp.zeros((2,self.m.nflexvert),dtype=wp.vec3)
        passive.flex_gather(self.m,d,gathered,wp.array(velocity))
        self.assert_relative(gathered.numpy().reshape(2,-1),velocity@self.J.T,2e-7)
        work=np.sum((res.numpy()-base)*velocity,axis=1)
        expected_work=-.7*np.sum(force*gathered.numpy(),axis=(1,2))
        np.testing.assert_allclose(work,expected_work,rtol=1e-6,atol=3e-6)

    def test_articulated_ancestor_projection_and_welded_vertices(self):
        cm=mujoco.MjModel.from_xml_string('''<mujoco><option integrator="discrete" jacobian="sparse" solver="CG"/>
          <worldbody><body name="parent" pos=".1 .2 .4"><freejoint/><geom size=".04" mass="1"/>
          <body name="hinged" pos=".1 0 .1"><joint axis="0 1 0"/><geom size=".03" mass=".2"/>
          <flexcomp name="patch" type="grid" count="3 3 1" spacing=".05 .05 .05" dim="2" mass=".1">
          <elasticity young="1000" poisson=".3" thickness=".001" elastic2d="both"/><pin id="0"/>
          </flexcomp></body></body></worldbody></mujoco>''')
        cd=mujoco.MjData(cm); cd.qpos[3:7]=[.9,.1,.2,.3]; cd.qpos[7]=.3
        mujoco.mj_normalizeQuat(cm,cd.qpos); mujoco.mj_forward(cm,cd)
        m=mjw.put_model(cm); d=mjw.put_data(cm,cd,nworld=2); p=OwnedFlexOperators(m,d)
        self.assertTrue(np.any(p.simple.numpy()!=2))
        force=np.random.default_rng(21).normal(size=(2,m.nflexvert,3)).astype(np.float32)
        result=wp.zeros((2,m.nv),dtype=float)
        p.scatter(m,d,result,wp.array(force,dtype=wp.vec3))
        expected=force.reshape(2,-1)@jacobian(cm,cd)
        self.assert_relative(result.numpy(),expected,4e-7)
        np.testing.assert_array_equal(p.status.numpy(),0)

    def test_stretch_matrix_symmetry_translation_and_rayleigh_scaling(self):
        d,p=self.fixture(); rng=np.random.default_rng(7)
        x=rng.normal(size=(2,self.m.nflexvert,3)).astype(np.float32)
        out=wp.zeros((2,self.m.nflexvert),dtype=wp.vec3)
        with p.capture_policy():
            d.flex_hessian_valid.zero_(); passive.flex_hessian(self.m,d)
            H=cartesian_hessian(self.cm,d)
            for s1,s2,use_time in ((1.,0.,False),(-1.,0.,True),(1.,1.,True),(.3,.7,False)):
                passive.flex_hessian_mul(self.m,d,out,wp.array(x,dtype=wp.vec3),s1,s2,use_time)
                scale=np.repeat(s1+s2*self.cm.flex_damping[np.repeat(np.arange(self.cm.nflex),self.cm.flex_vertnum)],3)
                if use_time:
                    scale=np.repeat(s1*(self.cm.opt.timestep if s1<0 else self.cm.opt.timestep**2)
                        +s2*self.cm.opt.timestep*self.cm.flex_damping[np.repeat(np.arange(self.cm.nflex),self.cm.flex_vertnum)],3)
                expected=(x.reshape(2,-1)@H.T)*scale
                self.assert_relative(out.numpy().reshape(2,-1),expected,8e-7)
            np.testing.assert_allclose(H,H.T,rtol=0,atol=0)
            # Translational invariance and bilinear symmetry are physical invariants.
            translation=np.tile([.4,-.2,.7],self.m.nflexvert)
            self.assertLess(np.linalg.norm(H@translation)/np.linalg.norm(H),2e-7)
            u,v=x.reshape(2,-1).astype(float)
            self.assertAlmostEqual(float(u@H@v),float(v@H@u),delta=1e-7)
            self.assertGreater(float(u@H@u),0)

    def test_atomic_and_owned_kernel_caches_are_isolated_in_both_directions(self):
        d,p=self.fixture(); reference=mjw.put_data(self.cm,self.cd,nworld=2)
        original_flags=self.m.opt.deterministic
        def run(state):
            state.flex_hessian_valid.zero_(); passive.flex_hessian(self.m,state)
            passive.passive(self.m,state); derivative.eff_shift(self.m,state)
        try:
            run(reference)
            fields=('flexedge_hessian','flexvert_hessian','qfrc_passive','efm_c')
            expected={n:getattr(reference,n).numpy() for n in fields}
            self.m.opt.deterministic=mjw.DeterminismType.NONE
            run(d)  # Populate upstream atomic factories before owned execution.
            for n in fields:self.assert_relative(getattr(d,n).numpy(),expected[n],3e-6)
            with p.capture_policy(): run(d)
            for n in fields:np.testing.assert_array_equal(getattr(d,n).numpy(),expected[n])
            run(d)  # Upstream atomic path must remain intact after leaving scope.
            for n in fields:self.assert_relative(getattr(d,n).numpy(),expected[n],3e-6)
            for name in ('_flex_hessian_elem','_flex_hessian_diag','_flex_elasticity','_flex_bending'):
                self.assertIsNot(p.factories[name](False),getattr(passive,name)(False))
        finally:self.m.opt.deterministic=original_flags

    def test_bending_independent_matrix_energy_and_disable_flags(self):
        d,p=self.fixture(); rng=np.random.default_rng(4)
        x=rng.normal(size=(2,self.m.nv)).astype(np.float32)
        H=self.J.T@bending_hessian(self.cm)@self.J
        out=wp.zeros((2,self.m.nv),dtype=float); vec=wp.array(x)
        with p.capture_policy():
            passive.flex_bend_mul(self.m,d,out,vec,use_timestep=False)
            self.assert_relative(out.numpy(),x@H.T,2e-6)
            np.testing.assert_allclose(H,H.T,rtol=0,atol=1e-10)
            self.assertGreater(float(x[0]@H@x[0]),0)
            # Negative s1 is -hK*v, not -h^2K*v.
            out.zero_(); passive.flex_bend_mul(self.m,d,out,vec,s1_in=-1.,use_timestep=True)
            self.assert_relative(out.numpy(),-self.cm.opt.timestep*(x@H.T),2e-6)
            flags=self.m.opt.disableflags
            try:
                self.m.opt.disableflags |= types.DisableBit.SPRING | types.DisableBit.DAMPER
                out.fill_(2.); passive.flex_bend_mul(self.m,d,out,vec,s1_in=1.,s2_in=1.)
                np.testing.assert_array_equal(out.numpy(),2.)
            finally: self.m.opt.disableflags=flags
        np.testing.assert_array_equal(p.status.numpy(),0)

    def test_cached_hessian_retains_valid_worlds_and_refreshes_invalid(self):
        d,p=self.fixture()
        with p.capture_policy():
            d.flex_hessian_valid.zero_(); passive.flex_hessian(self.m,d)
            before=(d.flexedge_hessian.numpy(),d.flexvert_hessian.numpy())
            d.flexedge_hessian.fill_(9); d.flexvert_hessian.fill_(8)
            valid=np.ones((2,self.m.nflex),bool); valid[1]=False
            d.flex_hessian_valid.assign(valid); passive.flex_hessian(self.m,d)
            np.testing.assert_array_equal(d.flexedge_hessian.numpy()[0],9)
            np.testing.assert_array_equal(d.flexvert_hessian.numpy()[0],8)
            np.testing.assert_array_equal(d.flexedge_hessian.numpy()[1],before[0][1])
            np.testing.assert_array_equal(d.flexvert_hessian.numpy()[1],before[1][1])
            saved=d.flexedge_hessian.numpy(); passive.flex_hessian(self.m,d)
            np.testing.assert_array_equal(saved,d.flexedge_hessian.numpy())

    def test_graph_replay_changed_input_skip_worlds_and_zero_transitions(self):
        d,p=self.fixture(); rng=np.random.default_rng(12)
        vec=wp.array(rng.normal(size=(2,self.m.nv)).astype(np.float32))
        out=wp.zeros((2,self.m.nv),dtype=float); skip=wp.zeros(2,dtype=bool)
        def run(): passive.flex_bend_mul(self.m,d,out,vec,use_timestep=False,skip=skip)
        with p.capture_policy():
            run(); wp.synchronize()
            with wp.ScopedCapture() as cap: run()
        changed=rng.normal(size=(2,self.m.nv)).astype(np.float32)
        vec.assign(changed); out.fill_(3.); skip.assign(np.array([True,False]))
        wp.capture_launch(cap.graph)
        H=self.J.T@bending_hessian(self.cm)@self.J
        np.testing.assert_array_equal(out.numpy()[0],3.)
        self.assert_relative(out.numpy()[1],3.+H@changed[1],2e-6)
        skip.zero_(); vec.zero_(); out.fill_(4.)
        wp.capture_launch(cap.graph)
        np.testing.assert_array_equal(out.numpy(),4.)
        # Disabled/zero contributions cannot survive from a previous graph launch.
        np.testing.assert_array_equal(p.bend_values.numpy(),0)
        np.testing.assert_array_equal(p.status.numpy(),0)

    def test_passive_disabled_transitions_have_no_stale_contributions(self):
        d,p=self.fixture(); a=mjw.put_data(self.cm,self.cd,nworld=2)
        flags=self.m.opt.disableflags
        try:
            for disabled in (0,types.DisableBit.SPRING,types.DisableBit.DAMPER,
                             types.DisableBit.SPRING | types.DisableBit.DAMPER,0):
                self.m.opt.disableflags=flags | disabled
                passive.passive(self.m,a)
                with p.capture_policy(): passive.passive(self.m,d)
                for name in ('qfrc_spring','qfrc_damper','qfrc_passive'):
                    self.assert_relative(getattr(d,name).numpy(),getattr(a,name).numpy()+1e-30,3e-6)
            np.testing.assert_array_equal(p.status.numpy(),0)
        finally: self.m.opt.disableflags=flags

    def test_graph_rebuilds_changed_geometry_and_preserves_skipped_hessian_products(self):
        from mujoco_warp._src import smooth
        d,p=self.fixture(); reference=mjw.put_data(self.cm,self.cd,nworld=2)
        def run():
            d.flex_hessian_valid.zero_(); passive.flex_hessian(self.m,d)
        with p.capture_policy():
            run(); wp.synchronize()
            with wp.ScopedCapture() as cap: run()
        qpos=d.qpos.numpy(); qpos[1,10:40]+=.0002
        for state in (d,reference):
            state.qpos.assign(qpos)
            smooth.kinematics(self.m,state); smooth.com_pos(self.m,state); smooth.flex(self.m,state)
        wp.capture_launch(cap.graph)
        reference.flex_hessian_valid.zero_(); passive.flex_hessian(self.m,reference)
        for name in ('flexedge_hessian','flexvert_hessian'):
            self.assert_relative(getattr(d,name).numpy(),getattr(reference,name).numpy(),3e-6)
        vec=wp.ones((2,self.m.nflexvert),dtype=wp.vec3)
        out=wp.full((2,self.m.nflexvert),wp.vec3(7.),dtype=wp.vec3)
        skip=wp.array(np.array([True,False]))
        with p.capture_policy():
            passive.flex_hessian_mul(self.m,d,out,vec,skip=skip)
            np.testing.assert_array_equal(out.numpy()[0],7.)
            # Excluding simple bodies must reset the active world's output.
            passive.flex_hessian_mul(self.m,d,out,vec,non_simple_only=True,skip=skip)
            np.testing.assert_array_equal(out.numpy()[0],7.)
            np.testing.assert_array_equal(out.numpy()[1],0.)
        np.testing.assert_array_equal(p.status.numpy(),0)

    def test_other_model_dispatch_and_product_argument_guards(self):
        d,p=self.fixture()
        cm=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>')
        cd=mujoco.MjData(cm); mujoco.mj_forward(cm,cd)
        m=mjw.put_model(cm); other=mjw.put_data(cm,cd)
        passive.passive(m,other); expected=other.qfrc_passive.numpy()
        with p.capture_policy():
            passive.passive(m,other)
            np.testing.assert_array_equal(other.qfrc_passive.numpy(),expected)
            vec=wp.ones((2,self.m.nv)); out=wp.zeros_like(vec)
            with self.assertRaises(ValueError):
                passive.flex_bend_mul(self.m,d,out,vec,skip=wp.zeros(1,dtype=bool))
            with self.assertRaisesRegex(ValueError,'alias'):
                passive.flex_bend_mul(self.m,d,vec,vec)
            with self.assertRaises(ValueError):
                passive.flex_hessian_mul(self.m,d,out,vec)
        np.testing.assert_array_equal(p.status.numpy(),0)

    def test_binned_reduction_matches_warp_with_cancellation_and_extreme_exponents(self):
        rng=np.random.default_rng(81)
        n,ng=1024,16
        groups=rng.integers(0,ng,size=n,dtype=np.int32)
        records=np.ldexp(rng.uniform(-1,1,(2,n,3)),rng.integers(-120,120,(2,n,3))).astype(np.float32)
        # Explicit severe cancellation, low terms, zeros and tiny values.
        records[:,0:8]=np.array([[1e20,1e-30,1.],[1.,1e20,-1e20],[-1e20,-1e20,1e20],
                                [0.,0.,0.],[1e-30,-1e-30,1e-30],[-1.,0.,1.],[3.,0.,-3.],[0.,0.,0.]],np.float32)
        groups[:8]=0
        initial=rng.normal(size=(2,ng,3)).astype(np.float32)
        data=wp.array(records,dtype=wp.vec3); out=wp.array(initial,dtype=wp.vec3)
        expected=wp.array(initial,dtype=wp.vec3)
        indices=np.argsort(groups,kind='stable').astype(np.int32)
        offsets=np.r_[0,np.cumsum(np.bincount(groups,minlength=ng))].astype(np.int32)
        wp.launch(upstream_binned_vec,dim=(2,n),inputs=[data,wp.array(groups),expected])
        status=wp.zeros(2,dtype=int)
        wp.launch(_reduce_components,dim=(2,ng,3),inputs=[wp.array(offsets),wp.array(indices),_components(data,3),_components(out,3),
            wp.zeros(2,dtype=bool),False,wp.zeros((2,1),dtype=bool),wp.zeros(ng,dtype=int),status])
        np.testing.assert_array_equal(out.numpy(),expected.numpy())
        np.testing.assert_array_equal(status.numpy(),0)

    def test_nonfinite_result_status_is_sticky(self):
        d,p=self.fixture()
        values=np.ones((2,self.m.nflexvert,3),np.float32); values[0]=np.nan
        out=wp.zeros((2,self.m.nv),dtype=float)
        p.scatter(self.m,d,out,wp.array(values,dtype=wp.vec3))
        np.testing.assert_array_equal(p.status.numpy(),[1,0])
        out.zero_(); p.scatter(self.m,d,out,wp.zeros((2,self.m.nflexvert),dtype=wp.vec3))
        np.testing.assert_array_equal(p.status.numpy(),[1,0])

    def test_domain_shape_source_topology_and_exception_guards(self):
        d,p=self.fixture(); original=passive.passive
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with p.capture_policy():
                with self.assertRaisesRegex(RuntimeError,'reentrant'):
                    with p.capture_policy(): pass
                raise RuntimeError('intentional')
        self.assertIs(passive.passive,original)
        with patch('deform3d_warp.mujoco_material.version',return_value='9.0'):
            with self.assertRaisesRegex(RuntimeError,'4.0.1'): OwnedFlexOperators(self.m,d)
        with patch.object(passive,'flex_bend_mul',lambda *args: None):
            with self.assertRaisesRegex(RuntimeError,'unreviewed'):
                with p.capture_policy(): pass
        with self.assertRaises(ValueError):
            p.scatter(self.m,d,wp.zeros((2,self.m.nv+1)),wp.zeros((2,self.m.nflexvert),dtype=wp.vec3))
        with self.assertRaises(ValueError):
            p.scatter(self.m,d,wp.zeros((2,self.m.nv)),wp.zeros((2,self.m.nflexvert),dtype=wp.vec3),skip=wp.zeros(1,dtype=bool))
        size=self.m.nflexedge
        try:
            self.m.nflexedge += 1
            with self.assertRaisesRegex(ValueError,'dimensions changed'):
                with p.capture_policy(): pass
        finally: self.m.nflexedge=size
        edge=self.m.flex_edgestiffness.numpy()
        try:
            self.m.flex_edgestiffness.fill_(1.)
            with self.assertRaisesRegex(ValueError,'edge springs'):
                with p.capture_policy(): pass
        finally: self.m.flex_edgestiffness.assign(edge)
        old=self.m.body_simple.numpy()
        try:
            self.m.body_simple.fill_(0)
            with self.assertRaisesRegex(RuntimeError,'topology changed'):
                with p.capture_policy(): pass
        finally: self.m.body_simple.assign(old)
        np.testing.assert_array_equal(p.status.numpy(),0)


if __name__=='__main__':
    with wp.ScopedDevice('cuda:0'): unittest.main(verbosity=2)
