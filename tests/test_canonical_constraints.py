"""Permutation invariants and captured replay for canonical constraint rows."""
from pathlib import Path
import sys
import unittest

import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import types
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from deform3d_warp.mujoco_ordering import CanonicalContactConstraints

DEVICE = 'cuda:0'


class CanonicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cpu = mujoco.MjModel.from_xml_path(str(ROOT/'examples/mujoco_bag_strong_ref.xml'))
        with wp.ScopedDevice(DEVICE):
            cls.model = mjw.put_model(cls.cpu)

    def fixture(self, orders=((2,0,1),(1,2,0)), sizes=(1,4,1)):
        with wp.ScopedDevice(DEVICE):
            d = mjw.make_data(self.cpu,nworld=2)
            d.nacon.assign(np.array([6],np.int32))
            d.nefc.assign(np.array([sum(sizes)]*2,np.int32))
            d.efc.jtdaj_nblock.fill_(3)
            fields = {name: getattr(d.efc,name).numpy() for name in CanonicalContactConstraints._ROW_FIELDS}
            blocks = d.efc.jtdaj_adr.numpy()
            lengths = d.efc.jtdaj_nrow.numpy()
            addresses = d.contact.efc_address.numpy(); addresses.fill(-1)
            world = d.contact.worldid.numpy()
            geom = d.contact.geom.numpy()
            pos = d.contact.pos.numpy()
            dim = d.contact.dim.numpy()
            # Contact IDs deliberately have a different permutation per world.
            ids = ((0,1,2),(5,3,4))
            for w in range(2):
                offset = 0
                for group, physical in enumerate(orders[w]):
                    count, c = sizes[physical], ids[w][physical]
                    blocks[w,group], lengths[w,group] = offset, count
                    addresses[c,:count] = np.arange(offset,offset+count)
                    world[c], geom[c], pos[c], dim[c] = w, [physical,physical+1], [physical,0,0], 1 if count == 1 else 3
                    for name, value in fields.items():
                        value[w,offset:offset+count] = 100*physical + np.arange(count)
                    fields['id'][w,offset:offset+count] = c
                    fields['type'][w,offset:offset+count] = int(types.ConstraintType.CONTACT_PYRAMIDAL)
                    offset += count
            for name,value in fields.items():
                getattr(d.efc,name).assign(value)
            d.efc.jtdaj_adr.assign(blocks); d.efc.jtdaj_nrow.assign(lengths)
            d.contact.efc_address.assign(addresses)
            d.contact.worldid.assign(world); d.contact.geom.assign(geom)
            d.contact.pos.assign(pos); d.contact.dim.assign(dim)
            # All physical key fields are defined; no dependence on unused slots.
            for name in ('dist','includemargin','frame','friction','solref','solreffriction','solimp'):
                getattr(d.contact,name).zero_()
            d.efc.J.fill_(7.25); d.efc.J_colind.fill_(2)
            return d, CanonicalContactConstraints(self.model,d), ids

    def verify(self,d,policy,ids,sizes=(1,4,1)):
        count = sum(sizes)
        expected = np.concatenate([100*p+np.arange(size) for p,size in enumerate(sizes)])
        for name in policy._ROW_FIELDS:
            actual = getattr(d.efc,name).numpy()[:,:count]
            if name == 'id':
                reference = np.array([[c for p in range(3) for c in [ids[w][p]]*sizes[p]] for w in range(2)])
            elif name == 'type':
                reference = np.full((2,count),int(types.ConstraintType.CONTACT_PYRAMIDAL))
            else:
                reference = np.tile(expected,(2,1))
            np.testing.assert_array_equal(actual,reference,err_msg=name)
        starts = np.array([0,sizes[0],sizes[0]+sizes[1]])
        np.testing.assert_array_equal(d.efc.jtdaj_adr.numpy()[:,:3],np.tile(starts,(2,1)))
        np.testing.assert_array_equal(d.efc.jtdaj_nrow.numpy()[:,:3],np.tile(sizes,(2,1)))
        addresses = d.contact.efc_address.numpy()
        for w in range(2):
            for p,c in enumerate(ids[w]):
                np.testing.assert_array_equal(addresses[c,:sizes[p]],np.arange(starts[p],starts[p]+sizes[p]))
        np.testing.assert_array_equal(d.efc.J.numpy(),7.25)
        np.testing.assert_array_equal(d.efc.J_colind.numpy(),2)
        np.testing.assert_array_equal(policy.status.numpy(),0)

    def test_all_row_and_contact_references(self):
        d,p,ids = self.fixture()
        p.enqueue(d)
        self.verify(d,p,ids)
        # Applying canonicalization twice is idempotent.
        p.enqueue(d)
        self.verify(d,p,ids)

    def test_graph_replay_with_different_layout(self):
        with wp.ScopedDevice(DEVICE):
            d,p,ids = self.fixture()
            p.enqueue(d)
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                p.enqueue(d)
            other,_,ids = self.fixture(((1,0,2),(0,2,1)))
            for name in p._ROW_FIELDS + ('jtdaj_adr','jtdaj_nrow','jtdaj_nblock'):
                wp.copy(getattr(d.efc,name),getattr(other.efc,name))
            wp.copy(d.contact.efc_address,other.contact.efc_address)
            wp.capture_launch(capture.graph)
            self.verify(d,p,ids)

    def test_duplicate_keys_keep_all_rows(self):
        d,p,_ = self.fixture(sizes=(1,1,1))
        d.contact.pos.zero_()
        d.contact.geom.zero_()
        p.enqueue(d)
        for w in range(2):
            self.assertEqual(sorted(d.efc.pos.numpy()[w,:3].tolist()),[0,100,200])
            self.assertEqual(sorted(p.permutation.numpy()[w,:3].tolist()),[0,1,2])
        np.testing.assert_array_equal(p.status.numpy(),0)

    def test_invalid_keys_and_layout_reported(self):
        d,p,_ = self.fixture()
        positions = d.contact.pos.numpy(); positions[0,0] = np.nan
        d.contact.pos.assign(positions)
        p.enqueue(d)
        self.assertEqual(p.status.numpy()[0]&1,1)
        self.assertEqual(sorted(p.permutation.numpy()[0,:6].tolist()),list(range(6)))
        d.contact.pos.zero_()
        d.nefc.fill_(d.njmax+1)
        p.enqueue(d)
        np.testing.assert_array_equal(p.status.numpy()&2,2)
        d.nefc.zero_(); d.efc.jtdaj_nblock.zero_()
        p.enqueue(d)
        np.testing.assert_array_equal(p.status.numpy()&2,2)

    def test_configuration_and_context_guards(self):
        from mujoco_warp._src import constraint
        d,p,_ = self.fixture()
        original = constraint.make_constraint
        with self.assertRaisesRegex(ValueError,'sentinel'):
            with p.capture_policy():
                with self.assertRaisesRegex(RuntimeError,'nested'):
                    with p.capture_policy():
                        pass
                raise ValueError('sentinel')
        self.assertIs(constraint.make_constraint,original)
        with wp.ScopedDevice(DEVICE):
            mismatch = mjw.make_data(self.cpu,nworld=1)
        with self.assertRaisesRegex(ValueError,'dimensions'):
            p.enqueue(mismatch)
        with p.capture_policy():
            with self.assertRaisesRegex(ValueError,'dimensions'):
                constraint.make_constraint(self.model,mismatch)
        old = self.model.opt.solver
        self.model.opt.solver = types.SolverType.CG
        try:
            with self.assertRaises(ValueError):
                CanonicalContactConstraints(self.model,d)
        finally:
            self.model.opt.solver = old

    def test_actual_contact_operator_preserved(self):
        from deform3d_warp.mujoco_contact import flex_contact_policy
        cpu = mujoco.MjData(self.cpu)
        for _ in range(140):
            mujoco.mj_step(self.cpu,cpu)
        with wp.ScopedDevice(DEVICE):
            original_cone = self.model.opt.cone
            self.addCleanup(setattr,self.model.opt,'cone',original_cone)
            for cone in (types.ConeType.PYRAMIDAL,types.ConeType.ELLIPTIC):
                self.model.opt.cone = cone
                d = mjw.put_data(self.cpu,cpu,nworld=2)
                with flex_contact_policy(self.model):
                    mjw.fwd_position(self.model,d,factorize=False)
                p = CanonicalContactConstraints(self.model,d)
                def operators():
                    nnz,adr,col,val = (getattr(d.efc,name).numpy() for name in ('J_rownnz','J_rowadr','J_colind','J'))
                    weights,offset = d.efc.D.numpy(),d.efc.pos.numpy()
                    result = []
                    for w,n in enumerate(d.nefc.numpy()):
                        self.assertGreater(n,0)
                        j = np.zeros((n,self.model.nv),np.float64)
                        for row in range(n):
                            start,count = adr[w,row],nnz[w,row]
                            j[row,col[w,0,start:start+count]] = val[w,0,start:start+count]
                        result.append((j.T@(weights[w,:n,None]*j),j.T@offset[w,:n].astype(np.float64)))
                    return result
                before = operators()
                rng = np.random.default_rng(201)
                d.efc.force.assign(rng.normal(size=d.efc.force.shape).astype(np.float32))
                contacts = wp.array(np.arange(d.naconmax,dtype=np.int32),dtype=int)
                force = wp.zeros(d.naconmax,dtype=wp.spatial_vector)
                mjw.contact_force(self.model,d,contacts,True,force)
                before_force = force.numpy().copy()
                p.enqueue(d)
                after = operators()
                mjw.contact_force(self.model,d,contacts,True,force)
                np.testing.assert_array_equal(force.numpy(),before_force)
                for (h0,g0),(h1,g1) in zip(before,after):
                    np.testing.assert_allclose(h0,h1,atol=1e-14,rtol=1e-13)
                    np.testing.assert_allclose(g0,g1,atol=1e-14,rtol=1e-13)
                np.testing.assert_array_equal(p.status.numpy(),0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
