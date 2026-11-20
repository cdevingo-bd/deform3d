"""Canonical contact constraint rows for MJWarp 4.0.1 sparse Newton.

MJWarp 4.0.1's CONTACTS/CONSTRAINT sorting hooks are stubs. This opt-in policy
orders actual contact blocks after assembly. It preserves their multiplicity,
equations and sparse Jacobian storage; every row/contact/block reference moves
together. It does not make other atomic reductions deterministic.
"""
from contextlib import contextmanager
from functools import lru_cache
import hashlib
from importlib.metadata import version
import inspect
import threading

import numpy as np
import warp as wp

_LOCK = threading.RLock()
_SOURCE_SHA = '8e62e9535654b5d74279ec5a6bfec149d22246c888e695d4eaeecbcdb2c360ed'
_STUB_SHA = 'edabab47d944dd2c5c76591615a6145e4e40cd7f6de955a814b6bed332cbd7b9'
_VEC5 = wp.types.vector(length=5, dtype=wp.float32)
_KEY_SIZE = 37


@wp.kernel(enable_backward=False)
def _contact_keys(nacon: wp.array(dtype=int), world: wp.array(dtype=int),
                  geom: wp.array(dtype=wp.vec2i), flex: wp.array(dtype=wp.vec2i),
                  elem: wp.array(dtype=wp.vec2i), vert: wp.array(dtype=wp.vec2i),
                  dim: wp.array(dtype=int), pos: wp.array(dtype=wp.vec3), frame: wp.array(dtype=wp.mat33),
                  dist: wp.array(dtype=float), margin: wp.array(dtype=float), friction: wp.array(dtype=_VEC5),
                  solref: wp.array(dtype=wp.vec2), solreffriction: wp.array(dtype=wp.vec2),
                  solimp: wp.array(dtype=_VEC5), keys: wp.array2d(dtype=wp.float64),
                  status: wp.array(dtype=int)):
    c = wp.tid()
    if c >= nacon[0]:
        return
    w = world[c]
    if w < 0 or w >= status.shape[0]:
        return
    for side in range(2):
        keys[c,side] = wp.float64(geom[c][side])
        # Rigid contact writers need not populate flex-only fields.
        f, e, v = int(-1), int(-1), int(-1)
        if geom[c][side] < 0:
            f, e, v = flex[c][side], elem[c][side], vert[c][side]
        keys[c,2+side] = wp.float64(f)
        keys[c,4+side] = wp.float64(e)
        keys[c,6+side] = wp.float64(v)
    keys[c,8] = wp.float64(dim[c])
    for i in range(3):
        keys[c,9+i] = wp.float64(pos[c][i])
        for j in range(3):
            keys[c,12+3*i+j] = wp.float64(frame[c][i,j])
    keys[c,21] = wp.float64(dist[c])
    keys[c,22] = wp.float64(margin[c])
    for i in range(5):
        keys[c,23+i] = wp.float64(friction[c][i])
        keys[c,32+i] = wp.float64(solimp[c][i])
    for i in range(2):
        keys[c,28+i] = wp.float64(solref[c][i])
        keys[c,30+i] = wp.float64(solreffriction[c][i])
    for i in range(_KEY_SIZE):
        if not wp.isfinite(keys[c,i]):
            wp.atomic_or(status,w,1)


@wp.func
def _before(a: int, b: int, keys: wp.array2d(dtype=wp.float64)):
    for i in range(_KEY_SIZE):
        x, y = keys[a,i], keys[b,i]
        # Total ordering even for invalid NaN inputs, so ranks cannot collide.
        nx, ny = wp.isnan(x), wp.isnan(y)
        if nx != ny:
            return nx
        if not nx:
            if x < y:
                return True
            if x > y:
                return False
    return a < b


@wp.kernel(enable_backward=False)
def _identity(permutation: wp.array2d(dtype=int), inverse: wp.array2d(dtype=int)):
    w, row = wp.tid()
    permutation[w,row] = row
    inverse[w,row] = row


@wp.kernel(enable_backward=False)
def _rank(nefc: wp.array(dtype=int), ne: wp.array(dtype=int), nf: wp.array(dtype=int), nl: wp.array(dtype=int),
          nblock: wp.array(dtype=int), block_adr: wp.array2d(dtype=int), block_size: wp.array2d(dtype=int),
          ids: wp.array2d(dtype=int), nacon: wp.array(dtype=int), world: wp.array(dtype=int),
          keys: wp.array2d(dtype=wp.float64), permutation: wp.array2d(dtype=int), inverse: wp.array2d(dtype=int),
          next_adr: wp.array2d(dtype=int), next_size: wp.array2d(dtype=int), status: wp.array(dtype=int)):
    w, group = wp.tid()
    capacity = permutation.shape[1]
    count = nblock[w]
    if (ne[w] != 0 or nf[w] != 0 or nl[w] != 0 or nefc[w] < 0 or nefc[w] > capacity
        or count < 0 or count > capacity or (count == 0 and nefc[w] != 0)
        or nacon[0] < 0 or nacon[0] > keys.shape[0]):
        if group == 0:
            wp.atomic_or(status,w,2)
        return
    if group >= count:
        return
    old = block_adr[w,group]
    rows = block_size[w,group]
    if old < 0 or rows < 1 or old+rows > nefc[w]:
        wp.atomic_or(status,w,2)
        return
    c = ids[w,old]
    if c < 0 or c >= nacon[0] or world[c] != w:
        wp.atomic_or(status,w,2)
        return
    new, ordinal, total = int(0), int(0), int(0)
    for other in range(count):
        address = block_adr[w,other]
        size = block_size[w,other]
        if address < 0 or size < 1 or address+size > nefc[w]:
            wp.atomic_or(status,w,2)
            return
        candidate = ids[w,address]
        if candidate < 0 or candidate >= nacon[0] or world[candidate] != w:
            wp.atomic_or(status,w,2)
            return
        if other != group and (candidate == c or (old < address+size and address < old+rows)):
            wp.atomic_or(status,w,2)
            return
        total += size
        if _before(candidate,c,keys):
            new += size
            ordinal += 1
    if total != nefc[w] or new+rows > nefc[w] or ordinal >= count:
        wp.atomic_or(status,w,2)
        return
    for i in range(rows):
        permutation[w,new+i] = old+i
        inverse[w,old+i] = new+i
    next_adr[w,ordinal] = new
    next_size[w,ordinal] = rows


@lru_cache(None)
def _gather(dtype):
    @wp.kernel(module='unique', enable_backward=False)
    def kernel(nefc: wp.array(dtype=int), permutation: wp.array2d(dtype=int),
               source: wp.array2d(dtype=dtype), target: wp.array2d(dtype=dtype)):
        w, row = wp.tid()
        if row < wp.min(nefc[w],permutation.shape[1]):
            target[w,row] = source[w,permutation[w,row]]
    return kernel


@wp.kernel(enable_backward=False)
def _remap_contacts(nacon: wp.array(dtype=int), world: wp.array(dtype=int),
                    nefc: wp.array(dtype=int), inverse: wp.array2d(dtype=int), address: wp.array2d(dtype=int)):
    c, axis = wp.tid()
    if c < nacon[0]:
        w = world[c]
        old = address[c,axis]
        if w >= 0 and w < nefc.shape[0] and old >= 0 and old < wp.min(nefc[w],inverse.shape[1]):
            address[c,axis] = inverse[w,old]


class CanonicalContactConstraints:
    """Opt-in contact-only sparse Newton ordering. Keep alive through replay.

    Status is sticky: 1 nonfinite contact key, 2 invalid/overflowed layout.
    Any nonzero status invalidates the trajectory. Ranking is quadratic in the
    number of active contact blocks per world; it is not a large-contact solver.
    """
    _ROW_FIELDS = ('type','id','J_rownnz','J_rowadr','pos','margin','D','vel','aref',
                   'frictionloss','force','state','island','Jqvel')

    def __init__(self, model, data):
        self.model = model
        self._validate_model()
        self.nworld, self.njmax, self.naconmax = data.nworld, data.njmax, data.naconmax
        self.device = model.body_mass.device
        if data.qpos.device != self.device or self.nworld < 1 or self.njmax < 1 or self.naconmax < 1:
            raise ValueError('invalid data dimensions/device')
        self.keys = wp.empty((self.naconmax,_KEY_SIZE),dtype=wp.float64,device=self.device)
        self.permutation = wp.empty((self.nworld,self.njmax),dtype=int,device=self.device)
        self.inverse = wp.empty_like(self.permutation)
        self.status = wp.zeros(self.nworld,dtype=int,device=self.device)
        self.rows = {name: wp.empty_like(getattr(data.efc,name)) for name in self._ROW_FIELDS}
        self.blocks = {name: wp.empty_like(getattr(data.efc,name)) for name in ('jtdaj_adr','jtdaj_nrow')}

    def _validate_model(self):
        from mujoco_warp._src import types
        m = self.model
        if version('mujoco-warp') != '4.0.1':
            raise RuntimeError('reviewed only for mujoco-warp 4.0.1')
        if (not m.body_mass.device.is_cuda or not m.is_sparse or m.opt.solver != types.SolverType.NEWTON
            or m.neq or m.ntendon or m.flg_adhesion or m.opt.enableflags & types.EnableBit.SLEEP
            or m.jnt_limited_ball_adr.size or m.jnt_limited_slide_hinge_adr.size
            or np.any(m.dof_frictionloss.numpy() != 0)):
            raise ValueError('requires CUDA sparse Newton, contact-only constraints, no adhesion or sleeping')

    def _validate_data(self, data):
        if (data.nworld != self.nworld or data.njmax != self.njmax or data.naconmax != self.naconmax
            or data.qpos.device != self.device):
            raise ValueError('data dimensions/device changed; rebuild the ordering workspace')

    def enqueue(self, data):
        self._validate_data(data)
        c, e = data.contact, data.efc
        wp.launch(_contact_keys,dim=self.naconmax,inputs=[data.nacon,c.worldid,c.geom,c.flex,c.elem,c.vert,
            c.dim,c.pos,c.frame,c.dist,c.includemargin,c.friction,c.solref,c.solreffriction,c.solimp,
            self.keys,self.status],device=self.device)
        wp.launch(_identity,dim=self.permutation.shape,inputs=[self.permutation,self.inverse],device=self.device)
        # Preserve unused entries, including padded storage, when copying back.
        for name, target in self.blocks.items():
            wp.copy(target,getattr(e,name))
        wp.launch(_rank,dim=self.permutation.shape,inputs=[data.nefc,data.ne,data.nf,data.nl,e.jtdaj_nblock,
            e.jtdaj_adr,e.jtdaj_nrow,e.id,data.nacon,c.worldid,self.keys,self.permutation,self.inverse,
            self.blocks['jtdaj_adr'],self.blocks['jtdaj_nrow'],self.status],device=self.device)
        for name, target in self.rows.items():
            source = getattr(e,name)
            wp.copy(target,source)
            wp.launch(_gather(source.dtype),dim=self.permutation.shape,
                inputs=[data.nefc,self.permutation,source,target],device=self.device)
        for name, target in self.rows.items():
            wp.copy(getattr(e,name),target)
        for name, target in self.blocks.items():
            wp.copy(getattr(e,name),target)
        wp.launch(_remap_contacts,dim=c.efc_address.shape,
            inputs=[data.nacon,c.worldid,data.nefc,self.inverse,c.efc_address],device=self.device)

    @contextmanager
    def capture_policy(self):
        from mujoco_warp._src import constraint
        self._validate_model()
        with _LOCK:
            original = constraint.make_constraint
            for function, expected in ((original,_SOURCE_SHA),(constraint._sort_constraints,_STUB_SHA)):
                if hashlib.sha256(inspect.getsource(function).encode()).hexdigest() != expected:
                    raise RuntimeError('unreviewed constraint assembly or nested ordering policy')
            def dispatch(model,data):
                if model is self.model:
                    self._validate_data(data)
                result = original(model,data)
                if model is self.model:
                    self.enqueue(data)
                return result
            constraint.make_constraint = dispatch
            # forward.py calls the function through the constraint module.
            try:
                yield self
            finally:
                constraint.make_constraint = original
