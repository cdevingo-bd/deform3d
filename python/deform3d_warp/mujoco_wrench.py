"""Reproducible sparse body-wrench projection for the reviewed MJWarp runtime.

Each generalized coordinate gathers the wrenches on bodies whose welded DOF
ancestor chain contains it. The local moment-arm arithmetic and binned reduction
match MJWarp's deterministic scatter, without global record sorting. Dense
apply_ft calls retain their upstream implementation and rounding.
"""
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import inspect
import threading

import numpy as np
import warp as wp

from ._warp_binned import Bins, add_record, rounded_sum

_LOCK = threading.RLock()
_ACTIVE = False
_HASHES = {
    'apply_ft': 'e001fa29415a16ae53b67425c277bc01171e0b8737ea86ebdfca04a20f181ccf',
    '_apply_ft_scatter': '6be852e36dfef531c08528d27e79aa5d451d4d5cd5b17f8ce1e7decdc12ad78b',
}
_TOPOLOGY = ('body_rootid', 'body_weldid', 'body_dofnum', 'body_dofadr', 'dof_parentid')


@wp.kernel(enable_backward=False)
def _project(offsets: wp.array[int], bodies: wp.array[int], roots: wp.array[int],
             xipos: wp.array2d[wp.vec3], com: wp.array2d[wp.vec3],
             cdof: wp.array2d[wp.spatial_vector], ft: wp.array2d[wp.spatial_vector],
             out: wp.array2d[float], status: wp.array[int]):
    w, dof = wp.tid()
    total = Bins()
    active = bool(False)
    motion = cdof[w, dof]
    for k in range(offsets[dof], offsets[dof+1]):
        b = bodies[k]
        wrench = ft[w, b]
        if not (wrench == wp.spatial_vector()):
            force = wp.spatial_top(wrench)
            offset = xipos[w, b] - com[w, roots[b]]
            torque = wp.spatial_bottom(wrench) + wp.cross(offset, force)
            value = wp.dot(wp.spatial_bottom(motion), force) + wp.dot(wp.spatial_top(motion), torque)
            if not wp.isfinite(value):
                wp.atomic_or(status, w, 1)
            total = add_record(total, value)
            active = True
    # A destination with no emitted records is untouched, including signed zero.
    if active:
        value = out[w, dof] + rounded_sum(total)
        if not wp.isfinite(value):
            wp.atomic_or(status, w, 1)
        out[w, dof] = value


class OwnedWrenchProjection:
    """Opt-in CUDA body-load adapter; no contact/material forces are added.

    Construct outside capture, then enter capture_policy while recording the
    simulation graph. Keep this object and topology alive until replay completes.
    Mutating topology requires rebuilding this object and recapturing. Workspaces
    are not reentrant; separate concurrent streams need separate owners.
    status is a sticky per-world bit (1: nonfinite projected force).
    """
    def __init__(self, model, data, *, max_workspace_bytes=256*1024*1024):
        self.model, self.nworld = model, data.nworld
        self.device = model.body_mass.device
        self.nv, self.nbody = model.nv, model.nbody
        if not self.device.is_cuda or self.nworld < 1 or self.nv < 0 or self.nbody < 1:
            raise ValueError('wrench projection requires CUDA and valid model/world dimensions')
        self.validate_source()
        roots, weld, count, first, parent = self._topology()
        if np.any(roots < 0) or np.any(roots >= self.nbody) or np.any(weld < 0) or np.any(weld >= self.nbody):
            raise ValueError('invalid body root/weld index')
        if np.any(count < 0) or np.any(count > self.nv):
            raise ValueError('invalid body DOF count')
        used = count > 0
        if np.any(first[used] < 0) or np.any(first[used] > self.nv-count[used]):
            raise ValueError('invalid body DOF range')
        if np.any(parent < -1) or np.any(parent >= np.arange(self.nv)):
            raise ValueError('DOF parents must precede children')
        # Cap before device allocation, and while growing host incidence storage.
        base_bytes = 4*(self.nv+1+self.nworld)
        if base_bytes > max_workspace_bytes:
            raise ValueError('wrench projection workspace exceeds memory bound')
        incidence = [[] for _ in range(self.nv)]
        nnz = 0
        for b in range(self.nbody):
            wb = int(weld[b])
            if count[wb] == 0:
                continue
            dof = int(first[wb]+count[wb]-1)
            while dof >= 0:
                nnz += 1
                if base_bytes+4*nnz > max_workspace_bytes or nnz >= np.iinfo(np.int32).max:
                    raise ValueError('wrench projection workspace exceeds memory bound')
                incidence[dof].append(b)
                dof = int(parent[dof])
        offsets = np.r_[0,np.cumsum([len(a) for a in incidence])].astype(np.int32)
        bodies = np.asarray([b for row in incidence for b in row],np.int32)
        self.offsets = wp.array(offsets,dtype=int,device=self.device)
        self.bodies = wp.array(bodies,dtype=int,device=self.device)
        self.status = wp.zeros(self.nworld,dtype=int,device=self.device)
        self.workspace_bytes = sum(a.capacity for a in (self.offsets,self.bodies,self.status))
        self.signature = self._fingerprint()

    @staticmethod
    def validate_source():
        from mujoco_warp._src import support
        if version('mujoco-warp') != '4.0.1' or version('warp-lang').split('+')[0] != '1.19.0':
            raise RuntimeError('wrench projection requires reviewed MJWarp 4.0.1 / Warp 1.19.0')
        for name,digest in _HASHES.items():
            if hashlib.sha256(inspect.getsource(getattr(support,name)).encode()).hexdigest() != digest:
                raise RuntimeError('unreviewed or already patched MJWarp '+name)

    def _array(self, array, shape, dtype):
        if array.shape != shape or array.dtype != dtype or array.device != self.device or not array.is_contiguous:
            raise ValueError('invalid wrench projection array')

    def _topology(self):
        arrays = [getattr(self.model,name) for name in _TOPOLOGY]
        for a,name in zip(arrays,_TOPOLOGY):
            self._array(a,(self.nv if name == 'dof_parentid' else self.nbody,),wp.int32)
        return tuple(a.numpy() for a in arrays)

    def _fingerprint(self):
        h = hashlib.sha256()
        for a in self._topology():
            h.update(str(a.shape).encode()); h.update(a.tobytes())
        return h.digest()

    def validate_setup(self):
        from mujoco_warp._src.types import DeterminismType
        self.validate_source()
        if (self.model.nv != self.nv or self.model.nbody != self.nbody
            or self.model.body_mass.device != self.device or self._fingerprint() != self.signature):
            raise ValueError('body wrench topology changed; rebuild and recapture')
        if not self.model.opt.deterministic & DeterminismType.ATOMICS:
            raise ValueError('owned wrench policy requires deterministic atomics')

    def project(self, data, ft, out, *, add=True):
        """Map world-space COM wrenches [force, torque] to generalized forces.

        Dynamic data/force values are read on device on every graph replay.
        add=False clears the entire destination, matching upstream sparse calls.
        """
        if data.nworld != self.nworld:
            raise ValueError('wrench projection world count changed')
        for a in (data.xipos,data.subtree_com):
            self._array(a,(self.nworld,self.nbody),wp.vec3)
        self._array(data.cdof,(self.nworld,self.nv),wp.spatial_vector)
        self._array(ft,(self.nworld,self.nbody),wp.spatial_vector)
        self._array(out,(self.nworld,self.nv),wp.float32)
        self._array(self.status,(self.nworld,),wp.int32)
        reads = [data.xipos,data.subtree_com,data.cdof,ft,self.offsets,self.bodies,self.model.body_rootid]
        for i,dest in enumerate((out,self.status)):
            if not dest.capacity:
                continue
            for src in reads+([out] if i else []):
                if src.capacity and dest.ptr < src.ptr+src.capacity and src.ptr < dest.ptr+dest.capacity:
                    raise ValueError('wrench projection outputs must not alias inputs or each other')
        if not add:
            out.zero_()
        if self.nv:
            wp.launch(_project,(self.nworld,self.nv),inputs=[self.offsets,self.bodies,self.model.body_rootid,
                data.xipos,data.subtree_com,data.cdof,ft,out,self.status],device=self.device)

    @contextmanager
    def capture_policy(self):
        global _ACTIVE
        from mujoco_warp._src import support
        with _LOCK:
            if _ACTIVE:
                raise RuntimeError('nested owned wrench policy is unsupported')
            self.validate_setup()
            original = support.apply_ft
            def dispatch(m,d,ft,qfrc,flg_add,sparse=False):
                if m is self.model and sparse:
                    return self.project(d,ft,qfrc,add=flg_add)
                return original(m,d,ft,qfrc,flg_add,sparse=sparse)
            _ACTIVE = True
            try:
                support.apply_ft = dispatch
                yield self
            finally:
                support.apply_ft = original
                _ACTIVE = False
