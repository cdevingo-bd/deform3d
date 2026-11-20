"""Owned assembly of the discrete flex effective metric for MJWarp 4.0.1.

Reviewed upstream kernels still evaluate h² K + h D in their original frames.
Only their atomic destinations become private slots. A fixed CSR incidence map
gathers each launch separately using Warp's reviewed reproducible accumulator.
"""
import ast
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import inspect
import linecache
import threading

import numpy as np
import warp as wp
from ._warp_binned import Bins, add_record, rounded_sum

_LOCK = threading.RLock()
_ACTIVE = False
_VERT = '_eff_flex_stretch_stiff_vert'
_EDGE = '_eff_flex_stretch_stiff_edge'
_BEND = '_eff_flex_bend_stiff'
_HASHES = {
    _VERT: '8e7ae2f890cd22b8cf7d1bd19d731e2bf391867da61474c522a5eab8e1f09def',
    _EDGE: 'cc56f838fbb4f2162e3c347c2e24aaeb72cf45db0db481f8526070857eb211d5',
    _BEND: '0cc2358fce21fa1d56934e67c33e623e5228c8b2f6cf386782a92b90005b17dc',
    'eff_build': '25ccfe35424dccef9648b535afa03df732ef566124b7fed6790f82a4922686ad',
    '_find_csr_col': '1af14ed51e29746eab1e9cb52b366f4c8178590f9980fd39b4a512eaee1dde0b',
}
_TOPOLOGY = ('body_weldid','body_dofnum','body_dofadr','flex_dim','flex_interp',
             'flex_vertadr','flex_vertnum','flex_edgeadr','flex_edgenum','flex_vertbodyid',
             'flex_edge','flex_edgeflap','flex_vertflexid','flex_edgeflexid',
             'flex_stiffnessadr','flex_bendingadr','flex_simple','flex_rigid',
             'efm_K_rownnz','efm_K_rowadr','efm_K_colind')
_COUNTS = ('nv','nbody','nflex','nflexvert','nflexedge','nefmK','nflexrodhinge',
           'nflexrodedge','flex_interp_assemblable')


@wp.kernel(enable_backward=False)
def _reduce(offset: wp.array[int], entries: wp.array[int],
            values: wp.array2d[float], emitted: wp.array2d[bool],
            out: wp.array2d[float], status: wp.array[int]):
    w, dst = wp.tid()
    total = Bins()
    active = bool(False)
    for k in range(offset[dst],offset[dst+1]):
        slot = entries[k]
        if emitted[w,slot]:
            value = values[w,slot]
            if not wp.isfinite(value):
                wp.atomic_or(status,w,1)
            total = add_record(total,value)
            active = True
    if active:
        value = out[w,dst] + rounded_sum(total)
        if not wp.isfinite(value):
            wp.atomic_or(status,w,1)
        out[w,dst] = value


def _source(module,name):
    obj = getattr(module,name)
    # _find_csr_col is a Warp Function, not a Python callable.
    source = inspect.getsource(obj.func if isinstance(obj,wp.Function) else obj)
    if hashlib.sha256(source.encode()).hexdigest() != _HASHES[name]:
        raise RuntimeError('unreviewed or already patched effective-metric '+name)
    return source


def _compile(source,namespace,name):
    path = '<deform3d_owned_effective_'+hashlib.sha256(source.encode()).hexdigest()+'>'
    linecache.cache[path] = (len(source),None,source.splitlines(True),path)
    namespace = dict(namespace)
    exec(compile(source,path,'exec'),namespace)
    return namespace[name]


def _replace(source,old,new):
    if source.count(old) != 1:
        raise RuntimeError('unexpected effective-metric output site')
    return source.replace(old,new)


def _factory(module,name):
    source = _source(module,name)
    source = _replace(source,'    efm_K_val_out: wp.array2d[float],',
                      '    efm_K_val_out: wp.array2d[float],\n    emitted_out: wp.array2d[bool],')
    slot = {'_eff_flex_stretch_stiff_vert':'9 * vertid',
            '_eff_flex_stretch_stiff_edge':'9 * (2 * edgeid + side)',
            '_eff_flex_bend_stiff':'9 * (16 * edgeid + 4 * i + j)'}[name]
    # Preserve expression, branch and per-launch rounding, including the bending
    # equal-quaternion diagonal shortcut. Masks distinguish absent and zero records.
    for suffix,value,indent in [('c','blkd[r, c]',22 if name==_BEND else 10)]+(
            [('r','q',20)] if name==_BEND else []):
        old=f'wp.atomic_add(efm_K_val_out, worldid, idx + {suffix}, {value})'
        dest=f'{slot} + 3 * r + {suffix}'
        new=f'efm_K_val_out[worldid, {dest}] = {value}\n'+(' '*indent)+f'emitted_out[worldid, {dest}] = True'
        source = _replace(source,old,new)
    if 'wp.atomic_add(' in source:
        raise RuntimeError('unreplaced effective-metric scatter')
    # cache_kernel keys names, not source: never occupy the upstream cache key.
    owned = '_deform3d_owned'+name+'_'+hashlib.sha256(source.encode()).hexdigest()[:12]
    source = _replace(source,'def '+name+'(','def '+owned+'(')
    return _compile(source,vars(module),owned)


class _Launches(ast.NodeTransformer):
    def __init__(self):self.counts=dict.fromkeys((_VERT,_EDGE,_BEND),0)
    def visit_Expr(self,node):
        node=self.generic_visit(node);call=node.value
        if not isinstance(call,ast.Call) or ast.unparse(call.func)!='wp.launch' or not call.args:return node
        factory=call.args[0]
        if not isinstance(factory,ast.Call) or not isinstance(factory.func,ast.Name):return node
        name=factory.func.id
        if name not in self.counts:return node
        self.counts[name]+=1
        call.func=ast.parse('_owner._launch',mode='eval').body
        call.args=[ast.Constant(name)]
        return node


class OwnedEffectiveMetric:
    """Experimental CUDA vertex-flex metric assembly, preserving native equations.

    Supports simple triangle/tetrahedral flexes (no rods/interpolated flexes).
    All unsupported host operations stay upstream. Topology is immutable; rebuild
    and recapture after changes. Dynamic hessians, rotations, timestep and material
    values remain device inputs. Keep the owner alive for graph replay; no concurrent
    use of its private storage. Sticky status 1 invalidates a nonfinite run.
    """
    def __init__(self,model,data,*,max_workspace_bytes=256*1024*1024):
        from mujoco_warp._src import derivative
        self.model,self.nworld,self.device=model,data.nworld,model.body_mass.device
        self.counts=tuple(getattr(model,n) for n in _COUNTS)
        self._validate_model()
        self.validate_source()
        topo=self._topology()
        self._validate_topology(topo)
        sizes={_VERT:9*model.nflexvert,_EDGE:18*model.nflexedge,_BEND:144*model.nflexedge}
        # Worst case: every scalar private slot occurs in the gather once.
        bound=4*self.nworld+sum(self.nworld*5*n+4*n+4*(model.nefmK+1) for n in sizes.values())
        if bound>max_workspace_bytes or max(sizes.values(),default=0)>=np.iinfo(np.int32).max:
            raise ValueError('effective-metric workspace exceeds memory bound')
        mappings=self._maps(topo,sizes)
        self.status=wp.zeros(self.nworld,dtype=int,device=self.device)
        self.buffers={};self.csr={};self.factories={};allocations=[self.status]
        for name,size in sizes.items():
            mapped=mappings[name];valid=np.flatnonzero(mapped>=0)
            order=np.argsort(mapped[valid],kind='stable');entries=valid[order].astype(np.int32)
            counts=np.bincount(mapped[valid],minlength=model.nefmK)
            offsets=np.r_[0,np.cumsum(counts)].astype(np.int32)
            self.csr[name]=(wp.array(offsets,dtype=int,device=self.device),wp.array(entries,dtype=int,device=self.device))
            self.buffers[name]=(wp.empty((self.nworld,size),dtype=float,device=self.device),
                                wp.zeros((self.nworld,size),dtype=bool,device=self.device))
            allocations.extend(self.csr[name]+self.buffers[name])
            self.factories[name]=_factory(derivative,name)
        self.workspace_bytes=sum(a.capacity for a in allocations)
        self.signature=self._fingerprint()
        tree=ast.parse(_source(derivative,'eff_build'));transform=_Launches()
        tree=ast.fix_missing_locations(transform.visit(tree))
        if any(n!=1 for n in transform.counts.values()):raise RuntimeError('unexpected effective-metric launch count')
        self._configured=_compile(ast.unparse(tree)+'\n',dict(vars(derivative),_owner=self),'eff_build')
        self._validate_data(data)

    @staticmethod
    def validate_source():
        from mujoco_warp._src import derivative
        if version('mujoco-warp')!='4.0.1' or version('warp-lang').split('+')[0]!='1.19.0':
            raise RuntimeError('effective metric requires reviewed MJWarp 4.0.1 / Warp 1.19.0')
        for name in _HASHES:_source(derivative,name)

    def _array(self,a,shape,dtype):
        if a.shape!=shape or a.dtype!=dtype or a.device!=self.device or not a.is_contiguous:
            raise ValueError('invalid effective-metric array')

    def _validate_model(self):
        m=self.model
        if not self.device.is_cuda or self.nworld<1 or min(m.nv,m.nbody,m.nflex,m.nflexvert,m.nflexedge,m.nefmK)<0:
            raise ValueError('effective metric requires CUDA and valid dimensions')
        if m.nflexrodedge or m.nflexrodhinge:
            raise ValueError('owned effective metric does not support rods or interpolation')

    def _topology(self):
        m=self.model;result={}
        for name in _TOPOLOGY:
            if name.startswith('body_'):size=m.nbody
            elif name in ('flex_vertbodyid','flex_vertflexid'):size=m.nflexvert
            elif name in ('flex_edge','flex_edgeflap','flex_edgeflexid'):size=m.nflexedge
            elif name in ('efm_K_rowadr','efm_K_rownnz'):size=m.nv
            elif name=='efm_K_colind':size=m.nefmK
            else:size=m.nflex
            dtype=wp.vec2i if name in ('flex_edge','flex_edgeflap') else (
                wp.bool if name in ('flex_simple','flex_rigid') else wp.int32)
            a=getattr(m,name);self._array(a,(size,),dtype);result[name]=a.numpy()
        return result

    def _validate_topology(self,t):
        m=self.model
        if np.any(t['flex_interp']!=0) or np.any(~np.isin(t['flex_dim'],[2,3])):
            raise ValueError('owned effective metric requires triangle/tetrahedral vertex flexes')
        for name,limit in [('body_weldid',m.nbody),('flex_vertbodyid',m.nbody),
                           ('flex_vertflexid',m.nflex),('flex_edgeflexid',m.nflex)]:
            if np.any(t[name]<0) or np.any(t[name]>=limit):raise ValueError('invalid '+name)
        for stem,total in [('vert',m.nflexvert),('edge',m.nflexedge)]:
            adr,num=t['flex_'+stem+'adr'],t['flex_'+stem+'num']
            if np.any(adr<0) or np.any(num<0) or np.any(adr>total-num):raise ValueError('invalid flex range')
            for f,(start,count) in enumerate(zip(adr,num)):
                if np.any(t['flex_'+stem+'flexid'][start:start+count]!=f):raise ValueError('inconsistent flex incidence')
        for e,f in enumerate(t['flex_edgeflexid']):
            n=t['flex_vertnum'][f]
            if np.any(t['flex_edge'][e]<0) or np.any(t['flex_edge'][e]>=n):raise ValueError('invalid flex edge')
            flap=t['flex_edgeflap'][e]
            if np.any(flap < -1) or np.any(flap>=n) or (flap[1]>=0 and flap[0]<0):raise ValueError('invalid flex flap')
        count,adr=t['body_dofnum'],t['body_dofadr'];used=count>0
        if np.any(count<0) or np.any(adr[used]<0) or np.any(adr[used]>m.nv-count[used]):raise ValueError('invalid body DOF range')
        occupied=np.zeros(m.nefmK,bool)
        for a,n in zip(t['efm_K_rowadr'],t['efm_K_rownnz']):
            if a<0 or n<0 or a>m.nefmK-n or occupied[a:a+n].any():raise ValueError('invalid or overlapping CSR rows')
            occupied[a:a+n]=True;cols=t['efm_K_colind'][a:a+n]
            if np.any(cols<0) or np.any(cols>=m.nv) or np.any(np.diff(cols)<=0):raise ValueError('CSR columns must be sorted and unique')
        self._validate_parameters(t)

    def _validate_parameters(self,t):
        m=self.model
        for name in ('flex_stiffness','flex_bending','flex_damping'):
            a=getattr(m,name);self._array(a,a.shape,wp.float32)
            if a.ndim!=1:raise ValueError('invalid effective-metric parameter dimensions')
        if m.flex_damping.size!=m.nflex:raise ValueError('invalid damping count')
        for f in range(m.nflex):
            s,b=int(t['flex_stiffnessadr'][f]),int(t['flex_bendingadr'][f])
            if s < -1 or s>=m.flex_stiffness.size:raise ValueError('invalid stiffness address')
            if b < -1 or (b>=0 and b+17*int(t['flex_edgenum'][f])>m.flex_bending.size):raise ValueError('invalid bending range')

    def _maps(self,t,sizes):
        maps={name:np.full(n,-1,np.int32) for name,n in sizes.items()}
        dofs=t['body_dofadr'][t['body_weldid'][t['flex_vertbodyid']]].copy()
        dofs[t['body_dofnum'][t['body_weldid'][t['flex_vertbodyid']]]!=3]=-1
        def block(name,slot,vi,vj):
            di,dj=int(dofs[vi]),int(dofs[vj])
            if di<0 or dj<0 or t['efm_K_rownnz'][di]==0:return
            for r in range(3):
                a,n=int(t['efm_K_rowadr'][di+r]),int(t['efm_K_rownnz'][di+r])
                cols=t['efm_K_colind'][a:a+n];k=int(np.searchsorted(cols,dj))
                if k==n or cols[k]!=dj:continue
                if k+3>n or not np.array_equal(cols[k:k+3],np.arange(dj,dj+3)):
                    raise ValueError('effective metric requires complete contiguous 3-column blocks')
                maps[name][slot+3*r:slot+3*r+3]=np.arange(a+k,a+k+3)
        for v,f in enumerate(t['flex_vertflexid']):
            if t['flex_simple'][f] and not t['flex_rigid'][f]:block(_VERT,9*v,v,v)
        for e,f in enumerate(t['flex_edgeflexid']):
            if not t['flex_simple'][f] or t['flex_rigid'][f]:continue
            base=t['flex_vertadr'][f];v0,v1=t['flex_edge'][e]+base
            block(_EDGE,18*e,v0,v1);block(_EDGE,18*e+9,v1,v0)
            if t['flex_dim'][f]==2 and t['flex_edgeflap'][e,1]>=0:
                vs=np.r_[t['flex_edge'][e],t['flex_edgeflap'][e]]+base
                for i,vi in enumerate(vs):
                    for j,vj in enumerate(vs):block(_BEND,9*(16*e+4*i+j),vi,vj)
        return maps

    def _fingerprint(self):
        h=hashlib.sha256()
        for a in self._topology().values():h.update(str(a.shape).encode());h.update(a.tobytes())
        return h.digest()

    def validate_setup(self):
        from mujoco_warp._src.types import DeterminismType,IntegratorType
        self.validate_source();self._validate_model()
        if tuple(getattr(self.model,n) for n in _COUNTS)!=self.counts or self._fingerprint()!=self.signature:
            raise ValueError('effective-metric topology changed; rebuild and recapture')
        self._validate_parameters(self._topology())
        if self.model.opt.integrator!=IntegratorType.DISCRETE or not self.model.opt.deterministic & DeterminismType.ATOMICS:
            raise ValueError('owned effective metric requires discrete integration and deterministic atomics')

    def _validate_data(self,d):
        from mujoco_warp._src.types import vec6
        m=self.model
        if d.nworld!=self.nworld:raise ValueError('effective-metric world count changed')
        for a,n,dtype in [(d.xmat,m.nbody,wp.mat33),(d.xquat,m.nbody,wp.quat),
                           (d.flexvert_hessian,m.nflexvert,vec6),(d.flexedge_hessian,m.nflexedge,wp.mat33),
                           (d.efm_K_val,m.nefmK,wp.float32)]:self._array(a,(self.nworld,n),dtype)
        self._array(m.opt.timestep,m.opt.timestep.shape,wp.float32)
        if m.opt.timestep.ndim!=1 or not m.opt.timestep.size:raise ValueError('invalid timestep array')
        self._array(self.status,(self.nworld,),wp.int32)
        reads=[d.xmat,d.xquat,d.flexvert_hessian,d.flexedge_hessian,m.opt.timestep,
               m.flex_stiffness,m.flex_bending,m.flex_damping]+[getattr(m,n) for n in _TOPOLOGY]
        for i,out in enumerate((d.efm_K_val,self.status)):
            if not out.capacity:continue
            for a in reads+([d.efm_K_val] if i else []):
                if a.capacity and out.ptr<a.ptr+a.capacity and a.ptr<out.ptr+out.capacity:
                    raise ValueError('effective-metric outputs must not alias inputs or each other')

    def _launch(self,name,*,dim,inputs,outputs):
        values,emitted=self.buffers[name]
        # Only the mask is cleared; absent records never read private values.
        emitted.zero_()
        wp.launch(self.factories[name](False),dim=dim,inputs=inputs,outputs=[values,emitted],device=self.device)
        if self.model.nefmK:
            wp.launch(_reduce,(self.nworld,self.model.nefmK),inputs=[*self.csr[name],values,emitted,outputs[0],self.status],device=self.device)

    @contextmanager
    def capture_policy(self):
        global _ACTIVE
        from mujoco_warp._src import derivative
        with _LOCK:
            if _ACTIVE:raise RuntimeError('nested owned effective-metric policy is unsupported')
            self.validate_setup();original=derivative.eff_build
            def dispatch(m,d):
                if m is not self.model:return original(m,d)
                self._validate_data(d)
                return self._configured(m,d)
            _ACTIVE=True
            try:
                derivative.eff_build=dispatch
                yield self
            finally:
                derivative.eff_build=original
                _ACTIVE=False
