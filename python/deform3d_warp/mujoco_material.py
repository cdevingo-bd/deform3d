"""Deterministic, topology-owned flex reductions for MJWarp 4.0.1.

The upstream element constitutive kernels are source-guarded and retain their
arithmetic. Only their output scatters are replaced by private contribution
slots and fixed-order binned gathers. This is an experimental optional adapter,
not a new material model. See docs/mjwarp_material.md for the domain/contract.
"""
import ast
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import inspect
import linecache
from pathlib import Path
import threading

import numpy as np
import warp as wp
from ._warp_binned import Bins, add_record, rounded_sum

_LOCK = threading.RLock()
_ACTIVE = False
_MODULE_HASH = 'c1cf7cd57ca4a6ee7c1e077295039bed4dee85c5f4f2c9920d3266bfd91dcef8'
_HASHES = {
    'flex_scatter': '972edadd84dca1cbe784369135544a1335c631022487d2a3d79a46385730b1c7',
    'flex_bend_mul': 'a802cca4806759f1ebef2c87da0db355c00a7d79d2a209b8f5055113fe27e5a0',
    'flex_hessian_mul': '21ce76a0ccb17ec58d0b619d0f62d61a5a760170d2c3ecc3942faae242495d91',
    'flex_hessian': '809412536ac06f0f050f855ffb550d7c895047d0f568e5fc31a9907aa4253eba',
    'passive': 'cd999d6a3080441a95c1efb358c48690b3129e423692179e82ae105eecc75081',
    '_flex_bend_mul': 'e89c87b5f82f5525d22f00cd73a0f4407f8a8cd92b1f14453d198fca053df39b',
    '_flex_hessian_mul_edge': '087b22a3dfc7fbbc8a4efaed87d7d54545bbcc8a1aaffc9ccadd7ae793c2d263',
    '_flex_elasticity': '6bbc429ce964e3018928f858e8d7faed526d7930e2610a814656d9a3168b9071',
    '_flex_bending': '302fa966d8a24fcf4eb1d69ffffd42e2b5459260dbdaa2a39383dec51934b387',
    '_flex_hessian_elem': '2b7868f8c31ed9327be2c610da6abc494199edb92739b3aba98e6996456ddcb4',
    '_flex_hessian_diag': '9d0f38a7ce9b4dff6f06a7bc9c2a0e1e4844ed8f1bcf08c359a7669ef422e838',
    'flex_scatter_vert': '1c8bf54372fc43021bbd871902b035fa1ba4a1090b20f6a02b67328e931f2480',
    '_spring_damper_flexedge_passive': '3e6d941b6ae8284061bd6c8eecba076ca996d368679c4a4c125704664106961c',
}
_FACTORIES = ('_flex_bend_mul', '_flex_hessian_mul_edge', '_flex_elasticity',
              '_flex_bending', '_flex_hessian_elem', '_flex_hessian_diag')
_HOOKS = ('flex_scatter', 'flex_bend_mul', 'flex_hessian_mul', 'flex_hessian', 'passive')
_TOPOLOGY = ('body_rootid', 'body_weldid', 'body_dofnum', 'body_dofadr', 'body_simple',
             'dof_parentid', 'flex_vertbodyid', 'flex_vertadr', 'flex_vertnum',
             'flex_dim', 'flex_interp', 'flex_edge', 'flex_edgeflap', 'flex_edgeflexid',
             'flex_elem', 'flex_elemflexid', 'flex_elemadr', 'flex_elemdataadr',
             'flex_elemedge', 'flex_elemedgeadr', 'flex_edgeadr')


@wp.kernel(enable_backward=False)
def _reduce_components(offset: wp.array[int], entries: wp.array[int],
                       source: wp.array3d[float], out: wp.array3d[float], skip: wp.array[bool],
                       cached: bool, valid: wp.array2d[bool], flexid: wp.array[int], status: wp.array[int]):
    # A scalar component has one owner. Scalar views avoid vector read/modify/
    # write races while spreading six/nine-component blocks across more threads.
    w, dst, component = wp.tid()
    if skip[w]:
        return
    if cached:
        if valid[w, flexid[dst]]:
            return
    total = Bins()
    for k in range(offset[dst], offset[dst+1]):
        total = add_record(total, source[w, entries[k], component])
    value = out[w, dst, component] + rounded_sum(total)
    if not wp.isfinite(value):
        wp.atomic_or(status, w, 1)
    out[w, dst, component] = value


def _components(array, width):
    if not array.is_contiguous:
        raise ValueError('owned reduction requires contiguous component storage')
    return array.view(dtype=float).reshape((array.shape[0],array.shape[1],width))


@wp.kernel(enable_backward=False)
def _project(offset: wp.array[int], entries: wp.array[int], vertices: wp.array[int],
             roots: wp.array[int], simple: wp.array[int],
             com: wp.array2d[wp.vec3], cdof: wp.array2d[wp.spatial_vector],
             xpos: wp.array2d[wp.vec3], values: wp.array2d[wp.vec3],
             scales: wp.array2d[float], per_edge: bool, scale: float,
             skip: wp.array[bool], out: wp.array2d[float], status: wp.array[int]):
    w, dof = wp.tid()
    if skip[w]:
        return
    total = Bins()
    motion = cdof[w, dof]
    for k in range(offset[dof], offset[dof+1]):
        slot = entries[k]
        vertex = vertices[slot]
        force = values[w, slot]
        # Preserve the host's zero-force early return (also avoids 0*NaN).
        if force[0] != 0.0 or force[1] != 0.0 or force[2] != 0.0:
            column = wp.spatial_bottom(motion)
            if simple[vertex] != 2:
                r = xpos[w, vertex] - com[w, roots[vertex]]
                column += wp.cross(wp.spatial_top(motion), r)
            coefficient = scale
            if per_edge:
                coefficient = scales[w, slot//4]
            # Local projection is host FP32 arithmetic; only the reduction uses binned arithmetic.
            total = add_record(total, coefficient * wp.dot(column, force))
    value = out[w, dof] + rounded_sum(total)
    if not wp.isfinite(value):
        wp.atomic_or(status, w, 1)
    out[w, dof] = value


def _checked_source(module, name):
    source = inspect.getsource(getattr(module, name))
    if hashlib.sha256(source.encode()).hexdigest() != _HASHES[name]:
        raise RuntimeError('unreviewed or already patched MJWarp ' + name)
    return source


def _compile(source, namespace, name):
    filename = '<deform3d_owned_material_' + hashlib.sha256(source.encode()).hexdigest() + '>'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(namespace)
    exec(compile(source, filename, 'exec'), namespace)
    return namespace[name]


def _replace(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('unexpected MJWarp material output site: ' + old)
    return source.replace(old, new)


def _factory(module, name, *, source=None, namespace=None):
    source = _checked_source(module, name) if source is None else source
    if name in ('_flex_elasticity', '_flex_bending'):
        slot = '4 * elemid + v' if name == '_flex_elasticity' else '4 * edgeid + i'
        for kind, suffix in (('spring', 's'), ('damper', 'd')):
            source = _replace(source,
                f'wp.atomic_add(flex_{kind}_body_force_out, worldid, bodyid, spatial_frc_{suffix})',
                f'flex_{kind}_body_force_out[worldid, {slot}] = spatial_frc_{suffix}')
    elif name == '_flex_hessian_elem':
        source = _replace(source, 'wp.atomic_add(flexedge_hessian_out, worldid, edge_id, block)',
                          'flexedge_hessian_out[worldid, 6 * elemid + e] = block')
    elif name == '_flex_hessian_diag':
        for i in range(2):
            source = _replace(source, f'wp.atomic_add(flexvert_hessian_out, worldid, va + vert[{i}], d{i})',
                              f'flexvert_hessian_out[worldid, 2 * edgeid + {i}] = d{i}')
    elif name == '_flex_hessian_mul_edge':
        source = _replace(source, 'wp.atomic_add(res_out, worldid, v0, scale * (block * y))',
                          'res_out[worldid, 2 * edgeid] = scale * (block * y)')
        source = _replace(source, 'wp.atomic_add(res_out, worldid, v1, scale * (wp.transpose(block) * x))',
                          'res_out[worldid, 2 * edgeid + 1] = scale * (wp.transpose(block) * x)')
    elif name == '_flex_bend_mul':
        source = _replace(source, '    res_out: wp.array2d[float],',
                          '    res_out: wp.array2d[wp.vec3],\n    scale_out: wp.array2d[float],')
        start = source.index('      flex_scatter_vert(\n')
        end = source.index('      )\n', start) + len('      )\n')
        source = source[:start] + ('      res_out[worldid, 4 * edgeid + i] = vw\n'
                                  '      scale_out[worldid, edgeid] = scale\n') + source[end:]
    # Any remaining floating-point atomic would violate the ownership contract.
    if 'wp.atomic_add(' in source or 'flex_scatter_vert(' in source:
        raise RuntimeError('unreplaced material scatter')
    # MJWarp caches factories by (module, function name, arguments), NOT source.
    # A distinct generated name prevents upstream atomic kernels from occupying
    # our private-slot cache entry, and prevents us poisoning upstream's cache.
    owned_name = '_deform3d_owned' + name + '_' + hashlib.sha256(source.encode()).hexdigest()[:12]
    source = _replace(source, 'def ' + name + '(', 'def ' + owned_name + '(')
    return _compile(source, vars(module) if namespace is None else namespace, owned_name)


class _Launches(ast.NodeTransformer):
    def __init__(self, hook):
        self.hook = hook
        self.count = 0

    def visit_Expr(self, node):
        node = self.generic_visit(node)
        call = node.value
        if not isinstance(call, ast.Call) or ast.unparse(call.func) != 'wp.launch':
            return node
        factory = call.args[0]
        if not isinstance(factory, ast.Call) or not isinstance(factory.func, ast.Name):
            return node
        name = factory.func.id
        # The adapter rejects nonzero edge spring/damper parameters. Eliminating
        # this identically-zero launch also avoids its nv-sized sorted scratch.
        if name == '_spring_damper_flexedge_passive':
            self.count += 1
            return ast.Pass()
        if name not in _FACTORIES:
            return node
        self.count += 1
        call.func = ast.parse('_owner._launch', mode='eval').body
        call.args = [ast.Constant(name), ast.Tuple(elts=factory.args, ctx=ast.Load()), ast.Name('d', ast.Load())]
        if self.hook in ('flex_bend_mul', 'flex_hessian_mul'):
            call.keywords.append(ast.keyword(arg='skip', value=ast.Name('skip', ast.Load())))
        return node


def _configured(module, name, owner):
    tree = ast.parse(_checked_source(module, name))
    transform = _Launches(name)
    tree = ast.fix_missing_locations(transform.visit(tree))
    expected = {'passive': 3, 'flex_hessian': 2, 'flex_hessian_mul': 1, 'flex_bend_mul': 1}[name]
    if transform.count != expected:
        raise RuntimeError('unexpected material launch count')
    return _compile(ast.unparse(tree)+'\n', dict(vars(module), _owner=owner), name)


class OwnedFlexOperators:
    """Fixed-topology CUDA 2D/3D vertex flexes; no rods/interpolation/edge springs.

    Construct outside capture, enter for warmup and graph construction, retain
    for replay. Model topology/edge parameters must stay immutable; rebuild and
    recapture after changes. One instance must not execute concurrently across
    streams. No host readback or allocation in the replaced operations. Sticky
    status 1 marks nonfinite gathered output and invalidates the run (no rollback).
    Other host operations retain their original determinism requirements.
    """
    def __init__(self, model, data):
        from mujoco_warp._src import passive
        self.model, self.nworld = model, data.nworld
        self.device = model.body_mass.device
        self._validate_model()
        self._data_shape = (model.nv, model.nbody, model.nflexvert, model.nflexedge, model.nflexelem)
        self._validate_data(data)
        self._topology = {name: getattr(model,name).numpy() for name in _TOPOLOGY}
        self._signature = self._fingerprint()
        self._check_module(passive)
        for name in _HASHES:
            _checked_source(passive, name)
        self.factories = {name: _factory(passive,name) for name in _FACTORIES}
        self.status = wp.zeros(self.nworld, dtype=int, device=self.device)
        self.skip = wp.zeros(self.nworld, dtype=bool, device=self.device)
        self._allocations = [self.status, self.skip]
        self._build_topology()
        from mujoco_warp._src.types import vec6
        specs = {
            '_flex_elasticity': (4*model.nflexelem, wp.spatial_vector, 6, False, self.element_bodies, 2),
            '_flex_bending': (4*model.nflexedge, wp.spatial_vector, 6, False, self.bend_bodies, 2),
            '_flex_hessian_elem': (6*model.nflexelem, wp.mat33, 9, True, self.element_edges, 1),
            '_flex_hessian_diag': (2*model.nflexedge, vec6, 6, False, self.edge_vertices, 1),
            '_flex_hessian_mul_edge': (2*model.nflexedge, wp.vec3, 3, False, self.edge_vertices, 1),
        }
        # Reject the total before allocating contribution storage.
        predicted = sum(a.capacity for a in self._allocations) + sum(
            self.nworld*n*width*4*count for n,_,width,_,_,count in specs.values())
        predicted += self.nworld*model.nflexedge*(4*12+4)
        if predicted > 256*1024**2:
            raise ValueError('owned flex workspace exceeds 256 MiB')
        self.buffers, self.reductions = {}, {}
        for name,(n,dtype,width,matrix,csr,count) in specs.items():
            self.buffers[name] = [self._zeros((self.nworld,n),dtype) for _ in range(count)]
            self.reductions[name] = (width,csr)
        self.bend_values = self._zeros((self.nworld,4*model.nflexedge),wp.vec3)
        self.bend_scale = self._zeros((self.nworld,model.nflexedge),float)
        self.buffers['_flex_bend_mul'] = [self.bend_values,self.bend_scale]
        self.workspace_bytes = sum(a.capacity for a in self._allocations)
        self._configured = {name:_configured(passive,name,self) for name in _HOOKS if name != 'flex_scatter'}
        self._signatures = {name:inspect.signature(getattr(passive,name)) for name in _HOOKS}

    @staticmethod
    def _check_module(module):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != _MODULE_HASH:
            raise RuntimeError('unreviewed MJWarp passive module or constitutive helpers')

    def _zeros(self, shape, dtype):
        a = wp.zeros(shape,dtype=dtype,device=self.device)
        self._allocations.append(a)
        return a

    def _upload(self, values):
        host = np.asarray(values,dtype=np.int32)
        if host.nbytes + sum(a.capacity for a in self._allocations) > 256*1024**2:
            raise ValueError('owned flex incidence exceeds 256 MiB')
        a = wp.array(host,dtype=int,device=self.device)
        self._allocations.append(a)
        return a

    def _csr(self, targets, count):
        lists = [[] for _ in range(count)]
        for slot, target in enumerate(targets):
            if target >= 0:
                if target >= count:
                    raise ValueError('invalid flex incidence target')
                lists[target].append(slot)
        return self._lists(lists)

    def _lists(self, lists):
        offsets = np.zeros(len(lists)+1,np.int32)
        offsets[1:] = np.cumsum([len(v) for v in lists],dtype=np.int64)
        return self._upload(offsets), self._upload([x for row in lists for x in row])

    def _fingerprint(self):
        h = hashlib.sha256()
        for name in _TOPOLOGY:
            a = getattr(self.model,name).numpy()
            h.update(name.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        return h.digest()

    def _validate_model(self):
        from mujoco_warp._src.types import EnableBit
        m = self.model
        shape = (m.nv,m.nbody,m.nflexvert,m.nflexedge,m.nflexelem)
        if hasattr(self,'_data_shape') and shape != self._data_shape:
            raise ValueError('model dimensions changed; recreate owned flex operators')
        if version('mujoco-warp') != '4.0.1':
            raise RuntimeError('reviewed only for mujoco-warp 4.0.1')
        if (not self.device.is_cuda or m.nflex < 1 or m.nv < 1 or self.nworld < 1
            or np.any(~np.isin(m.flex_dim.numpy(),[2,3])) or np.any(m.flex_interp.numpy()!=0)
            or m.opt.enableflags & EnableBit.SLEEP
            or np.any(m.flex_edgestiffness.numpy()!=0) or np.any(m.flex_edgedamping.numpy()!=0)):
            raise ValueError('requires CUDA vertex triangle/tet flexes, no sleep, rods, interpolation or edge springs/dampers')

    def _validate_data(self, d):
        m = self.model
        for name,shape,dtype in (
            ('qpos',(self.nworld,m.nq),float), ('qvel',(self.nworld,m.nv),float),
            ('cdof',(self.nworld,m.nv),wp.spatial_vector),
            ('flexvert_xpos',(self.nworld,m.nflexvert),wp.vec3),
            ('subtree_com',(self.nworld,m.nbody),wp.vec3),
            ('flex_hessian_valid',(self.nworld,m.nflex),bool)):
            a = getattr(d,name)
            if d.nworld != self.nworld or a.shape != shape or a.dtype != ({float: wp.float32, int: wp.int32, bool: wp.bool}.get(dtype, dtype)) or a.device != self.device:
                raise ValueError('data shape/type/device changed; recreate owned flex operators')

    def _build_topology(self):
        m,a = self.model,self._topology
        n = m.nflexvert
        roots = np.zeros(n,np.int32); simple = roots.copy()
        support = []
        for v,bodyid in enumerate(a['flex_vertbodyid']):
            if not 0 <= bodyid < m.nbody:
                raise ValueError('invalid flex vertex body')
            body = int(a['body_weldid'][bodyid])
            if not 0 <= body < m.nbody:
                raise ValueError('invalid welded body')
            roots[v], simple[v] = a['body_rootid'][body], a['body_simple'][body]
            if not 0 <= roots[v] < m.nbody:
                raise ValueError('invalid body root')
            count, start = int(a['body_dofnum'][body]), int(a['body_dofadr'][body])
            if count < 0 or (count and not 0 <= start <= m.nv-count):
                raise ValueError('invalid body DOFs')
            ids = []
            if count:
                if simple[v] == 2:
                    ids = list(range(start,start+count))
                else:
                    i = start+count-1
                    while i >= 0:
                        if i >= m.nv or i in ids:
                            raise ValueError('invalid DOF ancestor chain')
                        ids.append(i); i = int(a['dof_parentid'][i])
            support.append(ids)
        def project_csr(vertices):
            lists = [[] for _ in range(m.nv)]
            for slot,v in enumerate(vertices):
                if v >= 0:
                    if v >= n:
                        raise ValueError('invalid flex vertex')
                    for i in support[v]: lists[i].append(slot)
            return self._lists(lists)
        self.roots, self.simple = self._upload(roots), self._upload(simple)
        self.vertices = self._upload(np.arange(n))
        self.project_vertices = project_csr(np.arange(n))
        edgeverts = np.full(2*m.nflexedge,-1,np.int32)
        bendverts = np.full(4*m.nflexedge,-1,np.int32)
        for e,f in enumerate(a['flex_edgeflexid']):
            if not 0 <= f < m.nflex: raise ValueError('invalid edge flex')
            va = int(a['flex_vertadr'][f]); nv = int(a['flex_vertnum'][f])
            edge = a['flex_edge'][e]
            if np.any(edge < 0) or np.any(edge >= nv): raise ValueError('invalid edge vertices')
            edgeverts[2*e:2*e+2] = va+edge
            flap = a['flex_edgeflap'][e]
            if a['flex_dim'][f] == 2 and flap[1] != -1:
                if np.any(flap < 0) or np.any(flap >= nv): raise ValueError('invalid edge flap')
                bendverts[4*e:4*e+4] = va+np.r_[edge,flap]
        elem_bodies = np.full(4*m.nflexelem,-1,np.int32)
        elem_edges = np.full(6*m.nflexelem,-1,np.int32)
        for e,f in enumerate(a['flex_elemflexid']):
            if not 0 <= f < m.nflex: raise ValueError('invalid element flex')
            dim = int(a['flex_dim'][f]); local = e-int(a['flex_elemadr'][f])
            va = int(a['flex_vertadr'][f]); nv = int(a['flex_vertnum'][f])
            start = int(a['flex_elemdataadr'][f])+local*(dim+1)
            verts = a['flex_elem'][start:start+dim+1]
            if len(verts)!=dim+1 or np.any(verts<0) or np.any(verts>=nv): raise ValueError('invalid element vertices')
            elem_bodies[4*e:4*e+dim+1] = a['flex_vertbodyid'][va+verts]
            ne = 3 if dim==2 else 6
            start = int(a['flex_elemedgeadr'][f])+local*ne
            edges = a['flex_elemedge'][start:start+ne] + a['flex_edgeadr'][f]
            if len(edges)!=ne or np.any(edges<0) or np.any(edges>=m.nflexedge): raise ValueError('invalid element edges')
            elem_edges[6*e:6*e+ne] = edges
        bend_bodies = np.full_like(bendverts,-1)
        active = bendverts>=0
        bend_bodies[active] = a['flex_vertbodyid'][bendverts[active]]
        self.bend_vertices = self._upload(bendverts)
        self.project_bend = project_csr(bendverts)
        self.element_bodies = self._csr(elem_bodies,m.nbody)
        self.bend_bodies = self._csr(bend_bodies,m.nbody)
        self.element_edges = self._csr(elem_edges,m.nflexedge)
        self.edge_vertices = self._csr(edgeverts,n)

    def _array(self, a, shape, dtype):
        if a.shape != shape or a.dtype != ({float: wp.float32, int: wp.int32, bool: wp.bool}.get(dtype, dtype)) or a.device != self.device or not a.is_contiguous:
            raise ValueError('invalid owned flex operator array')

    def _skip(self, skip):
        if skip is None: return self.skip
        self._array(skip,(self.nworld,),bool)
        return skip

    def scatter(self, m, d, res, vec, scale=1.0, skip=None):
        if m is not self.model:
            raise ValueError('owned flex workspace belongs to a different model')
        self._validate_data(d)
        self._array(res,(self.nworld,m.nv),float)
        self._array(vec,(self.nworld,m.nflexvert),wp.vec3)
        if scale == 0.0: return
        wp.launch(_project,dim=(self.nworld,m.nv),inputs=[
            *self.project_vertices,self.vertices,self.roots,self.simple,d.subtree_com,d.cdof,
            d.flexvert_xpos,vec,self.bend_scale,False,float(scale),self._skip(skip),res,self.status],device=self.device)

    def _launch(self, name, args, d, dim, inputs, outputs, skip=None):
        skip = self._skip(skip)
        buffers = self.buffers[name]
        for a in buffers: a.zero_()  # Early returns must never expose stale slots.
        args = list(args); args[-1] = False  # Contributions have no atomics.
        wp.launch(self.factories[name](*args),dim=dim,inputs=inputs,outputs=buffers,device=self.device)
        if name == '_flex_bend_mul':
            wp.launch(_project,dim=(self.nworld,self.model.nv),inputs=[
                *self.project_bend,self.bend_vertices,self.roots,self.simple,d.subtree_com,d.cdof,
                d.flexvert_xpos,self.bend_values,self.bend_scale,True,1.0,skip,outputs[0],self.status],device=self.device)
        else:
            width,csr = self.reductions[name]
            cached = name in ('_flex_hessian_elem','_flex_hessian_diag')
            flexid = self.model.flex_edgeflexid if name=='_flex_hessian_elem' else self.model.flex_vertflexid
            for source,out in zip(buffers,outputs):
                wp.launch(_reduce_components,dim=(self.nworld,out.shape[1],width),inputs=[*csr,
                    _components(source,width),_components(out,width),skip,cached,
                    d.flex_hessian_valid,flexid,self.status],device=self.device)

    @contextmanager
    def capture_policy(self):
        global _ACTIVE
        from mujoco_warp._src import passive
        with _LOCK:
            if _ACTIVE: raise RuntimeError('owned flex setup is not reentrant')
            self._validate_model()
            if self._fingerprint() != self._signature:
                raise RuntimeError('model topology changed; recreate and recapture')
            self._check_module(passive)
            for name in _HASHES: _checked_source(passive,name)
            originals = {name:getattr(passive,name) for name in _HOOKS}
            def dispatch(name,original):
                def call(m,d,*args,**kwargs):
                    if m is not self.model: return original(m,d,*args,**kwargs)
                    self._validate_data(d)
                    if name in ('flex_bend_mul','flex_hessian_mul'):
                        bound = self._signatures[name].bind(m,d,*args,**kwargs)
                        bound.apply_defaults()
                        dtype = float if name=='flex_bend_mul' else wp.vec3
                        count = m.nv if name=='flex_bend_mul' else m.nflexvert
                        for key in ('res','vec'):
                            self._array(bound.arguments[key],(self.nworld,count),dtype)
                        if bound.arguments['res'].ptr == bound.arguments['vec'].ptr:
                            raise ValueError('flex product input and output must not alias')
                        self._skip(bound.arguments['skip'])
                    fn = self.scatter if name=='flex_scatter' else self._configured[name]
                    return fn(m,d,*args,**kwargs)
                return call
            _ACTIVE = True
            try:
                for name,original in originals.items(): setattr(passive,name,dispatch(name,original))
                yield self
            finally:
                for name,original in originals.items(): setattr(passive,name,original)
                _ACTIVE = False
