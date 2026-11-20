"""Shared C++ simplex constitutive kernel inside MJWarp 4.0.1.

This replaces the stretch force/damper arithmetic. Bending, collision, host
projected stiffness and integrator remain native; output reductions use the
existing deterministic OwnedFlexOperators. No host readbacks during stepping.
"""
from pathlib import Path
import hashlib
import numpy as np
import warp as wp
from .mujoco_material import OwnedFlexOperators, _checked_source, _factory, _compile, _replace
from contextlib import contextmanager
from importlib.metadata import version
import threading

mat43=wp.types.matrix(shape=(4,3),dtype=wp.float32)
mat83=wp.types.matrix(shape=(8,3),dtype=wp.float32)
vec6=wp.types.vector(length=6,dtype=wp.float32)
from .resources import simplex_material_header

HEADER = simplex_material_header()


@wp.func_native('''
float x[12], l[6], r[6], v[6], c[21];
for(int i=0;i<4;++i) for(int j=0;j<3;++j) x[3*i+j]=position.data[i][j];
for(int i=0;i<6;++i) { l[i]=length[i]; r[i]=rest[i]; v[i]=speed[i]; }
const int ne=dim==2?3:6;
for(int i=0;i<ne*(ne+1)/2;++i) c[i]=packed[adr+i];
const auto response=deform3d::SimplexMaterial<float>::evaluate(
    dim,x,l,r,v,c,dt,rayleigh,spring_enabled,damper_enabled);
wp::mat_t<8,3,float> out;
for(int i=0;i<4;++i) for(int j=0;j<3;++j) {
  out.data[i][j]=response.spring[3*i+j];
  out.data[4+i][j]=response.damper[3*i+j];
}
return out;
''')
def element_response(dim:int, position:mat43, length:vec6, rest:vec6, speed:vec6,
                     packed:wp.array[float], adr:int, dt:float, rayleigh:float,
                     spring_enabled:bool, damper_enabled:bool)->mat83:
    ...


def _validate(model):
    if version('mujoco-warp')!='4.0.1':
        raise RuntimeError('shared simplex adapter requires MJWarp 4.0.1')
    if np.any(~np.isin(model.flex_dim.numpy(),[2,3])) or np.any(model.flex_interp.numpy()!=0):
        raise ValueError('shared simplex requires triangle/tet vertex flexes')
    if np.any(model.flex_plasticadr.numpy()>=0):
        raise ValueError('shared simplex law does not implement plasticity')
    dims=model.flex_dim.numpy(); adr=model.flex_stiffnessadr.numpy()
    stiffness=model.flex_stiffness.numpy(); counts=model.flex_elemnum.numpy()
    for f in range(model.nflex):
        if dims[f]==3 and adr[f]>=0:
            if np.any(stiffness[adr[f]+21:adr[f]+24*counts[f]:24]!=0):
                raise ValueError('shared simplex law does not implement stable Neo-Hookean elasticity')


def _make_factory(owned,header_path=HEADER):
    from mujoco_warp._src import passive
    header_path=Path(header_path).resolve()
    header=header_path.read_text()
    source=_checked_source(passive,'_flex_elasticity')
    first=source.index('    elongation_spring = ')
    end=source.index('    for v in range(4):\n      if v < nvert:',first)
    replacement='''    lengths = vec6()
    rest = vec6()
    speeds = vec6()
    for e in range(nedge):
      eid = edge_adr + flex_elemedge[elemedge_adr + e]
      lengths[e] = flexedge_length_in[worldid, eid]
      rest[e] = flexedge_length0[eid]
      speeds[e] = flexedge_velocity_in[worldid, eid]
    response = element_response(dim, vert_xpos, lengths, rest, speeds,
        flex_stiffness, stiffness_adr, timestep, flex_damping[f], not dsbl_spring, not dsbl_damper)
    force_spring = mat43()
    force_damper = mat43()
    for v in range(4):
      force_spring[v] = response[v]
      force_damper[v] = response[4 + v]

'''
    source=source[:first]+replacement+source[end:]
    # Owned factory cache keys must also distinguish different installed core
    # headers; otherwise two policies could share one mutable module preamble.
    source+='\n# shared core sha256: '+hashlib.sha256(header.encode()).hexdigest()+'\n'
    namespace=dict(vars(passive),element_response=element_response)
    if owned:
        factory=_factory(passive,'_flex_elasticity',source=source,namespace=namespace)
    else:
        name='_deform3d_shared_simplex_'+hashlib.sha256((source+header).encode()).hexdigest()[:12]
        source=_replace(source,'def _flex_elasticity(','def '+name+'(')
        factory=_compile(source,namespace,name)
    kernels={}
    for deterministic in ((False,) if owned else (False,True)):
        kernel=factory(deterministic)
        wp.set_module_options({'extra_build_options':wp.ModuleBuildOptions(
            extra_cuda_preamble='#define DEFORM3D_ELEMENT_INLINE __host__ __device__ inline\n'+header,
            extra_cpu_preamble='#define DEFORM3D_ELEMENT_INLINE inline\n'+header,
            extra_build_dependencies=[header_path])},module=kernel.module)
        kernels[deterministic]=kernel
    return lambda deterministic=False:kernels[False if owned else bool(deterministic)]


class SimplexFlexOperators(OwnedFlexOperators):
    """Shared element law plus the existing owned deterministic reductions."""
    def _validate_model(self):
        super()._validate_model()
        _validate(self.model)

    def __init__(self,model,data,*,header_path=HEADER):
        super().__init__(model,data)
        self.core_sha256=hashlib.sha256(Path(header_path).read_bytes()).hexdigest()
        self.factories['_flex_elasticity']=_make_factory(True,header_path)


_LOCK=threading.RLock()
_ACTIVE=False
class SharedSimplexPolicy:
    """Replace ONLY the constitutive kernel; retain native host reductions.

    Scope is for single-threaded warmup/capture, then retain for graph replay.
    The host factory is process-global while entering the scope. No concurrent
    model warmup/capture, no topology/material-type mutation after construction.
    """
    def __init__(self,model,*,header_path=HEADER):
        _validate(model)
        self.model=model
        self.core_sha256=hashlib.sha256(Path(header_path).read_bytes()).hexdigest()
        self.factory=_make_factory(False,header_path)

    @contextmanager
    def capture_policy(self):
        global _ACTIVE
        from mujoco_warp._src import passive
        with _LOCK:
            if _ACTIVE:raise RuntimeError('shared simplex setup is not reentrant')
            _validate(self.model)
            _checked_source(passive,'_flex_elasticity')
            original=passive._flex_elasticity
            _ACTIVE=True
            try:
                passive._flex_elasticity=self.factory
                yield
            finally:
                passive._flex_elasticity=original
                _ACTIVE=False
