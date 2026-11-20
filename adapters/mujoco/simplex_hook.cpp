// Optional MuJoCo 4.0.1 source hook. Native collision, integration, bending and
// projected material stiffness remain in MuJoCo. No process-global model edits.
#include <deform3d/simplex_material.hpp>
#include <mujoco/mujoco.h>
#include <cmath>

namespace {
thread_local const mjModel* enabled_model=nullptr;
thread_local unsigned long long calls=0;
}

extern "C" {
// One model per calling thread; pass nullptr to disable. Validate once, then
// keep topology/material type immutable. Different threads may select models.
MJAPI int deform3d_simplex_enable(const mjModel* m) {
  if (!m) { enabled_model=nullptr; calls=0; return 1; }
  if (mj_version()!=4000001 || mjVERSION_HEADER!=4000001 ||
      m->opt.enableflags & mjENBL_SLEEP) return 0;
  for(int f=0;f<m->nflex;++f) {
    if((m->flex_dim[f]!=2 && m->flex_dim[f]!=3) || m->flex_interp[f] ||
       m->flex_plasticadr[f]>=0 || !std::isfinite(m->flex_damping[f]) ||
       m->flex_damping[f]<0) return 0;
    const int adr=m->flex_stiffnessadr[f];
    if(adr>=0 && m->flex_dim[f]==3)
      for(int e=0;e<m->flex_elemnum[f];++e)
        if(m->flex_stiffness[adr+24*e+21]!=0) return 0;
  }
  enabled_model=m; calls=0; return 1;
}
MJAPI int deform3d_simplex_enabled(const mjModel* m) { return m==enabled_model; }
MJAPI unsigned long long deform3d_simplex_calls() { return calls; }
MJAPI void deform3d_simplex_response(int dim, const mjtNum* x,
    const mjtNum* length, const mjtNum* rest, const mjtNum* speed,
    const mjtNum* packed, mjtNum dt, mjtNum damping, int spring, int damper,
    mjtNum* fs, mjtNum* fd) {
  const auto response=deform3d::SimplexMaterial<mjtNum>::evaluate(
      dim,x,length,rest,speed,packed,dt,damping,spring!=0,damper!=0);
  for(int i=0;i<3*(dim+1);++i) { fs[i]=response.spring[i]; fd[i]=response.damper[i]; }
  ++calls;
}
}
