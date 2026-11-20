#pragma once

// Optional host contact observation. No forces are fed back to the simulator.
#include "deform3d.hpp"
#include <mujoco/mujoco.h>
#include <cmath>
#include <cstddef>

namespace deform3d {

struct MujocoContactWrench {
  Vec3<mjtNum> force{};
  Vec3<mjtNum> torque{};  // about the supplied world-space origin
  std::size_t contacts = 0;
};

// Net world-frame contact wrench on one rigid geom. Includes normal, friction,
// torsional and rolling contact terms. Opposing contacts cancel algebraically;
// summing magnitudes instead would overestimate a gripper's load-cell reading.
//
// mj_step leaves contacts/forces at its preintegration state. Use mj_forward
// first when a measurement at the newly integrated state is required. This
// function never advances/reforwards the simulation or allocates frame memory.
// It reports contact contribution only, excluding gravity, actuation and other
// externally applied loads. The caller owns model/data lifetime and locking.
inline Status mujoco_contact_wrench(const mjModel* model, const mjData* data,
                                   int geom, Vec3<mjtNum> origin,
                                   MujocoContactWrench& output) noexcept {
  if (!model || !data || geom < 0 || geom >= model->ngeom ||
      mjVERSION_HEADER != mj_version() || data->ncon < 0 ||
      (data->ncon && !data->contact)) return Status::InvalidArgument;
  if (!finite(origin)) return Status::NonFinite;
  MujocoContactWrench result{};
  for (int i=0; i<data->ncon; ++i) {
    const mjContact& contact=data->contact[i];
    const int sign=(contact.geom[1]==geom ? 1 : 0)-(contact.geom[0]==geom ? 1 : 0);
    if (!sign || contact.efc_address<0) continue;
    mjtNum local[6]{};
    mj_contactForce(model,data,i,local);
    Vec3<mjtNum> f{}, t{};
    mjtNum* fv[3]={&f.x,&f.y,&f.z};
    mjtNum* tv[3]={&t.x,&t.y,&t.z};
    for (int j=0;j<3;++j) for (int k=0;k<3;++k) {
      *fv[j]+=sign*contact.frame[3*k+j]*local[k];
      *tv[j]+=sign*contact.frame[3*k+j]*local[k+3];
    }
    const Vec3<mjtNum> p{contact.pos[0],contact.pos[1],contact.pos[2]};
    if (!finite(f) || !finite(t) || !finite(p)) return Status::NonFinite;
    result.force+=f;
    result.torque+=t+cross(p-origin,f);
    ++result.contacts;
  }
  if (!finite(result.force) || !finite(result.torque)) return Status::NonFinite;
  output=result;  // transactional on invalid/non-finite input
  return Status::Ok;
}

} // namespace deform3d
