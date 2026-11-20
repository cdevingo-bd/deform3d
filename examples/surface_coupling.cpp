// c++ -std=c++17 -O3 -Iinclude examples/surface_coupling.cpp -o surface_coupling
// Host-independent attachment: a point 1 cm above a physical triangle.
#include <deform3d/surface_coupling.hpp>
#include <cstdio>

int main() {
  using V = deform3d::Vec3<double>;
  using S = deform3d::SurfaceEmbedStatus;
  const V rest[] = {{0,0,0}, {1,0,0}, {0,1,0}};
  const std::uint32_t triangles[] = {0,1,2};
  const V dense_rest[] = {{.25,.25,.01}};
  deform3d::SurfaceEmbedOptions<double> options;
  options.max_distance = .02;
  deform3d::SurfaceEmbedding<double> embedding;
  if (embedding.prepare(rest,3,triangles,1,dense_rest,1,options) != S::Ok) return 1;
  deform3d::SurfaceCoupling<double> coupling;
  if (coupling.prepare(embedding) != S::Ok) return 2;

  // Current world-space state comes from the host simulator. Storage is reused
  // each frame. Update once, then evaluate disjoint render ranges if desired.
  const V positions[] = {{0,0,0}, {1,0,.2}, {0,1,0}};
  const V velocities[] = {{0,0,0}, {0,0,.1}, {0,0,0}};
  V dense[1], dense_velocity[1], nodal_force[3];
  if (coupling.update(positions,3) != S::Ok ||
      coupling.evaluate_positions(dense,1) != S::Ok ||
      coupling.evaluate_velocities(velocities,3,dense_velocity,1) != S::Ok) return 3;

  // Example external load at the current dense point. The host applies the
  // returned nodal forces through its DOF mapping and advances its physics.
  // Do not feed back contacts that the simulator has already applied.
  const V external_force[] = {{0,3,-2}};
  if (coupling.transfer_forces(external_force,1,nodal_force,3) != S::Ok) return 4;
  std::printf("dense position: %.9f %.9f %.9f\n",dense[0].x,dense[0].y,dense[0].z);
  std::printf("dense velocity: %.9f %.9f %.9f\n",dense_velocity[0].x,dense_velocity[0].y,dense_velocity[0].z);
  for (int i=0;i<3;++i)
    std::printf("node %d force: %.9f %.9f %.9f\n",i,nodal_force[i].x,nodal_force[i].y,nodal_force[i].z);
}
