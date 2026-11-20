#include <deform3d/surface_coupling.hpp>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"CHECK %s:%d: %s\n",__FILE__,__LINE__,#x); std::abort(); } } while(false)
static std::size_t allocations=0;
void* operator new(std::size_t n) { ++allocations; if(auto p=std::malloc(n?n:1)) return p; throw std::bad_alloc(); }
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
void operator delete[](void* p,std::size_t) noexcept { std::free(p); }
using namespace deform3d;
template<class T> bool near(Vec3<T> a,Vec3<T> b,T eps) { return norm(a-b)<=eps*(T(1)+norm(a)+norm(b)); }

template<class T> void physical_identities() {
  using V=Vec3<T>; using S=SurfaceEmbedStatus;
  const V rest[5]={{0,0,0},{2,0,0},{0,2,0},{2,2,0},{9,9,9}};
  const std::uint32_t triangles[6]={0,1,2,1,3,2};
  const V points[4]={{T(.2),T(.3),T(.25)},{T(1.4),T(1.6),T(-.2)},
                     {T(.7),T(.5),0},{T(1.2),T(.4),T(.3)}};
  SurfaceEmbedOptions<T> options; options.max_distance=T(.4);
  SurfaceEmbedding<T> embedding;
  CHECK(embedding.prepare(rest,5,triangles,2,points,4,options)==S::Ok);
  SurfaceCoupling<T> coupling;
  CHECK(coupling.prepare(embedding)==S::Ok);
  V current[5],velocity[5],render_velocity[4],render[4],reference[4],force[5];
  const V loads[4]={{1,2,3},{-3,1,2},{4,-2,1},{-2,1,-4}};
  for(std::size_t i=0;i<5;++i) {
    current[i]={T(1)+T(.8)*rest[i].x,T(2)+T(1.3)*rest[i].y,T(.4)+T(.2)*rest[i].x-T(.3)*rest[i].y};
    velocity[i]={T(.1)*T(i+1),T(-.2)+T(.3)*T(i),T(.05)*T(i*i)};
  }
  CHECK(coupling.evaluate_positions(render,4)==S::InvalidInput);
  const auto before=allocations;
  CHECK(coupling.update(current,5)==S::Ok);
  CHECK(coupling.evaluate_positions(render,4)==S::Ok);
  CHECK(embedding.evaluate(current,5,reference,4)==S::Ok);
  CHECK(coupling.evaluate_velocities(velocity,5,render_velocity,4)==S::Ok);
  CHECK(coupling.transfer_forces(loads,4,force,5)==S::Ok);
  CHECK(allocations==before);
  const T eps=std::is_same<T,float>::value ? T(2e-5) : T(2e-12);
  for(int i=0;i<4;++i) CHECK(near(render[i],reference[i],eps));
  V total_source{},total_render{},torque_source{},torque_render{};
  T work_source=0,work_render=0;
  for(int i=0;i<5;++i) { total_source+=force[i];torque_source+=cross(current[i],force[i]);work_source+=dot(force[i],velocity[i]); }
  for(int i=0;i<4;++i) { total_render+=loads[i];torque_render+=cross(render[i],loads[i]);work_render+=dot(loads[i],render_velocity[i]); }
  CHECK(near(total_source,total_render,eps));
  CHECK(near(torque_source,torque_render,eps));
  CHECK(std::abs(work_source-work_render)<eps*(T(1)+std::abs(work_source)));
  CHECK(norm(force[4])==T(0)); // unused simulation vertex
  // Independent directional finite differences of the original position map.
  const T h=std::is_same<T,float>::value ? T(.001) : T(1e-5);
  V plus[5],minus[5],rp[4],rm[4];
  for(int i=0;i<5;++i) { plus[i]=current[i]+velocity[i]*h;minus[i]=current[i]-velocity[i]*h; }
  CHECK(embedding.evaluate(plus,5,rp,4)==S::Ok);
  CHECK(embedding.evaluate(minus,5,rm,4)==S::Ok);
  for(int i=0;i<4;++i) CHECK(near(render_velocity[i],(rp[i]-rm[i])/(T(2)*h),
      std::is_same<T,float>::value ? T(2e-4) : T(2e-9)));
  // Rigid angular velocity must move offset points with the same angular speed.
  const V omega{T(.3),T(-.6),T(.4)}, translation{T(.1),T(.2),T(-.2)};
  for(int i=0;i<5;++i) velocity[i]=cross(omega,current[i])+translation;
  CHECK(coupling.evaluate_velocities(velocity,5,render_velocity,4)==S::Ok);
  for(int i=0;i<4;++i) CHECK(near(render_velocity[i],cross(omega,render[i])+translation,eps));
  V split[5];
  std::thread first([&]{CHECK(coupling.transfer_forces(loads,4,split,5,SurfaceForceMode::Overwrite,0,2)==S::Ok);});
  std::thread second([&]{CHECK(coupling.transfer_forces(loads,4,split,5,SurfaceForceMode::Overwrite,2,5)==S::Ok);});
  first.join();second.join();
  for(int i=0;i<5;++i) CHECK(near(split[i],force[i],eps));
  CHECK(coupling.transfer_forces(loads,4,split,5,SurfaceForceMode::Accumulate)==S::Ok);
  for(int i=0;i<5;++i) CHECK(near(split[i],force[i]*T(2),eps));
  CHECK(coupling.transfer_forces(loads,4,split,5,SurfaceForceMode::Overwrite,0,0)==S::Ok);
  CHECK(near(split[0],force[0]*T(2),eps));
}

static void failures_are_transactional() {
  using V=Vec3<double>; using S=SurfaceEmbedStatus;
  V rest[3]={{0,0,0},{1,0,0},{0,1,0}},points[2]={{.2,.3,.2},{.3,.2,-.1}};
  const std::uint32_t tri[3]={0,1,2};
  SurfaceEmbedding<double> embedding,unprepared;
  SurfaceEmbedOptions<double> options;options.max_distance=.3;
  CHECK(embedding.prepare(rest,3,tri,1,points,2,options)==S::Ok);
  SurfaceCoupling<double> coupling;
  CHECK(coupling.prepare(embedding)==S::Ok);
  CHECK(coupling.update(rest,3)==S::Ok);
  CHECK(coupling.prepare(unprepared)==S::InvalidInput);
  V collapse[3]={{0,0,0},{1,0,0},{2,0,0}};
  CHECK(coupling.update(collapse,3)==S::DegenerateTriangle);
  V output[3]={{7,8,9},{7,8,9},{7,8,9}},saved[3];std::memcpy(saved,output,sizeof output);
  V forces[2]={{1,2,3},{1,2,3}};
  forces[1].x=std::numeric_limits<double>::infinity();
  CHECK(coupling.transfer_forces(forces,2,output,3)==S::NonFiniteInput);
  CHECK(std::memcmp(saved,output,sizeof output)==0);
  CHECK(coupling.transfer_forces(output,3,output,3)==S::Aliasing);
  CHECK(coupling.evaluate_velocities(rest,3,rest,3)==S::Aliasing);
  CHECK(coupling.evaluate_positions(output,3,2,1)==S::InvalidRange);
  CHECK(coupling.transfer_forces(forces,2,output,3,SurfaceForceMode::Overwrite,0,4)==S::InvalidRange);
  CHECK(coupling.evaluate_positions(output,3)==S::Ok);
  CHECK(near(output[0],points[0],1e-12)); // failed update retained the old frame
  forces[0]={std::numeric_limits<double>::max(),0,0};forces[1]=forces[0];
  std::fill(output,output+3,forces[0]);std::memcpy(saved,output,sizeof output);
  CHECK(coupling.transfer_forces(forces,2,output,3,SurfaceForceMode::Accumulate)==S::NumericalFailure);
  CHECK(std::memcmp(saved,output,sizeof output)==0);
}

int main() { physical_identities<float>();physical_identities<double>();failures_are_transactional();std::puts("surface coupling passed"); }
