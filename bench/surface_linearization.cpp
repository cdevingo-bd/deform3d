#include <deform3d/surface_linearization.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>

using namespace deform3d;
using V=Vec3<float>;
using S=SurfaceEmbedStatus;
using Clock=std::chrono::steady_clock;
double milliseconds(Clock::time_point a,Clock::time_point b) {return std::chrono::duration<double,std::milli>(b-a).count();}
double median(std::vector<double> v) {std::sort(v.begin(),v.end());return v[v.size()/2];}
int main(int argc,char** argv) {
  const std::size_t points=argc>1?std::strtoull(argv[1],nullptr,10):100000;
  if(!points || points>10000000) return 2;
  constexpr std::size_t side=16,vertices=side*side,triangles=2*(side-1)*(side-1);
  std::vector<V> rest(vertices),x(vertices),u(vertices),reference(vertices),product(vertices);
  std::vector<std::uint32_t> indices;
  for(std::size_t y=0;y<side;++y)for(std::size_t z=0;z<side;++z) {
    const auto id=y*side+z;rest[id]={float(z)/float(side-1),float(y)/float(side-1),0};
    x[id]={rest[id].x,rest[id].y,.2f*rest[id].x-.1f*rest[id].y};
    u[id]={.1f+rest[id].y,.2f-rest[id].x,.3f*rest[id].x+.1f*rest[id].y};
    if(y+1<side && z+1<side) {
      for(auto i:{id,id+1,id+side,id+1,id+side+1,id+side})indices.push_back(std::uint32_t(i));
    }
  }
  std::vector<V> dense(points),f(points),velocity(points),df(points);
  std::vector<Mat3<float>> k(points);
  for(std::size_t i=0;i<points;++i) {
    const auto t=i%triangles;
    dense[i]=rest[indices[3*t]]*.2f+rest[indices[3*t+1]]*.3f+rest[indices[3*t+2]]*.5f;
    f[i]=V{.1f,-.2f,.4f}/float(points);
    k[i].m={{-3.f,.2f,-.3f,.2f,-2.f,.1f,-.3f,.1f,-1.f}};
    for(auto& value:k[i].m)value/=float(points);
  }
  SurfaceEmbedding<float> embedding;SurfaceEmbedOptions<float> options;
  options.outside=SurfaceOutsidePolicy::ProjectToSurface;options.max_distance=1e-6f;
  if(embedding.prepare(rest.data(),vertices,indices.data(),triangles,dense.data(),points,options)!=S::Ok)return 3;
  SurfaceCoupling<float> coupling;SurfaceLinearization<float> op;
  if(coupling.prepare(embedding)!=S::Ok || op.prepare(embedding)!=S::Ok)return 4;
  if(coupling.update(x.data(),vertices)!=S::Ok)return 5;
  auto direct=[&] {
    if(coupling.evaluate_velocities(u.data(),vertices,velocity.data(),points)!=S::Ok)std::abort();
    for(std::size_t i=0;i<points;++i) {
      df[i]=k[i].transform(velocity[i]);
      if(!finite(df[i]))std::abort();
    }
    if(coupling.transfer_forces(df.data(),points,reference.data(),vertices)!=S::Ok)std::abort();
  };
  auto condensed=[&] {if(op.apply(u.data(),vertices,product.data(),vertices)!=S::Ok)std::abort();};
  std::vector<double> assembly,old_time,new_time;
  for(int trial=0;trial<9;++trial) {
    auto begin=Clock::now();
    if(op.update(x.data(),vertices,f.data(),k.data(),points)!=S::Ok)return 6;
    auto end=Clock::now();if(trial>=2)assembly.push_back(milliseconds(begin,end));
    for(int mode=0;mode<2;++mode) {
      const bool compressed=((trial+mode)%2)==0;
      begin=Clock::now();
      for(int j=0;j<20;++j) {
        // Vary the direction on every application, including the reference.
        // This prevents a compiler from hoisting repeated identical work.
        u[0].z=.001f*float(j+1);
        if(compressed)condensed();else direct();
      }
      end=Clock::now();
      if(trial>=2)(compressed?new_time:old_time).push_back(milliseconds(begin,end)/20);
    }
  }
  double error=0,scale=0;
  for(std::size_t i=0;i<vertices;++i){error=std::max(error,double(norm(product[i]-reference[i])));scale=std::max(scale,double(norm(reference[i])));}
  const double a=median(assembly),o=median(old_time),n=median(new_time);
  std::printf("{\"dense_points\":%zu,\"physical_vertices\":%zu,\"active_triangles\":%zu,\"update_median_ms\":%.6f,\"direct_apply_median_ms\":%.6f,\"condensed_apply_median_ms\":%.6f,\"update_plus_20_applies_ms\":%.6f,\"direct_20_applies_ms\":%.6f,\"max_relative_output_error\":%.9g,\"scope\":\"single-thread C++17; zero-offset local force derivative, 7 retained samples, alternating order; full input/output checks; setup, host physics and rendering excluded; update cost included separately\"}\n",points,vertices,op.active_triangle_count(),a,o,n,a+20*n,20*o,error/std::max(scale,1e-30));
  return error<=2e-5*std::max(scale,1e-30)?0:7;
}
