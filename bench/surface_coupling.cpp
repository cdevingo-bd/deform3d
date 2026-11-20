#include <deform3d/surface_coupling.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using V=deform3d::Vec3<float>;
using Clock=std::chrono::steady_clock;
int main(int argc,char** argv) {
  const std::size_t count=argc>1 ? std::strtoull(argv[1],nullptr,10) : 1000000;
  if(!count) return 2;
  const V source[4]={{0,0,0},{1,0,0},{0,1,0},{1,1,0}};
  const std::uint32_t triangles[6]={0,1,2,1,3,2};
  std::vector<V> rest(count),reference(count),cached(count);
  for(std::size_t i=0;i<count;++i) rest[i]={float((i*73)%1001)/1000,float((i*37)%1001)/1000,.002f};
  deform3d::SurfaceEmbedding<float> embedding;
  deform3d::SurfaceEmbedOptions<float> options;options.max_distance=.003f;
  using S=deform3d::SurfaceEmbedStatus;
  if(embedding.prepare(source,4,triangles,2,rest.data(),count,options)!=S::Ok) return 3;
  deform3d::SurfaceCoupling<float> coupling;
  if(coupling.prepare(embedding)!=S::Ok) return 4;
  std::vector<double> old_times,new_times;
  float max_delta=0;
  for(int trial=0;trial<14;++trial) {
    V current[4];for(int i=0;i<4;++i) current[i]={source[i].x,source[i].y,.02f*float(trial)*source[i].x};
    auto a=Clock::now();
    if(embedding.evaluate(current,4,reference.data(),count)!=S::Ok) return 5;
    auto b=Clock::now();
    if(coupling.update(current,4)!=S::Ok || coupling.evaluate_positions(cached.data(),count)!=S::Ok) return 6;
    auto c=Clock::now();
    if(trial>=2) {
      old_times.push_back(std::chrono::duration<double,std::milli>(b-a).count());
      new_times.push_back(std::chrono::duration<double,std::milli>(c-b).count());
    }
    for(std::size_t i=0;i<count;++i) max_delta=std::max(max_delta,deform3d::norm(cached[i]-reference[i]));
  }
  std::sort(old_times.begin(),old_times.end());std::sort(new_times.begin(),new_times.end());
  std::printf("{\"vertices\":%zu,\"triangles\":2,\"reference_median_ms\":%.6f,\"cached_median_ms\":%.6f,\"max_position_delta_m\":%.9g,\"scope\":\"single-thread float position update including frame-cache refresh; no physics or normals\"}\n",count,old_times[old_times.size()/2],new_times[new_times.size()/2],max_delta);
  return max_delta<=1e-6f ? 0:7;
}
