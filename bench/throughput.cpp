#include <deform3d/deform3d.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace deform3d;
using Clock=std::chrono::steady_clock;

int main(int argc,char** argv) {
  const size_t n=argc>1 ? size_t(std::strtoull(argv[1],nullptr,10)) : 250000;
  const int reps=argc>2 ? std::atoi(argv[2]) : 40;
  if (n==0 || reps<3) {std::cerr << "usage: deform3d_bench [vertices>0] [repetitions>=3]\n";return 2;}
  std::vector<Vec3f> positions(n),out(n);
  std::vector<VertexWeights4<float>> input_weights(n);
  for (size_t i=0;i<n;++i) {
    positions[i]={float(i%127)*0.004f,float(i%53)*0.009f,float(i%191)*0.002f};
    auto& w=input_weights[i];
    w.joint={uint32_t(i%64),uint32_t((i+1)%64),uint32_t((i+9)%64),uint32_t((i+17)%64)};
    w.weight={0.4f,0.3f,0.2f,0.1f};
  }
  SkinBinding<float> skin;
  if (skin.prepare(input_weights.data(),n,64)!=Status::Ok) return 2;
  std::vector<Affine3f> affine(64);
  std::vector<DualQuatf> rigid(64);
  for (size_t j=0;j<64;++j) {
    affine[j].m[3]=0.001f*float(j);
    rigid[j]=DualQuatf::from_rotation_translation({}, {affine[j].m[3],0,0});
  }
  Workspace<float> ws;ws.reserve(n,64);
  EvaluateParams<float> p;
  p.vertices=n;p.input_positions=view(positions.data());p.output_positions=view(out.data());
  p.skin=&skin;p.joint_count=64;p.workspace=&ws;
  p.affine_joints=affine.data();p.rigid_joints=rigid.data();
  auto run=[&](SkinMode mode,const char* name) {
    p.skin_mode=mode;
    std::vector<double> ms;
    ms.reserve(size_t(reps));
    for (int i=0;i<reps+5;++i) {
      const auto start=Clock::now();
      const Status s=evaluate(p);
      const auto end=Clock::now();
      if (s!=Status::Ok) {std::cerr << status_string(s) << '\n';std::exit(1);}
      if (i>=5) ms.push_back(std::chrono::duration<double,std::milli>(end-start).count());
    }
    std::sort(ms.begin(),ms.end());
    const double p50=ms[ms.size()/2],p95=ms[size_t(0.95*(ms.size()-1))];
    volatile float checksum=out[n/2].x;
    std::cout << name << ": vertices=" << n << " p50_ms=" << p50
              << " p95_ms=" << p95 << " million_vertices_per_s="
              << double(n)/(p50*1000.0) << " checksum=" << checksum << '\n';
  };
  run(SkinMode::LinearBlend,"LBS");
  run(SkinMode::DualQuaternion,"DQS");
}
