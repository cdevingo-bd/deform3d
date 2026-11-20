// Optional benchmark: requires MuJoCo SDK and the example cavity scene.
#include <deform3d/mujoco_adapter.hpp>
#include <array>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

using Clock=std::chrono::steady_clock;
using V=deform3d::Vec3<float>;

static void report(const char* label,std::vector<double>& values) {
  std::sort(values.begin(),values.end());
  const double p50=values[values.size()/2];
  const double p95=values[size_t(.95*(values.size()-1))];
  std::printf("%s p50_ms=%.6f p95_ms=%.6f\n",label,p50,p95);
}

int main(int argc,char** argv) {
  const char* scene=argc>1 ? argv[1] : "examples/mujoco_cavity.xml";
  const size_t n=argc>2 ? size_t(std::strtoull(argv[2],nullptr,10)) : 250000;
  if (!n) return 2;
  std::array<char,1024> error{};
  std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model(
      mj_loadXML(scene,nullptr,error.data(),int(error.size())),mj_deleteModel);
  if (!model) {std::fprintf(stderr,"%s\n",error.data());return 3;}
  std::unique_ptr<mjData,decltype(&mj_deleteData)> data(mj_makeData(model.get()),mj_deleteData);
  if (!data) return 4;
  mj_forward(model.get(),data.get());
  deform3d::MujocoFlexAdapter<float> adapter;
  const int flex_id=mj_name2id(model.get(),mjOBJ_FLEX,"cavity");
  if (adapter.bind(model.get(),data.get(),flex_id)!=deform3d::MujocoAdapterStatus::Ok) return 5;
  const auto& rest=adapter.rest_vertices();
  const auto& tets=adapter.tetrahedra();
  std::vector<V> render(n);
  for(size_t i=0;i<n;++i) {
    const size_t tet=(i*17)%adapter.tetrahedron_count();
    const float a=float(1+(i*7)%23),b=float(1+(i*11)%29);
    const float c=float(1+(i*13)%31),d=float(1+(i*19)%37);
    const float sum=a+b+c+d;
    render[i]=rest[tets[4*tet]]*(a/sum)+rest[tets[4*tet+1]]*(b/sum)+
              rest[tets[4*tet+2]]*(c/sum)+rest[tets[4*tet+3]]*(d/sum);
  }
  deform3d::TetraEmbedding<float> embedding;
  const auto start=Clock::now();
  if (adapter.prepare_embedding(embedding,render.data(),n)!=deform3d::EmbedStatus::Ok) return 6;
  const double setup_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
  std::vector<float> output(3*n);
  std::vector<double> physics,transfer,total;
  physics.reserve(240);transfer.reserve(240);total.reserve(240);
  int contacts=0;
  for(int i=0;i<260;++i) {
    if(i%120==0) {mj_resetData(model.get(),data.get());mj_forward(model.get(),data.get());}
    const auto a=Clock::now();
    mj_step(model.get(),data.get());
    deform3d::refresh_mujoco_flex_positions(model.get(),data.get());
    const auto b=Clock::now();
    if (adapter.evaluate_embedding(model.get(),data.get(),embedding,
                                   output.data(),n)!=deform3d::EmbedStatus::Ok) return 7;
    const auto c=Clock::now();
    contacts+=data->ncon;
    if(i>=20) {
      physics.push_back(std::chrono::duration<double,std::milli>(b-a).count());
      transfer.push_back(std::chrono::duration<double,std::milli>(c-b).count());
      total.push_back(std::chrono::duration<double,std::milli>(c-a).count());
    }
  }
  volatile float checksum=output[3*(n/2)];
  std::printf("MuJoCo version=%d tetrahedra=%zu render_vertices=%zu "
              "setup_ms=%.3f cumulative_contacts=%d checksum=%g\n",
              mj_version(),adapter.tetrahedron_count(),n,setup_ms,contacts,checksum);
  report("physics",physics);report("deformation",transfer);report("combined",total);
}
