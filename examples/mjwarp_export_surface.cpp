// Export prepared triangle-shell bindings for a dense XYZ rest mesh.
// Usage: deform3d_export_surface SCENE_XML FLEX_NAME OUTPUT.bind --mesh rest.xyz
//                               --max-distance METERS [--project]
#include <deform3d/mujoco_surface_adapter.hpp>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {
using V=deform3d::Vec3<float>;
bool u32(FILE* file,std::uint32_t value) {
  for(int shift=0;shift<32;shift+=8) {
    const auto byte=static_cast<unsigned char>(value>>shift);
    if(std::fwrite(&byte,1,1,file)!=1) return false;
  }
  return true;
}
bool u64(FILE* file,std::uint64_t value) {
  return u32(file,std::uint32_t(value))&&u32(file,std::uint32_t(value>>32));
}
bool f32(FILE* file,float value) {
  std::uint32_t bits=0; std::memcpy(&bits,&value,4); return u32(file,bits);
}
struct FileCloser { void operator()(FILE* file) const noexcept {if(file) std::fclose(file);} };
}

int main(int argc,char** argv) {
  static_assert(sizeof(float)==4&&std::numeric_limits<float>::is_iec559,"IEEE float32 required");
  if(argc<8) {
    std::fprintf(stderr,"usage: deform3d_export_surface SCENE_XML FLEX_NAME OUTPUT.bind --mesh rest.xyz --max-distance METERS [--project]\n");
    return 1;
  }
  const char* mesh_path=nullptr;
  float max_distance=-1;
  bool project=false;
  for(int i=4;i<argc;++i) {
    const std::string arg=argv[i];
    if(arg=="--mesh"&&i+1<argc) mesh_path=argv[++i];
    else if(arg=="--max-distance"&&i+1<argc) {
      char* end=nullptr; errno=0; max_distance=std::strtof(argv[++i],&end);
      if(errno || !end || end==argv[i] || *end || !std::isfinite(max_distance) || max_distance<0) return 2;
    } else if(arg=="--project") project=true;
    else return 2;
  }
  if(!mesh_path || max_distance<0) return 2;
  std::array<char,1024> error{};
  std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model(mj_loadXML(argv[1],nullptr,error.data(),int(error.size())),mj_deleteModel);
  if(!model) { std::fprintf(stderr,"MuJoCo: %s\n",error.data()); return 3; }
  std::unique_ptr<mjData,decltype(&mj_deleteData)> data(mj_makeData(model.get()),mj_deleteData);
  if(!data) return 4;
  mj_forward(model.get(),data.get());
  const int flex_id=mj_name2id(model.get(),mjOBJ_FLEX,argv[2]);
  deform3d::MujocoSurfaceAdapter<float> adapter;
  const auto bound=adapter.bind(model.get(),data.get(),flex_id);
  if(bound!=deform3d::MujocoSurfaceAdapterStatus::Ok) {
    std::fprintf(stderr,"surface bind: %s\n",deform3d::mujoco_surface_adapter_status_string(bound)); return 5;
  }
  std::ifstream input(mesh_path);
  if(!input) return 6;
  std::vector<V> render;
  std::string line;
  while(std::getline(input,line)) {
    const auto comment=line.find('#'); if(comment!=std::string::npos) line.erase(comment);
    std::istringstream row(line); row>>std::ws; if(row.eof()) continue;
    double x,y,z; std::string extra;
    if(!(row>>x>>y>>z)||(row>>extra)||!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(z)||
       std::abs(x)>std::numeric_limits<float>::max()||std::abs(y)>std::numeric_limits<float>::max()||
       std::abs(z)>std::numeric_limits<float>::max()) return 7;
    render.push_back({float(x),float(y),float(z)});
    if(render.size()>std::numeric_limits<std::uint32_t>::max()) return 7;
  }
  if(render.empty() || input.bad()) return 7;
  deform3d::SurfaceEmbedOptions<float> options;
  options.max_distance=max_distance;
  options.outside=project?deform3d::SurfaceOutsidePolicy::ProjectToSurface:deform3d::SurfaceOutsidePolicy::PreserveNormalOffset;
  deform3d::SurfaceEmbedding<float> embedding;
  std::size_t failed=0;
  const auto prepared=adapter.prepare_embedding(embedding,render.data(),render.size(),options,&failed);
  if(prepared!=deform3d::SurfaceEmbedStatus::Ok) {
    std::fprintf(stderr,"surface prepare: %s at index %zu\n",deform3d::surface_embed_status_string(prepared),failed); return 8;
  }
  std::unique_ptr<FILE,FileCloser> output(std::fopen(argv[3],"wb"));
  if(!output) return 9;
  // 40-byte header; 28-byte records. Each field is written individually in
  // canonical little-endian order, without C++ struct padding or host pointers.
  if(std::fwrite("D3SUR001",1,8,output.get())!=8 || !u32(output.get(),std::uint32_t(render.size())) ||
     !u32(output.get(),std::uint32_t(flex_id)) || !u32(output.get(),std::uint32_t(adapter.vertex_offset())) ||
     !u32(output.get(),std::uint32_t(adapter.vertex_count())) || !u64(output.get(),adapter.topology_hash()) ||
     !u64(output.get(),adapter.rest_hash())) return 10;
  for(const auto& binding:embedding.bindings()) {
    for(auto id:binding.vertices) if(!u32(output.get(),id)) return 10;
    for(auto weight:binding.weights) if(!f32(output.get(),weight)) return 10;
    if(!f32(output.get(),binding.normal_offset)) return 10;
  }
  if(std::fflush(output.get())!=0) return 10;
  std::printf("exported %zu surface bindings for flex %s (%d), %zu source vertices, %zu triangles\n",
              render.size(),argv[2],flex_id,adapter.vertex_count(),adapter.triangle_count());
  return 0;
}
