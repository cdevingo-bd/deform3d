// Optional MuJoCo SDK tool: exports prepared C++ tetra bindings for MJWarp.
// Usage: mjwarp_export_bindings examples/mujoco_cavity.xml /tmp/cavity.bind 6144
#include <deform3d/mujoco_adapter.hpp>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cerrno>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <memory>
#include <vector>

namespace {
using V=deform3d::Vec3<float>;
static_assert(sizeof(V)==12,"Vec3f must contain exactly three float32 scalars");
struct Record {
  std::uint32_t vertex[4];
  float weight[4];
};
static_assert(sizeof(Record)==32,"binding record ABI must be 32 bytes");
struct FileCloser {void operator()(FILE* f) const noexcept {if(f) std::fclose(f);}};

bool write_u32(FILE* file,std::uint32_t value) {
  const unsigned char bytes[4]{static_cast<unsigned char>(value),
      static_cast<unsigned char>(value>>8),static_cast<unsigned char>(value>>16),
      static_cast<unsigned char>(value>>24)};
  return std::fwrite(bytes,1,4,file)==4;
}
bool write_u64(FILE* file,std::uint64_t value) {
  for (int i=0;i<8;++i) {
    const unsigned char b=static_cast<unsigned char>(value>>(8*i));
    if (std::fwrite(&b,1,1,file)!=1) return false;
  }
  return true;
}
std::uint64_t topology_hash(const std::vector<std::uint32_t>& ids) {
  std::uint64_t hash=UINT64_C(14695981039346656037);
  for (std::uint32_t id:ids) for (int byte=0;byte<4;++byte) {
    hash^=static_cast<unsigned char>(id>>(8*byte));
    hash*=UINT64_C(1099511628211);
  }
  return hash;
}
std::uint64_t rest_hash(const std::vector<V>& vertices) {
  std::uint64_t hash=UINT64_C(14695981039346656037);
  for (const V& p:vertices) for (float component:{p.x,p.y,p.z}) {
    std::uint32_t bits=0;
    std::memcpy(&bits,&component,4);
    for (int byte=0;byte<4;++byte) {
      hash^=static_cast<unsigned char>(bits>>(8*byte));
      hash*=UINT64_C(1099511628211);
    }
  }
  return hash;
}

int run(const char* xml,const char* output_path,size_t count,const char* mesh_path,
        const char* flex_name, deform3d::EmbedOptions<float> options) {
  if ((!mesh_path && !count) || count>std::numeric_limits<std::uint32_t>::max()) return 2;
  const std::uint32_t endian_probe=1;
  if (*reinterpret_cast<const unsigned char*>(&endian_probe)!=1 ||
      sizeof(float)!=4 || !std::numeric_limits<float>::is_iec559) {
    std::fprintf(stderr,"binding export requires little-endian IEEE-754 float32\n");
    return 2;
  }
  std::array<char,1024> error{};
  std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model(
      mj_loadXML(xml,nullptr,error.data(),int(error.size())),mj_deleteModel);
  if (!model) {std::fprintf(stderr,"MuJoCo: %s\n",error.data());return 3;}
  std::unique_ptr<mjData,decltype(&mj_deleteData)> data(mj_makeData(model.get()),mj_deleteData);
  if (!data) return 4;
  mj_forward(model.get(),data.get());
  const int id=mj_name2id(model.get(),mjOBJ_FLEX,flex_name);
  deform3d::MujocoFlexAdapter<float> bridge;
  if (bridge.bind(model.get(),data.get(),id)!=deform3d::MujocoAdapterStatus::Ok) return 5;
  const auto& rest=bridge.rest_vertices();
  const auto& tets=bridge.tetrahedra();
  std::vector<V> render;
  if (mesh_path) {
    std::ifstream input(mesh_path);
    if (!input) return 11;
    std::string line;
    while (std::getline(input,line)) {
      const size_t comment=line.find('#');
      if (comment!=std::string::npos) line.erase(comment);
      std::istringstream row(line);
      row>>std::ws;
      if (row.eof()) continue;
      double x,y,z;
      std::string extra;
      if (!(row>>x>>y>>z) || (row>>extra) ||
          !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return 12;
      const V p{float(x),float(y),float(z)};
      if (!deform3d::finite(p)) return 12;
      render.push_back(p);
      if (render.size()>std::numeric_limits<std::uint32_t>::max()) return 2;
    }
    if (input.bad()) return 12;
    count=render.size();
    if (!count) return 12;
  } else {
    render.resize(count);
    for (size_t i=0;i<count;++i) {
      const size_t t=(i*17)%bridge.tetrahedron_count();
      const float weights[]{float(1+(i*7)%17),float(1+(i*11)%19),
                            float(1+(i*13)%23),float(1+(i*5)%29)};
      const float sum=weights[0]+weights[1]+weights[2]+weights[3];
      for (size_t k=0;k<4;++k)
        render[i]+=rest[tets[4*t+k]]*(weights[k]/sum);
    }
  }
  deform3d::TetraEmbedding<float> embed;
  size_t error_index=0;
  const auto status=bridge.prepare_embedding(embed,render.data(),count,options,&error_index);
  if (status!=deform3d::EmbedStatus::Ok) {
    std::fprintf(stderr,"binding preparation failed at index %zu: %s\n",error_index,
                 deform3d::embed_status_string(status));
    return 6;
  }
  size_t outside_count=0;
  float max_distance=0,minimum_weight=1;
  if (options.outside==deform3d::EmbedOutsidePolicy::Extrapolate) {
    // A separate clamped map measures geometric distance without changing the
    // signed barycentric weights written to the output file.
    deform3d::TetraEmbedding<float> projected;
    auto projection_options=options;
    projection_options.outside=deform3d::EmbedOutsidePolicy::ClampToBoundary;
    if (bridge.prepare_embedding(projected,render.data(),count,projection_options)!=
        deform3d::EmbedStatus::Ok) return 6;
    std::vector<V> closest(count);
    if (projected.evaluate(rest.data(),rest.size(),closest.data(),count)!=
        deform3d::EmbedStatus::Ok) return 6;
    for (size_t i=0;i<count;++i) {
      const float distance=deform3d::norm(render[i]-closest[i]);
      max_distance=std::max(max_distance,distance);
      if (distance>1e-6f) ++outside_count;
    }
  }
  for (const auto& binding:embed.bindings())
    for (float weight:binding.weights) minimum_weight=std::min(minimum_weight,weight);

  // Disk format: magic D3EMB003, little-endian u32 count, flex id, flex
  // vertex address, flex vertex count, u64 FNV-1a topology and rest hashes;
  // then records of 4 u32 local IDs
  // and 4 IEEE-754 float32 weights. No host pointers, padding or native size_t.
  if (id<0 || model->flex_vertadr[id]<0 ||
      bridge.vertex_count()>std::numeric_limits<std::uint32_t>::max()) return 7;
  std::unique_ptr<FILE,FileCloser> out(std::fopen(output_path,"wb"));
  if (!out) return 8;
  if (std::fwrite("D3EMB003",1,8,out.get())!=8 ||
      !write_u32(out.get(),std::uint32_t(count)) ||
      !write_u32(out.get(),std::uint32_t(id)) ||
      !write_u32(out.get(),std::uint32_t(model->flex_vertadr[id])) ||
      !write_u32(out.get(),std::uint32_t(bridge.vertex_count())) ||
      !write_u64(out.get(),topology_hash(tets)) ||
      !write_u64(out.get(),rest_hash(rest))) return 9;
  for (const auto& b : embed.bindings()) {
    Record record{};
    for (size_t k=0;k<4;++k) {record.vertex[k]=b.vertices[k];record.weight[k]=b.weights[k];}
    // This example targets common little-endian IEEE-754 hosts. The reader
    // validates header and ranges. Extend serialization for other hosts.
    if (std::fwrite(&record,sizeof(record),1,out.get())!=1) return 10;
  }
  // Small exports can fit entirely in stdio's buffer. Check the actual write
  // before reporting success, including disk-full and device errors.
  if (std::fflush(out.get())!=0) return 10;
  std::printf("exported %zu bindings for flex %d, offset %d\n",count,id,
              model->flex_vertadr[id]);
  std::printf("outside policy=%s; points farther than 1um=%zu; maximum distance=%.9g m; minimum weight=%.9g\n",
              options.outside==deform3d::EmbedOutsidePolicy::Extrapolate?"bounded extrapolate":"reject",
              outside_count,double(max_distance),double(minimum_weight));
  return 0;
}
} // namespace

int main(int argc,char** argv) {
  if (argc<3) {
    std::fprintf(stderr,"usage: mjwarp_export_bindings SCENE_XML OUTPUT.bind [render_count | --mesh rest.xyz] [--flex NAME] [--outside extrapolate --max-distance METERS]\n");
    return 1;
  }
  const char* mesh=nullptr;
  const char* flex="cavity";
  size_t count=6144;
  bool specified=false;
  deform3d::EmbedOptions<float> options;
  for (int i=3;i<argc;++i) {
    const std::string arg=argv[i];
    if (arg=="--flex" && i+1<argc) flex=argv[++i];
    else if (arg=="--outside" && i+1<argc) {
      const std::string policy=argv[++i];
      if (policy=="extrapolate") options.outside=deform3d::EmbedOutsidePolicy::Extrapolate;
      else if (policy=="reject") options.outside=deform3d::EmbedOutsidePolicy::Reject;
      else return 1;
    } else if (arg=="--max-distance" && i+1<argc) {
      char* end=nullptr;
      errno=0;
      const double distance=std::strtod(argv[++i],&end);
      if (errno || !end || end==argv[i] || *end || !std::isfinite(distance) ||
          distance<0 || distance>std::numeric_limits<float>::max()) return 1;
      options.max_outside_distance=float(distance);
    }
    else if (arg=="--mesh" && i+1<argc && !specified) {
      mesh=argv[++i]; count=0; specified=true;
    } else if (!specified && !arg.empty() && arg[0]>='0' && arg[0]<='9') {
      char* end=nullptr;
      errno=0;
      const auto parsed=std::strtoull(argv[i],&end,10);
      if (errno || !end || *end || !parsed || parsed>std::numeric_limits<size_t>::max()) return 1;
      count=static_cast<size_t>(parsed); specified=true;
    } else return 1;
  }
  if (options.outside==deform3d::EmbedOutsidePolicy::Extrapolate &&
      !std::isfinite(options.max_outside_distance)) {
    std::fprintf(stderr,"extrapolate requires an explicit finite --max-distance\n");
    return 1;
  }
  return run(argv[1],argv[2],count,mesh,flex,options);
}
