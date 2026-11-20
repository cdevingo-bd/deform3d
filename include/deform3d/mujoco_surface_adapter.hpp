#pragma once

// Optional MuJoCo dim=2 flex bridge. The core surface embedding has no MuJoCo dependency.
#include "surface_embed.hpp"
#include <mujoco/mujoco.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace deform3d {

enum class MujocoSurfaceAdapterStatus {
  Ok, InvalidInput, VersionMismatch, NotTriangular, InvalidTopology,
  NonFinitePosition, NotBound, InvalidRange
};
inline const char* mujoco_surface_adapter_status_string(MujocoSurfaceAdapterStatus status) noexcept {
  switch (status) {
    case MujocoSurfaceAdapterStatus::Ok: return "ok";
    case MujocoSurfaceAdapterStatus::InvalidInput: return "invalid MuJoCo input or output";
    case MujocoSurfaceAdapterStatus::VersionMismatch: return "MuJoCo header/library mismatch";
    case MujocoSurfaceAdapterStatus::NotTriangular: return "selected flex is not a triangle surface";
    case MujocoSurfaceAdapterStatus::InvalidTopology: return "invalid flex topology";
    case MujocoSurfaceAdapterStatus::NonFinitePosition: return "non-finite flex position";
    case MujocoSurfaceAdapterStatus::NotBound: return "adapter not bound to this model";
    case MujocoSurfaceAdapterStatus::InvalidRange: return "invalid vertex range";
  }
  return "unknown";
}

template<class T> class MujocoSurfaceAdapter {
  static_assert((std::is_same<T,float>::value || std::is_same<T,double>::value) &&
                std::numeric_limits<T>::is_iec559, "adapter requires IEEE float or double");
 public:
  // Call after mj_forward at the intended reference pose. Transactional.
  MujocoSurfaceAdapterStatus bind(const mjModel* model, const mjData* data,
                                  int flex_id, std::size_t* error_index=nullptr) {
    using S=MujocoSurfaceAdapterStatus;
    if(error_index) *error_index=0;
    if(!model || !data || !data->flexvert_xpos || flex_id<0 || flex_id>=model->nflex ||
       !model->flex_dim || !model->flex_vertadr || !model->flex_vertnum ||
       !model->flex_elemadr || !model->flex_elemnum || !model->flex_elemdataadr || !model->flex_elem)
      return S::InvalidInput;
    if(mjVERSION_HEADER!=mj_version()) return S::VersionMismatch;
    if(model->flex_dim[flex_id]!=2) return S::NotTriangular;
    const int va=model->flex_vertadr[flex_id],vn=model->flex_vertnum[flex_id];
    const int ea=model->flex_elemadr[flex_id],en=model->flex_elemnum[flex_id],da=model->flex_elemdataadr[flex_id];
    if(va<0 || vn<=0 || va>model->nflexvert || vn>model->nflexvert-va ||
       ea<0 || en<=0 || ea>model->nflexelem || en>model->nflexelem-ea ||
       da<0 || da>model->nflexelemdata || en>(model->nflexelemdata-da)/3)
      return S::InvalidTopology;
    std::vector<Vec3<T>> rest(static_cast<std::size_t>(vn));
    std::vector<std::uint32_t> triangles(static_cast<std::size_t>(en)*3);
    for(int i=0;i<vn;++i) {
      const mjtNum* p=data->flexvert_xpos+3*(std::size_t(va)+std::size_t(i));
      if(!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]) ||
         std::abs(static_cast<long double>(p[0]))>std::numeric_limits<T>::max() ||
         std::abs(static_cast<long double>(p[1]))>std::numeric_limits<T>::max() ||
         std::abs(static_cast<long double>(p[2]))>std::numeric_limits<T>::max()) {
        if(error_index) *error_index=std::size_t(i);
        return S::NonFinitePosition;
      }
      rest[std::size_t(i)]={T(p[0]),T(p[1]),T(p[2])};
    }
    for(int i=0;i<en;++i) for(std::size_t j=0;j<3;++j) {
      const int vertex=model->flex_elem[std::size_t(da)+3*std::size_t(i)+j];
      if(vertex<0 || vertex>=vn) {
        if(error_index) *error_index=std::size_t(i);
        return S::InvalidTopology;
      }
      triangles[3*std::size_t(i)+j]=std::uint32_t(vertex);
    }
    model_=model; flex_id_=flex_id; vertex_offset_=std::size_t(va);
    rest_.swap(rest); triangles_.swap(triangles);
    return S::Ok;
  }
  bool bound() const noexcept { return model_!=nullptr; }
  int flex_id() const noexcept { return flex_id_; }
  std::size_t vertex_offset() const noexcept { return vertex_offset_; }
  std::size_t vertex_count() const noexcept { return rest_.size(); }
  std::size_t triangle_count() const noexcept { return triangles_.size()/3; }
  const std::vector<Vec3<T>>& rest_vertices() const noexcept { return rest_; }
  const std::vector<std::uint32_t>& triangles() const noexcept { return triangles_; }

  // FNV-1a over canonical little-endian uint32 indices and IEEE reference
  // coordinates respectively. Reference hashes are specific to scalar type T.
  std::uint64_t topology_hash() const noexcept {
    std::uint64_t hash=UINT64_C(14695981039346656037);
    for(auto id:triangles_) for(int shift=0;shift<32;shift+=8) hash_byte(hash,(id>>shift)&255);
    return hash;
  }
  std::uint64_t rest_hash() const noexcept {
    using Bits=typename std::conditional<sizeof(T)==4,std::uint32_t,std::uint64_t>::type;
    std::uint64_t hash=UINT64_C(14695981039346656037);
    for(const auto& p:rest_) for(T value:{p.x,p.y,p.z}) {
      Bits bits=0; std::memcpy(&bits,&value,sizeof(T));
      for(std::size_t byte=0;byte<sizeof(T);++byte) hash_byte(hash,(bits>>(8*byte))&255);
    }
    return hash;
  }
  // Refresh mj_kinematics + mj_flex after mj_step before sampling integrated
  // state. This accessor does not change mjData; the host owns synchronization.
  const mjtNum* current_xyz(const mjModel* model,const mjData* data) const noexcept {
    if(!model_ || model!=model_ || !data || !data->flexvert_xpos ||
       flex_id_<0 || flex_id_>=model->nflex || !model->flex_vertadr || !model->flex_vertnum ||
       !model->flex_dim || model->flex_dim[flex_id_]!=2 ||
       model->flex_vertadr[flex_id_]!=static_cast<int>(vertex_offset_) ||
       model->flex_vertnum[flex_id_]!=static_cast<int>(rest_.size())) return nullptr;
    return data->flexvert_xpos+3*vertex_offset_;
  }
  SurfaceEmbedStatus prepare_embedding(SurfaceEmbedding<T>& embedding,
                                       const Vec3<T>* render,std::size_t count,
                                       SurfaceEmbedOptions<T> options={},std::size_t* error_index=nullptr) const {
    if(!bound()) return SurfaceEmbedStatus::InvalidInput;
    return embedding.prepare(rest_.data(),rest_.size(),triangles_.data(),triangle_count(),render,count,options,error_index);
  }
  template<class OutputScalar>
  SurfaceEmbedStatus evaluate_embedding(const mjModel* model,const mjData* data,
                                        const SurfaceEmbedding<T>& embedding,OutputScalar* output,
                                        std::size_t output_count,std::size_t output_stride=3,
                                        std::size_t begin=0,
                                        std::size_t end=std::numeric_limits<std::size_t>::max()) const noexcept {
    const mjtNum* xyz=current_xyz(model,data);
    if(!xyz) return SurfaceEmbedStatus::InvalidInput;
    return embedding.evaluate_strided(xyz,rest_.size(),3,output,output_count,output_stride,begin,end);
  }

 private:
  static void hash_byte(std::uint64_t& hash,std::uint64_t byte) noexcept {
    hash^=byte; hash*=UINT64_C(1099511628211);
  }
  const mjModel* model_=nullptr;
  int flex_id_=-1;
  std::size_t vertex_offset_=0;
  std::vector<Vec3<T>> rest_;
  std::vector<std::uint32_t> triangles_;
};

} // namespace deform3d
