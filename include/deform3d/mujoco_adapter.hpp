#pragma once

// Optional bridge for MuJoCo's CPU flex state. Include this header only in
// projects that have MuJoCo headers and link against a matching MuJoCo library.
// The core deformation and embedding headers do not depend on MuJoCo.

#include "embed.hpp"
#include <mujoco/mujoco.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace deform3d {

enum class MujocoAdapterStatus {
  Ok,
  InvalidInput,
  VersionMismatch,
  NotTetrahedral,
  InvalidTopology,
  NonFinitePosition,
  NotBound,
  InvalidRange
};

inline const char* mujoco_adapter_status_string(MujocoAdapterStatus status) noexcept {
  switch (status) {
    case MujocoAdapterStatus::Ok: return "ok";
    case MujocoAdapterStatus::InvalidInput: return "invalid MuJoCo input or output buffer";
    case MujocoAdapterStatus::VersionMismatch: return "MuJoCo headers and linked library differ";
    case MujocoAdapterStatus::NotTetrahedral: return "selected flex is not tetrahedral";
    case MujocoAdapterStatus::InvalidTopology: return "invalid flex element or vertex range";
    case MujocoAdapterStatus::NonFinitePosition: return "non-finite flex vertex position";
    case MujocoAdapterStatus::NotBound: return "adapter was not bound to this model";
    case MujocoAdapterStatus::InvalidRange: return "invalid vertex range";
  }
  return "unknown";
}

// A model/topology and reference-pose snapshot for a single dim=3 flex.
// MuJoCo owns model/data; keep the model alive and its topology unchanged.
// Bind after mj_forward has populated bind_data.flexvert_xpos
// at the desired reference pose. Bind is transactional: a failed call does not
// invalidate a previously prepared adapter.
template<class T> class MujocoFlexAdapter {
  static_assert(std::is_floating_point<T>::value, "position type must be float or double");

 public:
  MujocoAdapterStatus bind(const mjModel* model, const mjData* bind_data,
                           int flex_id, std::size_t* error_index = nullptr) {
    if (error_index) *error_index = 0;
    if (!model || !bind_data || !bind_data->flexvert_xpos ||
        flex_id < 0 || flex_id >= model->nflex || !model->flex_dim ||
        !model->flex_vertadr || !model->flex_vertnum ||
        !model->flex_elemadr || !model->flex_elemnum ||
        !model->flex_elemdataadr || !model->flex_elem) {
      return MujocoAdapterStatus::InvalidInput;
    }
    if (mjVERSION_HEADER != mj_version()) return MujocoAdapterStatus::VersionMismatch;
    if (model->flex_dim[flex_id] != 3) return MujocoAdapterStatus::NotTetrahedral;

    const int vertex_adr = model->flex_vertadr[flex_id];
    const int vertex_num = model->flex_vertnum[flex_id];
    const int element_adr = model->flex_elemadr[flex_id];
    const int element_num = model->flex_elemnum[flex_id];
    const int element_data_adr = model->flex_elemdataadr[flex_id];
    if (vertex_adr < 0 || vertex_num <= 0 ||
        vertex_num > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max()) ||
        vertex_adr > model->nflexvert || vertex_num > model->nflexvert - vertex_adr ||
        element_adr < 0 || element_num <= 0 ||
        element_adr > model->nflexelem || element_num > model->nflexelem - element_adr ||
        element_data_adr < 0 || element_data_adr > model->nflexelemdata ||
        element_num > (model->nflexelemdata - element_data_adr) / 4) {
      return MujocoAdapterStatus::InvalidTopology;
    }

    std::vector<Vec3<T>> rest(static_cast<std::size_t>(vertex_num));
    std::vector<std::uint32_t> tetrahedra(static_cast<std::size_t>(element_num) * 4);
    for (int v = 0; v < vertex_num; ++v) {
      const mjtNum* xyz = bind_data->flexvert_xpos + 3 * (vertex_adr + v);
      if (!std::isfinite(xyz[0]) || !std::isfinite(xyz[1]) || !std::isfinite(xyz[2])) {
        if (error_index) *error_index = static_cast<std::size_t>(v);
        return MujocoAdapterStatus::NonFinitePosition;
      }
      rest[static_cast<std::size_t>(v)] =
          {static_cast<T>(xyz[0]), static_cast<T>(xyz[1]), static_cast<T>(xyz[2])};
      if (!finite(rest[static_cast<std::size_t>(v)])) {
        if (error_index) *error_index = static_cast<std::size_t>(v);
        return MujocoAdapterStatus::NonFinitePosition;
      }
    }
    for (int e = 0; e < element_num; ++e) {
      for (int corner = 0; corner < 4; ++corner) {
        // flex_elem ids are local to the flex; flexvert_xpos is model-global.
        const int id = model->flex_elem[element_data_adr + 4 * e + corner];
        if (id < 0 || id >= vertex_num) {
          if (error_index) *error_index = static_cast<std::size_t>(e);
          return MujocoAdapterStatus::InvalidTopology;
        }
        tetrahedra[4 * static_cast<std::size_t>(e) + static_cast<std::size_t>(corner)] =
            static_cast<std::uint32_t>(id);
      }
    }
    model_ = model;
    flex_id_ = flex_id;
    vertex_adr_ = static_cast<std::size_t>(vertex_adr);
    rest_.swap(rest);
    tetrahedra_.swap(tetrahedra);
    return MujocoAdapterStatus::Ok;
  }

  bool bound() const noexcept { return model_ != nullptr; }
  int flex_id() const noexcept { return flex_id_; }
  std::size_t vertex_count() const noexcept { return rest_.size(); }
  std::size_t tetrahedron_count() const noexcept { return tetrahedra_.size() / 4; }
  const std::vector<Vec3<T>>& rest_vertices() const noexcept { return rest_; }
  const std::vector<std::uint32_t>& tetrahedra() const noexcept { return tetrahedra_; }

  // The returned pointer refers to the first coordinate of this flex in the
  // CPU mjData. It remains owned by MuJoCo. After mj_step, refresh kinematics
  // and flex positions before sampling the newly integrated state (see below).
  // Returns null on a missing/mismatched model, data, or stale flex range.
  const mjtNum* current_xyz(const mjModel* model, const mjData* data) const noexcept {
    if (!model_ || model != model_ || !data || !data->flexvert_xpos ||
        flex_id_ < 0 || flex_id_ >= model->nflex ||
        !model->flex_vertadr || !model->flex_vertnum ||
        model->flex_vertadr[flex_id_] != static_cast<int>(vertex_adr_) ||
        model->flex_vertnum[flex_id_] != static_cast<int>(rest_.size())) return nullptr;
    return data->flexvert_xpos + 3 * vertex_adr_;
  }

  // Writes local vertices in [begin,end) at their corresponding output indices.
  // Caller can split disjoint ranges across workers once simulation has stepped.
  // A non-finite sample can leave preceding vertices in this range updated.
  MujocoAdapterStatus copy_current(const mjModel* model, const mjData* data,
                                   Vec3<T>* output, std::size_t output_count,
                                   std::size_t begin = 0,
                                   std::size_t end = std::numeric_limits<std::size_t>::max()) const noexcept {
    const mjtNum* xyz = current_xyz(model, data);
    if (!xyz) return MujocoAdapterStatus::NotBound;
    if (!output || output_count < rest_.size()) return MujocoAdapterStatus::InvalidInput;
    if (end == std::numeric_limits<std::size_t>::max()) end = rest_.size();
    if (begin > end || end > rest_.size()) return MujocoAdapterStatus::InvalidRange;
    for (std::size_t i = begin; i < end; ++i) {
      const Vec3<T> p{static_cast<T>(xyz[3*i]), static_cast<T>(xyz[3*i+1]),
                      static_cast<T>(xyz[3*i+2])};
      if (!finite(p)) return MujocoAdapterStatus::NonFinitePosition;
      output[i] = p;
    }
    return MujocoAdapterStatus::Ok;
  }

  // Prepare once with the reference pose captured in bind. A render point
  // outside the volume follows the embedding's explicit outside policy.
  EmbedStatus prepare_embedding(TetraEmbedding<T>& embedding,
                                const Vec3<T>* rest_render_vertices,
                                std::size_t render_vertex_count,
                                EmbedOptions<T> options = {},
                                std::size_t* error_index = nullptr) const {
    if (!bound()) return EmbedStatus::InvalidInput;
    return embedding.prepare(rest_.data(), rest_.size(), tetrahedra_.data(),
                             tetrahedron_count(), rest_render_vertices,
                             render_vertex_count, options, error_index);
  }

  // Directly sample MuJoCo's mjtNum XYZ array into an arbitrary scalar render
  // buffer. There is no per-frame allocation or intermediate position copy.
  // Output strides count scalar components, not bytes. No mutation of mjData.
  template<class OutputScalar>
  EmbedStatus evaluate_embedding(const mjModel* model, const mjData* data,
                                 const TetraEmbedding<T>& embedding,
                                 OutputScalar* output_xyz,
                                 std::size_t output_vertex_count,
                                 std::size_t output_stride = 3,
                                 std::size_t begin = 0,
                                 std::size_t end = std::numeric_limits<std::size_t>::max()) const noexcept {
    const mjtNum* xyz = current_xyz(model, data);
    if (!xyz) return EmbedStatus::InvalidInput;
    return embedding.evaluate_strided(xyz, rest_.size(), 3, output_xyz,
                                      output_vertex_count, output_stride, begin, end);
  }

 private:
  const mjModel* model_ = nullptr;
  int flex_id_ = -1;
  std::size_t vertex_adr_ = 0;
  std::vector<Vec3<T>> rest_;
  std::vector<std::uint32_t> tetrahedra_;
};

// mj_step integrates qpos last; its derived positions still describe the
// pre-integration state. Refresh just geometry for render/coupling consumers.
// This mutates mjData and must finish before concurrent readers start. It does
// not refresh contacts, sensors or dynamics; use mj_forward when those are needed.
inline MujocoAdapterStatus refresh_mujoco_flex_positions(const mjModel* model,
                                                        mjData* data) noexcept {
  if (!model || !data) return MujocoAdapterStatus::InvalidInput;
  if (mjVERSION_HEADER != mj_version()) return MujocoAdapterStatus::VersionMismatch;
  mj_kinematics(model, data);
  mj_flex(model, data);
  return MujocoAdapterStatus::Ok;
}

} // namespace deform3d
