#pragma once

// Prepared differential surface attachment. Standard C++17 only.
// The host supplies physical vertex states and contact forces. This class
// computes positions, J*v, and J^T*f for the SAME offset-surface mapping.
#include "surface_differential.hpp"

namespace deform3d {

enum class SurfaceForceMode { Overwrite, Accumulate };

template<class T> class SurfaceCoupling {
  // Float inputs have ample exponent headroom in double; retain extended
  // precision for double inputs. Avoid scalar x87 arithmetic on dense float meshes.
  using L = typename std::conditional<std::is_same<T,float>::value,double,long double>::type;
  using P = Vec3<L>;
  using Binding = typename SurfaceEmbedding<T>::Binding;
  struct Frame : surface_detail::Differential<T> {
    std::array<std::uint32_t, 3> ids{};
    bool used = false;
  };
 public:
  // Setup owns a binding snapshot and a reverse vertex->attachment index.
  // A successful prepare invalidates the old frame; call update before use.
  // A failed prepare preserves the previous topology and frame.
  SurfaceEmbedStatus prepare(const SurfaceEmbedding<T>& embedding) {
    if (!embedding.ready() || embedding.render_vertex_count() >
        std::numeric_limits<std::size_t>::max() / 3) return SurfaceEmbedStatus::InvalidInput;
    SurfaceCoupling candidate;
    candidate.bindings_ = embedding.bindings();
    candidate.positions_.resize(embedding.simulation_vertex_count());
    candidate.frames_.resize(embedding.triangle_count());
    candidate.offsets_.resize(embedding.simulation_vertex_count() + 1);
    for (const auto& b : candidate.bindings_) {
      auto& frame = candidate.frames_[b.triangle];
      frame.ids = b.vertices;
      frame.used = true;
      for (auto vertex : b.vertices) ++candidate.offsets_[std::size_t(vertex) + 1];
    }
    for (std::size_t i = 1; i < candidate.offsets_.size(); ++i)
      candidate.offsets_[i] += candidate.offsets_[i - 1];
    candidate.incident_.resize(3 * candidate.bindings_.size());
    auto cursor = candidate.offsets_;
    for (std::size_t i = 0; i < candidate.bindings_.size(); ++i)
      for (std::size_t j = 0; j < 3; ++j)
        candidate.incident_[cursor[candidate.bindings_[i].vertices[j]]++] = 3 * i + j;
    candidate.staging_ = candidate.frames_;
    swap(candidate);
    return SurfaceEmbedStatus::Ok;
  }

  // Cache each referenced triangle's differential once per physical update.
  // No allocation; failure leaves the last valid frame usable. Finish update
  // before concurrent readers start. Unreferenced triangles are not evaluated.
  SurfaceEmbedStatus update(const Vec3<T>* positions, std::size_t count) noexcept {
    using S = SurfaceEmbedStatus;
    if (offsets_.empty() || !positions || count != positions_.size()) return S::InvalidInput;
    for (std::size_t i = 0; i < count; ++i)
      if (!finite(positions[i])) return S::NonFiniteInput;
    for (auto& frame : staging_) {
      if (!frame.used) continue;
      const S status=frame.update(positions[frame.ids[0]],positions[frame.ids[1]],positions[frame.ids[2]]);
      if (status!=S::Ok) return status;
    }
    std::copy(positions, positions + count, positions_.begin());
    frames_.swap(staging_);
    frame_ready_ = true;
    return S::Ok;
  }

  // Disjoint render ranges may execute concurrently. Every selected result is
  // checked before any write, including narrowing to float/double.
  SurfaceEmbedStatus evaluate_positions(Vec3<T>* output, std::size_t count,
      std::size_t begin = 0, std::size_t end = std::numeric_limits<std::size_t>::max()) const noexcept {
    using S = SurfaceEmbedStatus;
    if (!frame_ready_ || !output || count < bindings_.size() || !valid_count(count)) return S::InvalidInput;
    if (end == std::numeric_limits<std::size_t>::max()) end = bindings_.size();
    if (begin > end || end > bindings_.size()) return S::InvalidRange;
    return write_checked(output, begin, end, [&](std::size_t i) {
      const auto& b = bindings_[i];
      return wide(positions_[b.vertices[0]])*L(b.weights[0]) +
             wide(positions_[b.vertices[1]])*L(b.weights[1]) +
             wide(positions_[b.vertices[2]])*L(b.weights[2]) + frames_[b.triangle].normal*L(b.normal_offset);
    });
  }

  // Includes rotation of a nonzero normal offset. A barycentric-only velocity
  // would be inconsistent with evaluate_positions and would lose virtual work.
  SurfaceEmbedStatus evaluate_velocities(const Vec3<T>* velocities, std::size_t source_count,
      Vec3<T>* output, std::size_t count, std::size_t begin = 0,
      std::size_t end = std::numeric_limits<std::size_t>::max()) const noexcept {
    using S = SurfaceEmbedStatus;
    if (!frame_ready_ || !velocities || !output || source_count != positions_.size() ||
        count < bindings_.size() || !valid_count(count)) return S::InvalidInput;
    if (end == std::numeric_limits<std::size_t>::max()) end = bindings_.size();
    if (begin > end || end > bindings_.size()) return S::InvalidRange;
    if (overlap(velocities, source_count, output, count)) return S::Aliasing;
    for (std::size_t i = 0; i < source_count; ++i)
      if (!finite(velocities[i])) return S::NonFiniteInput;
    return write_checked(output, begin, end, [&](std::size_t i) {
      const auto& b = bindings_[i];
      const auto& frame = frames_[b.triangle];
      const P a = wide(velocities[b.vertices[0]]), v = wide(velocities[b.vertices[1]]), w = wide(velocities[b.vertices[2]]);
      P result = a*L(b.weights[0]) + v*L(b.weights[1]) + w*L(b.weights[2]);
      if (b.normal_offset != T(0)) {
        const P dn = cross(v-a, frame.e2) + cross(frame.e1, w-a);
        result += (dn-frame.normal*dot(frame.normal,dn))*(L(b.normal_offset)*frame.inverse_edge_area);
      }
      return result;
    });
  }

  // Deterministic gather via the prepared reverse index: no atomics, allocation
  // or shared output vertices. [begin,end) indexes SIMULATION vertices. Disjoint
  // ranges can run concurrently even when their attachments share triangles.
  // Overwrite replaces only this range, Accumulate adds to only this range.
  // All force inputs and selected accumulated outputs are checked before writes.
  SurfaceEmbedStatus transfer_forces(const Vec3<T>* forces, std::size_t force_count,
      Vec3<T>* output, std::size_t count, SurfaceForceMode mode = SurfaceForceMode::Overwrite,
      std::size_t begin = 0, std::size_t end = std::numeric_limits<std::size_t>::max()) const noexcept {
    using S = SurfaceEmbedStatus;
    if (!frame_ready_ || !forces || !output || force_count < bindings_.size() ||
        count != positions_.size() || !valid_count(force_count) ||
        (mode != SurfaceForceMode::Overwrite && mode != SurfaceForceMode::Accumulate)) return S::InvalidInput;
    if (end == std::numeric_limits<std::size_t>::max()) end = positions_.size();
    if (begin > end || end > positions_.size()) return S::InvalidRange;
    if (overlap(forces, force_count, output, count)) return S::Aliasing;
    for (std::size_t i = 0; i < bindings_.size(); ++i)
      if (!finite(forces[i])) return S::NonFiniteInput;
    if (mode == SurfaceForceMode::Accumulate)
      for (std::size_t i = begin; i < end; ++i)
        if (!finite(output[i])) return S::NonFiniteInput;
    return write_checked(output, begin, end, [&](std::size_t vertex) {
      P result = mode == SurfaceForceMode::Accumulate ? wide(output[vertex]) : P{};
      for (std::size_t entry = offsets_[vertex]; entry < offsets_[vertex+1]; ++entry) {
        const std::size_t id = incident_[entry]/3, corner = incident_[entry]%3;
        const auto& b = bindings_[id];
        const P force = wide(forces[id]);
        result += force*L(b.weights[corner]);
        if (b.normal_offset != T(0)) {
          const auto& frame = frames_[b.triangle];
          const P tangent = (force-frame.normal*dot(frame.normal,force))*(L(b.normal_offset)*frame.inverse_edge_area);
          const P fb = cross(frame.e2,tangent), fc = cross(tangent,frame.e1);
          result += corner == 0 ? -fb-fc : corner == 1 ? fb : fc;
        }
      }
      return result;
    });
  }

  bool frame_ready() const noexcept { return frame_ready_; }
  std::size_t simulation_vertex_count() const noexcept { return positions_.size(); }
  std::size_t render_vertex_count() const noexcept { return bindings_.size(); }
  void swap(SurfaceCoupling& other) noexcept {
    positions_.swap(other.positions_); bindings_.swap(other.bindings_);
    frames_.swap(other.frames_); staging_.swap(other.staging_);
    offsets_.swap(other.offsets_); incident_.swap(other.incident_);
    std::swap(frame_ready_, other.frame_ready_);
  }
 private:
  static P wide(Vec3<T> value) noexcept { return {L(value.x), L(value.y), L(value.z)}; }
  static L length(P value) noexcept { return std::hypot(value.x,value.y,value.z); }
  static bool valid_count(std::size_t count) noexcept { return count <= std::numeric_limits<std::size_t>::max()/sizeof(Vec3<T>); }
  static bool overlap(const Vec3<T>* a, std::size_t an, const Vec3<T>* b, std::size_t bn) noexcept {
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    return an && bn && (x <= y ? y-x < an*sizeof(Vec3<T>) : x-y < bn*sizeof(Vec3<T>));
  }
  template<class Compute> static SurfaceEmbedStatus write_checked(Vec3<T>* output,
      std::size_t begin, std::size_t end, Compute compute) noexcept {
    const L limit = std::numeric_limits<T>::max();
    for (std::size_t i = begin; i < end; ++i) {
      const P p = compute(i);
      if (!finite(p) || std::abs(p.x)>limit || std::abs(p.y)>limit || std::abs(p.z)>limit)
        return SurfaceEmbedStatus::NumericalFailure;
    }
    for (std::size_t i = begin; i < end; ++i) {
      const P p = compute(i);
      output[i] = {T(p.x),T(p.y),T(p.z)};
    }
    return SurfaceEmbedStatus::Ok;
  }
  bool frame_ready_ = false;
  std::vector<Vec3<T>> positions_;
  std::vector<Binding> bindings_;
  std::vector<Frame> frames_, staging_;
  std::vector<std::size_t> offsets_, incident_;
};

} // namespace deform3d
