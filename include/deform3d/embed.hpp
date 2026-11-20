#pragma once

// Dense render geometry embedded in a lower-resolution tetrahedral simulation mesh.
// C++17, standard library only. The simulation owns vertex positions and contacts;
// this class transfers their deformation to render vertices and transfers
// caller-computed point forces back through the transpose of the same mapping.

#include "deform3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace deform3d {

enum class EmbedStatus {
  Ok,
  InvalidInput,
  NonFiniteInput,
  InvalidIndex,
  DegenerateTetrahedron,
  OutsideVolume,
  InvalidRange,
  Aliasing
};

inline const char* embed_status_string(EmbedStatus status) noexcept {
  switch (status) {
    case EmbedStatus::Ok: return "ok";
    case EmbedStatus::InvalidInput: return "invalid input";
    case EmbedStatus::NonFiniteInput: return "non-finite input";
    case EmbedStatus::InvalidIndex: return "tetrahedron index out of range";
    case EmbedStatus::DegenerateTetrahedron: return "degenerate tetrahedron";
    case EmbedStatus::OutsideVolume: return "render vertex outside tetrahedral volume";
    case EmbedStatus::InvalidRange: return "invalid evaluation range";
    case EmbedStatus::Aliasing: return "input and output buffers overlap";
  }
  return "unknown";
}

enum class EmbedOutsidePolicy {
  Reject,
  ClampToBoundary,
  // Bind to the nearest tetrahedron using its unconstrained barycentric
  // coordinates. This preserves the exact rest position under affine motion,
  // including render detail just beyond the simulated volume. Requires a
  // finite max_outside_distance to bound extrapolation.
  Extrapolate
};

enum class EmbedForceMode {
  // Clear the entire simulation force array before transferring the selected
  // render forces, including simulation vertices unused by that selection.
  Overwrite,
  // Add to the existing simulation forces, e.g. from other embedded objects.
  Accumulate
};

template<class T> struct EmbedOptions {
  EmbedOutsidePolicy outside = EmbedOutsidePolicy::Reject;
  // Maximum distance from an uncontained render point to the tetrahedral volume.
  // Used with ClampToBoundary and Extrapolate. A projected point follows its
  // closest tetrahedron face. Extrapolate keeps the full rest offset; choose a
  // small finite limit to avoid large weights far from the simulated surface.
  T max_outside_distance = std::numeric_limits<T>::infinity();
  // Dimensionless tolerance for barycentric containment tests.
  T barycentric_tolerance = T(1e-6);
  std::size_t bvh_leaf_size = 8;
};

template<class T> class TetraEmbedding {
  static_assert(std::is_floating_point<T>::value, "TetraEmbedding needs float or double");

 public:
  struct Binding {
    std::array<std::uint32_t, 4> vertices{};
    std::array<T, 4> weights{};
    std::uint32_t tetrahedron = 0;
  };

  // prepare is transactional: a failed call leaves the previous embedding intact.
  // error_index is a tetrahedron index for InvalidIndex/DegenerateTetrahedron,
  // or a render vertex index for NonFiniteInput/OutsideVolume where applicable.
  EmbedStatus prepare(const Vec3<T>* rest_sim_vertices,
                      std::size_t sim_vertex_count,
                      const std::uint32_t* tetrahedra,
                      std::size_t tetrahedron_count,
                      const Vec3<T>* rest_render_vertices,
                      std::size_t render_vertex_count,
                      EmbedOptions<T> options = {},
                      std::size_t* error_index = nullptr) {
    if (error_index) *error_index = 0;
    if (!rest_sim_vertices || !tetrahedra || !sim_vertex_count || !tetrahedron_count ||
        (render_vertex_count && !rest_render_vertices) || !options.bvh_leaf_size ||
        !finite(options.barycentric_tolerance) || options.barycentric_tolerance < T(0) ||
        std::isnan(options.max_outside_distance) || options.max_outside_distance < T(0) ||
        (options.outside == EmbedOutsidePolicy::Extrapolate &&
         !std::isfinite(options.max_outside_distance)) ||
        tetrahedron_count > std::numeric_limits<std::uint32_t>::max() / 2 ||
        tetrahedron_count > std::numeric_limits<std::size_t>::max() / 4 ||
        sim_vertex_count > std::numeric_limits<std::size_t>::max() / sizeof(Vec3<T>) ||
        render_vertex_count > std::numeric_limits<std::size_t>::max() / sizeof(Vec3<T>)) {
      return EmbedStatus::InvalidInput;
    }

    TetraEmbedding candidate;
    candidate.rest_.assign(rest_sim_vertices, rest_sim_vertices + sim_vertex_count);
    candidate.tets_.resize(tetrahedron_count);
    candidate.order_.resize(tetrahedron_count);
    std::iota(candidate.order_.begin(), candidate.order_.end(), std::uint32_t(0));
    for (std::size_t i = 0; i < sim_vertex_count; ++i) {
      if (!finite(candidate.rest_[i])) {
        if (error_index) *error_index = i;
        return EmbedStatus::NonFiniteInput;
      }
    }
    for (std::size_t t = 0; t < tetrahedron_count; ++t) {
      Tet& tet = candidate.tets_[t];
      for (std::size_t j = 0; j < 4; ++j) {
        tet.ids[j] = tetrahedra[4 * t + j];
        if (tet.ids[j] >= sim_vertex_count) {
          if (error_index) *error_index = t;
          return EmbedStatus::InvalidIndex;
        }
      }
      const Vec3<T> a = candidate.rest_[tet.ids[0]];
      const Vec3<T> ab = candidate.rest_[tet.ids[1]] - a;
      const Vec3<T> ac = candidate.rest_[tet.ids[2]] - a;
      const Vec3<T> ad = candidate.rest_[tet.ids[3]] - a;
      const T det = dot(ab, cross(ac, ad));
      const T scale2 = std::max({dot(ab, ab), dot(ac, ac), dot(ad, ad)});
      const T scale = std::sqrt(scale2);
      // Compare volume with the tetrahedron's own scale, not scene coordinates.
      const T threshold = T(64) * std::numeric_limits<T>::epsilon() * scale2 * scale;
      if (!finite(det) || !(scale > T(0)) || std::abs(det) <= threshold) {
        if (error_index) *error_index = t;
        return EmbedStatus::DegenerateTetrahedron;
      }
      tet.origin = a;
      tet.gradient[0] = cross(ac, ad) / det;
      tet.gradient[1] = cross(ad, ab) / det;
      tet.gradient[2] = cross(ab, ac) / det;
      tet.box = box_of(candidate.rest_, tet.ids);
      tet.centroid = (a + candidate.rest_[tet.ids[1]] + candidate.rest_[tet.ids[2]] +
                      candidate.rest_[tet.ids[3]]) * T(0.25);
    }
    candidate.nodes_.reserve(2 * tetrahedron_count);
    candidate.build_node(0, tetrahedron_count, options.bvh_leaf_size);
    candidate.bindings_.resize(render_vertex_count);
    const Box& scene = candidate.nodes_[0].box;
    const T scene_scale = std::sqrt(dot(scene.hi - scene.lo, scene.hi - scene.lo));
    const T box_tolerance = options.barycentric_tolerance * scene_scale * T(2);
    for (std::size_t i = 0; i < render_vertex_count; ++i) {
      const Vec3<T>& p = rest_render_vertices[i];
      if (!finite(p)) {
        if (error_index) *error_index = i;
        return EmbedStatus::NonFiniteInput;
      }
      Binding binding;
      if (!candidate.find_containing(0, p, box_tolerance,
                                     options.barycentric_tolerance, binding)) {
        if (options.outside == EmbedOutsidePolicy::Reject) {
          if (error_index) *error_index = i;
          return EmbedStatus::OutsideVolume;
        }
        T best_distance2 = std::numeric_limits<T>::infinity();
        candidate.find_closest(0, p, best_distance2, binding);
        // Overflow in a squared distance can leave find_closest without any
        // candidate. In particular, infinity > infinity is false when the
        // caller uses the default unbounded clamp distance; never accept the
        // untouched, zero-weight Binding in that case.
        if (!finite(best_distance2) ||
            std::sqrt(best_distance2) > options.max_outside_distance) {
          if (error_index) *error_index = i;
          return EmbedStatus::OutsideVolume;
        }
        if (options.outside == EmbedOutsidePolicy::Extrapolate) {
          const Tet& tet = candidate.tets_[binding.tetrahedron];
          binding.vertices = tet.ids;
          binding.weights = barycentric(tet, p);
        }
      } else if (options.outside == EmbedOutsidePolicy::Extrapolate) {
        // A point barely outside a face may have passed the tolerance check.
        // Retain its signed weights instead of snapping it onto the face.
        const Tet& tet = candidate.tets_[binding.tetrahedron];
        binding.weights = barycentric(tet, p);
        if (std::any_of(binding.weights.begin(), binding.weights.end(),
                        [](T w) { return w < T(0); })) {
          T best_distance2 = std::numeric_limits<T>::infinity();
          Binding projected;
          candidate.find_closest(0, p, best_distance2, projected);
          if (!finite(best_distance2) ||
              std::sqrt(best_distance2) > options.max_outside_distance) {
            if (error_index) *error_index = i;
            return EmbedStatus::OutsideVolume;
          }
        }
      }
      candidate.bindings_[i] = binding;
    }
    swap(candidate);
    return EmbedStatus::Ok;
  }

  // The entire simulation position array is supplied, including vertices which
  // are not referenced by a particular render subset. Input/output may not overlap.
  // Separate output ranges can safely be evaluated on different worker threads.
  EmbedStatus evaluate(const Vec3<T>* current_sim_vertices,
                       std::size_t sim_vertex_count,
                       Vec3<T>* output_render_vertices,
                       std::size_t output_count) const noexcept {
    return evaluate_range(current_sim_vertices, sim_vertex_count,
                          output_render_vertices, output_count, 0, bindings_.size());
  }

  EmbedStatus evaluate_range(const Vec3<T>* current_sim_vertices,
                             std::size_t sim_vertex_count,
                             Vec3<T>* output_render_vertices,
                             std::size_t output_count,
                             std::size_t begin,
                             std::size_t end) const noexcept {
    if (!ready() || !current_sim_vertices || !output_render_vertices ||
        sim_vertex_count != rest_.size() || output_count < bindings_.size() ||
        output_count > std::numeric_limits<std::size_t>::max() / sizeof(Vec3<T>)) {
      return EmbedStatus::InvalidInput;
    }
    if (begin > end || end > bindings_.size()) return EmbedStatus::InvalidRange;
    if (overlap(current_sim_vertices, sim_vertex_count * sizeof(Vec3<T>),
                output_render_vertices, output_count * sizeof(Vec3<T>))) {
      return EmbedStatus::Aliasing;
    }
    for (std::size_t i = begin; i < end; ++i) {
      const Binding& b = bindings_[i];
      output_render_vertices[i] = current_sim_vertices[b.vertices[0]] * b.weights[0] +
                                  current_sim_vertices[b.vertices[1]] * b.weights[1] +
                                  current_sim_vertices[b.vertices[2]] * b.weights[2] +
                                  current_sim_vertices[b.vertices[3]] * b.weights[3];
    }
    return EmbedStatus::Ok;
  }

  // For raw XYZ arrays such as mjData::flexvert_xpos (mjtNum*) and a float
  // graphics buffer. Strides are in scalar elements, including padding.
  template<class InputScalar, class OutputScalar>
  EmbedStatus evaluate_strided(const InputScalar* current_xyz,
                               std::size_t sim_vertex_count,
                               std::size_t input_stride,
                               OutputScalar* output_xyz,
                               std::size_t output_count,
                               std::size_t output_stride,
                               std::size_t begin = 0,
                               std::size_t end = std::numeric_limits<std::size_t>::max()) const noexcept {
    static_assert(std::is_floating_point<InputScalar>::value &&
                  std::is_floating_point<OutputScalar>::value,
                  "XYZ input and output scalars must be floating point");
    if (!ready() || !current_xyz || !output_xyz || sim_vertex_count != rest_.size() ||
        output_count < bindings_.size() || input_stride < 3 || output_stride < 3 ||
        input_stride > std::numeric_limits<std::size_t>::max() / sizeof(InputScalar) ||
        output_stride > std::numeric_limits<std::size_t>::max() / sizeof(OutputScalar) ||
        sim_vertex_count > std::numeric_limits<std::size_t>::max() /
            (input_stride * sizeof(InputScalar)) ||
        (output_count && output_count > std::numeric_limits<std::size_t>::max() /
            (output_stride * sizeof(OutputScalar)))) {
      return EmbedStatus::InvalidInput;
    }
    if (end == std::numeric_limits<std::size_t>::max()) end = bindings_.size();
    if (begin > end || end > bindings_.size()) return EmbedStatus::InvalidRange;
    if (overlap(current_xyz, sim_vertex_count * input_stride * sizeof(InputScalar),
                output_xyz, output_count * output_stride * sizeof(OutputScalar))) {
      return EmbedStatus::Aliasing;
    }
    for (std::size_t i = begin; i < end; ++i) {
      const Binding& b = bindings_[i];
      Vec3<T> p{};
      for (std::size_t j = 0; j < 4; ++j) {
        const InputScalar* xyz = current_xyz + std::size_t(b.vertices[j]) * input_stride;
        p = p + Vec3<T>{T(xyz[0]), T(xyz[1]), T(xyz[2])} * b.weights[j];
      }
      OutputScalar* dst = output_xyz + i * output_stride;
      dst[0] = OutputScalar(p.x);
      dst[1] = OutputScalar(p.y);
      dst[2] = OutputScalar(p.z);
    }
    return EmbedStatus::Ok;
  }

  // Apply J^T to forces at embedded points, where evaluate computes x_render =
  // J*x_sim. This preserves virtual work, resultant force, and torque about
  // any origin (up to rounding), provided forces act at the evaluated positions.
  // Signed extrapolation weights are retained. No allocations or atomics occur.
  // The caller supplies forces/impulses; this does not generate contact forces.
  EmbedStatus transfer_forces(const Vec3<T>* render_forces,
                              std::size_t render_force_count,
                              Vec3<T>* output_sim_forces,
                              std::size_t sim_force_count,
                              EmbedForceMode mode = EmbedForceMode::Overwrite) const noexcept {
    return transfer_forces_range(render_forces, render_force_count,
                                 output_sim_forces, sim_force_count,
                                 0, bindings_.size(), mode);
  }

  // Source arrays use global render indexing, as in evaluate_range. Overwrite
  // clears ALL simulation forces even for an empty range; Accumulate clears
  // none. Range calls may share an embedding, but must not concurrently write
  // the same simulation buffer: different render ranges can share tet vertices.
  // Invalid arguments/nonfinite input leave the output unchanged. Only selected
  // source entries are checked; Accumulate also checks the full target array.
  // Finite inputs must be scaled so products and accumulated sums fit in T.
  EmbedStatus transfer_forces_range(const Vec3<T>* render_forces,
                                    std::size_t render_force_count,
                                    Vec3<T>* output_sim_forces,
                                    std::size_t sim_force_count,
                                    std::size_t begin,
                                    std::size_t end,
                                    EmbedForceMode mode = EmbedForceMode::Overwrite) const noexcept {
    if (!ready() || !render_forces || !output_sim_forces ||
        render_force_count < bindings_.size() || sim_force_count != rest_.size() ||
        render_force_count > std::numeric_limits<std::size_t>::max() / sizeof(Vec3<T>) ||
        (mode != EmbedForceMode::Overwrite && mode != EmbedForceMode::Accumulate)) {
      return EmbedStatus::InvalidInput;
    }
    if (begin > end || end > bindings_.size()) return EmbedStatus::InvalidRange;
    if (overlap(render_forces, render_force_count * sizeof(Vec3<T>),
                output_sim_forces, sim_force_count * sizeof(Vec3<T>))) {
      return EmbedStatus::Aliasing;
    }
    for (std::size_t i = begin; i < end; ++i) {
      if (!finite(render_forces[i])) return EmbedStatus::NonFiniteInput;
    }
    if (mode == EmbedForceMode::Accumulate) {
      for (std::size_t i = 0; i < sim_force_count; ++i) {
        if (!finite(output_sim_forces[i])) return EmbedStatus::NonFiniteInput;
      }
    } else {
      std::fill(output_sim_forces, output_sim_forces + sim_force_count, Vec3<T>{});
    }
    for (std::size_t i = begin; i < end; ++i) {
      const Binding& b = bindings_[i];
      for (std::size_t j = 0; j < 4; ++j)
        output_sim_forces[b.vertices[j]] += render_forces[i] * b.weights[j];
    }
    return EmbedStatus::Ok;
  }

  // Strides are scalar elements, including padding; only XYZ are written.
  // Uses the common precision of T, InputScalar and OutputScalar for each
  // contribution. Summation rounds to OutputScalar after each contribution.
  // The validation, range, mode and concurrency rules above apply here too;
  // padding is neither checked for finiteness nor modified.
  template<class InputScalar, class OutputScalar>
  EmbedStatus transfer_forces_strided(
      const InputScalar* render_force_xyz,
      std::size_t render_force_count,
      std::size_t input_stride,
      OutputScalar* output_sim_force_xyz,
      std::size_t sim_force_count,
      std::size_t output_stride,
      EmbedForceMode mode = EmbedForceMode::Overwrite,
      std::size_t begin = 0,
      std::size_t end = std::numeric_limits<std::size_t>::max()) const noexcept {
    static_assert(std::is_floating_point<InputScalar>::value &&
                  std::is_floating_point<OutputScalar>::value,
                  "XYZ input and output scalars must be floating point");
    using CalculationScalar = typename std::common_type<T, InputScalar, OutputScalar>::type;
    if (!ready() || !render_force_xyz || !output_sim_force_xyz ||
        render_force_count < bindings_.size() || sim_force_count != rest_.size() ||
        input_stride < 3 || output_stride < 3 ||
        (mode != EmbedForceMode::Overwrite && mode != EmbedForceMode::Accumulate) ||
        input_stride > std::numeric_limits<std::size_t>::max() / sizeof(InputScalar) ||
        output_stride > std::numeric_limits<std::size_t>::max() / sizeof(OutputScalar) ||
        render_force_count > std::numeric_limits<std::size_t>::max() /
            (input_stride * sizeof(InputScalar)) ||
        sim_force_count > std::numeric_limits<std::size_t>::max() /
            (output_stride * sizeof(OutputScalar))) {
      return EmbedStatus::InvalidInput;
    }
    if (end == std::numeric_limits<std::size_t>::max()) end = bindings_.size();
    if (begin > end || end > bindings_.size()) return EmbedStatus::InvalidRange;
    if (overlap(render_force_xyz, render_force_count * input_stride * sizeof(InputScalar),
                output_sim_force_xyz, sim_force_count * output_stride * sizeof(OutputScalar))) {
      return EmbedStatus::Aliasing;
    }
    for (std::size_t i = begin; i < end; ++i) {
      const InputScalar* src = render_force_xyz + i * input_stride;
      if (!std::isfinite(src[0]) || !std::isfinite(src[1]) || !std::isfinite(src[2]))
        return EmbedStatus::NonFiniteInput;
    }
    if (mode == EmbedForceMode::Accumulate) {
      for (std::size_t i = 0; i < sim_force_count; ++i) {
        const OutputScalar* dst = output_sim_force_xyz + i * output_stride;
        if (!std::isfinite(dst[0]) || !std::isfinite(dst[1]) || !std::isfinite(dst[2]))
          return EmbedStatus::NonFiniteInput;
      }
    } else {
      for (std::size_t i = 0; i < sim_force_count; ++i) {
        OutputScalar* dst = output_sim_force_xyz + i * output_stride;
        dst[0] = dst[1] = dst[2] = OutputScalar(0);
      }
    }
    for (std::size_t i = begin; i < end; ++i) {
      const Binding& b = bindings_[i];
      const InputScalar* src = render_force_xyz + i * input_stride;
      for (std::size_t j = 0; j < 4; ++j) {
        OutputScalar* dst = output_sim_force_xyz + std::size_t(b.vertices[j]) * output_stride;
        const CalculationScalar w = CalculationScalar(b.weights[j]);
        for (std::size_t axis = 0; axis < 3; ++axis)
          dst[axis] = OutputScalar(CalculationScalar(dst[axis]) + CalculationScalar(src[axis]) * w);
      }
    }
    return EmbedStatus::Ok;
  }

  bool ready() const noexcept { return !nodes_.empty(); }
  std::size_t render_vertex_count() const noexcept { return bindings_.size(); }
  std::size_t simulation_vertex_count() const noexcept { return rest_.size(); }
  std::size_t tetrahedron_count() const noexcept { return tets_.size(); }
  const std::vector<Binding>& bindings() const noexcept { return bindings_; }

  void swap(TetraEmbedding& other) noexcept {
    rest_.swap(other.rest_);
    tets_.swap(other.tets_);
    order_.swap(other.order_);
    nodes_.swap(other.nodes_);
    bindings_.swap(other.bindings_);
  }

 private:
  struct Box { Vec3<T> lo{}, hi{}; };
  struct Tet {
    std::array<std::uint32_t, 4> ids{};
    Vec3<T> origin{}, centroid{};
    std::array<Vec3<T>, 3> gradient{};
    Box box{};
  };
  struct Node {
    Box box{};
    std::size_t begin = 0, end = 0;
    std::uint32_t left = 0, right = 0;
    bool leaf = true;
  };

  static bool finite(T x) noexcept { return std::isfinite(x); }
  static bool finite(const Vec3<T>& v) noexcept {
    return finite(v.x) && finite(v.y) && finite(v.z);
  }
  static bool overlap(const void* p, std::size_t p_bytes,
                      const void* q, std::size_t q_bytes) noexcept {
    const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(p);
    const std::uintptr_t b = reinterpret_cast<std::uintptr_t>(q);
    if (!p_bytes || !q_bytes) return false;
    return a <= b ? b - a < p_bytes : a - b < q_bytes;
  }
  static Box box_of(const std::vector<Vec3<T>>& positions,
                    const std::array<std::uint32_t, 4>& ids) noexcept {
    Box b{positions[ids[0]], positions[ids[0]]};
    for (std::size_t j = 1; j < 4; ++j) grow(b, positions[ids[j]]);
    return b;
  }
  static void grow(Box& b, Vec3<T> p) noexcept {
    b.lo.x = std::min(b.lo.x, p.x); b.lo.y = std::min(b.lo.y, p.y); b.lo.z = std::min(b.lo.z, p.z);
    b.hi.x = std::max(b.hi.x, p.x); b.hi.y = std::max(b.hi.y, p.y); b.hi.z = std::max(b.hi.z, p.z);
  }
  static void grow(Box& b, const Box& x) noexcept { grow(b, x.lo); grow(b, x.hi); }
  static T component(Vec3<T> p, std::size_t axis) noexcept {
    return axis == 0 ? p.x : (axis == 1 ? p.y : p.z);
  }
  static T distance2(const Box& b, Vec3<T> p) noexcept {
    const T dx = std::max({b.lo.x - p.x, T(0), p.x - b.hi.x});
    const T dy = std::max({b.lo.y - p.y, T(0), p.y - b.hi.y});
    const T dz = std::max({b.lo.z - p.z, T(0), p.z - b.hi.z});
    return dx * dx + dy * dy + dz * dz;
  }

  std::uint32_t build_node(std::size_t begin, std::size_t end, std::size_t leaf_size) {
    const std::uint32_t node_index = std::uint32_t(nodes_.size());
    nodes_.emplace_back();
    Box box = tets_[order_[begin]].box;
    Vec3<T> cmin = tets_[order_[begin]].centroid;
    Vec3<T> cmax = cmin;
    for (std::size_t i = begin + 1; i < end; ++i) {
      const Tet& t = tets_[order_[i]];
      grow(box, t.box);
      cmin.x = std::min(cmin.x, t.centroid.x);
      cmin.y = std::min(cmin.y, t.centroid.y);
      cmin.z = std::min(cmin.z, t.centroid.z);
      cmax.x = std::max(cmax.x, t.centroid.x);
      cmax.y = std::max(cmax.y, t.centroid.y);
      cmax.z = std::max(cmax.z, t.centroid.z);
    }
    nodes_[node_index].box = box;
    nodes_[node_index].begin = begin;
    nodes_[node_index].end = end;
    const Vec3<T> span = cmax - cmin;
    const std::size_t axis = span.x >= span.y && span.x >= span.z ? 0 : (span.y >= span.z ? 1 : 2);
    if (end - begin <= leaf_size || !(component(span, axis) > T(0))) return node_index;
    const std::size_t middle = begin + (end - begin) / 2;
    std::nth_element(order_.begin() + begin, order_.begin() + middle,
                     order_.begin() + end, [&](std::uint32_t a, std::uint32_t b) {
      const T ca = component(tets_[a].centroid, axis), cb = component(tets_[b].centroid, axis);
      return ca < cb || (ca == cb && a < b);
    });
    const std::uint32_t left = build_node(begin, middle, leaf_size);
    const std::uint32_t right = build_node(middle, end, leaf_size);
    nodes_[node_index].left = left;
    nodes_[node_index].right = right;
    nodes_[node_index].leaf = false;
    return node_index;
  }

  static std::array<T, 4> barycentric(const Tet& tet, Vec3<T> p) noexcept {
    const Vec3<T> delta = p - tet.origin;
    const T w1 = dot(delta, tet.gradient[0]);
    const T w2 = dot(delta, tet.gradient[1]);
    const T w3 = dot(delta, tet.gradient[2]);
    return {{T(1) - w1 - w2 - w3, w1, w2, w3}};
  }
  void make_binding(std::uint32_t tet_id, std::array<T, 4> weights,
                    Binding& out) const noexcept {
    T total = T(0);
    for (T& w : weights) { w = std::max(T(0), w); total += w; }
    for (T& w : weights) w /= total;
    out.vertices = tets_[tet_id].ids;
    out.weights = weights;
    out.tetrahedron = tet_id;
  }
  bool find_containing(std::uint32_t node_index, Vec3<T> p, T box_tolerance,
                       T bary_tolerance, Binding& out) const noexcept {
    const Node& node = nodes_[node_index];
    if (distance2(node.box, p) > box_tolerance * box_tolerance) return false;
    if (!node.leaf) {
      return find_containing(node.left, p, box_tolerance, bary_tolerance, out) ||
             find_containing(node.right, p, box_tolerance, bary_tolerance, out);
    }
    for (std::size_t i = node.begin; i < node.end; ++i) {
      const std::uint32_t tet_id = order_[i];
      if (distance2(tets_[tet_id].box, p) > box_tolerance * box_tolerance) continue;
      std::array<T, 4> weights = barycentric(tets_[tet_id], p);
      if (std::all_of(weights.begin(), weights.end(), [&](T w) {
            return w >= -bary_tolerance && w <= T(1) + bary_tolerance;
          })) {
        make_binding(tet_id, weights, out);
        return true;
      }
    }
    return false;
  }

  // Closest point on a triangle, with barycentric weights (Ericson's Voronoi regions).
  static std::array<T, 3> closest_triangle(Vec3<T> p, Vec3<T> a,
                                           Vec3<T> b, Vec3<T> c) noexcept {
    const Vec3<T> ab = b - a, ac = c - a, ap = p - a;
    const T d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) return {{T(1), T(0), T(0)}};
    const Vec3<T> bp = p - b;
    const T d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return {{T(0), T(1), T(0)}};
    const T vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
      const T v = d1 / (d1 - d3);
      return {{T(1) - v, v, T(0)}};
    }
    const Vec3<T> cp = p - c;
    const T d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return {{T(0), T(0), T(1)}};
    const T vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
      const T w = d2 / (d2 - d6);
      return {{T(1) - w, T(0), w}};
    }
    const T va = d3 * d6 - d5 * d4;
    if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
      const T w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
      return {{T(0), T(1) - w, w}};
    }
    const T inv = T(1) / (va + vb + vc);
    const T v = vb * inv, w = vc * inv;
    return {{T(1) - v - w, v, w}};
  }
  void nearest_on_tet(std::uint32_t tet_id, Vec3<T> p,
                      T& best_distance2, Binding& out) const noexcept {
    const Tet& tet = tets_[tet_id];
    static constexpr std::size_t faces[4][3] = {{0, 1, 2}, {0, 1, 3},
                                                 {0, 2, 3}, {1, 2, 3}};
    for (const auto& face : faces) {
      const Vec3<T> a = rest_[tet.ids[face[0]]], b = rest_[tet.ids[face[1]]];
      const Vec3<T> c = rest_[tet.ids[face[2]]];
      const auto uvw = closest_triangle(p, a, b, c);
      const Vec3<T> q = a * uvw[0] + b * uvw[1] + c * uvw[2];
      const T d2 = dot(q - p, q - p);
      if (d2 < best_distance2) {
        best_distance2 = d2;
        std::array<T, 4> weights{};
        for (std::size_t j = 0; j < 3; ++j) weights[face[j]] = uvw[j];
        make_binding(tet_id, weights, out);
      }
    }
  }
  void find_closest(std::uint32_t node_id, Vec3<T> p,
                    T& best_distance2, Binding& out) const noexcept {
    const Node& node = nodes_[node_id];
    if (distance2(node.box, p) > best_distance2) return;
    if (node.leaf) {
      for (std::size_t i = node.begin; i < node.end; ++i) {
        const std::uint32_t tet_id = order_[i];
        if (distance2(tets_[tet_id].box, p) <= best_distance2)
          nearest_on_tet(tet_id, p, best_distance2, out);
      }
      return;
    }
    const T left_d = distance2(nodes_[node.left].box, p);
    const T right_d = distance2(nodes_[node.right].box, p);
    if (left_d <= right_d) {
      find_closest(node.left, p, best_distance2, out);
      find_closest(node.right, p, best_distance2, out);
    } else {
      find_closest(node.right, p, best_distance2, out);
      find_closest(node.left, p, best_distance2, out);
    }
  }

  std::vector<Vec3<T>> rest_;
  std::vector<Tet> tets_;
  std::vector<std::uint32_t> order_;
  std::vector<Node> nodes_;
  std::vector<Binding> bindings_;
};

} // namespace deform3d
