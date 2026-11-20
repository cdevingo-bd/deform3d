#pragma once

// Engine-independent dense surface attachment to triangle cloth/shell meshes.
// C++17, standard library only. The host owns shell physics and contact.
#include "deform3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

namespace deform3d {

enum class SurfaceEmbedStatus {
  Ok, InvalidInput, NonFiniteInput, InvalidIndex, DegenerateTriangle,
  OutsideSurface, InvalidRange, Aliasing, NumericalFailure
};

inline const char* surface_embed_status_string(SurfaceEmbedStatus status) noexcept {
  switch (status) {
    case SurfaceEmbedStatus::Ok: return "ok";
    case SurfaceEmbedStatus::InvalidInput: return "invalid input";
    case SurfaceEmbedStatus::NonFiniteInput: return "non-finite input";
    case SurfaceEmbedStatus::InvalidIndex: return "triangle index out of range";
    case SurfaceEmbedStatus::DegenerateTriangle: return "degenerate or collapsed triangle";
    case SurfaceEmbedStatus::OutsideSurface: return "point outside permitted surface attachment";
    case SurfaceEmbedStatus::InvalidRange: return "invalid evaluation range";
    case SurfaceEmbedStatus::Aliasing: return "input and output buffers overlap";
    case SurfaceEmbedStatus::NumericalFailure: return "result is not representable";
  }
  return "unknown";
}

enum class SurfaceOutsidePolicy {
  ProjectToSurface,
  PreserveNormalOffset
};

template<class T> struct SurfaceEmbedOptions {
  SurfaceOutsidePolicy outside = SurfaceOutsidePolicy::PreserveNormalOffset;
  // Absolute distance to the closest point on a triangle, in scene units.
  // Must be finite. The default only accepts points exactly on the surface.
  T max_distance = T(0);
  // PreserveNormalOffset cannot represent tangential displacement beyond an
  // open edge. Permit only this much residual relative to triangle edge length.
  T tangential_tolerance = T(64) * std::numeric_limits<T>::epsilon();
  std::size_t bvh_leaf_size = 8;
};

template<class T> class SurfaceEmbedding {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "SurfaceEmbedding needs float or double");
  using L = long double;
  using P = Vec3<L>;

 public:
  struct Binding {
    std::array<std::uint32_t, 3> vertices{};
    std::array<T, 3> weights{};
    std::uint32_t triangle = 0;
    T normal_offset = T(0);
  };

  // Transactional: every source vertex, triangle and render point is validated
  // before replacing an existing mapping. error_index identifies the offending
  // source/render vertex or triangle, according to the status and input phase.
  SurfaceEmbedStatus prepare(const Vec3<T>* rest_sim_vertices,
                             std::size_t sim_vertex_count,
                             const std::uint32_t* triangles,
                             std::size_t triangle_count,
                             const Vec3<T>* rest_render_vertices,
                             std::size_t render_vertex_count,
                             SurfaceEmbedOptions<T> options = {},
                             std::size_t* error_index = nullptr) {
    if (error_index) *error_index = 0;
    if (!rest_sim_vertices || !triangles || !sim_vertex_count || !triangle_count ||
        (render_vertex_count && !rest_render_vertices) || !options.bvh_leaf_size ||
        !std::isfinite(options.max_distance) || options.max_distance < T(0) ||
        !std::isfinite(options.tangential_tolerance) || options.tangential_tolerance < T(0) ||
        (options.outside != SurfaceOutsidePolicy::ProjectToSurface &&
         options.outside != SurfaceOutsidePolicy::PreserveNormalOffset) ||
        sim_vertex_count > std::numeric_limits<std::uint32_t>::max() ||
        sim_vertex_count > std::numeric_limits<std::size_t>::max() / sizeof(Vec3<T>) ||
        render_vertex_count > std::numeric_limits<std::size_t>::max() / sizeof(Vec3<T>) ||
        triangle_count > std::numeric_limits<std::uint32_t>::max() / 2 ||
        triangle_count > std::numeric_limits<std::size_t>::max() / 3) {
      return SurfaceEmbedStatus::InvalidInput;
    }
    SurfaceEmbedding candidate;
    candidate.vertex_count_ = sim_vertex_count;
    for (std::size_t i = 0; i < sim_vertex_count; ++i) {
      if (!finite(rest_sim_vertices[i])) {
        if (error_index) *error_index = i;
        return SurfaceEmbedStatus::NonFiniteInput;
      }
    }
    candidate.triangles_.resize(triangle_count);
    candidate.order_.resize(triangle_count);
    std::iota(candidate.order_.begin(), candidate.order_.end(), std::uint32_t(0));
    for (std::size_t i = 0; i < triangle_count; ++i) {
      Triangle& triangle = candidate.triangles_[i];
      for (std::size_t j = 0; j < 3; ++j) {
        triangle.ids[j] = triangles[3 * i + j];
        if (triangle.ids[j] >= sim_vertex_count) {
          if (error_index) *error_index = i;
          return SurfaceEmbedStatus::InvalidIndex;
        }
        triangle.rest[j] = wide(rest_sim_vertices[triangle.ids[j]]);
      }
      L edge = 0;
      if (!unit_normal(triangle.rest[0], triangle.rest[1], triangle.rest[2], triangle.normal, edge)) {
        if (error_index) *error_index = i;
        return SurfaceEmbedStatus::DegenerateTriangle;
      }
      triangle.edge = edge;
      triangle.box = {triangle.rest[0], triangle.rest[0]};
      grow(triangle.box, triangle.rest[1]);
      grow(triangle.box, triangle.rest[2]);
      triangle.centroid = triangle.rest[0] / L(3) + triangle.rest[1] / L(3) + triangle.rest[2] / L(3);
    }
    candidate.nodes_.reserve(2 * triangle_count);
    candidate.build_node(0, triangle_count, options.bvh_leaf_size);
    candidate.bindings_.resize(render_vertex_count);
    for (std::size_t i = 0; i < render_vertex_count; ++i) {
      if (!finite(rest_render_vertices[i])) {
        if (error_index) *error_index = i;
        return SurfaceEmbedStatus::NonFiniteInput;
      }
      const P point = wide(rest_render_vertices[i]);
      L best2 = std::numeric_limits<L>::infinity();
      std::uint32_t nearest = std::numeric_limits<std::uint32_t>::max();
      std::array<L, 3> weights{};
      candidate.find_closest(0, point, best2, nearest, weights);
      if (!std::isfinite(best2) || std::sqrt(best2) > L(options.max_distance)) {
        if (error_index) *error_index = i;
        return SurfaceEmbedStatus::OutsideSurface;
      }
      const Triangle& triangle = candidate.triangles_[nearest];
      const P projected = triangle.rest[0] * weights[0] + triangle.rest[1] * weights[1] +
                          triangle.rest[2] * weights[2];
      const P residual = point - projected;
      const L offset = dot(residual, triangle.normal);
      if (!std::isfinite(offset) || std::abs(offset) > L(std::numeric_limits<T>::max())) {
        if (error_index) *error_index = i;
        return SurfaceEmbedStatus::NumericalFailure;
      }
      if (options.outside == SurfaceOutsidePolicy::PreserveNormalOffset &&
          length(residual - triangle.normal * offset) > L(options.tangential_tolerance) * triangle.edge) {
        if (error_index) *error_index = i;
        return SurfaceEmbedStatus::OutsideSurface;
      }
      Binding& binding = candidate.bindings_[i];
      binding.vertices = triangle.ids;
      binding.triangle = nearest;
      for (std::size_t j = 0; j < 3; ++j) binding.weights[j] = T(weights[j]);
      // Store a partition of unity in the chosen scalar precision.
      const T sum = binding.weights[0] + binding.weights[1] + binding.weights[2];
      for (T& w : binding.weights) w /= sum;
      binding.normal_offset = options.outside == SurfaceOutsidePolicy::PreserveNormalOffset ? T(offset) : T(0);
    }
    swap(candidate);
    return SurfaceEmbedStatus::Ok;
  }

  SurfaceEmbedStatus evaluate(const Vec3<T>* current_sim_vertices,
                              std::size_t sim_vertex_count,
                              Vec3<T>* output_render_vertices,
                              std::size_t output_count) const noexcept {
    return evaluate_range(current_sim_vertices, sim_vertex_count, output_render_vertices,
                          output_count, 0, bindings_.size());
  }

  // Caller-owned buffers, no allocations. All source vertices/triangles and
  // selected results are checked before any output write. Collapsed triangles
  // fail even with zero offsets. Disjoint output ranges can run concurrently.
  SurfaceEmbedStatus evaluate_range(const Vec3<T>* current_sim_vertices,
                                    std::size_t sim_vertex_count,
                                    Vec3<T>* output_render_vertices,
                                    std::size_t output_count,
                                    std::size_t begin,
                                    std::size_t end) const noexcept {
    if (!ready() || !current_sim_vertices || !output_render_vertices ||
        sim_vertex_count != vertex_count_ || output_count < bindings_.size() ||
        output_count > std::numeric_limits<std::size_t>::max() / sizeof(Vec3<T>))
      return SurfaceEmbedStatus::InvalidInput;
    if (begin > end || end > bindings_.size()) return SurfaceEmbedStatus::InvalidRange;
    if (overlap(current_sim_vertices, sim_vertex_count * sizeof(Vec3<T>),
                output_render_vertices, output_count * sizeof(Vec3<T>))) return SurfaceEmbedStatus::Aliasing;
    const auto read = [=](std::size_t i) noexcept { return wide(current_sim_vertices[i]); };
    const auto write = [=](std::size_t i, P p) noexcept {
      output_render_vertices[i] = {T(p.x), T(p.y), T(p.z)};
    };
    return evaluate_impl(read, write, L(std::numeric_limits<T>::max()), begin, end);
  }

  template<class InputScalar, class OutputScalar>
  SurfaceEmbedStatus evaluate_strided(
      const InputScalar* current_xyz, std::size_t sim_vertex_count, std::size_t input_stride,
      OutputScalar* output_xyz, std::size_t output_count, std::size_t output_stride,
      std::size_t begin = 0, std::size_t end = std::numeric_limits<std::size_t>::max()) const noexcept {
    static_assert(std::is_floating_point<InputScalar>::value && std::is_floating_point<OutputScalar>::value,
                  "XYZ scalars must be floating point");
    if (!ready() || !current_xyz || !output_xyz || sim_vertex_count != vertex_count_ ||
        output_count < bindings_.size() || input_stride < 3 || output_stride < 3 ||
        input_stride > std::numeric_limits<std::size_t>::max() / sizeof(InputScalar) ||
        output_stride > std::numeric_limits<std::size_t>::max() / sizeof(OutputScalar) ||
        sim_vertex_count > std::numeric_limits<std::size_t>::max() / (input_stride * sizeof(InputScalar)) ||
        output_count > std::numeric_limits<std::size_t>::max() / (output_stride * sizeof(OutputScalar)))
      return SurfaceEmbedStatus::InvalidInput;
    if (end == std::numeric_limits<std::size_t>::max()) end = bindings_.size();
    if (begin > end || end > bindings_.size()) return SurfaceEmbedStatus::InvalidRange;
    if (overlap(current_xyz, sim_vertex_count * input_stride * sizeof(InputScalar),
                output_xyz, output_count * output_stride * sizeof(OutputScalar))) return SurfaceEmbedStatus::Aliasing;
    const auto read = [=](std::size_t i) noexcept {
      const InputScalar* p = current_xyz + i * input_stride;
      return P{L(p[0]), L(p[1]), L(p[2])};
    };
    const auto write = [=](std::size_t i, P p) noexcept {
      OutputScalar* out = output_xyz + i * output_stride;
      out[0] = OutputScalar(p.x); out[1] = OutputScalar(p.y); out[2] = OutputScalar(p.z);
    };
    return evaluate_impl(read, write, L(std::numeric_limits<OutputScalar>::max()), begin, end);
  }

  bool ready() const noexcept { return !nodes_.empty(); }
  std::size_t simulation_vertex_count() const noexcept { return vertex_count_; }
  std::size_t render_vertex_count() const noexcept { return bindings_.size(); }
  std::size_t triangle_count() const noexcept { return triangles_.size(); }
  const std::vector<Binding>& bindings() const noexcept { return bindings_; }
  void swap(SurfaceEmbedding& other) noexcept {
    std::swap(vertex_count_, other.vertex_count_);
    triangles_.swap(other.triangles_); order_.swap(other.order_);
    nodes_.swap(other.nodes_); bindings_.swap(other.bindings_);
  }

 private:
  struct Box { P lo{}, hi{}; };
  struct Triangle {
    std::array<std::uint32_t, 3> ids{};
    std::array<P, 3> rest{};
    P normal{}, centroid{};
    L edge = 0;
    Box box{};
  };
  struct Node {
    Box box{};
    std::size_t begin = 0, end = 0;
    std::uint32_t left = 0, right = 0;
    bool leaf = true;
  };
  template<class U> static P wide(Vec3<U> p) noexcept { return {L(p.x), L(p.y), L(p.z)}; }
  static L length(P p) noexcept { return std::hypot(p.x, p.y, p.z); }
  static bool unit_normal(P a, P b, P c, P& normal, L& edge) noexcept {
    const P ab = b - a, ac = c - a;
    edge = std::max({length(ab), length(ac), length(c - b)});
    if (!(edge > 0) || !std::isfinite(edge)) return false;
    const P cross_scaled = cross(ab / edge, ac / edge);
    const L magnitude = length(cross_scaled);
    if (!std::isfinite(magnitude) || magnitude <= L(64) * std::numeric_limits<T>::epsilon()) return false;
    normal = cross_scaled / magnitude;
    return true;
  }
  static bool overlap(const void* p, std::size_t p_bytes, const void* q, std::size_t q_bytes) noexcept {
    const auto a = reinterpret_cast<std::uintptr_t>(p), b = reinterpret_cast<std::uintptr_t>(q);
    return p_bytes && q_bytes && (a <= b ? b - a < p_bytes : a - b < q_bytes);
  }
  static void grow(Box& box, P p) noexcept {
    box.lo.x = std::min(box.lo.x, p.x); box.hi.x = std::max(box.hi.x, p.x);
    box.lo.y = std::min(box.lo.y, p.y); box.hi.y = std::max(box.hi.y, p.y);
    box.lo.z = std::min(box.lo.z, p.z); box.hi.z = std::max(box.hi.z, p.z);
  }
  static L component(P p, std::size_t axis) noexcept { return axis == 0 ? p.x : axis == 1 ? p.y : p.z; }
  static L distance2(const Box& box, P p) noexcept {
    const L dx = std::max({box.lo.x - p.x, L(0), p.x - box.hi.x});
    const L dy = std::max({box.lo.y - p.y, L(0), p.y - box.hi.y});
    const L dz = std::max({box.lo.z - p.z, L(0), p.z - box.hi.z});
    return dx * dx + dy * dy + dz * dz;
  }
  std::uint32_t build_node(std::size_t begin, std::size_t end, std::size_t leaf_size) {
    const auto id = std::uint32_t(nodes_.size());
    nodes_.emplace_back();
    Box box = triangles_[order_[begin]].box;
    Box centers{triangles_[order_[begin]].centroid, triangles_[order_[begin]].centroid};
    for (std::size_t i = begin + 1; i < end; ++i) {
      const Triangle& triangle = triangles_[order_[i]];
      grow(box, triangle.box.lo); grow(box, triangle.box.hi); grow(centers, triangle.centroid);
    }
    nodes_[id].box = box; nodes_[id].begin = begin; nodes_[id].end = end;
    const P extent = centers.hi - centers.lo;
    const std::size_t axis = extent.x >= extent.y && extent.x >= extent.z ? 0 : extent.y >= extent.z ? 1 : 2;
    if (end - begin <= leaf_size || !(component(extent, axis) > 0)) return id;
    const std::size_t middle = begin + (end - begin) / 2;
    std::nth_element(order_.begin() + begin, order_.begin() + middle, order_.begin() + end,
        [&](std::uint32_t a, std::uint32_t b) {
          const L x = component(triangles_[a].centroid, axis), y = component(triangles_[b].centroid, axis);
          return x < y || (x == y && a < b);
        });
    const auto left = build_node(begin, middle, leaf_size), right = build_node(middle, end, leaf_size);
    nodes_[id].left = left; nodes_[id].right = right; nodes_[id].leaf = false;
    return id;
  }
  // Ericson's triangle Voronoi regions, with normalized local coordinates to
  // keep intermediate products in range even for very large/small scene units.
  static std::array<L, 3> closest_weights(const Triangle& triangle, P point) noexcept {
    const P ab = (triangle.rest[1] - triangle.rest[0]) / triangle.edge;
    const P ac = (triangle.rest[2] - triangle.rest[0]) / triangle.edge;
    const P ap = (point - triangle.rest[0]) / triangle.edge;
    const L d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) return {{1,0,0}};
    const P bp = ap - ab;
    const L d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return {{0,1,0}};
    const L vc = d1*d4 - d3*d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
      const L v = d1 / (d1 - d3); return {{1-v,v,0}};
    }
    const P cp = ap - ac;
    const L d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return {{0,0,1}};
    const L vb = d5*d2 - d1*d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
      const L w = d2 / (d2 - d6); return {{1-w,0,w}};
    }
    const L va = d3*d6 - d5*d4;
    if (va <= 0 && d4-d3 >= 0 && d5-d6 >= 0) {
      const L w = (d4-d3) / ((d4-d3)+(d5-d6)); return {{0,1-w,w}};
    }
    const L inverse = 1 / (va + vb + vc), v = vb * inverse, w = vc * inverse;
    return {{1-v-w,v,w}};
  }
  void find_closest(std::uint32_t node_id, P point, L& best2,
                    std::uint32_t& nearest, std::array<L, 3>& weights) const noexcept {
    const Node& node = nodes_[node_id];
    if (distance2(node.box, point) > best2) return;
    if (node.leaf) {
      for (std::size_t i = node.begin; i < node.end; ++i) {
        const std::uint32_t id = order_[i];
        const Triangle& triangle = triangles_[id];
        if (distance2(triangle.box, point) > best2) continue;
        auto candidate = closest_weights(triangle, point);
        L total = 0;
        for (L& w : candidate) { w = std::max(L(0), w); total += w; }
        if (!(total > 0) || !std::isfinite(total)) continue;
        for (L& w : candidate) w /= total;
        const P q = triangle.rest[0] * candidate[0] + triangle.rest[1] * candidate[1] + triangle.rest[2] * candidate[2];
        const P delta = point - q;
        const L d2 = dot(delta, delta);
        if (d2 < best2 || (d2 == best2 && id < nearest)) {
          best2 = d2; nearest = id; weights = candidate;
        }
      }
      return;
    }
    const L left = distance2(nodes_[node.left].box, point), right = distance2(nodes_[node.right].box, point);
    if (left <= right) {
      find_closest(node.left, point, best2, nearest, weights);
      find_closest(node.right, point, best2, nearest, weights);
    } else {
      find_closest(node.right, point, best2, nearest, weights);
      find_closest(node.left, point, best2, nearest, weights);
    }
  }
  template<class Read> P evaluate_point(const Binding& binding, Read read) const noexcept {
    const P a = read(binding.vertices[0]), b = read(binding.vertices[1]), c = read(binding.vertices[2]);
    P result = a * L(binding.weights[0]) + b * L(binding.weights[1]) + c * L(binding.weights[2]);
    if (binding.normal_offset != T(0)) {
      P normal; L edge;
      unit_normal(a, b, c, normal, edge);  // all triangles have already been validated
      result += normal * L(binding.normal_offset);
    }
    return result;
  }
  template<class Read, class Write>
  SurfaceEmbedStatus evaluate_impl(Read read, Write write, L output_limit,
                                   std::size_t begin, std::size_t end) const noexcept {
    for (std::size_t i = 0; i < vertex_count_; ++i)
      if (!finite(read(i))) return SurfaceEmbedStatus::NonFiniteInput;
    for (const Triangle& triangle : triangles_) {
      P normal; L edge;
      if (!unit_normal(read(triangle.ids[0]), read(triangle.ids[1]), read(triangle.ids[2]), normal, edge))
        return SurfaceEmbedStatus::DegenerateTriangle;
    }
    // A separate representability pass guarantees no partial writes even if
    // an offset or output scalar conversion would overflow.
    for (std::size_t i = begin; i < end; ++i) {
      const P p = evaluate_point(bindings_[i], read);
      if (!finite(p) || std::abs(p.x) > output_limit || std::abs(p.y) > output_limit || std::abs(p.z) > output_limit)
        return SurfaceEmbedStatus::NumericalFailure;
    }
    for (std::size_t i = begin; i < end; ++i) write(i, evaluate_point(bindings_[i], read));
    return SurfaceEmbedStatus::Ok;
  }

  std::size_t vertex_count_ = 0;
  std::vector<Triangle> triangles_;
  std::vector<std::uint32_t> order_;
  std::vector<Node> nodes_;
  std::vector<Binding> bindings_;
};

} // namespace deform3d
