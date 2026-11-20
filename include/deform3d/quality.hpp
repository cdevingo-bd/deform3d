#ifndef DEFORM3D_QUALITY_HPP
#define DEFORM3D_QUALITY_HPP

// Optional diagnostics for tetrahedral deformation. The physics engine owns
// motion; this pass reports element inversion and stretch without allocations.
#include "deform3d.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace deform3d {

struct TetraQualityReport {
  size_t tetrahedra = 0;
  size_t inverted = 0;          // signed volume ratio < 0
  size_t near_zero_volume = 0;  // absolute ratio <= requested threshold
  long double min_volume_ratio = 0;
  long double max_volume_ratio = 0;
  long double min_edge_stretch = 0;  // current length / rest length
  long double max_edge_stretch = 0;
};

// Prepare once in rest state; evaluate after each external simulation step.
// This is a diagnostic, not a constitutive law or contact solver. Each tetra
// and each of its six edges are visited, O(tetrahedra), without frame allocation.
template<class T> class TetraQuality {
  static_assert(std::is_floating_point<T>::value,
                "TetraQuality requires floating point simulation positions");
 public:
  Status prepare(const Vec3<T>* rest_vertices, size_t vertex_count,
                 const uint32_t* tetrahedra, size_t tetrahedron_count) {
    if (!vertex_count || !tetrahedron_count || !rest_vertices || !tetrahedra ||
        vertex_count > std::numeric_limits<uint32_t>::max() ||
        tetrahedron_count > std::numeric_limits<size_t>::max()/4)
      return Status::InvalidArgument;
    for (size_t i=0;i<vertex_count;++i)
      if (!finite(rest_vertices[i])) return Status::NonFinite;
    std::vector<Tet> candidate;
    candidate.reserve(tetrahedron_count);
    for (size_t t=0;t<tetrahedron_count;++t) {
      Tet tet{};
      for (size_t j=0;j<4;++j) {
        tet.ids[j]=tetrahedra[4*t+j];
        if (tet.ids[j]>=vertex_count) return Status::InvalidIndex;
      }
      const auto p=read_positions(rest_vertices,tet.ids);
      tet.rest_volume6=determinant(p);
      long double max_edge=0;
      for (size_t e=0;e<6;++e) {
        tet.rest_edges[e]=distance(p[edges[e][0]],p[edges[e][1]]);
        if (!std::isfinite(tet.rest_edges[e])) return Status::NonFinite;
        max_edge=std::max(max_edge,tet.rest_edges[e]);
      }
      const long double threshold=64*std::numeric_limits<T>::epsilon()*
                                  max_edge*max_edge*max_edge;
      if (!std::isfinite(tet.rest_volume6)) return Status::NonFinite;
      if (!(max_edge>0) || std::abs(tet.rest_volume6)<=threshold)
        return Status::DegenerateGeometry;
      candidate.push_back(tet);
    }
    tets_.swap(candidate);
    vertices_=vertex_count;
    return Status::Ok;
  }

  size_t vertex_count() const noexcept {return vertices_;}
  size_t tetrahedron_count() const noexcept {return tets_.size();}

  // Runtime XYZ values may be float or double regardless of rest-data type.
  // Strides are measured in scalar elements, as with mjData::flexvert_xpos.
  template<class U> Status evaluate_strided(const U* current_xyz, size_t vertex_count,
                                            size_t stride, TetraQualityReport& report,
                                            long double near_zero_ratio=1e-4L) const noexcept {
    static_assert(std::is_floating_point<U>::value, "XYZ values must be float or double");
    if (tets_.empty() || !current_xyz || vertex_count!=vertices_ || stride<3 ||
        stride>std::numeric_limits<size_t>::max()/sizeof(U) ||
        vertex_count>std::numeric_limits<size_t>::max()/(stride*sizeof(U)) ||
        !std::isfinite(near_zero_ratio) || near_zero_ratio<0)
      return Status::InvalidArgument;
    return evaluate_impl([=](size_t i) noexcept {
      const U* p=current_xyz+i*stride;
      return Vec3<long double>{static_cast<long double>(p[0]),
                               static_cast<long double>(p[1]),
                               static_cast<long double>(p[2])};
    },report,near_zero_ratio);
  }

  Status evaluate(const Vec3<T>* current_vertices, size_t vertex_count,
                  TetraQualityReport& report,
                  long double near_zero_ratio=1e-4L) const noexcept {
    if (tets_.empty() || !current_vertices || vertex_count!=vertices_ ||
        !std::isfinite(near_zero_ratio) || near_zero_ratio<0)
      return Status::InvalidArgument;
    return evaluate_impl([=](size_t i) noexcept {
      const Vec3<T>& p=current_vertices[i];
      return Vec3<long double>{static_cast<long double>(p.x),
                               static_cast<long double>(p.y),
                               static_cast<long double>(p.z)};
    },report,near_zero_ratio);
  }

 private:
  struct Tet {
    std::array<uint32_t,4> ids{};
    long double rest_volume6=0;
    std::array<long double,6> rest_edges{};
  };
  static constexpr size_t edges[6][2]={{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

  static long double distance(Vec3<long double> a, Vec3<long double> b) noexcept {
    const Vec3<long double> d=a-b;
    return std::sqrt(dot(d,d));
  }
  static long double determinant(const std::array<Vec3<long double>,4>& p) noexcept {
    const auto a=p[1]-p[0], b=p[2]-p[0], c=p[3]-p[0];
    return dot(a,cross(b,c));
  }
  static std::array<Vec3<long double>,4> read_positions(
      const Vec3<T>* vertices, const std::array<uint32_t,4>& ids) noexcept {
    std::array<Vec3<long double>,4> result{};
    for (size_t j=0;j<4;++j) {
      const auto& p=vertices[ids[j]];
      result[j]={static_cast<long double>(p.x),static_cast<long double>(p.y),
                 static_cast<long double>(p.z)};
    }
    return result;
  }
  template<class Reader> Status evaluate_impl(Reader read,
                                               TetraQualityReport& report,
                                               long double threshold) const noexcept {
    TetraQualityReport next{};
    next.tetrahedra=tets_.size();
    next.min_volume_ratio=std::numeric_limits<long double>::infinity();
    next.max_volume_ratio=-std::numeric_limits<long double>::infinity();
    next.min_edge_stretch=std::numeric_limits<long double>::infinity();
    for (const Tet& tet:tets_) {
      std::array<Vec3<long double>,4> p{};
      for (size_t j=0;j<4;++j) {
        p[j]=read(tet.ids[j]);
        if (!finite(p[j])) return Status::NonFinite;
      }
      const long double ratio=determinant(p)/tet.rest_volume6;
      if (!std::isfinite(ratio)) return Status::NonFinite;
      next.min_volume_ratio=std::min(next.min_volume_ratio,ratio);
      next.max_volume_ratio=std::max(next.max_volume_ratio,ratio);
      if (ratio<0) ++next.inverted;
      if (std::abs(ratio)<=threshold) ++next.near_zero_volume;
      for (size_t e=0;e<6;++e) {
        const long double stretch=distance(p[edges[e][0]],p[edges[e][1]])/
                                  tet.rest_edges[e];
        if (!std::isfinite(stretch)) return Status::NonFinite;
        next.min_edge_stretch=std::min(next.min_edge_stretch,stretch);
        next.max_edge_stretch=std::max(next.max_edge_stretch,stretch);
      }
    }
    report=next;
    return Status::Ok;
  }

  size_t vertices_=0;
  std::vector<Tet> tets_;
};

} // namespace deform3d
#endif // DEFORM3D_QUALITY_HPP
