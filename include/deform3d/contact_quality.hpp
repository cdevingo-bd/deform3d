#ifndef DEFORM3D_CONTACT_QUALITY_HPP
#define DEFORM3D_CONTACT_QUALITY_HPP

// Analytic face-interior reference for collision diagnostics. This is not a
// general triangle collider: edge/vertex features require a host narrowphase.
#include "deform3d.hpp"

namespace deform3d {

enum class FaceContactStatus { Interior, OutsideFace, Degenerate, InvalidInput };

// The ellipsoid is center + sum(axes[i]*u[i]), where dot(u,u) <= 1.
// Columns may be rotated, reflected or sheared, but must span three dimensions.
template<class T> struct EllipsoidShape {
  Vec3<T> center{};
  std::array<Vec3<T>,3> axes{{{T(1),T(0),T(0)}, {T(0),T(1),T(0)}, {T(0),T(0),T(1)}}};
};

template<class T> struct FaceContact {
  T distance{};                    // negative for penetration
  Vec3<T> normal{};                 // from ellipsoid toward triangle
  Vec3<T> ellipsoid_witness{};
  Vec3<T> triangle_witness{};       // on the radius-inflated triangle face
  Vec3<T> position{};               // midpoint of the two witnesses
  std::array<T,3> barycentric{};    // underlying uninflated triangle face
};

// Allocation-free, failure-transactional face contact oracle. A successful
// result describes the planar feature only. It does not add contacts or forces
// to a simulator. At exactly zero plane/center distance, winding selects a side.
template<class T>
FaceContactStatus ellipsoid_triangle_face(const EllipsoidShape<T>& ellipsoid,
    const std::array<Vec3<T>,3>& triangle, T triangle_radius, FaceContact<T>& output) noexcept {
  static_assert(std::is_floating_point<T>::value, "floating point geometry required");
  if (!finite(ellipsoid.center) || !finite(triangle_radius) || triangle_radius < T(0))
    return FaceContactStatus::InvalidInput;
  for (const auto p : triangle) if (!finite(p)) return FaceContactStatus::InvalidInput;
  for (const auto p : ellipsoid.axes) if (!finite(p)) return FaceContactStatus::InvalidInput;
  using W = long double;
  using V = Vec3<W>;
  const auto wide = [](Vec3<T> p) noexcept -> V {return {W(p.x),W(p.y),W(p.z)};};
  const auto magnitude = [](V p) noexcept -> W {
    return std::max({std::abs(p.x),std::abs(p.y),std::abs(p.z)});
  };
  const V a=wide(triangle[0]), center=wide(ellipsoid.center);
  V e1=wide(triangle[1])-a, e2=wide(triangle[2])-a;
  const W edge_scale=std::max(magnitude(e1),magnitude(e2));
  if (!(edge_scale>0) || !finite(edge_scale)) return FaceContactStatus::Degenerate;
  e1=e1/edge_scale; e2=e2/edge_scale;
  V n=cross(e1,e2);
  const W area=norm(n), threshold=W(64)*std::numeric_limits<T>::epsilon();
  if (!finite(area) || area<=threshold) return FaceContactStatus::Degenerate;
  n=n/area;
  const V relative=center-a;
  if (dot(relative,n)>0) n=-n;
  const W signed_area=dot(cross(e1,e2),n);

  std::array<V,3> axes{{wide(ellipsoid.axes[0]),wide(ellipsoid.axes[1]),wide(ellipsoid.axes[2])}};
  const W axis_scale=std::max({magnitude(axes[0]),magnitude(axes[1]),magnitude(axes[2])});
  if (!(axis_scale>0) || !finite(axis_scale)) return FaceContactStatus::Degenerate;
  for (auto& axis : axes) axis=axis/axis_scale;
  if (std::abs(dot(axes[0],cross(axes[1],axes[2])))<=threshold)
    return FaceContactStatus::Degenerate;
  const V coefficients{dot(axes[0],n),dot(axes[1],n),dot(axes[2],n)};
  const W support_radius=norm(coefficients);
  if (!(support_radius>0) || !finite(support_radius)) return FaceContactStatus::Degenerate;
  const V support=(axes[0]*coefficients.x+axes[1]*coefficients.y+axes[2]*coefficients.z)
                  *(axis_scale/support_radius);
  const W gap=-dot(relative,n)-axis_scale*support_radius-W(triangle_radius);
  const V projection=(relative+support+n*(gap+W(triangle_radius)))/edge_scale;
  const W b1=dot(cross(projection,e2),n)/signed_area;
  const W b2=dot(cross(e1,projection),n)/signed_area;
  const W b0=W(1)-b1-b2;
  if (!finite(gap) || !finite(b0) || !finite(b1) || !finite(b2))
    return FaceContactStatus::InvalidInput;
  // No clamping to an edge: that would silently turn a planar result into an
  // unverified edge contact. The caller must fall back when the feature changes.
  if (b0<0 || b1<0 || b2<0) return FaceContactStatus::OutsideFace;
  const auto narrow = [](V p) noexcept -> Vec3<T> {return {T(p.x),T(p.y),T(p.z)};};
  FaceContact<T> result;
  result.distance=T(gap);
  result.normal=narrow(n);
  result.ellipsoid_witness=narrow(center+support);
  result.triangle_witness=narrow(center+support+n*gap);
  result.position=narrow(center+support+n*(gap/W(2)));
  result.barycentric={{T(b0),T(b1),T(b2)}};
  if (!finite(result.distance) || !finite(result.normal) || !finite(result.ellipsoid_witness)
      || !finite(result.triangle_witness) || !finite(result.position))
    return FaceContactStatus::InvalidInput;
  output=result;
  return FaceContactStatus::Interior;
}

} // namespace deform3d
#endif
