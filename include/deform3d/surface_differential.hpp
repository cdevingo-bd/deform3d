#pragma once

// Internal, scale-conditioned differential of the unit triangle normal.
#include "surface_embed.hpp"

namespace deform3d { namespace surface_detail {

template<class T> struct Differential {
  using L = typename std::conditional<std::is_same<T,float>::value,double,long double>::type;
  using P = Vec3<L>;
  P normal{}, e1{}, e2{};
  L inverse_edge_area = 0, inverse_scale = 0;

  static P wide(Vec3<T> v) noexcept { return {L(v.x),L(v.y),L(v.z)}; }
  static L length(P v) noexcept { return std::hypot(v.x,v.y,v.z); }

  SurfaceEmbedStatus update(Vec3<T> a0, Vec3<T> b0, Vec3<T> c0) noexcept {
    using S = SurfaceEmbedStatus;
    const P a=wide(a0), b=wide(b0), c=wide(c0);
    const L scale=std::max({length(b-a),length(c-a),length(c-b)});
    if (!(scale>0) || !std::isfinite(scale)) return S::DegenerateTriangle;
    e1=(b-a)/scale; e2=(c-a)/scale;
    const P area=cross(e1,e2);
    const L magnitude=length(area);
    if (!std::isfinite(magnitude) || magnitude<=L(64)*std::numeric_limits<T>::epsilon())
      return S::DegenerateTriangle;
    normal=area/magnitude;
    inverse_scale=L(1)/scale;
    inverse_edge_area=inverse_scale/magnitude;
    return std::isfinite(inverse_edge_area) && std::isfinite(inverse_scale)
        ? S::Ok : S::NumericalFailure;
  }

  P normal_change(const std::array<P,3>& u) const noexcept {
    const P dq=cross(u[1]-u[0],e2)+cross(e1,u[2]-u[0]);
    return (dq-normal*dot(normal,dq))*inverse_edge_area;
  }

  std::array<P,3> normal_transpose(P force) const noexcept {
    const P g=(force-normal*dot(normal,force))*inverse_edge_area;
    const P fb=cross(e2,g), fc=cross(g,e1);
    return {{-fb-fc,fb,fc}};
  }

  // D(Dn^T * force)[u], with force held fixed in world coordinates.
  // The scale is a fixed conditioning constant for this differential; no
  // derivative of max(edge lengths) enters the underlying smooth normal map.
  std::array<P,3> normal_geometric(P force, const std::array<P,3>& u) const noexcept {
    const P d1=u[1]-u[0], d2=u[2]-u[0];
    const P dq=cross(d1,e2)+cross(e1,d2);
    const L area_rate=dot(normal,dq)*inverse_edge_area;
    const P dn=(dq-normal*dot(normal,dq))*inverse_edge_area;
    const P g=(force-normal*dot(normal,force))*inverse_edge_area;
    const P dg=-(dn*dot(normal,force)+normal*dot(dn,force))*inverse_edge_area-g*area_rate;
    const P fb=cross(d2*inverse_scale,g)+cross(e2,dg);
    const P fc=cross(dg,e1)+cross(g,d1*inverse_scale);
    return {{-fb-fc,fb,fc}};
  }
};

}} // namespace deform3d::surface_detail
