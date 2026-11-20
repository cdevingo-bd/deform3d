#pragma once
#include "deform3d.hpp"
#include "simplex_material.hpp"
#include <type_traits>

namespace deform3d {

template<class T> struct SimplexRestMaterial {
  std::array<T,21> packed{};
  std::array<T,6> edge_lengths{};
  T measure{}; // reference volume, or triangle area times thickness
};

// Compile an isotropic Saint Venant-Kirchhoff material without a simulator.
// Triangles use plane stress. Tetrahedra use 3D elasticity; thickness is ignored.
// Failure leaves out unchanged. No allocation. Inputs may use either winding.
template<class T> Status prepare_simplex_material(int dim, const Vec3<T>* rest,
    T young, T poisson, T thickness, SimplexRestMaterial<T>& out) noexcept {
  static_assert(std::is_same<T,float>::value || std::is_same<T,double>::value,
                "material scalars must be float or double");
  if((dim!=2 && dim!=3) || !rest || !finite(young) || !finite(poisson) ||
     young<T(0) || poisson<=T(-1) || poisson>=T(.5) ||
     (dim==2 && (!finite(thickness) || thickness<=T(0)))) return Status::InvalidArgument;
  using L=long double; using P=Vec3<L>;
  P x[4]{},e[3]{},g[4]{};
  for(int i=0;i<dim+1;++i) {
    if(!finite(rest[i])) return Status::NonFinite;
    x[i]={L(rest[i].x),L(rest[i].y),L(rest[i].z)};
  }
  L scale=0;
  for(int i=0;i<dim;++i) {
    e[i]=x[i+1]-x[0];
    scale=std::max(scale,std::hypot(e[i].x,e[i].y,e[i].z));
  }
  if(!(scale>0) || !finite(scale)) return Status::DegenerateGeometry;
  for(int i=0;i<dim;++i)e[i]=e[i]/scale;
  const L threshold=64*std::numeric_limits<L>::epsilon();
  L measure;
  if(dim==2) {
    const P n=cross(e[0],e[1]);const L area2=std::hypot(n.x,n.y,n.z);
    if(!(area2>threshold))return Status::DegenerateGeometry;
    // Cross-product gradients avoid subtracting nearly equal Gram products.
    g[1]=cross(e[1],n)/(area2*area2*scale);
    g[2]=cross(n,e[0])/(area2*area2*scale);
    measure=area2*scale*scale*L(thickness)/2;
  } else {
    const L determinant=dot(e[0],cross(e[1],e[2]));
    if(!(std::abs(determinant)>threshold))return Status::DegenerateGeometry;
    g[1]=cross(e[1],e[2])/(determinant*scale);
    g[2]=cross(e[2],e[0])/(determinant*scale);
    g[3]=cross(e[0],e[1])/(determinant*scale);
    measure=std::abs(determinant)*scale*scale*scale/6;
  }
  for(int i=1;i<dim+1;++i)g[0]-=g[i];
  const L mu=L(young)/(2*(1+L(poisson)));
  const L lambda=dim==2 ? L(young)*L(poisson)/(1-L(poisson)*L(poisson))
      : L(young)*L(poisson)/((1+L(poisson))*(1-2*L(poisson)));
  const int ne=dim==2?3:6;
  L basis[6][9]{},trace[6]{};
  SimplexRestMaterial<T> candidate;
  auto convert=[](L value,T& destination) {
    if(!finite(value) || std::abs(value)>L(std::numeric_limits<T>::max()))return false;
    destination=T(value);return true;
  };
  if(!convert(measure,candidate.measure))return Status::NonFinite;
  for(int a=0;a<ne;++a) {
    const int i=SimplexMaterial<T>::endpoint(dim,a,0),j=SimplexMaterial<T>::endpoint(dim,a,1);
    const P d=x[i]-x[j];
    if(!convert(std::hypot(d.x,d.y,d.z),candidate.edge_lengths[a]))return Status::NonFinite;
    const L u[3]={g[i].x,g[i].y,g[i].z},v[3]={g[j].x,g[j].y,g[j].z};
    // F^T F - I = sum_a e_a B_a in the reference material tangent space.
    for(int r=0;r<3;++r)for(int c=0;c<3;++c)basis[a][3*r+c]=-(u[r]*v[c]+v[r]*u[c])/2;
    trace[a]=basis[a][0]+basis[a][4]+basis[a][8];
  }
  int k=0;
  for(int a=0;a<ne;++a)for(int b=a;b<ne;++b) {
    L inner=0;for(int c=0;c<9;++c)inner+=basis[a][c]*basis[b][c];
    if(!convert(measure*(mu*inner+lambda*trace[a]*trace[b]/2),candidate.packed[k++]))return Status::NonFinite;
  }
  out=candidate;return Status::Ok;
}
} // namespace deform3d
