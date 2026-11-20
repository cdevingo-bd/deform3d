#include <deform3d/contact_quality.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#define CHECK(x) do {if (!(x)) {std::fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);std::abort();}} while(false)
using namespace deform3d;

template<class T> void check_geometry() {
  const T eps=std::is_same<T,float>::value ? T(2e-5) : T(2e-12);
  for (T scale : {T(1e-6),T(1),T(1e6)}) {
    EllipsoidShape<T> e;
    e.center={0,0,T(.3)*scale};
    e.axes={{{T(.1)*scale,0,0},{0,T(.2)*scale,0},{0,0,T(.25)*scale}}};
    std::array<Vec3<T>,3> tri{{{-scale,-scale,0},{scale,-scale,0},{0,scale,0}}};
    FaceContact<T> c;
    CHECK(ellipsoid_triangle_face(e,tri,T(.01)*scale,c)==FaceContactStatus::Interior);
    CHECK(std::abs(c.distance/scale-T(.04))<eps);
    CHECK(norm(c.normal-Vec3<T>{0,0,-1})<eps);
    CHECK(norm(c.triangle_witness-(c.ellipsoid_witness+c.normal*c.distance))/scale<eps);
    CHECK(norm(c.position-(c.triangle_witness+c.ellipsoid_witness)*T(.5))/scale<eps);
    CHECK(std::abs(c.barycentric[0]-T(.25))<eps && std::abs(c.barycentric[2]-T(.5))<eps);
    std::swap(tri[0],tri[1]);
    FaceContact<T> reversed;
    CHECK(ellipsoid_triangle_face(e,tri,T(.01)*scale,reversed)==FaceContactStatus::Interior);
    CHECK(norm(reversed.normal-c.normal)<eps);
    e.center.z=T(.2)*scale;
    CHECK(ellipsoid_triangle_face(e,tri,T(.01)*scale,c)==FaceContactStatus::Interior);
    CHECK(std::abs(c.distance/scale+T(.06))<eps);
  }

  // Rotate and translate an anisotropic shape together with its triangle.
  const Quat<T> q=normalized(Quat<T>{T(.8),T(.2),T(-.3),T(.4)});
  const Vec3<T> translation{T(5),T(-8),T(12)};
  EllipsoidShape<T> e; e.center=rotate(q,Vec3<T>{0,0,T(.3)})+translation;
  e.axes={{rotate(q,Vec3<T>{T(.1),0,0}),rotate(q,Vec3<T>{0,T(.2),0}),rotate(q,Vec3<T>{0,0,T(.25)})}};
  std::array<Vec3<T>,3> tri{{{-1,-1,0},{1,-1,0},{0,1,0}}};
  for(auto& p:tri) p=rotate(q,p)+translation;
  FaceContact<T> c;
  CHECK(ellipsoid_triangle_face(e,tri,T(.01),c)==FaceContactStatus::Interior);
  CHECK(std::abs(c.distance-T(.04))<eps);
  CHECK(norm(c.normal-rotate(q,Vec3<T>{0,0,-1}))<eps);
  // Independent support identity: the reported point lies on the ellipsoid.
  const Vec3<T> local=rotate(q.conjugate(),c.ellipsoid_witness-e.center);
  const T unit=local.x*local.x/T(.01)+local.y*local.y/T(.04)+local.z*local.z/T(.0625);
  CHECK(std::abs(unit-T(1))<eps*T(4));

  // Nonorthogonal (sheared) and reflected axes: independently verify both
  // the implicit unit sphere equation and its world-space surface gradient.
  e.center={0,0,T(.6)};
  e.axes={{{T(.2),0,T(.03)},{T(.1),T(.3),T(.08)},{T(.04),T(.08),T(.4)}}};
  tri={{{-2,-2,0},{2,-2,0},{0,2,0}}};
  for (int reflected=0;reflected<2;++reflected) {
    if(reflected) e.axes[0]=-e.axes[0];
    CHECK(ellipsoid_triangle_face(e,tri,T(.01),c)==FaceContactStatus::Interior);
    const T determinant=dot(e.axes[0],cross(e.axes[1],e.axes[2]));
    const std::array<Vec3<T>,3> reciprocal{{cross(e.axes[1],e.axes[2])/determinant,
      cross(e.axes[2],e.axes[0])/determinant,cross(e.axes[0],e.axes[1])/determinant}};
    const Vec3<T> p=c.ellipsoid_witness-e.center;
    const Vec3<T> u{dot(reciprocal[0],p),dot(reciprocal[1],p),dot(reciprocal[2],p)};
    CHECK(std::abs(dot(u,u)-T(1))<eps*T(4));
    const Vec3<T> gradient=reciprocal[0]*u.x+reciprocal[1]*u.y+reciprocal[2]*u.z;
    CHECK(norm(gradient/norm(gradient)-c.normal)<eps*T(4));
  }
}

int main() {
  check_geometry<float>(); check_geometry<double>();
  EllipsoidShape<double> e; e.center={10,0,2};
  std::array<Vec3d,3> tri{{{-1,-1,0},{1,-1,0},{0,1,0}}};
  FaceContact<double> output; output.distance=123;
  CHECK(ellipsoid_triangle_face(e,tri,0.,output)==FaceContactStatus::OutsideFace);
  CHECK(output.distance==123);
  e.center={0,0,2};e.axes[2]={0,0,0};
  CHECK(ellipsoid_triangle_face(e,tri,0.,output)==FaceContactStatus::Degenerate);
  CHECK(output.distance==123);
  e.axes[2]={0,0,1};tri[2]=tri[1];
  CHECK(ellipsoid_triangle_face(e,tri,0.,output)==FaceContactStatus::Degenerate);
  CHECK(ellipsoid_triangle_face(e,tri,-1.,output)==FaceContactStatus::InvalidInput);
  e.center.x=std::numeric_limits<double>::quiet_NaN();
  CHECK(ellipsoid_triangle_face(e,tri,0.,output)==FaceContactStatus::InvalidInput);
  CHECK(output.distance==123);
  std::puts("contact quality tests passed");
}
