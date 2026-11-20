#include <deform3d/surface_linearization.hpp>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"CHECK %s:%d: %s\n",__FILE__,__LINE__,#x); std::abort(); } } while(false)
static std::atomic<std::size_t> allocations{0};
void* operator new(std::size_t n) { ++allocations; if(auto p=std::malloc(n?n:1)) return p; throw std::bad_alloc(); }
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
void operator delete[](void* p,std::size_t) noexcept { std::free(p); }
using namespace deform3d;
using S=SurfaceEmbedStatus;
template<class T> bool near(Vec3<T> a,Vec3<T> b,T eps) { return norm(a-b)<=eps*(T(1)+norm(a)+norm(b)); }

template<class T> void derivatives(bool symmetric,T scale) {
  using V=Vec3<T>;
  const V base[6]={{0,0,0},{2,0,0},{0,2,0},{2,2,0},{4,2,0},{9,9,9}};
  const std::uint32_t tri[9]={0,1,2,1,3,2,1,4,3}; // third triangle is unreferenced
  const V samples[5]={{T(.2),T(.3),T(.25)},{T(1.4),T(1.6),T(-.2)},
                     {T(.7),T(.5),0},{T(1.2),T(.4),T(.3)},{T(.9),T(.2),T(-.1)}};
  V rest[6],points[5],x[6],u[6],v[6];
  for(int i=0;i<6;++i) {
    rest[i]=base[i]*scale;
    x[i]={scale*(T(1)+T(.8)*base[i].x),scale*(T(2)+T(1.3)*base[i].y),scale*(T(.4)+T(.2)*base[i].x-T(.3)*base[i].y)};
    u[i]=V{T(.1)*T(i+1),T(-.2)+T(.3)*T(i),T(.05)*T(i*i)}*scale;
    v[i]=V{T(.4)-T(.1)*T(i),T(.3)*T(i%3),T(.3)-T(.09)*T(i*i)}*scale;
  }
  for(int i=0;i<5;++i) points[i]=samples[i]*scale;
  SurfaceEmbedOptions<T> options;options.max_distance=T(.4)*scale;
  SurfaceEmbedding<T> embedding;
  CHECK(embedding.prepare(rest,6,tri,3,points,5,options)==S::Ok);
  SurfaceCoupling<T> reference;CHECK(reference.prepare(embedding)==S::Ok);
  SurfaceLinearization<T> op;CHECK(op.prepare(embedding)==S::Ok);
  CHECK(op.active_triangle_count()==2);
  Mat3<T> k[5];V f[5],y[5],ju[5],df[5],expected[6],actual[6],au[6],av[6];
  for(int i=0;i<5;++i) {
    k[i].m={{T(-3-i)/scale,T(.2)/scale,T(-.3)/scale,
             T(symmetric?.2:-.4)/scale,T(-2-i)/scale,T(.1)/scale,
             T(symmetric?-.3:.7)/scale,T(symmetric?.1:-.2)/scale,T(-1-i)/scale}};
    f[i]={T(.4+i),T(.2-i),T(.3*i-1)};
  }
  const auto before=allocations.load();
  CHECK(reference.update(x,6)==S::Ok);
  CHECK(reference.evaluate_positions(y,5)==S::Ok);
  CHECK(reference.evaluate_velocities(u,6,ju,5)==S::Ok);
  CHECK(reference.transfer_forces(f,5,expected,6)==S::Ok);
  CHECK(op.update(x,6,f,k,5)==S::Ok);
  CHECK(op.evaluate_forces(actual,6)==S::Ok);
  CHECK(op.apply(u,6,au,6)==S::Ok);CHECK(op.apply(v,6,av,6)==S::Ok);
  CHECK(allocations.load()==before);
  const T eps=std::is_same<T,float>::value?T(1e-5):T(2e-12);
  for(int i=0;i<6;++i) CHECK(near(actual[i],expected[i],eps));
  CHECK(norm(au[5])==0);
  T uv=0,vu=0;V dnet{},point_dnet{},torque{},point_torque{};
  for(int i=0;i<5;++i) {
    df[i]=k[i].transform(ju[i]);point_dnet+=df[i];
    point_torque+=cross(ju[i],f[i])+cross(y[i],df[i]);
  }
  for(int i=0;i<6;++i) {
    uv+=dot(u[i],av[i]);vu+=dot(v[i],au[i]);dnet+=au[i];
    torque+=cross(u[i],actual[i])+cross(x[i],au[i]);
  }
  CHECK(near(dnet,point_dnet,eps));CHECK(near(torque,point_torque,eps));
  if(symmetric) CHECK(std::abs(uv-vu)<eps*(T(1)+std::abs(uv)+std::abs(vu)));
  // Differentiate the existing, uncompressed nonlinear force map. Test a
  // general direction and every coordinate column, including zero-offset and
  // positive/negative-offset attachments on shared physical vertices.
  const T h=std::is_same<T,float>::value?T(.002):T(2e-5);
  for(int direction=-1;direction<18;++direction) {
    V delta[6]{};
    if(direction<0) std::copy(u,u+6,delta);
    else {if(direction%3==0)delta[direction/3].x=scale;else if(direction%3==1)delta[direction/3].y=scale;else delta[direction/3].z=scale;}
    V plus[6],minus[6],yp[5],ym[5],fp[5],fm[5],gp[6],gm[6],product[6];
    for(int i=0;i<6;++i) {plus[i]=x[i]+delta[i]*h;minus[i]=x[i]-delta[i]*h;}
    CHECK(reference.update(plus,6)==S::Ok);CHECK(reference.evaluate_positions(yp,5)==S::Ok);
    for(int i=0;i<5;++i) fp[i]=f[i]+k[i].transform(yp[i]-y[i]);
    CHECK(reference.transfer_forces(fp,5,gp,6)==S::Ok);
    CHECK(reference.update(minus,6)==S::Ok);CHECK(reference.evaluate_positions(ym,5)==S::Ok);
    for(int i=0;i<5;++i) fm[i]=f[i]+k[i].transform(ym[i]-y[i]);
    CHECK(reference.transfer_forces(fm,5,gm,6)==S::Ok);
    CHECK(op.apply(delta,6,product,6)==S::Ok);
    for(int i=0;i<6;++i) CHECK(near(product[i],(gp[i]-gm[i])/(T(2)*h),
        std::is_same<T,float>::value?T(7e-4):T(2e-8)));
  }
  // A null force derivative must retain geometric stiffness, not reduce to 0.
  CHECK(op.update(x,6,f,nullptr,5)==S::Ok);CHECK(op.apply(u,6,au,6)==S::Ok);
  T size=0;for(auto a:au)size+=norm(a);CHECK(size>T(.01));
  V uniform[6];std::fill(uniform,uniform+6,V{scale,scale,-scale});
  CHECK(op.apply(uniform,6,av,6)==S::Ok);for(auto a:av) CHECK(norm(a)<eps);
  V plus[6],minus[6],gp[6],gm[6];
  for(int i=0;i<6;++i) {plus[i]=x[i]+u[i]*h;minus[i]=x[i]-u[i]*h;}
  CHECK(reference.update(plus,6)==S::Ok);CHECK(reference.transfer_forces(f,5,gp,6)==S::Ok);
  CHECK(reference.update(minus,6)==S::Ok);CHECK(reference.transfer_forces(f,5,gm,6)==S::Ok);
  for(int i=0;i<6;++i) CHECK(near(au[i],(gp[i]-gm[i])/(T(2)*h),std::is_same<T,float>::value?T(5e-4):T(1e-8)));
  V parallel[6];
  std::thread a([&]{CHECK(op.apply(u,6,parallel,6,SurfaceForceMode::Overwrite,0,2)==S::Ok);});
  std::thread b([&]{CHECK(op.apply(u,6,parallel,6,SurfaceForceMode::Overwrite,2,6)==S::Ok);});
  a.join();b.join();CHECK(std::memcmp(parallel,au,sizeof au)==0);
  CHECK(op.apply(u,6,parallel,6,SurfaceForceMode::Accumulate)==S::Ok);
  for(int i=0;i<6;++i) CHECK(near(parallel[i],au[i]*T(2),eps));
  V saved[6];std::memcpy(saved,parallel,sizeof saved);
  CHECK(op.apply(u,6,parallel,6,SurfaceForceMode::Overwrite,2,2)==S::Ok);
  CHECK(std::memcmp(saved,parallel,sizeof saved)==0);
}

void transactions_and_resolution() {
  using V=Vec3<double>;
  const V rest[3]={{0,0,0},{1,0,0},{0,1,0}},p{.2,.3,.2},f{1,-2,3},u[3]={{.1,.2,-.1},{-.3,.1,.2},{.4,.3,-.2}};
  const std::uint32_t tri[3]={0,1,2};SurfaceEmbedOptions<double> options;options.max_distance=.3;
  SurfaceEmbedding<double> embedding,empty;
  CHECK(embedding.prepare(rest,3,tri,1,&p,1,options)==S::Ok);
  SurfaceLinearization<double> op;V result[3],expected[3],expected_force[3];
  CHECK(op.prepare(embedding)==S::Ok);CHECK(op.apply(u,3,result,3)==S::InvalidInput);
  Mat3<double> k{{-2,.2,.3,.2,-3,.1,.3,.1,-4}};
  CHECK(op.update(rest,3,&f,&k,1)==S::Ok);CHECK(op.apply(u,3,expected,3)==S::Ok);
  CHECK(op.evaluate_forces(expected_force,3)==S::Ok);
  CHECK(op.prepare(empty)==S::InvalidInput);
  V collapse[3]={{0,0,0},{1,0,0},{2,0,0}};
  CHECK(op.update(collapse,3,&f,&k,1)==S::DegenerateTriangle);
  Mat3<double> bad=k;bad.m[4]=std::numeric_limits<double>::quiet_NaN();
  CHECK(op.update(rest,3,&f,&bad,1)==S::NonFiniteInput);
  V bad_force=f;bad_force.z=std::numeric_limits<double>::infinity();
  CHECK(op.update(rest,3,&bad_force,&k,1)==S::NonFiniteInput);
  CHECK(op.update(rest,2,&f,&k,1)==S::InvalidInput);
  CHECK(op.apply(u,3,result,3)==S::Ok);CHECK(std::memcmp(expected,result,sizeof result)==0);
  CHECK(op.evaluate_forces(result,3)==S::Ok);CHECK(std::memcmp(expected_force,result,sizeof result)==0);
  V saved[3];std::memcpy(saved,result,sizeof result);
  CHECK(op.apply(result,3,result,3)==S::Aliasing);
  CHECK(op.apply(u,3,result,3,SurfaceForceMode::Overwrite,2,4)==S::InvalidRange);
  CHECK(op.apply(u,3,result,3,SurfaceForceMode(99))==S::InvalidInput);
  CHECK(std::memcmp(saved,result,sizeof result)==0);
  V bad_u[3];std::copy(u,u+3,bad_u);bad_u[2].x=std::numeric_limits<double>::infinity();
  CHECK(op.apply(bad_u,3,result,3)==S::NonFiniteInput);CHECK(std::memcmp(saved,result,sizeof result)==0);
  // Refining a quadrature sample divides its force AND tangent by its weight.
  // The physical load and derivative must not grow with render resolution.
  std::vector<V> points(1000,p),forces(1000,f/1000.);
  std::vector<Mat3<double>> ks(1000,k);for(auto& m:ks)for(auto& value:m.m)value/=1000.;
  SurfaceEmbedding<double> dense;
  CHECK(dense.prepare(rest,3,tri,1,points.data(),points.size(),options)==S::Ok);
  CHECK(op.prepare(dense)==S::Ok);CHECK(!op.ready());
  CHECK(op.update(rest,3,forces.data(),ks.data(),points.size())==S::Ok);
  CHECK(op.apply(u,3,result,3)==S::Ok);for(int i=0;i<3;++i)CHECK(near(result[i],expected[i],1e-12));
  CHECK(op.evaluate_forces(result,3)==S::Ok);for(int i=0;i<3;++i)CHECK(near(result[i],expected_force[i],1e-12));
  SurfaceLinearization<double> moved; moved.swap(op);CHECK(moved.ready());CHECK(!op.ready());
  // Empty dense plans still have a valid physical output domain.
  CHECK(dense.prepare(rest,3,tri,1,nullptr,0,options)==S::Ok);
  CHECK(op.prepare(dense)==S::Ok);CHECK(op.update(rest,3,&f,nullptr,0)==S::Ok);
  CHECK(op.apply(u,3,result,3)==S::Ok);for(auto value:result)CHECK(norm(value)==0);
}

void overflow_is_transactional() {
  using V=Vec3<float>;
  V rest[3]={{0,0,0},{1,0,0},{0,1,0}},p{.2f,.3f,.2f},f{0,0,0};
  std::uint32_t tri[3]={0,1,2};SurfaceEmbedOptions<float> options;options.max_distance=.3f;
  SurfaceEmbedding<float> binding;CHECK(binding.prepare(rest,3,tri,1,&p,1,options)==S::Ok);
  SurfaceLinearization<float> op;CHECK(op.prepare(binding)==S::Ok);
  Mat3<float> k;k.m[0]=k.m[4]=k.m[8]=std::numeric_limits<float>::max();
  CHECK(op.update(rest,3,&f,&k,1)==S::Ok);
  V u[3]={{100,100,100},{100,100,100},{100,100,100}},out[3]={{7,8,9},{7,8,9},{7,8,9}},saved[3];
  std::memcpy(saved,out,sizeof out);
  CHECK(op.apply(u,3,out,3)==S::NumericalFailure);CHECK(std::memcmp(saved,out,sizeof out)==0);
}

int main() {
  for(bool symmetric:{true,false})for(double scale:{1e-4,1.,1e4}) {
    derivatives<float>(symmetric,float(scale));derivatives<double>(symmetric,scale);
  }
  transactions_and_resolution();overflow_is_transactional();
  std::puts("surface linearization: derivatives, conservation, symmetry, scaling, ranges, transactions and allocation checks passed");
}
