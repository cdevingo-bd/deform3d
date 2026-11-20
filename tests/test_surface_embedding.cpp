#include <deform3d/surface_embed.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <random>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"CHECK %s:%d: %s\n",__FILE__,__LINE__,#x); std::abort(); } } while(false)
static std::size_t allocations=0;
void* operator new(std::size_t n) { ++allocations; if(void* p=std::malloc(n?n:1)) return p; throw std::bad_alloc(); }
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
void operator delete[](void* p,std::size_t) noexcept { std::free(p); }
using namespace deform3d;
template<class T> bool near(T a,T b,T e=T(2e-5)) { return std::abs(a-b)<=e*(T(1)+std::max(std::abs(a),std::abs(b))); }
template<class T> bool near(Vec3<T> a,Vec3<T> b,T e=T(2e-5)) { return near(a.x,b.x,e)&&near(a.y,b.y,e)&&near(a.z,b.z,e); }
template<class T> Vec3<T> rigid(Vec3<T> p) { return {-p.z+T(2),p.x+T(3),-p.y+T(4)}; }

template<class T> void covariance() {
  const Vec3<T> rest[4]={{0,0,0},{2,0,0},{0,2,0},{2,2,0}};
  const std::uint32_t triangles[6]={0,1,2,1,3,2};
  const Vec3<T> points[4]={{T(.2),T(.3),T(.1)},{T(1.4),T(1.6),T(-.2)},
                         {T(.5),T(.5),0},{T(1.2),T(.4),T(.3)}};
  SurfaceEmbedOptions<T> options; options.max_distance=T(.4);
  SurfaceEmbedding<T> embedding;
  CHECK(embedding.prepare(rest,4,triangles,2,points,4,options)==SurfaceEmbedStatus::Ok);
  CHECK(embedding.ready()&&embedding.triangle_count()==2&&embedding.simulation_vertex_count()==4);
  Vec3<T> output[4],moved[4];
  for(std::size_t i=0;i<4;++i) moved[i]=rigid(rest[i]);
  const auto allocation_count=allocations;
  CHECK(embedding.evaluate(moved,4,output,4)==SurfaceEmbedStatus::Ok);
  CHECK(allocations==allocation_count);
  for(std::size_t i=0;i<4;++i) CHECK(near(output[i],rigid(points[i])));
  CHECK(embedding.bindings()[0].normal_offset>T(0));
  CHECK(embedding.bindings()[1].normal_offset<T(0));
  output[0]={99,99,99};
  CHECK(embedding.evaluate_range(moved,4,output,4,1,4)==SurfaceEmbedStatus::Ok);
  CHECK(output[0].x==T(99));
  CHECK(embedding.evaluate_range(moved,4,output,4,0,1)==SurfaceEmbedStatus::Ok);
  CHECK(near(output[0],rigid(points[0])));
  // Offset is physical thickness and remains fixed under in-plane scaling.
  for(std::size_t i=0;i<4;++i) moved[i]=rest[i]*T(2);
  CHECK(embedding.evaluate(moved,4,output,4)==SurfaceEmbedStatus::Ok);
  for(std::size_t i=0;i<4;++i) CHECK(near(output[i],Vec3<T>{points[i].x*T(2),points[i].y*T(2),points[i].z}));
  // Zero-offset points follow any affine map of a planar triangle exactly.
  options.outside=SurfaceOutsidePolicy::ProjectToSurface;
  CHECK(embedding.prepare(rest,4,triangles,2,points,4,options)==SurfaceEmbedStatus::Ok);
  const auto affine=[](Vec3<T> p) { return Vec3<T>{T(2)*p.x+T(.5)*p.y+T(1),T(.3)*p.x+p.y,T(.2)*p.x-T(.4)*p.y}; };
  for(std::size_t i=0;i<4;++i) moved[i]=affine(rest[i]);
  CHECK(embedding.evaluate(moved,4,output,4)==SurfaceEmbedStatus::Ok);
  for(std::size_t i=0;i<4;++i) CHECK(near(output[i],affine(Vec3<T>{points[i].x,points[i].y,0})));
}

static void validation() {
  using V=Vec3<double>; using S=SurfaceEmbedStatus;
  const V rest[4]={{0,0,0},{1,0,0},{0,1,0},{9,9,9}};
  const std::uint32_t triangle[3]={0,1,2},bad[3]={0,1,4},degenerate[3]={0,1,1};
  const V points[2]={{.2,.3,.1},{.4,.2,-.1}};
  V output[2]={{37,37,37},{37,37,37}},before[2];
  std::memcpy(before,output,sizeof(output));
  SurfaceEmbedding<double> embedding;
  SurfaceEmbedOptions<double> options; options.max_distance=.25;
  CHECK(embedding.evaluate(rest,4,output,2)==S::InvalidInput);
  CHECK(embedding.prepare(rest,4,triangle,1,points,2,options)==S::Ok);
  CHECK(embedding.prepare(rest,4,bad,1,points,2,options)==S::InvalidIndex);
  CHECK(embedding.prepare(rest,4,degenerate,1,points,2,options)==S::DegenerateTriangle);
  options.max_distance=std::numeric_limits<double>::infinity();
  CHECK(embedding.prepare(rest,4,triangle,1,points,2,options)==S::InvalidInput);
  options.max_distance=.01;
  CHECK(embedding.prepare(rest,4,triangle,1,points,2,options)==S::OutsideSurface);
  options.max_distance=.25;
  const V lateral[1]={{1.1,0,.1}};
  CHECK(embedding.prepare(rest,4,triangle,1,lateral,1,options)==S::OutsideSurface);
  V nonfinite[4]; std::copy(std::begin(rest),std::end(rest),nonfinite);
  nonfinite[3].x=std::numeric_limits<double>::quiet_NaN(); // Unreferenced source still validated.
  std::size_t error=0;
  CHECK(embedding.prepare(nonfinite,4,triangle,1,points,2,options,&error)==S::NonFiniteInput&&error==3);
  CHECK(embedding.render_vertex_count()==2); // all failures leave old map intact
  const auto count=allocations;
  CHECK(embedding.evaluate(nonfinite,4,output,2)==S::NonFiniteInput);
  V collapsed[4]={{0,0,0},{1,0,0},{2,0,0},{9,9,9}};
  CHECK(embedding.evaluate(collapsed,4,output,2)==S::DegenerateTriangle);
  CHECK(embedding.evaluate_range(rest,4,output,2,2,1)==S::InvalidRange);
  CHECK(embedding.evaluate_range(rest,4,output,2,0,3)==S::InvalidRange);
  CHECK(embedding.evaluate(nullptr,4,output,2)==S::InvalidInput);
  CHECK(embedding.evaluate(rest,3,output,2)==S::InvalidInput);
  CHECK(embedding.evaluate(collapsed,4,collapsed,4)==S::Aliasing);
  CHECK(embedding.evaluate(rest,4,output,std::numeric_limits<std::size_t>::max())==S::InvalidInput);
  CHECK(allocations==count);
  CHECK(std::memcmp(output,before,sizeof(output))==0);
  CHECK(embedding.evaluate(rest,4,output,2)==S::Ok);
  for(std::size_t i=0;i<2;++i) CHECK(near(output[i],points[i],1e-13));
  double xyz[4*4]={0,0,0,91,1,0,0,92,0,1,0,93,9,9,9,94};
  float out[2*4]={0,0,0,55,0,0,0,56};
  CHECK(embedding.evaluate_strided(xyz,4,4,out,2,4)==S::Ok);
  CHECK(near(double(out[0]),points[0].x,1e-7)&&out[3]==55&&out[7]==56);
  CHECK(embedding.evaluate_strided(xyz,4,2,out,2,4)==S::InvalidInput);
  CHECK(embedding.evaluate_strided(xyz,4,4,out,2,std::numeric_limits<std::size_t>::max())==S::InvalidInput);
  CHECK(embedding.evaluate_strided(xyz,4,4,xyz,4,4)==S::Aliasing);
  // Valid source geometry can still exceed a narrower output's numeric range.
  double enormous[4*3]={1e40,0,0,2e40,0,0,1e40,1e40,0,9,9,9};
  float sentinel[2*4]; std::copy(std::begin(out),std::end(out),sentinel);
  CHECK(embedding.evaluate_strided(enormous,4,3,out,2,4)==S::NumericalFailure);
  CHECK(std::memcmp(out,sentinel,sizeof(out))==0);
  options.outside=SurfaceOutsidePolicy::ProjectToSurface;
  CHECK(embedding.prepare(rest,4,triangle,1,lateral,1,options)==S::Ok);
  CHECK(embedding.evaluate(rest,4,output,2)==S::Ok&&near(output[0],V{1,0,0}));
  CHECK(embedding.evaluate(collapsed,4,output,2)==S::DegenerateTriangle);
  // A huge finite point must not produce an all-zero or nonfinite binding.
  options.max_distance=std::numeric_limits<double>::max();
  const V huge[1]={{1e200,1e200,1e200}};
  CHECK(embedding.prepare(rest,4,triangle,1,huge,1,options)==S::Ok);
  CHECK(embedding.evaluate(rest,4,output,2)==S::Ok);
  CHECK(finite(output[0]));
  double sum=0; for(auto w:embedding.bindings()[0].weights) sum+=w;
  CHECK(near(sum,1.,1e-14));
  options.max_distance=1e200;
  CHECK(embedding.prepare(rest,4,triangle,1,huge,1,options)==S::OutsideSurface);
  options.max_distance=2e200;
  CHECK(embedding.prepare(rest,4,triangle,1,huge,1,options)==S::Ok);
  CHECK(embedding.prepare(rest,4,triangle,1,nullptr,0)==S::Ok);
  CHECK(embedding.evaluate(rest,4,output,2)==S::Ok);
}

static void extreme_distances() {
  using V=Vec3<double>; using S=SurfaceEmbedStatus;
  // Distinct tiny distances must not all become zero and select triangle 0.
  // Also exercise BVH pruning and large distances whose squares overflow.
  for(double scale:{1e-200,1e200}) {
    const V rest[6]={{0,0,0},{scale,0,0},{0,scale,0},
                     {0,0,3*scale},{scale,0,3*scale},{0,scale,3*scale}};
    const std::uint32_t triangles[6]={0,1,2,3,4,5};
    const V point{.25*scale,.25*scale,3.125*scale};
    for(std::size_t leaf_size:{1,8}) {
      SurfaceEmbedOptions<double> options;
      options.max_distance=.2*scale; options.bvh_leaf_size=leaf_size;
      SurfaceEmbedding<double> embedding; V output;
      CHECK(embedding.prepare(rest,6,triangles,2,&point,1,options)==S::Ok);
      CHECK(embedding.bindings()[0].triangle==1);
      CHECK(embedding.evaluate(rest,6,&output,1)==S::Ok);
      CHECK(near(output/scale,point/scale,1e-13));
      options.max_distance=.1*scale;
      CHECK(embedding.prepare(rest,6,triangles,2,&point,1,options)==S::OutsideSurface);
      CHECK(embedding.bindings()[0].triangle==1);
    }
  }
}

template<class T> void dense_reference() {
  // A tiled rectangular shell gives an independent closest-point reference:
  // clamp X/Y to the rectangle and set Z=0, including open edges and corners.
  constexpr std::size_t n=9;
  std::vector<Vec3<T>> vertices;
  std::vector<std::uint32_t> triangles;
  for(std::size_t y=0;y<=n;++y) for(std::size_t x=0;x<=n;++x)
    vertices.push_back({T(x),T(y),0});
  for(std::uint32_t y=0;y<n;++y) for(std::uint32_t x=0;x<n;++x) {
    const auto a=y*(n+1)+x,b=a+1,c=a+(n+1),d=c+1;
    for(auto i:{a,b,c,b,d,c}) triangles.push_back(std::uint32_t(i));
  }
  std::mt19937 random(5324);
  std::uniform_real_distribution<double> xy(-2.,11.),height(-.3,.3);
  std::vector<Vec3<T>> points(8192),output(points.size());
  for(auto& p:points) p={T(xy(random)),T(xy(random)),T(height(random))};
  SurfaceEmbedOptions<T> options; options.outside=SurfaceOutsidePolicy::ProjectToSurface;
  options.max_distance=T(4); options.bvh_leaf_size=3;
  SurfaceEmbedding<T> embedding;
  CHECK(embedding.prepare(vertices.data(),vertices.size(),triangles.data(),triangles.size()/3,
                           points.data(),points.size(),options)==SurfaceEmbedStatus::Ok);
  CHECK(embedding.evaluate(vertices.data(),vertices.size(),output.data(),output.size())==SurfaceEmbedStatus::Ok);
  T worst=0;
  for(std::size_t i=0;i<points.size();++i) {
    const Vec3<T> expected{std::clamp(points[i].x,T(0),T(n)),std::clamp(points[i].y,T(0),T(n)),0};
    CHECK(near(output[i],expected)); worst=std::max(worst,norm(output[i]-expected));
  }
  std::printf("surface %s: 8192-point rectangle reference max error %.6g\n",
      std::is_same<T,float>::value?"float":"double",double(worst));
  // Known signed offsets on random interior points must survive rigid motion.
  std::uniform_real_distribution<double> interior(.1,8.9);
  for(auto& p:points) p={T(interior(random)),T(interior(random)),T(height(random))};
  options.outside=SurfaceOutsidePolicy::PreserveNormalOffset; options.max_distance=T(.31);
  CHECK(embedding.prepare(vertices.data(),vertices.size(),triangles.data(),triangles.size()/3,
                           points.data(),points.size(),options)==SurfaceEmbedStatus::Ok);
  for(auto& p:vertices) p=rigid(p);
  CHECK(embedding.evaluate(vertices.data(),vertices.size(),output.data(),output.size())==SurfaceEmbedStatus::Ok);
  for(std::size_t i=0;i<points.size();++i) CHECK(near(output[i],rigid(points[i])));
}

template<class T> void scales_and_nonplanar_reference() {
  const T scale=std::is_same<T,float>::value?T(1e-20):T(1e-200);
  const Vec3<T> tiny[3]={{0,0,0},{scale,0,0},{0,scale,0}};
  const std::uint32_t ids[3]={0,1,2};
  const Vec3<T> point[1]={{scale*T(.25),scale*T(.25),scale*T(.125)}};
  SurfaceEmbedOptions<T> options; options.max_distance=scale*T(.2);
  SurfaceEmbedding<T> embedding; Vec3<T> output;
  CHECK(embedding.prepare(tiny,3,ids,1,point,1,options)==SurfaceEmbedStatus::Ok);
  CHECK(embedding.evaluate(tiny,3,&output,1)==SurfaceEmbedStatus::Ok);
  CHECK(near(output/scale,Vec3<T>{T(.25),T(.25),T(.125)}));
  const Vec3<T> skinny[3]={{0,0,0},{1,0,0},{1,T(8)*std::numeric_limits<T>::epsilon(),0}};
  CHECK(embedding.prepare(skinny,3,ids,1,point,1,options)==SurfaceEmbedStatus::DegenerateTriangle);

  // Disconnected, separated triangles make the identity of the nearest
  // triangle known independently of BVH traversal. Each triangle has random
  // nonorthogonal edges and a different normal; points use known barycentrics.
  std::mt19937 generator(53642);
  std::uniform_real_distribution<double> random(-1.,1.);
  std::vector<Vec3<T>> source,moved,points,expected;
  std::vector<std::uint32_t> triangles;
  for(std::uint32_t face=0;face<64;++face) {
    const Vec3<T> a{T(face)*T(4),T(random(generator)),T(random(generator))};
    const Vec3<T> ab{T(1+.1*random(generator)),T(.2*random(generator)),T(.2*random(generator))};
    const Vec3<T> ac{T(.2*random(generator)),T(1+.1*random(generator)),T(.2*random(generator))};
    const Vec3<T> normal=normalized(cross(ab,ac));
    const Vec3<T> deformed_ab=ab+Vec3<T>{T(.1),T(.2),T(.3)};
    const Vec3<T> deformed_ac=ac+Vec3<T>{T(.2),T(.1),T(-.2)};
    const Vec3<T> deformed_normal=normalized(cross(deformed_ab,deformed_ac));
    const auto base=std::uint32_t(source.size());
    source.insert(source.end(),{a,a+ab,a+ac});
    moved.insert(moved.end(),{a,a+deformed_ab,a+deformed_ac});
    triangles.insert(triangles.end(),{base,base+1,base+2});
    for(std::size_t sample=0;sample<32;++sample) {
      const T w1=T(.2+.1*random(generator)),w2=T(.2+.1*random(generator)),offset=T(.02*random(generator));
      points.push_back(a+ab*w1+ac*w2+normal*offset);
      expected.push_back(a+deformed_ab*w1+deformed_ac*w2+deformed_normal*offset);
    }
  }
  options.max_distance=T(.04); options.bvh_leaf_size=2;
  CHECK(embedding.prepare(source.data(),source.size(),triangles.data(),triangles.size()/3,
                           points.data(),points.size(),options)==SurfaceEmbedStatus::Ok);
  std::vector<Vec3<T>> dense(points.size());
  CHECK(embedding.evaluate(moved.data(),moved.size(),dense.data(),dense.size())==SurfaceEmbedStatus::Ok);
  const T tolerance=std::is_same<T,float>::value?T(1e-4):T(1e-10);
  for(std::size_t i=0;i<points.size();++i) {
    CHECK(embedding.bindings()[i].triangle==i/32);
    CHECK(norm(dense[i]-expected[i])<tolerance);
  }
}

int main() {
  covariance<float>(); covariance<double>(); validation(); extreme_distances();
  dense_reference<float>(); dense_reference<double>();
  scales_and_nonplanar_reference<float>(); scales_and_nonplanar_reference<double>();
}
