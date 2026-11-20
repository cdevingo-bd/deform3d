#include <deform3d/deform3d.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

using namespace deform3d;

static void check(bool b, const char* msg) {
  if (!b) {std::cerr << "FAIL: " << msg << '\n'; std::exit(1);}
}
static bool near(Vec3f a, Vec3f b, float e=2e-5f) {
  return norm(a-b)<=e;
}
static Quat<float> zrotation(float a) {return {std::cos(a/2),0,0,std::sin(a/2)};}

static void test_skin_and_morph() {
  struct Vertex {Vec3f p; float u; Vec3f n;};
  Vertex in[3]{{{1,0,0},0,{1,0,1}},{{0,1,0},0,{0,1,0}},{{3,1,0},0,{0,0,1}}};
  Vertex out[3]{};
  VertexWeights4<float> raw[3]{};
  raw[0].joint[0]=0;raw[0].weight[0]=2;
  raw[0].joint[1]=1;raw[0].weight[1]=2;
  raw[1].joint[0]=1;raw[1].weight[0]=1;
  SkinBinding<float> skin;
  check(skin.prepare(raw,3,2)==Status::Ok,"prepare skin");
  check(std::abs(skin[0].weight[0]-0.5f)<1e-6f,"weight normalization");
  Affine3f transforms[2];transforms[0].m[3]=2;transforms[1].m[7]=4;
  Workspace<float> ws;ws.reserve(3,2);
  EvaluateParams<float> p;
  p.vertices=3;
  p.input_positions={&in[0].p,sizeof(Vertex)};
  p.input_normals={&in[0].n,sizeof(Vertex)};
  p.output_positions={&out[0].p,sizeof(Vertex)};
  p.output_normals={&out[0].n,sizeof(Vertex)};
  p.skin=&skin;p.skin_mode=SkinMode::LinearBlend;
  p.affine_joints=transforms;p.joint_count=2;
  p.normal_mode=NormalMode::Skin;p.workspace=&ws;
  check(evaluate(p)==Status::Ok,"LBS strided");
  check(near(out[0].p,{2,2,0}) && near(out[1].p,{0,5,0}),"LBS blended points");
  check(near(out[2].p,in[2].p),"unweighted passthrough");
  check(near(out[0].n,normalized(in[0].n)),"LBS normals");

  uint32_t idx[]={0,2};Vec3f delta[]={{0,0,2},{0,1,0}};
  MorphTargetView<float> target{idx,delta,nullptr,2};
  MorphSet<float> morph;
  check(morph.prepare(3,&target,1)==Status::Ok,"prepare morph");
  float weight=0.5f;
  p.skin_mode=SkinMode::None;p.normal_mode=NormalMode::None;
  p.morphs=&morph;p.morph_weights=&weight;p.morph_weight_count=1;
  check(evaluate(p)==Status::Ok,"evaluate sparse morph");
  check(near(out[0].p,{1,0,1}) && near(out[2].p,{3,1.5f,0}),"morph deltas");
  p.input_positions={&out[0].p,sizeof(Vertex)};
  check(evaluate(p)==Status::Ok,"in-place morph");
  check(near(out[0].p,{1,0,2}),"in-place result");
}

static void test_dq_and_normal_matrix() {
  Vec3f in[1]{{1,0,0}},out[1]{};
  Vec3f normal[1]{{1,0,1}},n_out[1]{};
  VertexWeights4<float> weights[1]{};
  weights[0].joint[0]=0;weights[0].weight[0]=1;
  weights[0].joint[1]=1;weights[0].weight[1]=1;
  SkinBinding<float> skin;
  check(skin.prepare(weights,1,2)==Status::Ok,"prepare two joints");
  const float pi=3.14159265358979323846f;
  DualQuatf dq[2]{DualQuatf::from_rotation_translation(zrotation(pi/3),{}),
                  DualQuatf::from_rotation_translation(zrotation(-pi/3),{})};
  Workspace<float> ws;
  EvaluateParams<float> p;
  p.vertices=1;p.input_positions=view(in);p.output_positions=view(out);
  p.input_normals=view(normal);p.output_normals=view(n_out);
  p.skin=&skin;p.skin_mode=SkinMode::DualQuaternion;
  p.rigid_joints=dq;p.joint_count=2;p.workspace=&ws;
  p.normal_mode=NormalMode::Skin;
  check(evaluate(p)==Status::Ok,"DQ skin");
  check(near(out[0],in[0]),"DQ maintains bend volume");
  dq[1].real=-dq[1].real;dq[1].dual=-dq[1].dual;
  check(evaluate(p)==Status::Ok && near(out[0],in[0]),"DQ antipodal joint sign");
  Affine3f affine[2];
  affine[0].m={{2,0,0,0,0,1,0,0,0,0,0.5f,0}};
  affine[1]=affine[0];
  p.skin_mode=SkinMode::LinearBlend;p.affine_joints=affine;
  check(evaluate(p)==Status::Ok,"affine normals");
  check(near(out[0],{2,0,0}) && near(n_out[0],normalized(Vec3f{0.5f,0,2})),
        "inverse transpose normal under scale");
  affine[0].m[0]=0;
  check(evaluate(p)==Status::SingularTransform,"singular normal transform rejected");
}

static void test_normals_and_errors() {
  const uint32_t triangles[]={0,1,2,0,2,3};
  NormalTopology topology;
  check(topology.prepare(4,triangles,6)==Status::Ok,"normal topology");
  Vec3f p0[4]{{0,0,0},{1,0,0},{1,1,0},{0,1,0}}, out[4], normals[4];
  Workspace<float> ws;
  EvaluateParams<float> p;
  p.vertices=4;p.input_positions=view(p0);p.output_positions=view(out);
  p.output_normals=view(normals);p.normal_mode=NormalMode::Recompute;
  p.topology=&topology;p.workspace=&ws;
  check(evaluate(p)==Status::Ok,"recompute normals");
  for (auto n : normals) check(near(n,{0,0,1}),"area weighted normal");
  // Dense meshes expressed in meters routinely have millimeter edges. Their
  // area vectors are small but must still produce unit shading normals.
  for (auto& position : p0) position *= 1e-3f;
  check(evaluate(p)==Status::Ok,"recompute millimeter mesh normals");
  for (auto n : normals) check(near(n,{0,0,1}),"millimeter unit normal");
  const uint32_t bad[]={0,1,4};
  check(topology.prepare(4,bad,3)==Status::InvalidIndex,"topology rejects invalid index");
  check(topology.triangle_count()==2,"failed prepare preserves previous topology");
  VertexWeights4<float> bad_weights[1]{};bad_weights[0].weight[0]=-1;
  SkinBinding<float> skin;
  check(skin.prepare(bad_weights,1,1)==Status::InvalidWeight,"negative weight rejected");
  p.workspace=nullptr;
  check(evaluate(p)==Status::MissingWorkspace,"workspace required");
}

template<class T> static void test_vector_normalization() {
  const T small=std::numeric_limits<T>::min();
  const T large=std::numeric_limits<T>::max()/T(4);
  const T tolerance=std::numeric_limits<T>::epsilon()*T(8);
  const Vec3<T> expected{T(0),T(0.6),T(0.8)};
  for (T scale : {small, T(1), large}) {
    const auto n=normalized(Vec3<T>{T(0),scale*T(3),scale*T(4)});
    check(finite(n) && norm(n-expected)<=tolerance,
          "normalization preserves direction across finite scales");
  }
  const T subnormal=std::numeric_limits<T>::denorm_min();
  if (subnormal>T(0)) {
    const auto n=normalized(Vec3<T>{subnormal,T(0),T(0)});
    check(n.x==T(1) && n.y==T(0) && n.z==T(0),"subnormal direction");
  }
  const Vec3<T> fallback{T(1),T(0),T(0)};
  check(normalized(Vec3<T>{},fallback).x==T(1),"zero-vector fallback");
  check(normalized(Vec3<T>{T(1),std::numeric_limits<T>::quiet_NaN(),T(0)},
                   fallback).x==T(1),"nonfinite-vector fallback");
}

static void threaded(void*,size_t count,size_t grain,void* job,
                     void (*run)(void*,size_t,size_t)) {
  std::vector<std::thread> threads;
  for (size_t begin=0;begin<count;begin+=grain) {
    const size_t end=std::min(begin+grain,count);
    threads.emplace_back([=]{run(job,begin,end);});
  }
  for (auto& t:threads)t.join();
}
static void test_parallel_equivalence() {
  const size_t n=50000;
  std::vector<Vec3f> in(n),serial(n),parallel(n);
  std::vector<VertexWeights4<float>> w(n);
  for(size_t i=0;i<n;++i) {
    in[i]={float(i%17),float(i%31),float(i%13)};
    w[i].joint[0]=0;w[i].weight[0]=1;
  }
  SkinBinding<float> skin;check(skin.prepare(w.data(),n,1)==Status::Ok,"dense binding");
  Affine3f joint;joint.m[3]=2;
  EvaluateParams<float> p;
  p.vertices=n;p.input_positions=view(in.data());p.output_positions=view(serial.data());
  p.skin=&skin;p.skin_mode=SkinMode::LinearBlend;
  p.affine_joints=&joint;p.joint_count=1;
  check(evaluate(p)==Status::Ok,"serial dense");
  p.output_positions=view(parallel.data());p.executor.parallel_for=threaded;p.grain=8192;
  check(evaluate(p)==Status::Ok,"parallel dense");
  for(size_t i=0;i<n;++i) check(near(serial[i],parallel[i],0),"parallel agrees exactly");
}

int main() {
  test_skin_and_morph();test_dq_and_normal_matrix();
  test_vector_normalization<float>();test_vector_normalization<double>();
  test_normals_and_errors();test_parallel_equivalence();
  std::cout << "core tests passed\n";
}
