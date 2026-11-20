// deform3d: dependency-free, header-only deformation kernels. C++17.
// Copyright (c) 2026. MIT license; see LICENSE.
#ifndef DEFORM3D_DEFORM3D_HPP
#define DEFORM3D_DEFORM3D_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace deform3d {

enum class Status {
  Ok, InvalidArgument, InvalidIndex, InvalidWeight, NonFinite,
  DegenerateGeometry, SingularTransform, MissingWorkspace
};

inline const char* status_string(Status s) noexcept {
  switch (s) {
    case Status::Ok: return "ok";
    case Status::InvalidArgument: return "invalid argument";
    case Status::InvalidIndex: return "index outside vertex or joint range";
    case Status::InvalidWeight: return "invalid skinning weight";
    case Status::NonFinite: return "non-finite value";
    case Status::DegenerateGeometry: return "degenerate geometry";
    case Status::SingularTransform: return "singular affine transform";
    case Status::MissingWorkspace: return "workspace required";
  }
  return "unknown status";
}

template <class T> struct Vec3 {
  T x = T(0), y = T(0), z = T(0);
  constexpr Vec3() noexcept = default;
  constexpr Vec3(T x_, T y_, T z_) noexcept : x(x_), y(y_), z(z_) {}
  constexpr Vec3 operator+(Vec3 b) const noexcept { return {x+b.x,y+b.y,z+b.z}; }
  constexpr Vec3 operator-(Vec3 b) const noexcept { return {x-b.x,y-b.y,z-b.z}; }
  constexpr Vec3 operator-() const noexcept { return {-x,-y,-z}; }
  constexpr Vec3 operator*(T a) const noexcept { return {x*a,y*a,z*a}; }
  constexpr Vec3 operator/(T a) const noexcept { return {x/a,y/a,z/a}; }
  Vec3& operator+=(Vec3 b) noexcept { x+=b.x; y+=b.y; z+=b.z; return *this; }
  Vec3& operator-=(Vec3 b) noexcept { x-=b.x; y-=b.y; z-=b.z; return *this; }
  Vec3& operator*=(T a) noexcept { x*=a; y*=a; z*=a; return *this; }
};
template<class T> constexpr Vec3<T> operator*(T a, Vec3<T> b) noexcept { return b*a; }
template<class T> constexpr T dot(Vec3<T> a, Vec3<T> b) noexcept { return a.x*b.x+a.y*b.y+a.z*b.z; }
template<class T> constexpr Vec3<T> cross(Vec3<T> a, Vec3<T> b) noexcept {
  return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
template<class T> inline T norm(Vec3<T> a) noexcept { return std::sqrt(dot(a,a)); }
template<class T> inline Vec3<T> normalized(Vec3<T> a, Vec3<T> fallback = {}) noexcept {
  // A normal's area magnitude depends on mesh units and tessellation density.
  // Every finite nonzero vector has a direction; an absolute epsilon here
  // incorrectly discards valid normals on small triangles. Scale first so the
  // squared norm cannot underflow or overflow for finite inputs.
  const T scale = std::max({std::abs(a.x),std::abs(a.y),std::abs(a.z)});
  if (!(scale > T(0)) || !std::isfinite(scale)) return fallback;
  const Vec3<T> scaled = a/scale;
  const T n = std::sqrt(dot(scaled,scaled));
  return std::isfinite(n) && n > T(0) ? scaled/n : fallback;
}
template<class T> inline bool finite(T v) noexcept { return std::isfinite(v); }
template<class T> inline bool finite(Vec3<T> v) noexcept {
  return finite(v.x)&&finite(v.y)&&finite(v.z);
}

template<class T> struct Quat {
  T w = T(1), x = T(0), y = T(0), z = T(0);
  constexpr Quat() noexcept = default;
  constexpr Quat(T w_, T x_, T y_, T z_) noexcept : w(w_),x(x_),y(y_),z(z_) {}
  constexpr Quat operator+(Quat q) const noexcept { return {w+q.w,x+q.x,y+q.y,z+q.z}; }
  constexpr Quat operator-(Quat q) const noexcept { return {w-q.w,x-q.x,y-q.y,z-q.z}; }
  constexpr Quat operator-() const noexcept { return {-w,-x,-y,-z}; }
  constexpr Quat operator*(T a) const noexcept { return {w*a,x*a,y*a,z*a}; }
  Quat& operator+=(Quat q) noexcept {w+=q.w;x+=q.x;y+=q.y;z+=q.z;return *this;}
  constexpr Quat conjugate() const noexcept {return {w,-x,-y,-z};}
};
template<class T> constexpr T dot(Quat<T> a, Quat<T> b) noexcept {
  return a.w*b.w+a.x*b.x+a.y*b.y+a.z*b.z;
}
template<class T> constexpr Quat<T> operator*(Quat<T> a, Quat<T> b) noexcept {
  return {a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z,
          a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
          a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
          a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w};
}
template<class T> inline bool finite(Quat<T> q) noexcept {
  return finite(q.w)&&finite(q.x)&&finite(q.y)&&finite(q.z);
}
template<class T> inline Quat<T> normalized(Quat<T> q) noexcept {
  const T n = std::sqrt(dot(q,q));
  return n > std::numeric_limits<T>::epsilon()*T(16) ? q*(T(1)/n) : Quat<T>{};
}
template<class T> inline Vec3<T> rotate(Quat<T> q, Vec3<T> p) noexcept {
  const Vec3<T> v{q.x,q.y,q.z};
  const Vec3<T> t = T(2)*cross(v,p);
  return p + q.w*t + cross(v,t);
}

template<class T> struct DualQuat {
  Quat<T> real, dual{T(0),T(0),T(0),T(0)};
  static DualQuat from_rotation_translation(Quat<T> q, Vec3<T> t) noexcept {
    q = normalized(q);
    return {q, (Quat<T>{T(0),t.x,t.y,t.z}*q)*T(0.5)};
  }
  Vec3<T> translation() const noexcept {
    const Quat<T> t = dual*real.conjugate();
    return {T(2)*t.x,T(2)*t.y,T(2)*t.z};
  }
  Vec3<T> transform_point(Vec3<T> p) const noexcept {
    return rotate(real,p)+translation();
  }
};
template<class T> inline bool finite(DualQuat<T> q) noexcept {
  return finite(q.real)&&finite(q.dual);
}

// Row-major 3x4 affine transform. Transform points in the model's coordinate system.
template<class T> struct Affine3 {
  std::array<T,12> m{{T(1),T(0),T(0),T(0),
                      T(0),T(1),T(0),T(0),
                      T(0),T(0),T(1),T(0)}};
  Vec3<T> transform_point(Vec3<T> p) const noexcept {
    return {m[0]*p.x+m[1]*p.y+m[2]*p.z+m[3],
            m[4]*p.x+m[5]*p.y+m[6]*p.z+m[7],
            m[8]*p.x+m[9]*p.y+m[10]*p.z+m[11]};
  }
  Vec3<T> transform_vector(Vec3<T> p) const noexcept {
    return {m[0]*p.x+m[1]*p.y+m[2]*p.z,
            m[4]*p.x+m[5]*p.y+m[6]*p.z,
            m[8]*p.x+m[9]*p.y+m[10]*p.z};
  }
};
template<class T> inline bool finite(const Affine3<T>& a) noexcept {
  for (T v : a.m) if (!finite(v)) return false;
  return true;
}

template<class T> struct Mat3 {
  std::array<T,9> m{};
  Vec3<T> transform(Vec3<T> p) const noexcept {
    return {m[0]*p.x+m[1]*p.y+m[2]*p.z,
            m[3]*p.x+m[4]*p.y+m[5]*p.z,
            m[6]*p.x+m[7]*p.y+m[8]*p.z};
  }
};
template<class T> inline Status inverse_transpose(const Affine3<T>& a, Mat3<T>& out) noexcept {
  const T x=a.m[0], y=a.m[1], z=a.m[2];
  const T u=a.m[4], v=a.m[5], w=a.m[6];
  const T p=a.m[8], q=a.m[9], r=a.m[10];
  const std::array<T,9> cof{{v*r-w*q,w*p-u*r,u*q-v*p,
                              z*q-y*r,x*r-z*p,y*p-x*q,
                              y*w-z*v,z*u-x*w,x*v-y*u}};
  const T det=x*cof[0]+y*cof[1]+z*cof[2];
  const T max_entry = std::max({std::abs(x),std::abs(y),std::abs(z),
                                std::abs(u),std::abs(v),std::abs(w),
                                std::abs(p),std::abs(q),std::abs(r)});
  if (max_entry == T(0) || std::abs(det) <=
      std::numeric_limits<T>::epsilon()*T(32)*max_entry*max_entry*max_entry)
    return Status::SingularTransform;
  for (size_t i=0;i<9;++i) out.m[i]=cof[i]/det;
  return Status::Ok;
}

// Strided xyz triples, supporting interleaved engine buffers. No alignment assumption.
// Strides are bytes. Memory ranges of distinct output attributes must not overlap.
template<class T> struct ConstVec3View {
  const void* data = nullptr;
  size_t stride = sizeof(Vec3<T>);
  Vec3<T> get(size_t i) const noexcept {
    Vec3<T> p;
    std::memcpy(&p,static_cast<const unsigned char*>(data)+i*stride,sizeof(p));
    return p;
  }
  bool valid(size_t n) const noexcept {
    return (n==0 || data!=nullptr) && stride>=sizeof(Vec3<T>) &&
           (n<2 || stride<=(std::numeric_limits<size_t>::max()-sizeof(Vec3<T>))/(n-1));
  }
};
template<class T> struct Vec3View {
  void* data = nullptr;
  size_t stride = sizeof(Vec3<T>);
  operator ConstVec3View<T>() const noexcept {return {data,stride};}
  Vec3<T> get(size_t i) const noexcept {
    Vec3<T> p;
    std::memcpy(&p,static_cast<const unsigned char*>(data)+i*stride,sizeof(p));
    return p;
  }
  void set(size_t i, Vec3<T> p) const noexcept {
    std::memcpy(static_cast<unsigned char*>(data)+i*stride,&p,sizeof(p));
  }
  bool valid(size_t n) const noexcept {
    return (n==0 || data!=nullptr) && stride>=sizeof(Vec3<T>) &&
           (n<2 || stride<=(std::numeric_limits<size_t>::max()-sizeof(Vec3<T>))/(n-1));
  }
};
template<class T> inline ConstVec3View<T> view(const Vec3<T>* p) noexcept {return {p,sizeof(Vec3<T>)};}
template<class T> inline Vec3View<T> view(Vec3<T>* p) noexcept {return {p,sizeof(Vec3<T>)};}

template<class T> struct VertexWeights4 {
  std::array<uint32_t,4> joint{};
  std::array<T,4> weight{};
};

// Immutable after preparation. Zero-influence vertices keep their input attributes.
template<class T> class SkinBinding {
 public:
  Status prepare(const VertexWeights4<T>* source, size_t count, size_t joint_count) {
    if (count && !source) return Status::InvalidArgument;
    std::vector<VertexWeights4<T>> temp(count);
    for (size_t i=0;i<count;++i) {
      T sum=T(0);
      for (size_t k=0;k<4;++k) {
        const T w=source[i].weight[k];
        if (!finite(w) || w<T(0)) return Status::InvalidWeight;
        if (w>T(0) && source[i].joint[k]>=joint_count) return Status::InvalidIndex;
        sum+=w;
      }
      if (!finite(sum)) return Status::InvalidWeight;
      for (size_t k=0;k<4;++k) {
        temp[i].joint[k]=source[i].joint[k];
        temp[i].weight[k]=sum>T(0) ? source[i].weight[k]/sum : T(0);
      }
    }
    weights_.swap(temp); joint_count_=joint_count;
    return Status::Ok;
  }
  size_t size() const noexcept {return weights_.size();}
  size_t joint_count() const noexcept {return joint_count_;}
  const VertexWeights4<T>& operator[](size_t i) const noexcept {return weights_[i];}
 private:
  std::vector<VertexWeights4<T>> weights_;
  size_t joint_count_=0;
};

template<class T> struct MorphTargetView {
  const uint32_t* indices=nullptr;
  const Vec3<T>* position_deltas=nullptr;
  const Vec3<T>* normal_deltas=nullptr; // optional
  size_t count=0;
};

// Sparse indexed correctives. Duplicates are intentionally additive, in target order.
template<class T> class MorphSet {
 public:
  struct Delta { uint32_t vertex; Vec3<T> position, normal; };
  struct Range { size_t begin, end; };
  Status prepare(size_t vertices, const MorphTargetView<T>* source, size_t target_count) {
    if (target_count && !source) return Status::InvalidArgument;
    std::vector<Range> ranges;
    std::vector<Delta> deltas;
    ranges.reserve(target_count);
    for (size_t t=0;t<target_count;++t) {
      const auto& s=source[t];
      if (s.count && (!s.indices || !s.position_deltas)) return Status::InvalidArgument;
      if (s.count > deltas.max_size()-deltas.size()) return Status::InvalidArgument;
      const size_t begin=deltas.size();
      for (size_t i=0;i<s.count;++i) {
        if (s.indices[i]>=vertices) return Status::InvalidIndex;
        const Vec3<T> n=s.normal_deltas ? s.normal_deltas[i] : Vec3<T>{};
        if (!finite(s.position_deltas[i]) || !finite(n)) return Status::NonFinite;
        deltas.push_back({s.indices[i],s.position_deltas[i],n});
      }
      ranges.push_back({begin,deltas.size()});
    }
    vertices_=vertices; ranges_.swap(ranges); deltas_.swap(deltas);
    return Status::Ok;
  }
  size_t vertex_count() const noexcept {return vertices_;}
  size_t target_count() const noexcept {return ranges_.size();}
  Range range(size_t i) const noexcept {return ranges_[i];}
  const Delta& delta(size_t i) const noexcept {return deltas_[i];}
 private:
  size_t vertices_=0;
  std::vector<Range> ranges_;
  std::vector<Delta> deltas_;
};

// Area-weighted shared-vertex normals. Duplicate seam vertices for hard edges.
class NormalTopology {
 public:
  Status prepare(size_t vertices, const uint32_t* indices, size_t index_count) {
    if (index_count%3 || (index_count && !indices) ||
        index_count>std::numeric_limits<uint32_t>::max() ||
        vertices>std::numeric_limits<uint32_t>::max()) return Status::InvalidArgument;
    std::vector<uint32_t> tris(index_count), offsets(vertices+1,0), adjacency(index_count);
    for (size_t i=0;i<index_count;++i) {
      if (indices[i]>=vertices) return Status::InvalidIndex;
      tris[i]=indices[i]; ++offsets[indices[i]+1];
    }
    for (size_t i=1;i<offsets.size();++i) offsets[i]+=offsets[i-1];
    std::vector<uint32_t> cursors=offsets;
    for (size_t i=0;i<index_count;++i) adjacency[cursors[tris[i]]++]=uint32_t(i/3);
    vertices_=vertices; tris_.swap(tris); offsets_.swap(offsets); adjacent_.swap(adjacency);
    return Status::Ok;
  }
  size_t vertex_count() const noexcept {return vertices_;}
  size_t triangle_count() const noexcept {return tris_.size()/3;}
  const uint32_t* triangle(size_t t) const noexcept {return tris_.data()+3*t;}
  uint32_t first_face(size_t v) const noexcept {return offsets_[v];}
  uint32_t end_face(size_t v) const noexcept {return offsets_[v+1];}
  uint32_t face_at(size_t offset) const noexcept {return adjacent_[offset];}
 private:
  size_t vertices_=0;
  std::vector<uint32_t> tris_,offsets_,adjacent_;
};

// The callback must synchronously process disjoint, complete ranges [begin,end).
// It can call run(job,begin,end) on worker threads and must join before returning.
struct Executor {
  void* user=nullptr;
  void (*parallel_for)(void* user, size_t count, size_t grain, void* job,
                       void (*run)(void*,size_t,size_t))=nullptr;
};
inline void dispatch(Executor e, size_t n, size_t grain, void* job,
                     void (*run)(void*,size_t,size_t)) {
  if (e.parallel_for && n>grain) e.parallel_for(e.user,n,grain,job,run);
  else run(job,0,n);
}

enum class SkinMode { None, LinearBlend, DualQuaternion };
enum class NormalMode { None, Skin, Recompute };

template<class T> struct Workspace {
  std::vector<Vec3<T>> morphed_positions, morphed_normals, face_cross;
  std::vector<Mat3<T>> normal_matrices;
  std::vector<DualQuat<T>> rigid_joints;
  void reserve(size_t vertices, size_t joints=0, size_t faces=0) {
    morphed_positions.reserve(vertices); morphed_normals.reserve(vertices);
    face_cross.reserve(faces); normal_matrices.reserve(joints); rigid_joints.reserve(joints);
  }
};

template<class T> struct EvaluateParams {
  size_t vertices=0;
  ConstVec3View<T> input_positions, input_normals;
  Vec3View<T> output_positions, output_normals;
  const SkinBinding<T>* skin=nullptr;
  SkinMode skin_mode=SkinMode::None;
  const Affine3<T>* affine_joints=nullptr;
  const DualQuat<T>* rigid_joints=nullptr;
  size_t joint_count=0;
  const MorphSet<T>* morphs=nullptr;
  const T* morph_weights=nullptr;
  size_t morph_weight_count=0;
  NormalMode normal_mode=NormalMode::None;
  const NormalTopology* topology=nullptr;
  Workspace<T>* workspace=nullptr;
  Executor executor{};
  size_t grain=16384;
};

namespace detail {
template<class T> struct VertexJob {
  const EvaluateParams<T>* p;
  const Workspace<T>* ws;
  bool morph;
};
template<class T> void vertex_run(void* raw, size_t begin, size_t end) {
  const auto& c=*static_cast<VertexJob<T>*>(raw);
  const auto& p=*c.p;
  for (size_t i=begin;i<end;++i) {
    const Vec3<T> pos=c.morph ? c.ws->morphed_positions[i] : p.input_positions.get(i);
    const bool skin_normals=p.normal_mode==NormalMode::Skin;
    const Vec3<T> normal=skin_normals ?
      (c.morph ? c.ws->morphed_normals[i] : p.input_normals.get(i)) : Vec3<T>{};
    Vec3<T> out=pos, nout=normal;
    if (p.skin_mode!=SkinMode::None) {
      const auto& weights=(*p.skin)[i];
      T total=T(0);
      for (size_t k=0;k<4;++k) total+=weights.weight[k];
      if (total>T(0)) {
        if (p.skin_mode==SkinMode::LinearBlend) {
          out={}; nout={};
          for (size_t k=0;k<4;++k) {
            const T w=weights.weight[k];
            if (w<=T(0)) continue;
            const size_t j=weights.joint[k];
            out+=p.affine_joints[j].transform_point(pos)*w;
            if (skin_normals) nout+=c.ws->normal_matrices[j].transform(normal)*w;
          }
        } else {
          Quat<T> qr{T(0),T(0),T(0),T(0)}, qd{T(0),T(0),T(0),T(0)};
          Quat<T> reference;
          bool first=true;
          for (size_t k=0;k<4;++k) {
            const T w=weights.weight[k];
            if (w<=T(0)) continue;
            const auto& q=c.ws->rigid_joints[weights.joint[k]];
            if (first) {reference=q.real;first=false;}
            const T signed_weight=dot(q.real,reference)<T(0) ? -w : w;
            qr+=q.real*signed_weight; qd+=q.dual*signed_weight;
          }
          T len=std::sqrt(dot(qr,qr));
          if (len<=std::numeric_limits<T>::epsilon()*T(16)) {
            for (size_t k=0;k<4;++k) if (weights.weight[k]>T(0)) {
              qr=c.ws->rigid_joints[weights.joint[k]].real;
              qd=c.ws->rigid_joints[weights.joint[k]].dual;break;
            }
          } else {qr=qr*(T(1)/len);qd=qd*(T(1)/len);}
          qd=qd-qr*dot(qr,qd);
          const DualQuat<T> blend{qr,qd};
          out=blend.transform_point(pos);
          if (skin_normals) nout=rotate(qr,normal);
        }
      }
    }
    p.output_positions.set(i,out);
    if (skin_normals) p.output_normals.set(i,normalized(nout,normal));
  }
}
template<class T> struct FaceJob {const NormalTopology* topology;Vec3View<T> positions;Vec3<T>* crosses;};
template<class T> void face_run(void* raw,size_t begin,size_t end) {
  auto& c=*static_cast<FaceJob<T>*>(raw);
  for (size_t t=begin;t<end;++t) {
    const uint32_t* id=c.topology->triangle(t);
    const Vec3<T> a=c.positions.get(id[0]),b=c.positions.get(id[1]),d=c.positions.get(id[2]);
    c.crosses[t]=cross(b-a,d-a);
  }
}
template<class T> struct NormalJob {const NormalTopology* topology;const Vec3<T>* crosses;Vec3View<T> normals;};
template<class T> void normal_run(void* raw,size_t begin,size_t end) {
  auto& c=*static_cast<NormalJob<T>*>(raw);
  for (size_t v=begin;v<end;++v) {
    Vec3<T> n{};
    for (uint32_t k=c.topology->first_face(v);k<c.topology->end_face(v);++k)
      n+=c.crosses[c.topology->face_at(k)];
    c.normals.set(v,normalized(n));
  }
}
} // namespace detail

// Prevalidate at setup; evaluate performs no per-vertex allocations. A workspace is
// required for morphs, normal skinning, DQ skinning, or normal recomputation.
// Input positions and output positions may be exactly in-place. Partial overlap is
// unsupported. Evaluations sharing a workspace must not run concurrently.
template<class T> Status evaluate(const EvaluateParams<T>& p) {
  static_assert(std::is_floating_point<T>::value,"deform3d requires floating point scalar");
  if (!p.input_positions.valid(p.vertices) || !p.output_positions.valid(p.vertices) ||
      p.grain==0) return Status::InvalidArgument;
  if (p.normal_mode==NormalMode::Skin &&
      (!p.input_normals.valid(p.vertices)||!p.output_normals.valid(p.vertices))) return Status::InvalidArgument;
  if (p.normal_mode==NormalMode::Recompute &&
      (!p.output_normals.valid(p.vertices)||!p.topology||
       p.topology->vertex_count()!=p.vertices)) return Status::InvalidArgument;
  if (p.skin_mode!=SkinMode::None &&
      (!p.skin || p.skin->size()!=p.vertices || p.joint_count<p.skin->joint_count() ||
       (p.joint_count && !(p.skin_mode==SkinMode::LinearBlend ?
                           static_cast<bool>(p.affine_joints):static_cast<bool>(p.rigid_joints)))))
    return Status::InvalidArgument;
  if (p.morphs && (p.morphs->vertex_count()!=p.vertices ||
      p.morph_weight_count!=p.morphs->target_count() ||
      (p.morph_weight_count && !p.morph_weights))) return Status::InvalidArgument;
  if (!p.morphs && p.morph_weight_count) return Status::InvalidArgument;
  bool active_morph=false;
  for (size_t t=0;t<p.morph_weight_count;++t) {
    if (!finite(p.morph_weights[t])) return Status::NonFinite;
    active_morph|=p.morph_weights[t]!=T(0);
  }
  const bool need_workspace=active_morph || p.normal_mode==NormalMode::Recompute ||
      (p.normal_mode==NormalMode::Skin && p.skin_mode==SkinMode::LinearBlend) ||
      p.skin_mode==SkinMode::DualQuaternion;
  if (need_workspace && !p.workspace) return Status::MissingWorkspace;
  if (p.skin_mode==SkinMode::LinearBlend) {
    for (size_t j=0;j<p.joint_count;++j) {
      if (!finite(p.affine_joints[j])) return Status::NonFinite;
    }
    if (p.normal_mode==NormalMode::Skin) {
      p.workspace->normal_matrices.resize(p.joint_count);
      for (size_t j=0;j<p.joint_count;++j) {
        Status s=inverse_transpose(p.affine_joints[j],p.workspace->normal_matrices[j]);
        if (s!=Status::Ok) return s;
      }
    }
  }
  if (p.skin_mode==SkinMode::DualQuaternion) {
    p.workspace->rigid_joints.resize(p.joint_count);
    for (size_t j=0;j<p.joint_count;++j) {
      auto q=p.rigid_joints[j];
      if (!finite(q)) return Status::NonFinite;
      const T n=std::sqrt(dot(q.real,q.real));
      if (n<=std::numeric_limits<T>::epsilon()*T(16)) return Status::SingularTransform;
      q.real=q.real*(T(1)/n);q.dual=q.dual*(T(1)/n);
      q.dual=q.dual-q.real*dot(q.real,q.dual);
      p.workspace->rigid_joints[j]=q;
    }
  }
  if (active_morph) {
    auto& ws=*p.workspace;
    ws.morphed_positions.resize(p.vertices);
    const bool normals=p.normal_mode==NormalMode::Skin;
    if (normals) ws.morphed_normals.resize(p.vertices);
    for (size_t i=0;i<p.vertices;++i) {
      ws.morphed_positions[i]=p.input_positions.get(i);
      if (normals) ws.morphed_normals[i]=p.input_normals.get(i);
    }
    for (size_t t=0;t<p.morph_weight_count;++t) {
      const T weight=p.morph_weights[t];
      if (weight==T(0)) continue;
      const auto range=p.morphs->range(t);
      for (size_t k=range.begin;k<range.end;++k) {
        const auto& delta=p.morphs->delta(k);
        ws.morphed_positions[delta.vertex]+=delta.position*weight;
        if (normals) ws.morphed_normals[delta.vertex]+=delta.normal*weight;
      }
    }
  }
  detail::VertexJob<T> job{&p,p.workspace,active_morph};
  dispatch(p.executor,p.vertices,p.grain,&job,detail::vertex_run<T>);
  if (p.normal_mode==NormalMode::Recompute) {
    auto& crosses=p.workspace->face_cross;
    crosses.resize(p.topology->triangle_count());
    detail::FaceJob<T> face{p.topology,p.output_positions,crosses.data()};
    dispatch(p.executor,crosses.size(),p.grain,&face,detail::face_run<T>);
    detail::NormalJob<T> normal{p.topology,crosses.data(),p.output_normals};
    dispatch(p.executor,p.vertices,p.grain,&normal,detail::normal_run<T>);
  }
  return Status::Ok;
}

using Vec3f=Vec3<float>;
using Vec3d=Vec3<double>;
using Affine3f=Affine3<float>;
using DualQuatf=DualQuat<float>;

} // namespace deform3d
#endif // DEFORM3D_DEFORM3D_HPP
