#pragma once

// Exact local-force linearization through an offset surface. C++17 only.
#include "surface_coupling.hpp"

namespace deform3d {

// Given point forces f_i and local world-space derivatives K_i=df_i/dy_i,
// this snapshot represents F=J^T f and A=J^T K J + D(J^T)[.] f.
// The latter is a FORCE Jacobian, not an energy Hessian (their signs differ).
// Point laws must be local: cross-point/off-object derivatives are not included.
template<class T> class SurfaceLinearization {
  using D = surface_detail::Differential<T>;
  using L = typename D::L;
  using P = Vec3<L>;
  using S = SurfaceEmbedStatus;
  using Binding = typename SurfaceEmbedding<T>::Binding;
  struct Triangle {
    std::array<std::uint32_t,3> ids{};
    D geometry{};
    std::array<P,3> force{};
    std::array<L,81> jacobian{}; // row-major 9x9, three XYZ corners
  };
  struct Moments {
    std::array<P,4> force{};
    std::array<Mat3<L>,10> tangent{};
  };
 public:
  // Own a compact plan of referenced triangles and bindings. Preparation may
  // allocate/throw; failures leave an existing plan/snapshot unchanged.
  SurfaceEmbedStatus prepare(const SurfaceEmbedding<T>& embedding) {
    if (!embedding.ready() || embedding.simulation_vertex_count()==std::numeric_limits<std::size_t>::max())
      return S::InvalidInput;
    SurfaceLinearization candidate;
    candidate.vertex_count_=embedding.simulation_vertex_count();
    candidate.bindings_=embedding.bindings();
    const auto invalid=std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> remap(embedding.triangle_count(),invalid);
    for (auto& b:candidate.bindings_) {
      auto& mapped=remap[b.triangle];
      if (mapped==invalid) {
        mapped=std::uint32_t(candidate.triangles_.size());
        Triangle tri; tri.ids=b.vertices;
        candidate.triangles_.push_back(tri);
      }
      b.triangle=mapped;
    }
    if (candidate.triangles_.size()>std::numeric_limits<std::size_t>::max()/3) return S::InvalidInput;
    candidate.staging_=candidate.triangles_;
    candidate.moments_.resize(candidate.triangles_.size());
    candidate.offsets_.resize(candidate.vertex_count_+1);
    for (const auto& tri:candidate.triangles_)
      for (auto vertex:tri.ids) ++candidate.offsets_[std::size_t(vertex)+1];
    for (std::size_t i=1;i<candidate.offsets_.size();++i) candidate.offsets_[i]+=candidate.offsets_[i-1];
    candidate.incident_.resize(3*candidate.triangles_.size());
    auto cursor=candidate.offsets_;
    for (std::size_t i=0;i<candidate.triangles_.size();++i)
      for (std::size_t j=0;j<3;++j)
        candidate.incident_[cursor[candidate.triangles_[i].ids[j]]++]=3*i+j;
    swap(candidate);
    return S::Ok;
  }

  // Linearize at one host state. K is row-major df/dy, and may be nonsymmetric.
  // A null K means fixed world-space forces; geometric stiffness still applies.
  // No allocation. A failed update preserves the previous valid snapshot.
  SurfaceEmbedStatus update(const Vec3<T>* positions, std::size_t vertex_count,
      const Vec3<T>* forces, const Mat3<T>* point_jacobians, std::size_t point_count) noexcept {
    if (offsets_.empty() || !positions || !forces || vertex_count!=vertex_count_ ||
        point_count<bindings_.size() || !valid_count(point_count,sizeof(Vec3<T>)) ||
        (point_jacobians && !valid_count(point_count,sizeof(Mat3<T>)))) return S::InvalidInput;
    for (std::size_t i=0;i<vertex_count;++i) if (!finite(positions[i])) return S::NonFiniteInput;
    for (std::size_t i=0;i<staging_.size();++i) {
      auto& tri=staging_[i];
      const auto status=tri.geometry.update(positions[tri.ids[0]],positions[tri.ids[1]],positions[tri.ids[2]]);
      if (status!=S::Ok) return status;
      moments_[i]=Moments{};
    }
    // y_i = [w0 I, w1 I, w2 I, h I] [a,b,c,n]. Ten coefficient
    // pairs suffice even for nonsymmetric K: Q_ab=c_a*c_b*K=Q_ba,
    // without transposing K. No per-point 9x9 matrix is constructed.
    for (std::size_t i=0;i<bindings_.size();++i) {
      if (!finite(forces[i])) return S::NonFiniteInput;
      const auto& b=bindings_[i];
      const std::array<L,4> c{{L(b.weights[0]),L(b.weights[1]),L(b.weights[2]),L(b.normal_offset)}};
      auto& moment=moments_[b.triangle];
      for (std::size_t a=0;a<4;++a) moment.force[a]+=D::wide(forces[i])*c[a];
      if (point_jacobians) {
        const auto& k=point_jacobians[i].m;
        for (T value:k) if (!finite(value)) return S::NonFiniteInput;
        std::size_t pair=0;
        for (std::size_t a=0;a<4;++a)
          for (std::size_t z=a;z<4;++z,++pair) {
            const L weight=c[a]*c[z];
            if (weight==L(0)) continue;
            for (std::size_t entry=0;entry<9;++entry)
              moment.tangent[pair].m[entry]+=weight*L(k[entry]);
          }
      }
    }
    for (std::size_t t=0;t<staging_.size();++t) {
      auto& tri=staging_[t]; const auto& moment=moments_[t];
      for (const auto& v:moment.force) if (!finite(v)) return S::NumericalFailure;
      for (const auto& m:moment.tangent)
        for (L value:m.m) if (!finite(value)) return S::NumericalFailure;
      const auto offset_force=tri.geometry.normal_transpose(moment.force[3]);
      for (std::size_t a=0;a<3;++a) {
        tri.force[a]=moment.force[a]+offset_force[a];
        if (!finite(tri.force[a])) return S::NumericalFailure;
      }
      std::array<P,9> normal_columns;
      for (std::size_t col=0;col<9;++col) {
        std::array<P,3> u{}; u[col/3]=axis(col%3);
        normal_columns[col]=tri.geometry.normal_change(u);
      }
      for (std::size_t col=0;col<9;++col) {
        const P unit=axis(col%3), dn=normal_columns[col];
        const P normal_force=moment.tangent[pair_index(3,col/3)].transform(unit)+
                             moment.tangent[pair_index(3,3)].transform(dn);
        const auto pulled_normal=tri.geometry.normal_transpose(normal_force);
        std::array<P,3> u{}; u[col/3]=unit;
        const auto geometric=tri.geometry.normal_geometric(moment.force[3],u);
        for (std::size_t a=0;a<3;++a) {
          const P value=moment.tangent[pair_index(a,col/3)].transform(unit)+
                        moment.tangent[pair_index(a,3)].transform(dn)+pulled_normal[a]+geometric[a];
          if (!finite(value)) return S::NumericalFailure;
          for (std::size_t component=0;component<3;++component)
            tri.jacobian[(3*a+component)*9+col]=entry(value,component);
        }
      }
    }
    triangles_.swap(staging_); ready_=true;
    return S::Ok;
  }

  SurfaceEmbedStatus evaluate_forces(Vec3<T>* output, std::size_t count,
      SurfaceForceMode mode=SurfaceForceMode::Overwrite, std::size_t begin=0,
      std::size_t end=std::numeric_limits<std::size_t>::max()) const noexcept {
    const auto status=validate_output(output,count,mode,begin,end);
    if (status!=S::Ok) return status;
    return write_checked(output,begin,end,[&](std::size_t vertex) {
      P value=mode==SurfaceForceMode::Accumulate ? D::wide(output[vertex]) : P{};
      for (std::size_t e=offsets_[vertex];e<offsets_[vertex+1];++e)
        value+=triangles_[incident_[e]/3].force[incident_[e]%3];
      return value;
    });
  }

  // Apply the frozen force Jacobian without dense-point traversal or a global
  // matrix. Disjoint PHYSICAL vertex ranges may execute concurrently. Inputs
  // and output must not overlap. Full input validation is included in each call.
  SurfaceEmbedStatus apply(const Vec3<T>* displacement, std::size_t source_count,
      Vec3<T>* output, std::size_t count, SurfaceForceMode mode=SurfaceForceMode::Overwrite,
      std::size_t begin=0, std::size_t end=std::numeric_limits<std::size_t>::max()) const noexcept {
    if (!displacement || source_count!=vertex_count_) return S::InvalidInput;
    const auto status=validate_output(output,count,mode,begin,end);
    if (status!=S::Ok) return status;
    const auto x=reinterpret_cast<std::uintptr_t>(displacement), y=reinterpret_cast<std::uintptr_t>(output);
    if (count && (x<=y ? y-x<count*sizeof(Vec3<T>) : x-y<count*sizeof(Vec3<T>))) return S::Aliasing;
    for (std::size_t i=0;i<source_count;++i) if (!finite(displacement[i])) return S::NonFiniteInput;
    return write_checked(output,begin,end,[&](std::size_t vertex) {
      P value=mode==SurfaceForceMode::Accumulate ? D::wide(output[vertex]) : P{};
      for (std::size_t e=offsets_[vertex];e<offsets_[vertex+1];++e) {
        const auto& tri=triangles_[incident_[e]/3];
        const std::size_t corner=incident_[e]%3;
        P delta{};
        for (std::size_t row=0;row<3;++row) {
          L sum=0;
          for (std::size_t col=0;col<9;++col)
            sum+=tri.jacobian[(3*corner+row)*9+col]*entry(D::wide(displacement[tri.ids[col/3]]),col%3);
          if (row==0) delta.x=sum; else if (row==1) delta.y=sum; else delta.z=sum;
        }
        value+=delta;
      }
      return value;
    });
  }

  bool ready() const noexcept { return ready_; }
  std::size_t simulation_vertex_count() const noexcept { return vertex_count_; }
  std::size_t render_vertex_count() const noexcept { return bindings_.size(); }
  std::size_t active_triangle_count() const noexcept { return triangles_.size(); }
  void swap(SurfaceLinearization& other) noexcept {
    std::swap(vertex_count_,other.vertex_count_); std::swap(ready_,other.ready_);
    bindings_.swap(other.bindings_); triangles_.swap(other.triangles_); staging_.swap(other.staging_);
    moments_.swap(other.moments_); offsets_.swap(other.offsets_); incident_.swap(other.incident_);
  }
 private:
  static P axis(std::size_t i) noexcept { return {i==0?L(1):L(0),i==1?L(1):L(0),i==2?L(1):L(0)}; }
  static L entry(P p,std::size_t i) noexcept { return i==0?p.x:i==1?p.y:p.z; }
  static std::size_t pair_index(std::size_t a,std::size_t b) noexcept {
    if (a>b) std::swap(a,b);
    // Upper triangle of a 4x4 matrix: 00,01,02,03,11,12,13,22,23,33.
    return a*(9-a)/2+b-a;
  }
  static bool valid_count(std::size_t n,std::size_t width) noexcept {
    return n<=std::numeric_limits<std::size_t>::max()/width;
  }
  SurfaceEmbedStatus validate_output(Vec3<T>* output,std::size_t count,SurfaceForceMode mode,
      std::size_t begin,std::size_t& end) const noexcept {
    if (!ready_ || !output || count!=vertex_count_ || !valid_count(count,sizeof(Vec3<T>)) ||
        (mode!=SurfaceForceMode::Overwrite && mode!=SurfaceForceMode::Accumulate)) return S::InvalidInput;
    if (end==std::numeric_limits<std::size_t>::max()) end=vertex_count_;
    if (begin>end || end>vertex_count_) return S::InvalidRange;
    if (mode==SurfaceForceMode::Accumulate)
      for (std::size_t i=begin;i<end;++i) if (!finite(output[i])) return S::NonFiniteInput;
    return S::Ok;
  }
  template<class Compute> static SurfaceEmbedStatus write_checked(Vec3<T>* output,
      std::size_t begin,std::size_t end,Compute compute) noexcept {
    const L limit=std::numeric_limits<T>::max();
    for (std::size_t i=begin;i<end;++i) {
      const P p=compute(i);
      if (!finite(p) || std::abs(p.x)>limit || std::abs(p.y)>limit || std::abs(p.z)>limit)
        return S::NumericalFailure;
    }
    for (std::size_t i=begin;i<end;++i) {
      const P p=compute(i); output[i]={T(p.x),T(p.y),T(p.z)};
    }
    return S::Ok;
  }
  bool ready_=false;
  std::size_t vertex_count_=0;
  std::vector<Binding> bindings_;
  std::vector<Triangle> triangles_,staging_;
  std::vector<Moments> moments_;
  std::vector<std::size_t> offsets_,incident_;
};

} // namespace deform3d
