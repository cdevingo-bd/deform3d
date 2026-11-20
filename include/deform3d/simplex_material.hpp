#pragma once

// Allocation-free element constitutive law. No simulator or GPU dependency.
// The optional qualifier also lets the exact same source execute in CUDA.
#ifndef DEFORM3D_ELEMENT_INLINE
#define DEFORM3D_ELEMENT_INLINE inline
#define DEFORM3D_ELEMENT_LOCAL_INLINE
#endif

namespace deform3d {

// Quadratic energy in squared edge-length changes: E = (1/4) e^T C e.
// C is the symmetric material metric, packed by upper-triangular rows.
// Triangles: edges (1,2),(2,0),(0,1). Tets: (0,1),(1,2),(2,0),(2,3),(0,3),(1,3).
// The host supplies material/rest data; its continuum compiler can construct C.
// No plasticity or stable Neo-Hookean terms are implied by this law.
template<class T> struct SimplexMaterial {
  struct Response { T spring[12]{}, damper[12]{}; };

  DEFORM3D_ELEMENT_INLINE static int endpoint(int dim, int edge, int side) {
    const int triangle[6]={1,2,2,0,0,1};
    const int tetra[12]={0,1,1,2,2,0,2,3,0,3,1,3};
    return dim==2 ? triangle[2*edge+side] : tetra[2*edge+side];
  }

  // Preconditions: dim is 2 or 3; finite inputs; dt>0; rayleigh>=0;
  // packed has nedge*(nedge+1)/2 entries; positions has 3*(dim+1).
  // Lengths/speeds come from the host's kinematics to preserve its rounding.
  DEFORM3D_ELEMENT_INLINE static Response evaluate(int dim, const T* positions,
      const T* length, const T* rest_length, const T* speed, const T* packed,
      T dt, T rayleigh, bool spring_enabled=true, bool damper_enabled=true) {
    Response out;
    const int ne=dim==2 ? 3 : 6;
    T metric[36]{}, strain[6]{}, viscous[6]{}, tension[6]{}, damping[6]{};
    int k=0;
    for(int a=0;a<ne;++a) for(int b=a;b<ne;++b) {
      metric[a*ne+b]=metric[b*ne+a]=packed[k++];
    }
    const T kd=damper_enabled && dt>T(0) ? rayleigh/dt : T(0);
    for(int a=0;a<ne;++a) {
      strain[a]=spring_enabled ? length[a]*length[a]-rest_length[a]*rest_length[a] : T(0);
      const T dl=speed[a]*dt;
      viscous[a]=kd>T(0) ? dl*(T(2)*length[a]-dl)*kd : T(0);
    }
    for(int a=0;a<ne;++a) for(int b=0;b<ne;++b) {
      tension[a]+=metric[a*ne+b]*strain[b];
      damping[a]+=metric[a*ne+b]*viscous[b];
    }
    for(int a=0;a<ne;++a) {
      const int i=endpoint(dim,a,0), j=endpoint(dim,a,1);
      for(int c=0;c<3;++c) {
        const T edge=positions[3*i+c]-positions[3*j+c];
        const T fs=tension[a]*edge, fd=damping[a]*edge;
        out.spring[3*i+c]-=fs; out.spring[3*j+c]+=fs;
        out.damper[3*i+c]-=fd; out.damper[3*j+c]+=fd;
      }
    }
    return out;
  }

  // Exact elastic energy Hessian times u, or the host-compatible positive
  // geometric approximation (negative edge tensions clipped individually).
  // The approximation is NOT the exact Hessian under compression.
  DEFORM3D_ELEMENT_INLINE static void stiffness_apply(int dim, const T* positions,
      const T* rest_length, const T* packed, const T* u, T* out,
      bool positive_geometric=false) {
    const int ne=dim==2?3:6;
    T metric[36]{}, edge[18]{}, du[18]{}, strain[6]{}, dstrain[6]{}, tension[6]{}, dtension[6]{};
    int k=0;
    for(int a=0;a<ne;++a) for(int b=a;b<ne;++b)
      metric[a*ne+b]=metric[b*ne+a]=packed[k++];
    for(int a=0;a<ne;++a) {
      int i=endpoint(dim,a,0),j=endpoint(dim,a,1);
      strain[a]=-rest_length[a]*rest_length[a];
      for(int c=0;c<3;++c) {
        edge[3*a+c]=positions[3*i+c]-positions[3*j+c];
        du[3*a+c]=u[3*i+c]-u[3*j+c];
        strain[a]+=edge[3*a+c]*edge[3*a+c];
        dstrain[a]+=T(2)*edge[3*a+c]*du[3*a+c];
      }
    }
    for(int a=0;a<ne;++a) for(int b=0;b<ne;++b) {
      tension[a]+=metric[a*ne+b]*strain[b];
      dtension[a]+=metric[a*ne+b]*dstrain[b];
    }
    for(int c=0;c<3*(dim+1);++c) out[c]=T(0);
    for(int a=0;a<ne;++a) {
      int i=endpoint(dim,a,0),j=endpoint(dim,a,1);
      const T t=positive_geometric && tension[a]<T(0) ? T(0) : tension[a];
      for(int c=0;c<3;++c) {
        T v=dtension[a]*edge[3*a+c]+t*du[3*a+c];
        out[3*i+c]+=v; out[3*j+c]-=v;
      }
    }
  }
};

} // namespace deform3d
#ifdef DEFORM3D_ELEMENT_LOCAL_INLINE
#undef DEFORM3D_ELEMENT_INLINE
#undef DEFORM3D_ELEMENT_LOCAL_INLINE
#endif
