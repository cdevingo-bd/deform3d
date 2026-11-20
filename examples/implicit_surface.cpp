// Standalone backward-Euler step with dense offset-surface spring forces.
// c++ -std=c++17 -O3 -Iinclude examples/implicit_surface.cpp -o implicit_surface
// This small host example assembles its 9x9 Newton matrix through apply(). A
// larger host can instead use the same operator in its own iterative solver.
#include <deform3d/surface_linearization.hpp>
#include <cstdio>

using namespace deform3d;
using V=Vec3<double>;
using S=SurfaceEmbedStatus;

static bool solve(double a[9][10],double x[9]) {
  for(int col=0;col<9;++col) {
    int pivot=col;for(int row=col+1;row<9;++row)if(std::abs(a[row][col])>std::abs(a[pivot][col]))pivot=row;
    if(std::abs(a[pivot][col])<1e-14)return false;
    for(int j=col;j<10;++j)std::swap(a[col][j],a[pivot][j]);
    for(int row=col+1;row<9;++row) {
      const double factor=a[row][col]/a[col][col];
      for(int j=col;j<10;++j)a[row][j]-=factor*a[col][j];
    }
  }
  for(int row=8;row>=0;--row) {
    double value=a[row][9];for(int j=row+1;j<9;++j)value-=a[row][j]*x[j];
    x[row]=value/a[row][row];
  }
  return true;
}

int main() {
  const V rest[3]={{0,0,0},{1,0,0},{0,1,0}};
  const std::uint32_t tri[3]={0,1,2};
  const V points[4]={{.2,.2,.08},{.6,.1,-.06},{.1,.6,.05},{.3,.3,-.04}};
  V targets[4],x[3];std::copy(rest,rest+3,x);
  for(int i=0;i<4;++i)targets[i]=points[i]+V{-.08*points[i].x,.04*points[i].y,.14+.25*points[i].x};
  SurfaceEmbedOptions<double> options;options.max_distance=.1;
  SurfaceEmbedding<double> embedding;
  if(embedding.prepare(rest,3,tri,1,points,4,options)!=S::Ok)return 1;
  SurfaceCoupling<double> coupling;SurfaceLinearization<double> op;
  if(coupling.prepare(embedding)!=S::Ok || op.prepare(embedding)!=S::Ok)return 2;
  constexpr double stiffness=7500.,dt=.01,mass=1.,inertia=mass/(dt*dt);
  Mat3<double> k[4];for(auto& m:k)m.m[0]=m.m[4]=m.m[8]=-stiffness;
  V y[4],f[4],nodal[3];
  auto evaluate=[&](const V* state,double& energy) {
    if(coupling.update(state,3)!=S::Ok || coupling.evaluate_positions(y,4)!=S::Ok)return false;
    energy=0;
    for(int i=0;i<4;++i){f[i]=(targets[i]-y[i])*stiffness;energy+=.5*stiffness*dot(y[i]-targets[i],y[i]-targets[i]);}
    for(int i=0;i<3;++i)energy+=.5*inertia*dot(state[i]-rest[i],state[i]-rest[i]);
    return std::isfinite(energy);
  };
  int iterations=0;double residual=1,energy=0;
  for(;iterations<30;++iterations) {
    if(!evaluate(x,energy) || op.update(x,3,f,k,4)!=S::Ok || op.evaluate_forces(nodal,3)!=S::Ok)return 3;
    V r[3];double norm2=0,force2=0;
    for(int i=0;i<3;++i){r[i]=(x[i]-rest[i])*inertia-nodal[i];norm2+=dot(r[i],r[i]);force2+=dot(nodal[i],nodal[i]);}
    residual=std::sqrt(norm2/std::max(force2,1.));
    if(residual<1e-9)break;
    double matrix[9][10]{};
    for(int col=0;col<9;++col) {
      V basis[3]{},product[3];
      if(col%3==0)basis[col/3].x=1;else if(col%3==1)basis[col/3].y=1;else basis[col/3].z=1;
      if(op.apply(basis,3,product,3)!=S::Ok)return 4;
      for(int row=0;row<9;++row)matrix[row][col]=(row==col?inertia:0)-
          (row%3==0?product[row/3].x:row%3==1?product[row/3].y:product[row/3].z);
    }
    for(int row=0;row<9;++row)matrix[row][9]=-(row%3==0?r[row/3].x:row%3==1?r[row/3].y:r[row/3].z);
    double delta[9]{};if(!solve(matrix,delta))return 5;
    V direction[3];double slope=0;
    for(int i=0;i<3;++i){direction[i]={delta[3*i],delta[3*i+1],delta[3*i+2]};slope+=dot(r[i],direction[i]);}
    if(!(slope<0))return 6;
    bool accepted=false;double alpha=1;
    for(int search=0;search<24;++search,alpha*=.5) {
      V trial[3];for(int i=0;i<3;++i)trial[i]=x[i]+direction[i]*alpha;
      double candidate;
      if(evaluate(trial,candidate) && candidate<=energy+1e-4*alpha*slope) {
        std::copy(trial,trial+3,x);accepted=true;break;
      }
    }
    if(!accepted)return 7;
  }
  // Validate with the original uncompressed force transfer, not cached forces.
  if(!evaluate(x,energy) || coupling.transfer_forces(f,4,nodal,3)!=S::Ok)return 8;
  double norm2=0,force2=0;
  for(int i=0;i<3;++i) {const V r=(x[i]-rest[i])*inertia-nodal[i];norm2+=dot(r,r);force2+=dot(nodal[i],nodal[i]);}
  residual=std::sqrt(norm2/std::max(force2,1.));
  std::printf("{\"newton_iterations\":%d,\"relative_backward_euler_residual\":%.12g,\"dt_s\":%.3f,\"scope\":\"standalone nonlinear offset-surface spring step; no contact or material calibration\"}\n",iterations,residual,dt);
  return iterations<30 && residual<1e-9?0:9;
}
