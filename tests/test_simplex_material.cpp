#include <deform3d/simplex_material.hpp>
#include <deform3d/simplex_rest.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#define CHECK(x) do { if(!(x)) { std::cerr<<__LINE__<<": "<<#x<<"\n"; std::abort(); } } while(0)
using Law=deform3d::SimplexMaterial<double>;
void lengths(int dim,const double* x,double* l) {
  for(int a=0;a<(dim==2?3:6);++a) {
    int i=Law::endpoint(dim,a,0),j=Law::endpoint(dim,a,1);
    double s=0;for(int c=0;c<3;++c)s+=std::pow(x[3*i+c]-x[3*j+c],2);
    l[a]=std::sqrt(s);
  }
}
double energy(int dim,const double* x,const double* r,const double* packed) {
  int ne=dim==2?3:6,k=0;double l[6],sum=0;lengths(dim,x,l);
  for(int i=0;i<ne;++i)for(int j=i;j<ne;++j)
    sum+=(i==j?.25:.5)*packed[k++]*(l[i]*l[i]-r[i]*r[i])*(l[j]*l[j]-r[j]*r[j]);
  return sum;
}
int main() {
  for(int dim:{2,3}) {
    int n=3*(dim+1),ne=dim==2?3:6,k=0;
    double restx[12]={0,0,0,1,0,0,.1,1,0,.2,.3,1};
    double x[12]={.1,-.2,.05,.85,.04,-.1,.3,.8,.2,.4,.2,.75};
    double r[6],l[6],speed[6]{},packed[21]{},u[12],h[12];
    lengths(dim,restx,r);lengths(dim,x,l);
    for(int a=0;a<ne;++a)for(int b=a;b<ne;++b)packed[k++]=a==b?3.+a:.15;
    for(int c=0;c<n;++c)u[c]=std::sin(c+1.);
    auto f=Law::evaluate(dim,x,l,r,speed,packed,.001,.02);
    Law::stiffness_apply(dim,x,r,packed,u,h);
    double xp[12],xm[12],lp[6],lm[6];
    const double eps=1e-6;
    for(int c=0;c<n;++c) { xp[c]=x[c]+eps*u[c];xm[c]=x[c]-eps*u[c]; }
    lengths(dim,xp,lp);lengths(dim,xm,lm);
    auto fp=Law::evaluate(dim,xp,lp,r,speed,packed,.001,.02);
    auto fm=Law::evaluate(dim,xm,lm,r,speed,packed,.001,.02);
    for(int c=0;c<n;++c) {
      CHECK(std::abs((fm.spring[c]-fp.spring[c])/(2*eps)-h[c])<2e-8);
      CHECK(f.damper[c]==0);
      std::copy(x,x+n,xp);std::copy(x,x+n,xm);xp[c]+=eps;xm[c]-=eps;
      CHECK(std::abs((energy(dim,xp,r,packed)-energy(dim,xm,r,packed))/(2*eps)+f.spring[c])<2e-8);
    }
    for(int c=0;c<3;++c) { double total=0;for(int v=0;v<dim+1;++v)total+=f.spring[3*v+c]; CHECK(std::abs(total)<1e-14); }
    for(int c=0;c<3;++c) {
      double torque=0;int a=(c+1)%3,b=(c+2)%3;
      for(int v=0;v<dim+1;++v)torque+=x[3*v+a]*f.spring[3*v+b]-x[3*v+b]*f.spring[3*v+a];
      CHECK(std::abs(torque)<1e-13);
    }
    double v[12],hv[12];for(int c=0;c<n;++c)v[c]=std::cos(c+.3);
    Law::stiffness_apply(dim,x,r,packed,v,hv);
    double uv=0,vu=0;for(int c=0;c<n;++c){uv+=u[c]*hv[c];vu+=v[c]*h[c];}CHECK(std::abs(uv-vu)<1e-12);
    auto off=Law::evaluate(dim,x,l,r,speed,packed,.001,.02,false,false);
    for(int c=0;c<12;++c)CHECK(off.spring[c]==0 && off.damper[c]==0);
    Law::stiffness_apply(dim,x,r,packed,u,h,true);
    double quadratic=0;for(int c=0;c<n;++c)quadratic+=u[c]*h[c];CHECK(quadratic>=0);
    // Independent continuum energy under uniform stretch, including area/thickness.
    deform3d::Vec3<double> reference[4];
    for(int v=0;v<dim+1;++v)reference[v]={restx[3*v],restx[3*v+1],restx[3*v+2]};
    deform3d::SimplexRestMaterial<double> material;
    CHECK(deform3d::prepare_simplex_material(dim,reference,12000.,.3,.002,material)==deform3d::Status::Ok);
    for(int c=0;c<n;++c)xp[c]=1.17*restx[c];
    double mu=12000./2.6,lambda=dim==2?12000.*.3/(1-.09):12000.*.3/(1.3*.4);
    double strain=(1.17*1.17-1)/2;
    double expected=material.measure*(mu*dim+.5*lambda*dim*dim)*strain*strain;
    CHECK(std::abs(energy(dim,xp,material.edge_lengths.data(),material.packed.data())-expected)<1e-11);
    const double F[9]={1.1,.25,.1,.1,.87,-.2,.15,-.1,1.07};
    for(int v=0;v<dim+1;++v)for(int c=0;c<3;++c) {
      xp[3*v+c]=.7;
      for(int a=0;a<3;++a)xp[3*v+c]+=F[3*c+a]*restx[3*v+a];
    }
    double trace=0,squared=0;
    for(int i=0;i<dim;++i)for(int j=0;j<dim;++j) {
      double gij=i==j?-1.:0.;for(int c=0;c<3;++c)gij+=F[3*c+i]*F[3*c+j];
      double E=gij/2;squared+=E*E;if(i==j)trace+=E;
    }
    expected=material.measure*(mu*squared+.5*lambda*trace*trace);
    CHECK(std::abs(energy(dim,xp,material.edge_lengths.data(),material.packed.data())-expected)<1e-10);
    for(double scale:{1e-4,1e4}) {
      deform3d::Vec3<double> scaled[4];deform3d::Vec3<float> small[4];
      for(int v=0;v<dim+1;++v) { scaled[v]=reference[v]*scale;small[v]={float(scaled[v].x),float(scaled[v].y),float(scaled[v].z)}; }
      deform3d::SimplexRestMaterial<double> large;deform3d::SimplexRestMaterial<float> fp32;
      CHECK(deform3d::prepare_simplex_material(dim,scaled,12000.,.3,.002,large)==deform3d::Status::Ok);
      CHECK(deform3d::prepare_simplex_material(dim,small,12000.f,.3f,.002f,fp32)==deform3d::Status::Ok);
      for(int a=0;a<ne*(ne+1)/2;++a) {
        double expected_k=material.packed[a]*std::pow(scale,dim-4);
        CHECK(std::abs(large.packed[a]-expected_k)<1e-11*std::max(1.,std::abs(expected_k)));
        CHECK(std::abs(double(fp32.packed[a])-large.packed[a])<3e-6*std::max(1.,std::abs(large.packed[a])));
      }
    }
    const auto saved=material;
    reference[1]=reference[0];
    CHECK(deform3d::prepare_simplex_material(dim,reference,12000.,.3,.002,material)==deform3d::Status::DegenerateGeometry);
    CHECK(material.packed==saved.packed && material.edge_lengths==saved.edge_lengths);
  }
  std::cout<<"simplex energy, force, tangent, invariants and disabled response passed\n";
}
