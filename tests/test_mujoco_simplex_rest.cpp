#include <deform3d/simplex_rest.hpp>
#include <mujoco/mujoco.h>
#include <cstdlib>
#include <iostream>

#define CHECK(x) do { if(!(x)) { std::cerr<<__LINE__<<": "<<#x<<"\n"; std::abort(); } } while(0)
int main(int argc,char** argv) {
  CHECK(argc==2);char error[1024]{};
  mjModel* m=mj_loadXML(argv[1],nullptr,error,sizeof(error));
  if(!m){std::cerr<<error<<'\n';return 1;}
  mjData* d=mj_makeData(m);mj_forward(m,d);
  double maximum=0;int count=0;
  for(int f=0;f<m->nflex;++f) {
    int dim=m->flex_dim[f],ne=dim==2?3:6;
    for(int e=0;e<m->flex_elemnum[f];++e) {
      deform3d::Vec3<double> rest[4];
      for(int v=0;v<dim+1;++v) {
        int id=m->flex_vertadr[f]+m->flex_elem[m->flex_elemdataadr[f]+(dim+1)*e+v];
        rest[v]={d->flexvert_xpos[3*id],d->flexvert_xpos[3*id+1],d->flexvert_xpos[3*id+2]};
      }
      deform3d::SimplexRestMaterial<double> material;
      CHECK(deform3d::prepare_simplex_material(dim,rest,dim==2?50e6:12000.,dim==2?.35:.45,25.4e-6,material)==deform3d::Status::Ok);
      const auto* native=m->flex_stiffness+m->flex_stiffnessadr[f]+e*(dim==2?21:24);
      double delta=0,scale=0;
      for(int k=0;k<ne*(ne+1)/2;++k) { delta+=std::pow(material.packed[k]-native[k],2);scale+=native[k]*native[k]; }
      double relative=std::sqrt(delta/scale);maximum=std::max(maximum,relative);
      CHECK(relative<1e-11);++count;
    }
  }
  std::cout<<"{\"elements\":"<<count<<",\"max_relative_metric_error\":"<<maximum<<"}\n";
  mj_deleteData(d);mj_deleteModel(m);
}
