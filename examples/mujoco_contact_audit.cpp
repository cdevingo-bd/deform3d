// Compare MuJoCo ellipsoid/triangle face contacts to a simulator-independent
// analytic reference at one fixed settling checkpoint. No physical parameters
// or integration timestep are changed; only collision termination is swept.
#include <deform3d/contact_quality.hpp>
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cstdio>
#include <memory>

int main(int argc,char** argv) {
  if(argc!=2) {std::fprintf(stderr,"usage: contact_audit scene.xml\n");return 2;}
  char error[1024]{};
  std::unique_ptr<mjModel,decltype(&mj_deleteModel)> m(mj_loadXML(argv[1],nullptr,error,sizeof(error)),mj_deleteModel);
  if(!m) {std::fprintf(stderr,"%s\n",error);return 2;}
  std::unique_ptr<mjData,decltype(&mj_deleteData)> d(mj_makeData(m.get()),mj_deleteData);
  mj_forward(m.get(),d.get());
  for(int step=0;step<int(std::round(.12/m->opt.timestep));++step) mj_step(m.get(),d.get());
  std::printf("{\"checkpoint_s\":%.17g,\"scope\":\"same CPU state; analytic face-interior contact audit\",\"results\":[",d->time);
  bool valid=true;int sample=0;
  for(double tolerance : {1e-6,1e-9,1e-10}) {
    m->opt.ccd_tolerance=tolerance;
    mj_forward(m.get(),d.get());
    int accepted=0, skipped=0;double normal_error=0,distance_error=0;
    for(int i=0;i<d->ncon;++i) {
      const mjContact& c=d->contact[i];
      const int side=c.geom[0]>=0 ? 0 : 1;
      const int g=c.geom[side],f=c.flex[1-side],e=c.elem[1-side];
      if(g<0 || f<0 || e<0 || m->geom_type[g]!=mjGEOM_ELLIPSOID || m->flex_dim[f]!=2) continue;
      deform3d::EllipsoidShape<double> shape;
      shape.center={d->geom_xpos[3*g],d->geom_xpos[3*g+1],d->geom_xpos[3*g+2]};
      for(int axis=0;axis<3;++axis) {
        const double length=m->geom_size[3*g+axis];
        shape.axes[axis]={length*d->geom_xmat[9*g+axis],length*d->geom_xmat[9*g+3+axis],length*d->geom_xmat[9*g+6+axis]};
      }
      std::array<deform3d::Vec3d,3> triangle;
      for(int v=0;v<3;++v) {
        const int id=m->flex_vertadr[f]+m->flex_elem[m->flex_elemdataadr[f]+3*e+v];
        triangle[v]={d->flexvert_xpos[3*id],d->flexvert_xpos[3*id+1],d->flexvert_xpos[3*id+2]};
      }
      deform3d::FaceContact<double> exact;
      if(deform3d::ellipsoid_triangle_face(shape,triangle,m->flex_radius[f],exact)
          !=deform3d::FaceContactStatus::Interior) {++skipped;continue;}
      ++accepted;
      const deform3d::Vec3d actual=deform3d::Vec3d{c.frame[0],c.frame[1],c.frame[2]}*(side==0?1.:-1.);
      normal_error=std::max(normal_error,deform3d::norm(actual-exact.normal));
      distance_error=std::max(distance_error,std::abs(c.dist-exact.distance));
    }
    if(sample++) std::printf(",");
    std::printf("{\"ccd_tolerance\":%.17g,\"interior_contacts\":%d,\"skipped_contacts\":%d,\"max_unit_normal_delta\":%.17g,\"max_distance_delta_m\":%.17g}",
                tolerance,accepted,skipped,normal_error,distance_error);
    if(tolerance==1e-10 && (accepted==0 || normal_error>1e-5 || distance_error>1e-8)) valid=false;
  }
  std::printf("],\"tight_contact_geometry_passed\":%s}\n",valid?"true":"false");
  return valid?0:1;
}
