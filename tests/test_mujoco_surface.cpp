#include <deform3d/mujoco_surface_adapter.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#define CHECK(x) do { if(!(x)) {std::fprintf(stderr,"CHECK %s:%d: %s\n",__FILE__,__LINE__,#x);std::abort();} } while(false)

int main() {
  using namespace deform3d;
  // The preceding volumetric flex makes the surface's model-global vertex
  // offset nonzero and verifies that its triangle indices remain local.
  constexpr const char* xml=R"(<mujoco model="surface_adapter_test">
    <option timestep="0.001" gravity="0 0 0"/>
    <worldbody>
      <flexcomp name="solid" type="direct" dim="3" mass="1"
                point="10 0 0 11 0 0 10 1 0 10 0 1" element="0 1 2 3">
        <edge equality="true"/>
      </flexcomp>
      <flexcomp name="cloth" type="direct" dim="2" mass="1" radius="0.001"
                point="0 0 0 1 0 0 0 1 0 1 1 0" element="0 1 2 1 3 2">
        <edge equality="true"/>
      </flexcomp>
    </worldbody>
  </mujoco>)";
  std::array<char,1024> error{};
  std::unique_ptr<mjSpec,decltype(&mj_deleteSpec)> spec(
      mj_parseXMLString(xml,nullptr,error.data(),int(error.size())),mj_deleteSpec);
  if(!spec) {std::fprintf(stderr,"%s\n",error.data());return 1;}
  std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model(mj_compile(spec.get(),nullptr),mj_deleteModel);
  CHECK(model);
  std::unique_ptr<mjData,decltype(&mj_deleteData)> data(mj_makeData(model.get()),mj_deleteData);
  CHECK(data);
  mj_forward(model.get(),data.get());
  MujocoSurfaceAdapter<double> adapter;
  CHECK(adapter.bind(model.get(),data.get(),1)==MujocoSurfaceAdapterStatus::Ok);
  CHECK(adapter.vertex_offset()==4&&adapter.vertex_count()==4&&adapter.triangle_count()==2);
  const auto original_hash=adapter.topology_hash(),original_rest_hash=adapter.rest_hash();
  CHECK(adapter.bind(model.get(),data.get(),0)==MujocoSurfaceAdapterStatus::NotTriangular);
  CHECK(adapter.bind(nullptr,data.get(),1)==MujocoSurfaceAdapterStatus::InvalidInput);
  CHECK(adapter.flex_id()==1&&adapter.topology_hash()==original_hash&&adapter.rest_hash()==original_rest_hash);
  auto points=adapter.rest_vertices();
  for(auto& p:points) p.z+=.02;
  SurfaceEmbedOptions<double> options; options.max_distance=.03;
  SurfaceEmbedding<double> embedding;
  CHECK(adapter.prepare_embedding(embedding,points.data(),points.size(),options)==SurfaceEmbedStatus::Ok);
  for(int i=0;i<model->nv;++i) data->qvel[i]=.25;
  mj_step(model.get(),data.get());
  mj_kinematics(model.get(),data.get()); mj_flex(model.get(),data.get());
  std::vector<double> output(points.size()*4,37.);
  CHECK(adapter.evaluate_embedding(model.get(),data.get(),embedding,output.data(),points.size(),4)==SurfaceEmbedStatus::Ok);
  double worst=0;
  for(std::size_t i=0;i<points.size();++i) {
    const auto expected=points[i]+Vec3<double>{.00025,.00025,.00025};
    CHECK(std::isfinite(output[4*i])&&std::isfinite(output[4*i+1])&&std::isfinite(output[4*i+2]));
    worst=std::max({worst,std::abs(output[4*i]-expected.x),std::abs(output[4*i+1]-expected.y),std::abs(output[4*i+2]-expected.z)});
    CHECK(output[4*i+3]==37.);
  }
  CHECK(worst<1e-10);
  const auto original=output;
  for(std::size_t i=0;i<adapter.vertex_count();++i) for(std::size_t j=0;j<3;++j)
    data->flexvert_xpos[3*(adapter.vertex_offset()+i)+j]=0;
  CHECK(adapter.evaluate_embedding(model.get(),data.get(),embedding,output.data(),points.size(),4)==SurfaceEmbedStatus::DegenerateTriangle);
  CHECK(output==original);
  std::printf("MuJoCo dim2 adapter: nonzero global offset, step/refresh, signed thickness and collapse passed; max_error=%g\n",worst);
}
