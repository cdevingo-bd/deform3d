#include <deform3d/mujoco_adapter.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 2) return 1;
  std::array<char,1024> error{};
  std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model(
      mj_loadXML(argv[1],nullptr,error.data(),int(error.size())),mj_deleteModel);
  if (!model) { std::fprintf(stderr,"%s\n",error.data()); return 2; }
  std::unique_ptr<mjData,decltype(&mj_deleteData)> data(mj_makeData(model.get()),mj_deleteData);
  std::unique_ptr<mjData,decltype(&mj_deleteData)> reference(mj_makeData(model.get()),mj_deleteData);
  if (!data || !reference) return 3;
  mj_forward(model.get(),data.get());
  using V=deform3d::Vec3<double>;
  deform3d::MujocoFlexAdapter<double> adapter;
  if (adapter.bind(model.get(),data.get(),mj_name2id(model.get(),mjOBJ_FLEX,"cavity")) !=
      deform3d::MujocoAdapterStatus::Ok) return 4;
  deform3d::TetraEmbedding<double> embedding;
  if (adapter.prepare_embedding(embedding,adapter.rest_vertices().data(),adapter.vertex_count()) !=
      deform3d::EmbedStatus::Ok) return 5;
  // Deliberately move every generalized velocity on the very first step.
  for (int i=0;i<model->nv;++i) data->qvel[i]=0.25;
  mj_step(model.get(),data.get());
  std::vector<double> before(3*adapter.vertex_count()), after(before.size()), expected(before.size());
  if (adapter.evaluate_embedding(model.get(),data.get(),embedding,before.data(),adapter.vertex_count()) !=
      deform3d::EmbedStatus::Ok) return 6;
  mj_copyData(reference.get(),model.get(),data.get());
  mj_forward(model.get(),reference.get());
  if (deform3d::refresh_mujoco_flex_positions(model.get(),data.get()) !=
      deform3d::MujocoAdapterStatus::Ok) return 7;
  if (adapter.evaluate_embedding(model.get(),data.get(),embedding,after.data(),adapter.vertex_count()) !=
      deform3d::EmbedStatus::Ok ||
      adapter.evaluate_embedding(model.get(),reference.get(),embedding,expected.data(),adapter.vertex_count()) !=
      deform3d::EmbedStatus::Ok) return 8;
  double movement=0, max_error=0;
  for (std::size_t i=0;i<after.size();++i) {
    if (!std::isfinite(before[i]) || !std::isfinite(after[i]) || !std::isfinite(expected[i])) return 11;
    movement=std::max(movement,std::abs(after[i]-before[i]));
    max_error=std::max(max_error,std::abs(after[i]-expected[i]));
  }
  std::printf("first-step geometry refresh: movement=%g error_vs_forward=%g\n",movement,max_error);
  if (!(movement>1e-5) || max_error>1e-12) return 9;
  if (deform3d::refresh_mujoco_flex_positions(nullptr,data.get()) !=
      deform3d::MujocoAdapterStatus::InvalidInput) return 10;
  return 0;
}
