// Opt-in integration test; requires MuJoCo headers and a matching linked library.
// Run from the repository root: ./optional_mujoco examples/mujoco_cavity.xml

#include <deform3d/mujoco_adapter.hpp>
#include <deform3d/quality.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <vector>

namespace {
using V = deform3d::Vec3<float>;

float distance(V a, V b) {
  return deform3d::norm(a - b);
}

int run(const char* scene) {
  std::array<char, 1024> error{};
  std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(
      mj_loadXML(scene, nullptr, error.data(), static_cast<int>(error.size())),
      mj_deleteModel);
  if (!model) {
    std::fprintf(stderr, "MuJoCo load failed: %s\n", error.data());
    return 1;
  }
  std::unique_ptr<mjData, decltype(&mj_deleteData)> data(
      mj_makeData(model.get()), mj_deleteData);
  if (!data) return 2;
  mj_forward(model.get(), data.get());

  const int flex_id = mj_name2id(model.get(), mjOBJ_FLEX, "cavity");
  const int inner_id = mj_name2id(model.get(), mjOBJ_FLEX, "inner");
  const int geom_id = mj_name2id(model.get(), mjOBJ_GEOM, "probe_geom");
  const int body_id = mj_name2id(model.get(), mjOBJ_BODY, "probe");
  if (flex_id < 0 || inner_id < 0 || geom_id < 0 || body_id < 0) return 3;

  deform3d::MujocoFlexAdapter<float> bridge;
  const auto bind = bridge.bind(model.get(), data.get(), flex_id);
  if (bind != deform3d::MujocoAdapterStatus::Ok) {
    std::fprintf(stderr, "bridge bind failed: %s\n",
                 deform3d::mujoco_adapter_status_string(bind));
    return 4;
  }
  if (bridge.vertex_count() != 32 || bridge.tetrahedron_count() != 48) {
    std::fprintf(stderr, "unexpected cavity topology\n");
    return 5;
  }
  deform3d::TetraQuality<float> quality;
  if (quality.prepare(bridge.rest_vertices().data(), bridge.vertex_count(),
                      bridge.tetrahedra().data(), bridge.tetrahedron_count()) !=
      deform3d::Status::Ok) return 15;
  deform3d::MujocoFlexAdapter<float> inner_bridge;
  if (inner_bridge.bind(model.get(), data.get(), inner_id) !=
          deform3d::MujocoAdapterStatus::Ok ||
      inner_bridge.vertex_count() != 4 || inner_bridge.tetrahedron_count() != 1)
    return 13;

  // Extract the outer and cavity walls by cancelling shared tetra faces.
  struct Face {
    std::array<std::uint32_t, 3> oriented{}, key{};
  };
  std::vector<Face> all_faces;
  all_faces.reserve(4 * bridge.tetrahedron_count());
  std::vector<V> rest_render;
  std::vector<std::uint32_t> render_triangles;
  const auto& rest = bridge.rest_vertices();
  const auto& tet = bridge.tetrahedra();
  for (std::size_t t = 0; t < bridge.tetrahedron_count(); ++t) {
    const std::array<std::uint32_t, 4> id{{tet[4*t], tet[4*t+1], tet[4*t+2], tet[4*t+3]}};
    for (int opposite = 0; opposite < 4; ++opposite) {
      Face face;
      int corner = 0;
      for (int j = 0; j < 4; ++j) if (j != opposite) face.oriented[corner++] = id[j];
      const V a = rest[face.oriented[0]], b = rest[face.oriented[1]];
      const V c = rest[face.oriented[2]], inside = rest[id[opposite]];
      if (deform3d::dot(deform3d::cross(b-a, c-a), inside-a) > 0)
        std::swap(face.oriented[1], face.oriented[2]);
      face.key = face.oriented;
      std::sort(face.key.begin(), face.key.end());
      all_faces.push_back(face);
    }
  }
  std::sort(all_faces.begin(), all_faces.end(),
            [](const Face& a, const Face& b) { return a.key < b.key; });
  std::vector<std::array<std::uint32_t, 3>> boundary;
  for (std::size_t i = 0; i < all_faces.size();) {
    std::size_t j = i + 1;
    while (j < all_faces.size() && all_faces[j].key == all_faces[i].key) ++j;
    if (j == i + 1) boundary.push_back(all_faces[i].oriented);
    else if (j != i + 2) return 17; // nonmanifold face
    i = j;
  }
  if (boundary.size() != 64) return 18;

  // Each original boundary triangle becomes 16^2 render triangles. Vertices
  // are shared within each face and duplicated at face borders (hard seams).
  constexpr std::uint32_t subdivision = 16;
  rest_render.reserve(boundary.size() * (subdivision + 1) * (subdivision + 2) / 2);
  render_triangles.reserve(boundary.size() * subdivision * subdivision * 3);
  for (const auto& f : boundary) {
    const V a = rest[f[0]], b = rest[f[1]], c = rest[f[2]];
    const std::uint32_t base = static_cast<std::uint32_t>(rest_render.size());
    const auto index = [base](std::uint32_t i, std::uint32_t j) {
      return base + i*(subdivision+1) - i*(i-1)/2 + j;
    };
    for (std::uint32_t i = 0; i <= subdivision; ++i)
      for (std::uint32_t j = 0; j <= subdivision-i; ++j)
        rest_render.push_back(a + (b-a)*(float(i)/subdivision) +
                              (c-a)*(float(j)/subdivision));
    for (std::uint32_t i = 0; i < subdivision; ++i) {
      for (std::uint32_t j = 0; j < subdivision-i; ++j) {
        render_triangles.insert(render_triangles.end(),
                                {index(i,j), index(i+1,j), index(i,j+1)});
        if (i+j+2 <= subdivision)
          render_triangles.insert(render_triangles.end(),
                                  {index(i+1,j), index(i+1,j+1), index(i,j+1)});
      }
    }
  }
  if (rest_render.size() != 9792 || render_triangles.size()/3 != 16384) return 19;

  deform3d::NormalTopology normal_topology;
  if (normal_topology.prepare(rest_render.size(), render_triangles.data(),
                              render_triangles.size()) != deform3d::Status::Ok) return 20;
  deform3d::Workspace<float> normal_workspace;
  normal_workspace.reserve(rest_render.size(), 0, render_triangles.size()/3);
  std::vector<V> render_normals(rest_render.size());
  const auto update_normals = [&](std::vector<float>& xyz) {
    deform3d::EvaluateParams<float> params;
    params.vertices = rest_render.size();
    params.input_positions = {xyz.data(), 3*sizeof(float)};
    params.output_positions = {xyz.data(), 3*sizeof(float)};
    params.output_normals = deform3d::view(render_normals.data());
    params.normal_mode = deform3d::NormalMode::Recompute;
    params.topology = &normal_topology;
    params.workspace = &normal_workspace;
    if (deform3d::evaluate(params) != deform3d::Status::Ok) return false;
    for (V n : render_normals)
      if (!deform3d::finite(n) || std::abs(deform3d::norm(n)-1.0f) > 1e-3f)
        return false;
    return true;
  };

  deform3d::TetraEmbedding<float> embedding;
  const auto prepared = bridge.prepare_embedding(embedding, rest_render.data(), rest_render.size());
  if (prepared != deform3d::EmbedStatus::Ok) {
    std::fprintf(stderr, "embedding failed: %s\n", deform3d::embed_status_string(prepared));
    return 6;
  }
  std::vector<float> output(3 * rest_render.size());
  if (bridge.evaluate_embedding(model.get(), data.get(), embedding,
                                output.data(), rest_render.size()) != deform3d::EmbedStatus::Ok) return 7;
  for (std::size_t i = 0; i < rest_render.size(); ++i) {
    if (distance({output[3*i], output[3*i+1], output[3*i+2]}, rest_render[i]) > 2e-5f)
      return 8;
  }
  if (!update_normals(output)) return 21;

  std::vector<V> current(bridge.vertex_count());
  std::vector<V> inner_current(inner_bridge.vertex_count());
  const V initial_probe{float(data->xpos[3*body_id]), float(data->xpos[3*body_id+1]),
                        float(data->xpos[3*body_id+2])};
  int flex_geom_contacts = 0;
  int flex_flex_contacts = 0;
  float max_flex_motion = 0;
  float max_inner_motion = 0;
  for (int step = 0; step < 120; ++step) {
    mj_step(model.get(), data.get());
    deform3d::refresh_mujoco_flex_positions(model.get(), data.get());
    for (int c = 0; c < data->ncon; ++c) {
      const mjContact& hit = data->contact[c];
      if ((hit.flex[0] == flex_id && hit.geom[1] == geom_id) ||
          (hit.flex[1] == flex_id && hit.geom[0] == geom_id)) ++flex_geom_contacts;
      if ((hit.flex[0] == flex_id && hit.flex[1] == inner_id) ||
          (hit.flex[1] == flex_id && hit.flex[0] == inner_id)) ++flex_flex_contacts;
    }
    if (bridge.copy_current(model.get(), data.get(), current.data(), current.size()) !=
        deform3d::MujocoAdapterStatus::Ok) return 9;
    deform3d::TetraQualityReport report;
    if (quality.evaluate(current.data(), current.size(), report) !=
        deform3d::Status::Ok || report.inverted || report.near_zero_volume)
      return 16;
    for (std::size_t i = 0; i < current.size(); ++i)
      max_flex_motion = std::max(max_flex_motion, distance(current[i], rest[i]));
    if (inner_bridge.copy_current(model.get(), data.get(),
                                  inner_current.data(), inner_current.size()) !=
        deform3d::MujocoAdapterStatus::Ok) return 14;
    for (std::size_t i = 0; i < inner_current.size(); ++i)
      max_inner_motion = std::max(max_inner_motion,
                                  distance(inner_current[i], inner_bridge.rest_vertices()[i]));
  }
  if (bridge.evaluate_embedding(model.get(), data.get(), embedding,
                                output.data(), rest_render.size()) != deform3d::EmbedStatus::Ok) return 10;
  for (float coordinate : output) if (!std::isfinite(coordinate)) return 11;
  if (!update_normals(output)) return 22;

  const V final_probe{float(data->xpos[3*body_id]), float(data->xpos[3*body_id+1]),
                      float(data->xpos[3*body_id+2])};
  const float rigid_motion = distance(initial_probe, final_probe);
  std::printf("flex vertices=%zu tetrahedra=%zu render vertices=%zu render triangles=%zu "
              "flex-geom contacts=%d flex-flex contacts=%d "
              "outer motion=%.6g inner motion=%.6g rigid motion=%.6g\n",
              bridge.vertex_count(), bridge.tetrahedron_count(), rest_render.size(),
              render_triangles.size()/3,
              flex_geom_contacts, flex_flex_contacts, max_flex_motion,
              max_inner_motion, rigid_motion);
  // These checks establish solver contact and two-way response, not physical
  // calibration to a real material. Tolerances may need scene-specific tuning.
  if (flex_geom_contacts == 0 || flex_flex_contacts == 0 ||
      max_flex_motion < 1e-7f || max_inner_motion < 1e-7f ||
      rigid_motion < 1e-7f) return 12;
  return 0;
}
} // namespace

int main(int argc, char** argv) {
  return run(argc > 1 ? argv[1] : "examples/mujoco_cavity.xml");
}
