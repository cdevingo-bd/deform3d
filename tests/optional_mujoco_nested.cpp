// Closed-cavity integration fixture. This tests contact/embedding behavior,
// not material calibration or continuous collision guarantees.
#include <deform3d/mujoco_adapter.hpp>
#include <deform3d/quality.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

namespace {
using V = deform3d::Vec3<double>;
using Model = std::unique_ptr<mjModel, decltype(&mj_deleteModel)>;
using Data = std::unique_ptr<mjData, decltype(&mj_deleteData)>;

int fail(const char* message) {
  std::fprintf(stderr, "nested fixture: %s\n", message);
  return 1;
}

struct FlexCheck {
  deform3d::MujocoFlexAdapter<double> adapter;
  deform3d::TetraEmbedding<double> embedding;
  deform3d::TetraQuality<double> quality;
  std::vector<V> current, rest_samples;
  std::vector<double> dense;
  double max_motion = 0;
  long double min_volume_ratio = std::numeric_limits<long double>::infinity();
  long double max_volume_ratio = 0;
  long double max_edge_stretch = 0;

  bool prepare(const mjModel* model, const mjData* data, int flex_id) {
    if (adapter.bind(model, data, flex_id) != deform3d::MujocoAdapterStatus::Ok)
      return false;
    const auto& rest = adapter.rest_vertices();
    const auto& tet = adapter.tetrahedra();
    if (quality.prepare(rest.data(), rest.size(), tet.data(), adapter.tetrahedron_count()) !=
        deform3d::Status::Ok) return false;
    // Interior samples exercise independent embeddings for both flexes.
    // They remain attached to the chosen physical solid rather than nesting
    // an inner object's transform under the outer object's deformation.
    for (std::size_t t = 0; t < adapter.tetrahedron_count(); ++t) {
      V center{};
      for (int j = 0; j < 4; ++j) center += rest[tet[4*t+j]] * 0.25;
      rest_samples.push_back(center);
      for (int j = 0; j < 4; ++j)
        rest_samples.push_back(center * 0.4 + rest[tet[4*t+j]] * 0.6);
    }
    current.resize(rest.size());
    dense.resize(3 * rest_samples.size());
    if (adapter.prepare_embedding(embedding, rest_samples.data(), rest_samples.size()) !=
        deform3d::EmbedStatus::Ok) return false;
    if (!sample(model, data)) return false;
    for (std::size_t i = 0; i < rest_samples.size(); ++i)
      if (deform3d::norm(V{dense[3*i], dense[3*i+1], dense[3*i+2]} - rest_samples[i]) >
          1e-12) return false;
    return true;
  }

  bool sample(const mjModel* model, const mjData* data) {
    if (adapter.copy_current(model, data, current.data(), current.size()) !=
        deform3d::MujocoAdapterStatus::Ok) return false;
    deform3d::TetraQualityReport report;
    if (quality.evaluate(current.data(), current.size(), report) != deform3d::Status::Ok ||
        report.inverted || report.near_zero_volume) return false;
    min_volume_ratio = std::min(min_volume_ratio, report.min_volume_ratio);
    max_volume_ratio = std::max(max_volume_ratio, report.max_volume_ratio);
    max_edge_stretch = std::max(max_edge_stretch, report.max_edge_stretch);
    const auto& rest = adapter.rest_vertices();
    for (std::size_t i = 0; i < current.size(); ++i)
      max_motion = std::max(max_motion, deform3d::norm(current[i] - rest[i]));
    if (adapter.evaluate_embedding(model, data, embedding, dense.data(), rest_samples.size()) !=
        deform3d::EmbedStatus::Ok) return false;
    return std::all_of(dense.begin(), dense.end(), [](double x) { return std::isfinite(x); });
  }
};

int run(const char* path) {
  std::array<char, 1024> error{};
  Model model(mj_loadXML(path, nullptr, error.data(), int(error.size())), mj_deleteModel);
  if (!model) return fail(error.data());
  Data data(mj_makeData(model.get()), mj_deleteData);
  if (!data) return fail("data allocation failed");
  mj_forward(model.get(), data.get());
  const int shell_id = mj_name2id(model.get(), mjOBJ_FLEX, "shell");
  const int inner_id = mj_name2id(model.get(), mjOBJ_FLEX, "inner");
  const int probe_id = mj_name2id(model.get(), mjOBJ_GEOM, "probe_geom");
  const int body_id = mj_name2id(model.get(), mjOBJ_BODY, "probe");
  if (shell_id < 0 || inner_id < 0 || probe_id < 0 || body_id < 0)
    return fail("missing named objects");
  if (data->ncon) return fail("fixture must start without contacts");

  FlexCheck shell, inner;
  if (!shell.prepare(model.get(), data.get(), shell_id) ||
      !inner.prepare(model.get(), data.get(), inner_id)) return fail("flex setup failed");
  if (shell.adapter.vertex_count() != 64 || shell.adapter.tetrahedron_count() != 156 ||
      inner.adapter.tetrahedron_count() != 1) return fail("unexpected closed-shell topology");

  // Explicit initial clearance from all six walls of the empty reference
  // cavity. Contact radii are included in these bounds.
  for (V p : inner.adapter.rest_vertices())
    if (std::max({std::abs(p.x), std::abs(p.y), std::abs(p.z)}) +
        model->flex_radius[inner_id] + model->flex_radius[shell_id] >= 0.03)
      return fail("inner flex does not begin inside the cavity with clearance");
  const V probe_start{data->xpos[3*body_id], data->xpos[3*body_id+1],
                      data->xpos[3*body_id+2]};
  if (std::max({std::abs(probe_start.x), std::abs(probe_start.y), std::abs(probe_start.z)}) +
      model->geom_size[3*probe_id] + model->flex_radius[shell_id] >= 0.03)
    return fail("probe does not begin inside the cavity with clearance");

  // A no-contact control separates shell self-weight deformation from its
  // response to the internal objects. All other properties remain identical.
  Model control_model(mj_copyModel(nullptr, model.get()), mj_deleteModel);
  if (!control_model) return fail("control model allocation failed");
  control_model->opt.disableflags |= mjDSBL_CONTACT;
  Data control(mj_makeData(control_model.get()), mj_deleteData);
  if (!control) return fail("control data allocation failed");
  mj_forward(control_model.get(), control.get());

  std::size_t soft_contact_samples = 0, rigid_contact_samples = 0;
  std::size_t soft_contact_steps = 0, rigid_contact_steps = 0;
  double max_penetration = 0, max_normal_force = 0;
  constexpr int steps = 500;
  for (int step = 0; step < steps; ++step) {
    mj_step(model.get(), data.get());
    mj_step(control_model.get(), control.get());
    bool soft = false, rigid = false;
    // Contacts and forces belong to the dynamics evaluation of this step.
    // Inspect them before refreshing positions to the integrated state.
    for (int c = 0; c < data->ncon; ++c) {
      const mjContact& hit = data->contact[c];
      const bool ss = (hit.flex[0] == shell_id && hit.flex[1] == inner_id) ||
                      (hit.flex[1] == shell_id && hit.flex[0] == inner_id);
      const bool sr = (hit.flex[0] == shell_id && hit.geom[1] == probe_id) ||
                      (hit.flex[1] == shell_id && hit.geom[0] == probe_id);
      if (!ss && !sr) continue;
      soft |= ss; rigid |= sr;
      soft_contact_samples += ss; rigid_contact_samples += sr;
      max_penetration = std::max(max_penetration, -double(hit.dist));
      mjtNum force[6]{};
      mj_contactForce(model.get(), data.get(), c, force);
      for (double f : force) if (!std::isfinite(f)) return fail("non-finite contact force");
      max_normal_force = std::max(max_normal_force, double(force[0]));
    }
    soft_contact_steps += soft; rigid_contact_steps += rigid;
    if (deform3d::refresh_mujoco_flex_positions(model.get(), data.get()) !=
        deform3d::MujocoAdapterStatus::Ok) return fail("position refresh failed");
    if (!shell.sample(model.get(), data.get()) || !inner.sample(model.get(), data.get()))
      return fail("non-finite, inverted or collapsed flex/embedding");
    for (int q = 0; q < model->nq; ++q)
      if (!std::isfinite(data->qpos[q])) return fail("non-finite qpos");
    for (int q = 0; q < model->nv; ++q)
      if (!std::isfinite(data->qvel[q])) return fail("non-finite qvel");
    for (int w = 0; w < mjNWARNING; ++w)
      if (data->warning[w].number || control->warning[w].number)
        return fail("MuJoCo reported a warning");
  }
  if (deform3d::refresh_mujoco_flex_positions(control_model.get(), control.get()) !=
      deform3d::MujocoAdapterStatus::Ok) return fail("control refresh failed");
  const int shell_offset = model->flex_vertadr[shell_id];
  double contact_response = 0;
  for (std::size_t i = 0; i < shell.current.size(); ++i) {
    const mjtNum* p = control->flexvert_xpos + 3 * (shell_offset + i);
    contact_response = std::max(contact_response,
        deform3d::norm(shell.current[i] - V{p[0], p[1], p[2]}));
  }
  const V probe_end{data->xpos[3*body_id], data->xpos[3*body_id+1],
                    data->xpos[3*body_id+2]};
  std::printf("nested sim_seconds=%.3f shell_tets=%zu inner_tets=%zu "
              "soft_contact_samples=%zu rigid_contact_samples=%zu "
              "soft_contact_steps=%zu rigid_contact_steps=%zu "
              "max_penetration_m=%.6g max_normal_force_N=%.6g\n",
              double(data->time), shell.adapter.tetrahedron_count(),
              inner.adapter.tetrahedron_count(), soft_contact_samples,
              rigid_contact_samples, soft_contact_steps, rigid_contact_steps,
              max_penetration, max_normal_force);
  std::printf("shell_motion_m=%.6g inner_motion_m=%.6g probe_motion_m=%.6g "
              "shell_contact_response_m=%.6g "
              "shell_volume_ratio=[%.6Lg,%.6Lg] inner_volume_ratio=[%.6Lg,%.6Lg] "
              "shell_max_edge_stretch=%.6Lg inner_max_edge_stretch=%.6Lg\n",
              shell.max_motion, inner.max_motion, deform3d::norm(probe_end-probe_start),
              contact_response, shell.min_volume_ratio, shell.max_volume_ratio,
              inner.min_volume_ratio, inner.max_volume_ratio,
              shell.max_edge_stretch, inner.max_edge_stretch);
  // Broad behavioral gates, not physical accuracy claims. Many contact steps
  // guard against a single accidental touch passing this closed-cavity test.
  if (soft_contact_steps < 50 || rigid_contact_steps < 50 || max_normal_force <= 0 ||
      shell.max_motion < 1e-6 || inner.max_motion < 0.01 ||
      deform3d::norm(probe_end-probe_start) < 0.01 || contact_response < 1e-7 ||
      std::abs(data->time - steps*model->opt.timestep) > 1e-10)
    return fail("expected persistent internal contact and coupled motion were absent");
  return 0;
}
} // namespace

int main(int argc, char** argv) {
  return run(argc > 1 ? argv[1] : "examples/mujoco_nested.xml");
}
