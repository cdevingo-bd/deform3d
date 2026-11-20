// Controlled loading fixture for future force/displacement calibration.
// These elastic parameters are illustrative, not a fit to foam or silicone.
// Usage: optional_mujoco_gripper [scene.xml] [force_displacement.csv]
#include <deform3d/mujoco_adapter.hpp>
#include <deform3d/quality.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <vector>

namespace {
using V = deform3d::Vec3<double>;
int fail(const char* message) {
  std::fprintf(stderr, "gripper fixture: %s\n", message);
  return 1;
}

struct FlexMetrics {
  deform3d::MujocoFlexAdapter<double> adapter;
  deform3d::TetraQuality<double> quality;
  deform3d::TetraQualityReport report;
  std::vector<V> current;
  long double minimum_volume_ratio = 1;
  long double maximum_edge_stretch = 1;
  double maximum_motion = 0;

  bool prepare(const mjModel* model, const mjData* data, int id) {
    if (adapter.bind(model, data, id) != deform3d::MujocoAdapterStatus::Ok) return false;
    current.resize(adapter.vertex_count());
    if (quality.prepare(adapter.rest_vertices().data(), current.size(),
                        adapter.tetrahedra().data(), adapter.tetrahedron_count()) !=
        deform3d::Status::Ok) return false;
    return sample(model, data);
  }
  bool sample(const mjModel* model, const mjData* data) {
    if (adapter.copy_current(model, data, current.data(), current.size()) !=
        deform3d::MujocoAdapterStatus::Ok ||
        quality.evaluate(current.data(), current.size(), report) != deform3d::Status::Ok ||
        report.inverted || report.near_zero_volume) return false;
    minimum_volume_ratio = std::min(minimum_volume_ratio, report.min_volume_ratio);
    maximum_edge_stretch = std::max(maximum_edge_stretch, report.max_edge_stretch);
    for (std::size_t i = 0; i < current.size(); ++i)
      maximum_motion = std::max(maximum_motion,
          deform3d::norm(current[i] - adapter.rest_vertices()[i]));
    return true;
  }
};

double smooth(double x) { return x*x*(3-2*x); }
double closure(double t) {
  if (t < .25) return 0;
  if (t < .65) return smooth((t-.25)/.4);
  if (t < .75) return 1;
  if (t < 1.15) return 1-smooth((t-.75)/.4);
  return 0;
}
const char* phase(double t) {
  return t < .25 ? "settle" : t < .65 ? "close" : t < .75 ? "hold" :
         t < 1.15 ? "release" : "recover";
}

int run(const char* path, const char* csv_path) {
  std::array<char, 1024> error{};
  std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(
      mj_loadXML(path, nullptr, error.data(), int(error.size())), mj_deleteModel);
  if (!model) return fail(error.data());
  std::unique_ptr<mjData, decltype(&mj_deleteData)> data(mj_makeData(model.get()), mj_deleteData);
  if (!data) return fail("data allocation failed");
  mj_forward(model.get(), data.get());
  const int shell_id = mj_name2id(model.get(), mjOBJ_FLEX, "shell");
  const int inner_id = mj_name2id(model.get(), mjOBJ_FLEX, "inner");
  const int probe_id = mj_name2id(model.get(), mjOBJ_GEOM, "probe_geom");
  const int pad_geom[2] = {mj_name2id(model.get(), mjOBJ_GEOM, "left_pad_geom"),
                           mj_name2id(model.get(), mjOBJ_GEOM, "right_pad_geom")};
  const int pad_body[2] = {mj_name2id(model.get(), mjOBJ_BODY, "left_pad"),
                           mj_name2id(model.get(), mjOBJ_BODY, "right_pad")};
  if (shell_id < 0 || inner_id < 0 || probe_id < 0 || pad_geom[0] < 0 ||
      pad_geom[1] < 0 || pad_body[0] < 0 || pad_body[1] < 0)
    return fail("missing named objects");
  const int mocap[2] = {model->body_mocapid[pad_body[0]], model->body_mocapid[pad_body[1]]};
  if (mocap[0] < 0 || mocap[1] < 0 || data->ncon != 0)
    return fail("pads must be kinematic and all objects initially separated");

  FlexMetrics shell, inner;
  if (!shell.prepare(model.get(), data.get(), shell_id) ||
      !inner.prepare(model.get(), data.get(), inner_id)) return fail("flex setup failed");
  for (V p : inner.current)
    if (std::max({std::abs(p.x), std::abs(p.y), std::abs(p.z)}) +
        model->flex_radius[inner_id] + model->flex_radius[shell_id] >= .03)
      return fail("inner flex must initially lie inside the cavity with clearance");
  const mjtNum* sphere = data->geom_xpos + 3*probe_id;
  if (std::max({std::abs(sphere[0]), std::abs(sphere[1]), std::abs(sphere[2])}) +
      model->geom_size[3*probe_id] + model->flex_radius[shell_id] >= .03)
    return fail("sphere must initially lie inside the cavity with clearance");

  // Measure the sides at the pad's free middle region. The mounted base and
  // unloaded top can bulge; a whole-body AABB would not measure compression.
  std::array<std::vector<std::size_t>, 2> side;
  const auto& rest = shell.adapter.rest_vertices();
  for (std::size_t i = 0; i < rest.size(); ++i)
    if (std::abs(std::abs(rest[i].x)-.06) < 1e-10 &&
        std::abs(rest[i].y) < .031 && std::abs(rest[i].z) < .031)
      side[rest[i].x > 0 ? 1 : 0].push_back(i);
  if (side[0].size() != 4 || side[1].size() != 4)
    return fail("missing middle-region measurement vertices");
  const auto width = [&] {
    double mean[2]{};
    for (int s = 0; s < 2; ++s)
      for (auto i : side[s]) mean[s] += shell.current[i].x / double(side[s].size());
    return mean[1]-mean[0];
  };
  double settled_width = width(), minimum_width = settled_width;
  std::ofstream csv;
  if (csv_path) {
    csv.open(csv_path);
    if (!csv) return fail("could not open requested CSV output");
    csv << "time_s,phase,pad_gap_m,command_compression_m,shell_width_m,"
           "measured_compression_m,left_force_N,right_force_N,shell_inner_contacts,"
           "shell_probe_contacts,pad_shell_contacts,max_penetration_m,"
           "pad_max_penetration_m,shell_min_volume_ratio,inner_min_volume_ratio\n";
    csv << std::setprecision(12);
  }
  std::size_t soft_contact_steps = 0, rigid_contact_steps = 0, pad_contact_steps[2]{};
  double peak_force[2]{}, max_penetration = 0, max_pad_penetration = 0;
  double final_force[2]{};
  const int steps = int(std::ceil(1.3/model->opt.timestep));
  for (int step = 0; step < steps; ++step) {
    const double t = data->time;
    const double u = closure(t);
    const double center = .075-.011*u;
    data->mocap_pos[3*mocap[0]] = -center;
    data->mocap_pos[3*mocap[1]] = center;
    mj_step(model.get(), data.get());

    // Forces/contacts are evaluated before integration. Sample their matching
    // derived flex positions and log timestamp t, then refresh to the new state.
    if (!shell.sample(model.get(), data.get()) || !inner.sample(model.get(), data.get()))
      return fail("invalid flex state during force sampling");
    const double measured_width = width();
    if (t < .25) settled_width = measured_width;
    minimum_width = std::min(minimum_width, measured_width);
    int soft_contacts = 0, rigid_contacts = 0, pad_contacts[2]{};
    double signed_pad_force[2]{}, penetration = 0, pad_penetration = 0;
    for (int c = 0; c < data->ncon; ++c) {
      const mjContact& hit = data->contact[c];
      if (!std::isfinite(hit.dist)) return fail("non-finite contact distance");
      const bool soft = (hit.flex[0] == shell_id && hit.flex[1] == inner_id) ||
                        (hit.flex[1] == shell_id && hit.flex[0] == inner_id);
      const bool rigid = (hit.flex[0] == shell_id && hit.geom[1] == probe_id) ||
                         (hit.flex[1] == shell_id && hit.geom[0] == probe_id);
      soft_contacts += soft; rigid_contacts += rigid;
      penetration = std::max(penetration, -double(hit.dist));
      for (int pad = 0; pad < 2; ++pad) {
        if (!((hit.flex[0] == shell_id && hit.geom[1] == pad_geom[pad]) ||
              (hit.flex[1] == shell_id && hit.geom[0] == pad_geom[pad]))) continue;
        ++pad_contacts[pad];
        mjtNum force[6]{};
        mj_contactForce(model.get(), data.get(), c, force);
        for (double f : force) if (!std::isfinite(f)) return fail("non-finite pad force");
        // Contact frame axes are rows. Transform the linear force to world X;
        // the force on the first member is opposite to the second member.
        const double world_x = hit.frame[0]*force[0] + hit.frame[3]*force[1] +
                               hit.frame[6]*force[2];
        signed_pad_force[pad] += (hit.geom[0] == pad_geom[pad] ? -1 : 1) * world_x;
        pad_penetration = std::max(pad_penetration, -double(hit.dist));
      }
    }
    if (t >= .25) {
      soft_contact_steps += soft_contacts > 0;
      rigid_contact_steps += rigid_contacts > 0;
    }
    for (int pad = 0; pad < 2; ++pad) {
      pad_contact_steps[pad] += pad_contacts[pad] > 0;
      final_force[pad] = std::abs(signed_pad_force[pad]);
      peak_force[pad] = std::max(peak_force[pad], final_force[pad]);
    }
    max_penetration = std::max(max_penetration, penetration);
    max_pad_penetration = std::max(max_pad_penetration, pad_penetration);
    if (csv.is_open())
      csv << t << ',' << phase(t) << ',' << 2*(center-.01) << ','
          << std::max(0., .12-2*(center-.01)) << ',' << measured_width << ','
          << settled_width-measured_width << ',' << final_force[0] << ',' << final_force[1] << ','
          << soft_contacts << ',' << rigid_contacts << ',' << pad_contacts[0]+pad_contacts[1] << ','
          << penetration << ',' << pad_penetration << ','
          << shell.report.min_volume_ratio << ',' << inner.report.min_volume_ratio << '\n';

    if (deform3d::refresh_mujoco_flex_positions(model.get(), data.get()) !=
        deform3d::MujocoAdapterStatus::Ok || !shell.sample(model.get(), data.get()) ||
        !inner.sample(model.get(), data.get())) return fail("invalid integrated flex state");
    for (int q = 0; q < model->nq; ++q)
      if (!std::isfinite(data->qpos[q])) return fail("non-finite qpos");
    for (int q = 0; q < model->nv; ++q)
      if (!std::isfinite(data->qvel[q])) return fail("non-finite qvel");
    for (int w = 0; w < mjNWARNING; ++w)
      if (data->warning[w].number) return fail("MuJoCo reported a warning");
  }
  if (csv.is_open()) {
    csv.flush();
    if (!csv) return fail("CSV write failed");
  }
  const double final_width = width();
  std::printf("gripper sim_seconds=%.3f pad_contact_steps=[%zu,%zu] "
              "internal_soft_contact_steps=%zu internal_rigid_contact_steps=%zu "
              "peak_pad_force_N=[%.6g,%.6g] final_pad_force_N=[%.6g,%.6g]\n",
              double(data->time), pad_contact_steps[0], pad_contact_steps[1],
              soft_contact_steps, rigid_contact_steps, peak_force[0], peak_force[1],
              final_force[0], final_force[1]);
  std::printf("settled_width_m=%.6g minimum_width_m=%.6g recovered_width_m=%.6g "
              "max_penetration_m=%.6g max_pad_penetration_m=%.6g "
              "shell_min_volume_ratio=%.6Lg inner_min_volume_ratio=%.6Lg "
              "shell_max_edge_stretch=%.6Lg inner_max_edge_stretch=%.6Lg\n",
              settled_width, minimum_width, final_width, max_penetration, max_pad_penetration,
              shell.minimum_volume_ratio, inner.minimum_volume_ratio,
              shell.maximum_edge_stretch, inner.maximum_edge_stretch);
  // Qualitative functional gates: sustained loading, internal contact,
  // compression and recovery. These are not material-accuracy tolerances.
  if (pad_contact_steps[0] < 50 || pad_contact_steps[1] < 50 ||
      soft_contact_steps < 50 || rigid_contact_steps < 50 ||
      peak_force[0] < .1 || peak_force[1] < .1 ||
      final_force[0] > .1*peak_force[0] || final_force[1] > .1*peak_force[1] ||
      settled_width-minimum_width < .003 || final_width-minimum_width < .003 ||
      std::abs(data->time-steps*model->opt.timestep) > 1e-10)
    return fail("expected loading/contact/compression/recovery was absent");
  if (csv_path) std::printf("force_displacement_csv=%s\n", csv_path);
  return 0;
}
} // namespace

int main(int argc, char** argv) {
  return run(argc > 1 ? argv[1] : "examples/mujoco_gripper.xml", argc > 2 ? argv[2] : nullptr);
}
