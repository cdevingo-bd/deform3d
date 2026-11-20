#include <deform3d/embed.hpp>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

#define CHECK(condition) do { if (!(condition)) {                         \
  std::fprintf(stderr, "CHECK failed: %s:%d: %s\n", __FILE__, __LINE__, #condition); \
  std::abort();                                                            \
} } while (false)

using deform3d::Vec3;

template<class T> static bool near(T a, T b, T eps = T(1e-5)) {
  return std::abs(a - b) <= eps;
}
template<class T> static bool near(Vec3<T> a, Vec3<T> b, T eps = T(1e-5)) {
  return near(a.x,b.x,eps) && near(a.y,b.y,eps) && near(a.z,b.z,eps);
}

int main() {
  using namespace deform3d;
  const Vec3<double> rest[4] = {{0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}};
  const std::uint32_t tets[4] = {0, 1, 2, 3};
  const Vec3<double> render[3] = {{.25,.25,.25}, {1,0,0}, {0,0,0}};
  TetraEmbedding<double> embedding;
  CHECK(embedding.prepare(rest, 4, tets, 1, render, 3) == EmbedStatus::Ok);
  CHECK(embedding.simulation_vertex_count() == 4);
  CHECK(embedding.render_vertex_count() == 3);
  CHECK(embedding.tetrahedron_count() == 1);
  for (const auto& b : embedding.bindings()) {
    double total = 0;
    for (double w : b.weights) total += w;
    CHECK(near(total, 1.0));
  }

  Vec3<double> moved[4];
  for (int i=0; i<4; ++i) moved[i] = rest[i] + Vec3<double>{3,4,5};
  Vec3<double> output[3]{};
  CHECK(embedding.evaluate(moved, 4, output, 3) == EmbedStatus::Ok);
  for (int i=0; i<3; ++i) CHECK(near(output[i], render[i] + Vec3<double>{3,4,5}));
  CHECK(embedding.evaluate(moved, 4, output, 3) == EmbedStatus::Ok);

  // A sheared tetrahedron interpolates every embedded point affinely.
  for (int i=0; i<4; ++i) moved[i] = {rest[i].x + .5*rest[i].y, rest[i].y,
                                      2*rest[i].z};
  CHECK(embedding.evaluate_range(moved, 4, output, 3, 0, 1) == EmbedStatus::Ok);
  CHECK(embedding.evaluate_range(moved, 4, output, 3, 1, 3) == EmbedStatus::Ok);
  CHECK(near(output[0], Vec3<double>{.375,.25,.5}));
  CHECK(embedding.evaluate_range(moved, 4, output, 3, 2, 4) == EmbedStatus::InvalidRange);
  CHECK(embedding.evaluate(moved, 3, output, 3) == EmbedStatus::InvalidInput);
  CHECK(embedding.evaluate(moved, 4, moved, 4) == EmbedStatus::Aliasing);

  // Failed preparation does not disturb an established mapping.
  const Vec3<double> outside[1] = {{2,0,0}};
  std::size_t bad_index = 42;
  CHECK(embedding.prepare(rest, 4, tets, 1, outside, 1, {}, &bad_index) ==
         EmbedStatus::OutsideVolume);
  CHECK(bad_index == 0 && embedding.render_vertex_count() == 3);

  EmbedOptions<double> options;
  options.outside = EmbedOutsidePolicy::ClampToBoundary;
  options.max_outside_distance = .5;
  CHECK(embedding.prepare(rest, 4, tets, 1, outside, 1, options) ==
         EmbedStatus::OutsideVolume);
  options.max_outside_distance = 1.01;
  CHECK(embedding.prepare(rest, 4, tets, 1, outside, 1, options) == EmbedStatus::Ok);
  Vec3<double> projected{};
  CHECK(embedding.evaluate(rest, 4, &projected, 1) == EmbedStatus::Ok);
  CHECK(near(projected, Vec3<double>{1,0,0}));

  // Bounded extrapolation preserves a detailed render surface outside the
  // coarser simulation hull, including the undeformed pose and translations.
  options.outside = EmbedOutsidePolicy::Extrapolate;
  options.max_outside_distance = std::numeric_limits<double>::infinity();
  CHECK(embedding.prepare(rest, 4, tets, 1, outside, 1, options) ==
        EmbedStatus::InvalidInput);
  options.max_outside_distance = .5;
  CHECK(embedding.prepare(rest, 4, tets, 1, outside, 1, options) ==
        EmbedStatus::OutsideVolume);
  options.max_outside_distance = 1.01;
  CHECK(embedding.prepare(rest, 4, tets, 1, outside, 1, options) == EmbedStatus::Ok);
  CHECK(embedding.evaluate(rest, 4, &projected, 1) == EmbedStatus::Ok);
  CHECK(near(projected, outside[0]));
  Vec3<double> translated[4];
  for (int i=0; i<4; ++i) translated[i] = rest[i] + Vec3<double>{.5,2,-3};
  CHECK(embedding.evaluate(translated, 4, &projected, 1) == EmbedStatus::Ok);
  CHECK(near(projected, outside[0] + Vec3<double>{.5,2,-3}));

  // A point just beyond a face can pass the containment tolerance; it must
  // still retain its signed barycentric weight in extrapolation mode.
  const Vec3<double> barely_outside[1] = {{-1e-7,.2,.2}};
  options.max_outside_distance = 0;
  CHECK(embedding.prepare(rest, 4, tets, 1, barely_outside, 1, options) ==
        EmbedStatus::OutsideVolume);
  options.max_outside_distance = .01;
  CHECK(embedding.prepare(rest, 4, tets, 1, barely_outside, 1, options) == EmbedStatus::Ok);
  CHECK(embedding.evaluate(rest, 4, &projected, 1) == EmbedStatus::Ok);
  CHECK(near(projected.x, -1e-7, 1e-10));

  options.max_outside_distance = 1.01;
  CHECK(embedding.prepare(rest, 4, tets, 1, outside, 1, options) == EmbedStatus::Ok);

  // Raw mjtNum-style XYZ arrays can feed a float rendering buffer directly.
  double raw[4][4] = {{0,0,0,99}, {1,0,0,99}, {0,1,0,99}, {0,0,1,99}};
  float vertices[1][4] = {{0,0,0,37}};
  CHECK(embedding.evaluate_strided(&raw[0][0], 4, 4, &vertices[0][0], 1, 4) ==
         EmbedStatus::Ok);
  CHECK(near(vertices[0][0], 2.f) && near(vertices[0][3], 37.f));

  const std::uint32_t invalid[4] = {0,1,2,4};
  CHECK(embedding.prepare(rest, 4, invalid, 1, render, 3) == EmbedStatus::InvalidIndex);
  const std::uint32_t degenerate[4] = {0,1,2,2};
  CHECK(embedding.prepare(rest, 4, degenerate, 1, render, 3) ==
         EmbedStatus::DegenerateTetrahedron);

  // Validate lookup across neighboring tetrahedra sharing a face.
  const Vec3<double> five[5] = {{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,1,1}};
  const std::uint32_t pair[8] = {0,1,2,3, 1,2,3,4};
  const Vec3<double> inside_second[1] = {{.75,.75,.75}};
  CHECK(embedding.prepare(five, 5, pair, 2, inside_second, 1) == EmbedStatus::Ok);
  CHECK(embedding.bindings()[0].tetrahedron == 1);
  Vec3<double> result;
  CHECK(embedding.evaluate(five, 5, &result, 1) == EmbedStatus::Ok);
  CHECK(near(result, inside_second[0]));

  // Force a multi-node BVH and exercise both containment and closest searches.
  std::vector<Vec3<double>> many_vertices;
  std::vector<std::uint32_t> many_tets;
  std::vector<Vec3<double>> many_render;
  for (std::uint32_t i=0; i<32; ++i) {
    const double x = double(i) * 2;
    const auto base = std::uint32_t(many_vertices.size());
    many_vertices.insert(many_vertices.end(), {{x,0,0},{x+1,0,0},{x,1,0},{x,0,1}});
    for (std::uint32_t k=0; k<4; ++k) many_tets.push_back(base+k);
    many_render.push_back({x+.2,.3,.1});
  }
  EmbedOptions<double> bvh_options;
  bvh_options.bvh_leaf_size = 2;
  CHECK(embedding.prepare(many_vertices.data(), many_vertices.size(),
                           many_tets.data(), many_tets.size()/4,
                           many_render.data(), many_render.size(), bvh_options) ==
         EmbedStatus::Ok);
  std::vector<Vec3<double>> many_out(many_render.size());
  CHECK(embedding.evaluate(many_vertices.data(), many_vertices.size(),
                            many_out.data(), many_out.size()) == EmbedStatus::Ok);
  for (std::size_t i=0; i<many_render.size(); ++i)
    CHECK(near(many_out[i], many_render[i]));

  return 0;
}
