#include <deform3d/embed.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <random>
#include <vector>

#define CHECK(condition) do { if (!(condition)) { \
  std::fprintf(stderr, "CHECK failed: %s:%d: %s\n", __FILE__, __LINE__, #condition); \
  std::abort(); \
} } while (false)

// Count standard dynamic allocations to verify that prepared transfers do not
// need a hidden workspace, including the first call and error paths.
static std::size_t allocation_count = 0;
void* operator new(std::size_t n) {
  ++allocation_count;
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

using namespace deform3d;

template<class T> static bool near(T a, T b, T eps = T(3e-5)) {
  return std::abs(a - b) <= eps * (T(1) + std::max(std::abs(a), std::abs(b)));
}
template<class T> static bool near(Vec3<T> a, Vec3<T> b, T eps = T(3e-5)) {
  return near(a.x, b.x, eps) && near(a.y, b.y, eps) && near(a.z, b.z, eps);
}

template<class T> static void invariants() {
  // Two neighboring tets share three vertices; the sixth vertex is unused.
  const Vec3<T> rest[6] = {{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,1,1},{4,4,4}};
  const std::uint32_t tets[8] = {0,1,2,3, 1,2,3,4};
  const Vec3<T> current[6] = {{T(.2),T(.1),T(-.3)}, {T(1.2),T(.2),T(.1)},
      {T(.3),T(.7),T(-.2)}, {T(-.1),T(.3),T(1.4)},
      {T(.8),T(1.4),T(1.1)}, {4,4,4}};
  const Vec3<T> velocity[6] = {{1,2,-3},{3,-2,1},{-2,1,4},{4,2,3},{1,-2,-1},{6,7,8}};
  std::mt19937 rng(20261117);
  std::uniform_real_distribution<double> uniform(0.01, 1.0);
  std::vector<Vec3<T>> points(4096), forces(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    T w[4] = {T(uniform(rng)), T(uniform(rng)), T(uniform(rng)), T(uniform(rng))};
    const T total = w[0] + w[1] + w[2] + w[3];
    for (std::size_t j = 0; j < 4; ++j)
      points[i] += rest[tets[4 * (i % 2) + j]] * (w[j] / total);
    forces[i] = {T(uniform(rng) - .5), T(uniform(rng) - .5), T(uniform(rng) - .5)};
  }
  TetraEmbedding<T> embedding;
  CHECK(embedding.prepare(rest, 6, tets, 2, points.data(), points.size()) == EmbedStatus::Ok);
  std::vector<Vec3<T>> position(points.size()), point_velocity(points.size());
  Vec3<T> coarse[6], sequential[6], left[6], right[6];
  const std::size_t allocations_before = allocation_count;
  CHECK(embedding.evaluate(current, 6, position.data(), position.size()) == EmbedStatus::Ok);
  CHECK(embedding.evaluate(velocity, 6, point_velocity.data(), point_velocity.size()) == EmbedStatus::Ok);
  CHECK(embedding.transfer_forces(forces.data(), forces.size(), coarse, 6) == EmbedStatus::Ok);
  CHECK(allocation_count == allocations_before);
  CHECK(near(coarse[5], Vec3<T>{}));

  // These identities hold for arbitrary nonaffine nodal motion and forces.
  // Accumulate diagnostics in double to measure transfer error separately.
  Vec3<double> fine_force{}, coarse_force{}, fine_torque{}, coarse_torque{};
  double fine_power = 0, coarse_power = 0;
  const Vec3<T> origin{T(2.1), T(-.8), T(.4)};
  for (std::size_t i = 0; i < forces.size(); ++i) {
    const auto f = forces[i];
    const auto moment = cross(position[i] - origin, f);
    fine_force += Vec3<double>{double(f.x), double(f.y), double(f.z)};
    fine_torque += Vec3<double>{double(moment.x), double(moment.y), double(moment.z)};
    fine_power += double(dot(f, point_velocity[i]));
  }
  for (std::size_t i = 0; i < 6; ++i) {
    const auto moment = cross(current[i] - origin, coarse[i]);
    coarse_force += Vec3<double>{double(coarse[i].x), double(coarse[i].y), double(coarse[i].z)};
    coarse_torque += Vec3<double>{double(moment.x), double(moment.y), double(moment.z)};
    coarse_power += double(dot(coarse[i], velocity[i]));
  }
  const double eps = std::is_same<T, float>::value ? 3e-5 : 1e-12;
  CHECK(near(fine_force, coarse_force, eps));
  CHECK(near(fine_torque, coarse_torque, eps));
  CHECK(near(fine_power, coarse_power, eps));

  // Serial batches and per-worker buffers followed by reduction both compose.
  const std::size_t middle = forces.size() / 2;
  CHECK(embedding.transfer_forces_range(forces.data(), forces.size(), sequential, 6,
                                         0, middle) == EmbedStatus::Ok);
  CHECK(embedding.transfer_forces_range(forces.data(), forces.size(), sequential, 6,
                                         middle, forces.size(), EmbedForceMode::Accumulate) == EmbedStatus::Ok);
  CHECK(embedding.transfer_forces_range(forces.data(), forces.size(), left, 6,
                                         0, middle) == EmbedStatus::Ok);
  CHECK(embedding.transfer_forces_range(forces.data(), forces.size(), right, 6,
                                         middle, forces.size()) == EmbedStatus::Ok);
  for (std::size_t i = 0; i < 6; ++i) {
    CHECK(near(coarse[i], sequential[i]));
    CHECK(near(coarse[i], left[i] + right[i]));
  }
  std::printf("%s: force error %.3g, torque error %.3g, power error %.3g\n",
              std::is_same<T, float>::value ? "float" : "double",
              norm(coarse_force - fine_force), norm(coarse_torque - fine_torque),
              std::abs(coarse_power - fine_power));
}

static void outside_and_contacts() {
  const Vec3<double> rest[4] = {{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
  const std::uint32_t tet[4] = {0,1,2,3};
  const Vec3<double> outside[1] = {{1.25,0,0}};
  const Vec3<double> force[1] = {{0,2,3}};
  TetraEmbedding<double> embedding;
  EmbedOptions<double> options;
  options.outside = EmbedOutsidePolicy::Extrapolate;
  options.max_outside_distance = .3;
  CHECK(embedding.prepare(rest, 4, tet, 1, outside, 1, options) == EmbedStatus::Ok);
  Vec3<double> result[4];
  CHECK(embedding.transfer_forces(force, 1, result, 4) == EmbedStatus::Ok);
  CHECK(near(result[0], force[0] * -.25));
  CHECK(near(result[1], force[0] * 1.25));
  CHECK(near(cross(rest[0], result[0]) + cross(rest[1], result[1]), cross(outside[0], force[0])));

  // Projection preserves torque at the projected point, not the discarded
  // external rest point. This distinction matters for offset contact geometry.
  options.outside = EmbedOutsidePolicy::ClampToBoundary;
  CHECK(embedding.prepare(rest, 4, tet, 1, outside, 1, options) == EmbedStatus::Ok);
  CHECK(embedding.transfer_forces(force, 1, result, 4) == EmbedStatus::Ok);
  Vec3<double> projected;
  CHECK(embedding.evaluate(rest, 4, &projected, 1) == EmbedStatus::Ok);
  CHECK(near(result[1], force[0]));
  CHECK(near(cross(rest[1], result[1]), cross(projected, force[0])));
  CHECK(!near(cross(rest[1], result[1]), cross(outside[0], force[0])));

  // Two separate hosts/meshes can scatter equal-and-opposite contact forces.
  // Force and torque cancel globally when both forces act at the same point.
  const Vec3<double> other_rest[4] = {{-.1,-.1,-.1},{2,-.1,-.1},{-.1,2,-.1},{-.1,-.1,2}};
  const Vec3<double> contact[1] = {{.2,.3,.1}};
  const Vec3<double> reaction[1] = {-force[0]};
  TetraEmbedding<double> other;
  CHECK(embedding.prepare(rest, 4, tet, 1, contact, 1) == EmbedStatus::Ok);
  CHECK(other.prepare(other_rest, 4, tet, 1, contact, 1) == EmbedStatus::Ok);
  Vec3<double> other_result[4], net_force{}, net_torque{};
  CHECK(embedding.transfer_forces(force, 1, result, 4) == EmbedStatus::Ok);
  CHECK(other.transfer_forces(reaction, 1, other_result, 4) == EmbedStatus::Ok);
  for (std::size_t i = 0; i < 4; ++i) {
    net_force += result[i] + other_result[i];
    net_torque += cross(rest[i], result[i]) + cross(other_rest[i], other_result[i]);
  }
  CHECK(near(net_force, Vec3<double>{}, 1e-13));
  CHECK(near(net_torque, Vec3<double>{}, 1e-13));
}

static void potential_gradient() {
  const Vec3<double> rest[4] = {{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
  const std::uint32_t tet[4] = {0,1,2,3};
  const Vec3<double> points[2] = {{.1,.2,.3},{.3,.1,.4}};
  const Vec3<double> target[2] = {{.3,-.4,.7},{-.1,.3,.6}};
  Vec3<double> current[4] = {{.2,.1,0},{1.2,.2,0},{.3,1.1,.2},{.1,-.1,.7}};
  Vec3<double> position[2], forces[2], coarse[4];
  TetraEmbedding<double> embedding;
  CHECK(embedding.prepare(rest, 4, tet, 1, points, 2) == EmbedStatus::Ok);
  CHECK(embedding.evaluate(current, 4, position, 2) == EmbedStatus::Ok);
  for (std::size_t i = 0; i < 2; ++i) forces[i] = (target[i] - position[i]) * 7.0;
  CHECK(embedding.transfer_forces(forces, 2, coarse, 4) == EmbedStatus::Ok);
  const auto energy = [&] {
    CHECK(embedding.evaluate(current, 4, position, 2) == EmbedStatus::Ok);
    double u = 0;
    for (std::size_t i = 0; i < 2; ++i) u += 3.5 * dot(position[i] - target[i], position[i] - target[i]);
    return u;
  };
  const double h = 1e-6;
  for (std::size_t i = 0; i < 4; ++i) {
    double* coordinate[3] = {&current[i].x, &current[i].y, &current[i].z};
    const double f[3] = {coarse[i].x, coarse[i].y, coarse[i].z};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const double original = *coordinate[axis];
      *coordinate[axis] = original + h;
      const double plus = energy();
      *coordinate[axis] = original - h;
      const double minus = energy();
      *coordinate[axis] = original;
      CHECK(near(f[axis], -(plus - minus) / (2 * h), 1e-9));
    }
  }
}

static void validation_and_strides() {
  const Vec3<double> rest[5] = {{0,0,0},{1,0,0},{0,1,0},{0,0,1},{2,2,2}};
  const std::uint32_t tet[4] = {0,1,2,3};
  const Vec3<double> points[2] = {{.25,.25,.25},{.5,.25,.125}};
  Vec3<double> forces[2] = {{4,8,12},{8,-4,2}};
  const Vec3<double> sentinel{17,18,19};
  std::array<Vec3<double>, 5> output, before;
  output.fill(sentinel);
  before = output;
  TetraEmbedding<double> embedding;
  CHECK(embedding.transfer_forces(forces, 2, output.data(), 5) == EmbedStatus::InvalidInput);
  CHECK(embedding.prepare(rest, 5, tet, 1, points, 2) == EmbedStatus::Ok);
  const std::size_t allocations_before = allocation_count;
  CHECK(embedding.transfer_forces(nullptr, 2, output.data(), 5) == EmbedStatus::InvalidInput);
  CHECK(embedding.transfer_forces(forces, 1, output.data(), 5) == EmbedStatus::InvalidInput);
  CHECK(embedding.transfer_forces(forces, 2, output.data(), 4) == EmbedStatus::InvalidInput);
  CHECK(embedding.transfer_forces(forces, 2, nullptr, 5) == EmbedStatus::InvalidInput);
  CHECK(embedding.transfer_forces(forces, std::numeric_limits<std::size_t>::max(),
                                   output.data(), 5) == EmbedStatus::InvalidInput);
  CHECK(embedding.transfer_forces(forces, 2, output.data(), 5,
                                   static_cast<EmbedForceMode>(42)) == EmbedStatus::InvalidInput);
  CHECK(embedding.transfer_forces_range(forces, 2, output.data(), 5, 2, 1) == EmbedStatus::InvalidRange);
  CHECK(embedding.transfer_forces_range(forces, 2, output.data(), 5, 0, 3) == EmbedStatus::InvalidRange);
  CHECK(embedding.transfer_forces(output.data(), 2, output.data(), 5) == EmbedStatus::Aliasing);
  CHECK(embedding.transfer_forces(output.data() + 1, 2, output.data(), 5) == EmbedStatus::Aliasing);
  CHECK(std::memcmp(output.data(), before.data(), sizeof(output)) == 0);

  forces[1].z = std::numeric_limits<double>::quiet_NaN();
  CHECK(embedding.transfer_forces(forces, 2, output.data(), 5) == EmbedStatus::NonFiniteInput);
  CHECK(embedding.transfer_forces(forces, 2, output.data(), 5, EmbedForceMode::Accumulate) == EmbedStatus::NonFiniteInput);
  CHECK(std::memcmp(output.data(), before.data(), sizeof(output)) == 0);
  // Nonfinite values outside the chosen range are not read.
  CHECK(embedding.transfer_forces_range(forces, 2, output.data(), 5, 0, 1) == EmbedStatus::Ok);
  for (std::size_t i = 0; i < 4; ++i) CHECK(near(output[i], Vec3<double>{1,2,3}));
  CHECK(near(output[4], Vec3<double>{}));
  forces[1].z = 2;
  output[4].x = std::numeric_limits<double>::infinity();
  before = output;
  CHECK(embedding.transfer_forces(forces, 2, output.data(), 5, EmbedForceMode::Accumulate) == EmbedStatus::NonFiniteInput);
  CHECK(std::memcmp(output.data(), before.data(), sizeof(output)) == 0);
  CHECK(embedding.transfer_forces(forces, 2, output.data(), 5) == EmbedStatus::Ok);
  Vec3<double> reference[5];
  std::copy(output.begin(), output.end(), reference);
  CHECK(embedding.transfer_forces(forces, 2, output.data(), 5, EmbedForceMode::Accumulate) == EmbedStatus::Ok);
  for (std::size_t i = 0; i < 5; ++i) CHECK(near(output[i], reference[i] * 2.0));
  before = output;
  CHECK(embedding.transfer_forces_range(forces, 2, output.data(), 5, 1, 1,
                                         EmbedForceMode::Accumulate) == EmbedStatus::Ok);
  CHECK(std::memcmp(output.data(), before.data(), sizeof(output)) == 0);
  CHECK(embedding.transfer_forces_range(forces, 2, output.data(), 5, 1, 1) == EmbedStatus::Ok);
  for (const auto& f : output) CHECK(near(f, Vec3<double>{}));

  double raw[2 * 5] = {4,8,12,91,92, 8,-4,2,93,94};
  float destination[5 * 4];
  std::fill(std::begin(destination), std::end(destination), 37.f);
  CHECK(embedding.transfer_forces_strided(raw, 2, 5, destination, 5, 4) == EmbedStatus::Ok);
  for (std::size_t i = 0; i < 5; ++i) {
    CHECK(near(Vec3<double>{destination[4*i], destination[4*i+1], destination[4*i+2]}, reference[i]));
    CHECK(destination[4*i+3] == 37.f);
  }
  CHECK(embedding.transfer_forces_strided(raw, 2, 5, destination, 5, 4,
                                           EmbedForceMode::Accumulate) == EmbedStatus::Ok);
  for (std::size_t i = 0; i < 5; ++i)
    CHECK(near(Vec3<double>{destination[4*i], destination[4*i+1], destination[4*i+2]}, reference[i] * 2.0));
  float previous[5 * 4];
  std::copy(std::begin(destination), std::end(destination), previous);
  CHECK(embedding.transfer_forces_strided(raw, 2, 2, destination, 5, 4) == EmbedStatus::InvalidInput);
  CHECK(embedding.transfer_forces_strided(raw, 2, 5, destination, 5, 2) == EmbedStatus::InvalidInput);
  CHECK(embedding.transfer_forces_strided(raw, 2, std::numeric_limits<std::size_t>::max(),
                                           destination, 5, 4) == EmbedStatus::InvalidInput);
  CHECK(embedding.transfer_forces_strided(raw, 2, 5, destination, 5,
                                           std::numeric_limits<std::size_t>::max()) == EmbedStatus::InvalidInput);
  CHECK(embedding.transfer_forces_strided(raw, std::numeric_limits<std::size_t>::max(),
                                           5, destination, 5, 4) == EmbedStatus::InvalidInput);
  CHECK(embedding.transfer_forces_strided(raw, 2, 5, destination, 5, 4,
                                           EmbedForceMode::Overwrite, 1, 0) == EmbedStatus::InvalidRange);
  CHECK(embedding.transfer_forces_strided(destination, 2, 4, destination, 5, 4) == EmbedStatus::Aliasing);
  raw[7] = std::numeric_limits<double>::infinity();
  CHECK(embedding.transfer_forces_strided(raw, 2, 5, destination, 5, 4) == EmbedStatus::NonFiniteInput);
  CHECK(std::memcmp(destination, previous, sizeof(destination)) == 0);
  raw[7] = 2;
  destination[0] = std::numeric_limits<float>::quiet_NaN();
  std::copy(std::begin(destination), std::end(destination), previous);
  CHECK(embedding.transfer_forces_strided(raw, 2, 5, destination, 5, 4,
                                           EmbedForceMode::Accumulate) == EmbedStatus::NonFiniteInput);
  CHECK(std::memcmp(destination, previous, sizeof(destination)) == 0);
  raw[3] = std::numeric_limits<double>::quiet_NaN();  // padding is ignored
  CHECK(embedding.transfer_forces_strided(raw, 2, 5, destination, 5, 4) == EmbedStatus::Ok);
  CHECK(embedding.transfer_forces_strided(raw, 2, 5, destination, 5, 4,
                                           EmbedForceMode::Overwrite, 0, 1) == EmbedStatus::Ok);
  for (std::size_t i = 0; i < 4; ++i)
    CHECK(near(Vec3<float>{destination[4*i], destination[4*i+1], destination[4*i+2]}, Vec3<float>{1,2,3}));
  CHECK(allocation_count == allocations_before);

  // As with evaluate, zero render bindings are valid but nonnull buffers are
  // still required. Overwrite zeroes all coarse forces in this case.
  CHECK(embedding.prepare(rest, 5, tet, 1, nullptr, 0) == EmbedStatus::Ok);
  output.fill(sentinel);
  CHECK(embedding.transfer_forces(forces, 0, output.data(), 5) == EmbedStatus::Ok);
  for (const auto& f : output) CHECK(near(f, Vec3<double>{}));
}

static void overflowing_boundary_search() {
  const Vec3<float> rest[4] = {{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
  const std::uint32_t tet[4] = {0,1,2,3};
  const Vec3<float> valid[1] = {{.25f,.25f,.25f}};
  const Vec3<float> huge[1] = {{1e20f,1e20f,1e20f}};
  const Vec3<float> force[1] = {{1,2,3}};
  TetraEmbedding<float> embedding;
  CHECK(embedding.prepare(rest, 4, tet, 1, valid, 1) == EmbedStatus::Ok);
  EmbedOptions<float> options;
  options.outside = EmbedOutsidePolicy::ClampToBoundary;
  std::size_t error_index = 19;
  CHECK(embedding.prepare(rest, 4, tet, 1, huge, 1, options, &error_index) == EmbedStatus::OutsideVolume);
  CHECK(error_index == 0);
  Vec3<float> output[4];
  // Failed preparation is transactional and cannot introduce a zero-sum map.
  CHECK(embedding.transfer_forces(force, 1, output, 4) == EmbedStatus::Ok);
  for (const auto& f : output) CHECK(near(f, force[0] * .25f));
}

int main() {
  invariants<float>();
  invariants<double>();
  outside_and_contacts();
  potential_gradient();
  validation_and_strides();
  overflowing_boundary_search();
  return 0;
}
