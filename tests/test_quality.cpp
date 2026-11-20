#include <deform3d/quality.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#define CHECK(x) do {if (!(x)) {std::fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x);std::abort();}} while(false)

using namespace deform3d;

static bool near(long double a,long double b,long double eps=1e-10L) {
  return std::abs(a-b)<=eps;
}

int main() {
  Vec3d rest[]={{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
  const uint32_t tet[]={0,1,2,3};
  TetraQuality<double> quality;
  CHECK(quality.prepare(rest,4,tet,1)==Status::Ok);
  TetraQualityReport report;
  CHECK(quality.evaluate(rest,4,report)==Status::Ok);
  CHECK(report.tetrahedra==1 && report.inverted==0 && report.near_zero_volume==0);
  CHECK(near(report.min_volume_ratio,1) && near(report.max_volume_ratio,1));
  CHECK(near(report.min_edge_stretch,1) && near(report.max_edge_stretch,1));

  Vec3d current[]={{0,0,0},{2,0,0},{0,2,0},{0,0,2}};
  CHECK(quality.evaluate(current,4,report)==Status::Ok);
  CHECK(near(report.min_volume_ratio,8) && near(report.max_edge_stretch,2));
  current[3]={0,0,-1};
  CHECK(quality.evaluate(current,4,report)==Status::Ok);
  CHECK(report.inverted==1 && near(report.min_volume_ratio,-4));
  current[3]={0,0,0};
  CHECK(quality.evaluate(current,4,report)==Status::Ok);
  CHECK(report.inverted==0 && report.near_zero_volume==1 && near(report.min_volume_ratio,0));
  CHECK(near(report.min_edge_stretch,0));

  // Source positions can be a MuJoCo-style array of mjtNum triples.
  float strided[4][4]={{0,0,0,99},{1,0,0,99},{0,1,0,99},{0,0,1,99}};
  CHECK(quality.evaluate_strided(&strided[0][0],4,4,report)==Status::Ok);
  CHECK(near(report.max_volume_ratio,1) && near(report.max_edge_stretch,1));
  strided[3][2]=std::numeric_limits<float>::quiet_NaN();
  CHECK(quality.evaluate_strided(&strided[0][0],4,4,report)==Status::NonFinite);
  CHECK(near(report.min_volume_ratio,1)); // failure preserves the previous report

  const uint32_t bad[]={0,1,2,4};
  CHECK(quality.prepare(rest,4,bad,1)==Status::InvalidIndex);
  CHECK(quality.tetrahedron_count()==1); // failed prepare preserves prior topology
  const uint32_t degenerate[]={0,1,2,2};
  CHECK(quality.prepare(rest,4,degenerate,1)==Status::DegenerateGeometry);
  CHECK(quality.evaluate(rest,4,report)==Status::Ok);
  rest[0].x=std::numeric_limits<double>::infinity();
  CHECK(quality.prepare(rest,4,tet,1)==Status::NonFinite);
  CHECK(quality.evaluate_strided(&strided[0][0],4,2,report)==Status::InvalidArgument);

  std::puts("quality tests passed");
}
