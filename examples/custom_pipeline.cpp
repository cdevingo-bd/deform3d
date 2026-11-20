#include <deform3d/embed.hpp>
#include <iostream>

// A simulation or engine supplies its own tetrahedral vertex stream. This
// example uses four vertices; the same binding can drive millions of samples.
int main() {
  using namespace deform3d;
  const Vec3d rest[]{{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
  const uint32_t tetrahedra[]{0,1,2,3};
  const Vec3d dense_rest[]{{0.25,0.25,0.25},{0.1,0.2,0.3}};
  TetraEmbedding<double> embed;
  const auto status=embed.prepare(rest,4,tetrahedra,1,dense_rest,2);
  if (status!=EmbedStatus::Ok) {
    std::cerr << embed_status_string(status) << '\n';return 1;
  }

  // Stand-in for world-space positions supplied by mjData.flexvert_xpos or
  // another host. The host remains responsible for physics and contact.
  const double current_xyz[]{0,0,0,  1.1,0,0,  0,1,0,  0,0,1};
  float output_xyz[2*3]{};
  if (embed.evaluate_strided(current_xyz,4,3,output_xyz,2,3)!=EmbedStatus::Ok)
    return 2;
  for (size_t i=0;i<2;++i)
    std::cout << output_xyz[3*i] << ' ' << output_xyz[3*i+1]
              << ' ' << output_xyz[3*i+2] << '\n';
}
