#include <deform3d/embed.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace deform3d;
using Clock=std::chrono::steady_clock;

int main(int argc,char** argv) {
  const size_t n=argc>1 ? size_t(std::strtoull(argv[1],nullptr,10)) : 250000;
  const int reps=argc>2 ? std::atoi(argv[2]) : 40;
  const size_t grid=argc>3 ? size_t(std::strtoull(argv[3],nullptr,10)) : 10;
  if (!n || reps<3 || grid<1 || grid>100) return 2;
  const size_t pitch=grid+1;
  const auto id=[=](size_t x,size_t y,size_t z) {
    return uint32_t((z*pitch+y)*pitch+x);
  };
  std::vector<Vec3d> rest(pitch*pitch*pitch);
  for(size_t z=0;z<pitch;++z) for(size_t y=0;y<pitch;++y)
    for(size_t x=0;x<pitch;++x) rest[id(x,y,z)]={double(x),double(y),double(z)};
  std::vector<uint32_t> tetra;
  tetra.reserve(24*grid*grid*grid);
  for(size_t z=0;z<grid;++z) for(size_t y=0;y<grid;++y)
    for(size_t x=0;x<grid;++x) {
      const uint32_t a=id(x,y,z),b=id(x+1,y,z),c=id(x+1,y+1,z),d=id(x,y+1,z);
      const uint32_t e=id(x,y,z+1),f=id(x+1,y,z+1),g=id(x+1,y+1,z+1),h=id(x,y+1,z+1);
      const uint32_t cells[24]={a,b,c,g,a,c,d,g,a,d,h,g,a,h,e,g,a,e,f,g,a,f,b,g};
      tetra.insert(tetra.end(),cells,cells+24);
    }
  std::vector<Vec3d> render(n);
  for (size_t i=0;i<n;++i) {
    const uint64_t a=uint64_t(i)*0x9e3779b97f4a7c15ULL;
    const uint64_t b=uint64_t(i)*0xbf58476d1ce4e5b9ULL;
    const uint64_t c=uint64_t(i)*0x94d049bb133111ebULL;
    render[i]={double(a%1000003)/1000003.0*grid,
               double(b%1000033)/1000033.0*grid,
               double(c%1000037)/1000037.0*grid};
  }
  TetraEmbedding<double> embed;
  const auto setup_start=Clock::now();
  if (embed.prepare(rest.data(),rest.size(),tetra.data(),tetra.size()/4,
                    render.data(),n)!=EmbedStatus::Ok) return 3;
  const auto setup_end=Clock::now();
  std::vector<double> current_xyz(3*rest.size());
  for(size_t i=0;i<rest.size();++i) {
    current_xyz[3*i]=rest[i].x+0.05*std::sin(rest[i].y);
    current_xyz[3*i+1]=rest[i].y;
    current_xyz[3*i+2]=rest[i].z;
  }
  std::vector<float> out(3*n);
  std::vector<double> ms;ms.reserve(size_t(reps));
  for (int i=0;i<reps+5;++i) {
    const auto start=Clock::now();
    const EmbedStatus s=embed.evaluate_strided(current_xyz.data(),rest.size(),3,out.data(),n,3);
    const auto end=Clock::now();
    if (s!=EmbedStatus::Ok) return 4;
    if (i>=5) ms.push_back(std::chrono::duration<double,std::milli>(end-start).count());
  }
  std::sort(ms.begin(),ms.end());
  const double p50=ms[ms.size()/2],p95=ms[size_t(.95*(ms.size()-1))];
  volatile float checksum=out[3*(n/2)];
  std::cout << "embedding: vertices=" << n << " tetrahedra=" << tetra.size()/4
            << " setup_ms=" << std::chrono::duration<double,std::milli>(setup_end-setup_start).count()
            << " p50_ms=" << p50 << " p95_ms=" << p95
            << " million_vertices_per_s=" << double(n)/(p50*1000)
            << " checksum=" << checksum << '\n';
}
