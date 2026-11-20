#include <deform3d/softbody.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc,char** argv) {
    const int cells=argc>1 ? std::atoi(argv[1]) : 12;
    const int repeats=argc>2 ? std::atoi(argv[2]) : 15;
    const int substeps=argc>3 ? std::atoi(argv[3]) : 4;
    const int iterations=argc>4 ? std::atoi(argv[4]) : 8;
    if (cells<1 || cells>128 || repeats<3 || substeps<1 || iterations<1) {
        std::cerr << "usage: softbody_bench [cells_per_axis=12] "
                     "[repeats>=3] [substeps=4] [iterations=8]\n";
        return 2;
    }
    using Point=deform3d::Vec3<float>;
    using Clock=std::chrono::steady_clock;
    const int side=cells+1;
    const auto index=[=](int x,int y,int z) {
        return std::uint32_t((z*side+y)*side+x);
    };
    std::vector<Point> rest;
    rest.reserve(std::size_t(side)*side*side);
    for (int z=0;z<side;++z)
        for (int y=0;y<side;++y)
            for (int x=0;x<side;++x)
                rest.push_back({0.01f*x,0.01f*y,0.01f*z});
    std::vector<std::uint32_t> tetra;
    tetra.reserve(std::size_t(cells)*cells*cells*6*4);
    constexpr int pieces[6][4]={{0,1,3,7},{0,3,2,7},{0,2,6,7},
                                 {0,6,4,7},{0,4,5,7},{0,5,1,7}};
    for (int z=0;z<cells;++z)
        for (int y=0;y<cells;++y)
            for (int x=0;x<cells;++x) {
                const std::uint32_t v[8]={index(x,y,z),index(x+1,y,z),
                    index(x,y+1,z),index(x+1,y+1,z),index(x,y,z+1),
                    index(x+1,y,z+1),index(x,y+1,z+1),index(x+1,y+1,z+1)};
                for (auto& piece:pieces)
                    for (int corner:piece) tetra.push_back(v[corner]);
            }
    deform3d::SoftbodySolver<float> body;
    const auto prepare_start=Clock::now();
    if (body.prepare(rest.data(),rest.size(),tetra.data(),tetra.size())!=
        deform3d::SoftbodyStatus::ok) return 1;
    const auto prepare_end=Clock::now();
    body.set_compliance(1e-7f,1e-10f);
    for (int i=0;i<side;++i)
        for (int j=0;j<side;++j)
            body.set_fixed(index(i,j,cells),rest[index(i,j,cells)]);
    std::vector<double> timings;
    timings.reserve(std::size_t(repeats));
    for (int i=0;i<repeats+3;++i) {
        const auto begin=Clock::now();
        if (body.step(1.0f/60.0f,unsigned(substeps),unsigned(iterations))!=
            deform3d::SoftbodyStatus::ok) return 1;
        const auto end=Clock::now();
        if (i>=3)
            timings.push_back(std::chrono::duration<double,std::milli>(end-begin).count());
    }
    std::sort(timings.begin(),timings.end());
    const double p50=timings[timings.size()/2];
    const double p95=timings[std::size_t(0.95*(timings.size()-1))];
    volatile float checksum=body.positions()[body.vertex_count()/2].y;
    std::cout << "vertices=" << body.vertex_count()
              << " tetrahedra=" << body.tetrahedron_count()
              << " substeps=" << substeps << " iterations=" << iterations
              << " prepare_ms=" << std::chrono::duration<double,std::milli>(
                   prepare_end-prepare_start).count()
              << " step_p50_ms=" << p50 << " step_p95_ms=" << p95
              << " tetrahedra_per_second=" << double(body.tetrahedron_count())*
                   1000.0/p50
              << " checksum=" << checksum << '\n';
}
