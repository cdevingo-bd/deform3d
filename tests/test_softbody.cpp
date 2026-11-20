#include "deform3d/softbody.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {

int failures=0;
#define CHECK(expr) do { if (!(expr)) { \
    std::cerr << __FILE__ << ':' << __LINE__ << ": " #expr " failed\n"; \
    ++failures; \
} } while (false)

using deform3d::SoftbodySolver;
using deform3d::SoftbodySphere;
using deform3d::SoftbodyStatus;
using V=deform3d::Vec3<double>;

double distance(V a,V b) { return deform3d::norm(a-b); }

double signed_volume(const std::vector<V>& p) {
    return deform3d::dot(deform3d::cross(p[1]-p[0],p[2]-p[0]),
                         p[3]-p[0])/6.0;
}

void rest_and_gravity() {
    const V rest[4]={{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
    const std::uint32_t tet[4]={0,1,2,3};
    SoftbodySolver<double> body;
    CHECK(body.prepare(rest,4,tet,4)==SoftbodyStatus::ok);
    CHECK(body.vertex_count()==4);
    CHECK(body.tetrahedron_count()==1);
    CHECK(body.edge_count()==6);
    CHECK(body.boundary_face_count()==4);
    CHECK(body.set_gravity({0,0,0})==SoftbodyStatus::ok);
    CHECK(body.step(1.0/60.0,4,8)==SoftbodyStatus::ok);
    for (int i=0;i<4;++i) CHECK(distance(body.positions()[i],rest[i])<1e-12);

    // With no pins or contacts, gravity produces a rigid translation and
    // leaves all six edge lengths and the signed volume intact.
    CHECK(body.set_gravity({0,-9.81,0})==SoftbodyStatus::ok);
    CHECK(body.step(1.0/60.0,4,8)==SoftbodyStatus::ok);
    const auto& p=body.positions();
    const V displacement=p[0]-rest[0];
    CHECK(displacement.y<0);
    for (int i=1;i<4;++i)
        CHECK(distance(p[i]-rest[i],displacement)<1e-10);
    CHECK(std::abs(signed_volume(p)-1.0/6.0)<1e-10);
}

void pins_and_elastic_recovery() {
    const V rest[4]={{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
    const std::uint32_t tet[4]={0,1,2,3};
    SoftbodySolver<double> body;
    CHECK(body.prepare(rest,4,tet,4)==SoftbodyStatus::ok);
    CHECK(body.set_gravity({0,0,0})==SoftbodyStatus::ok);
    CHECK(body.set_compliance(1e-8,1e-9)==SoftbodyStatus::ok);
    CHECK(body.set_fixed(0,{0,0,0})==SoftbodyStatus::ok);
    CHECK(body.set_position(1,{2,0,0})==SoftbodyStatus::ok);
    CHECK(body.step(1.0/60.0,4,16)==SoftbodyStatus::ok);
    CHECK(body.positions()[0].x==0 && body.positions()[0].y==0 &&
          body.positions()[0].z==0);
    CHECK(distance(body.positions()[0],body.positions()[1])<1.1);
    CHECK(signed_volume(body.positions())>0);

    CHECK(body.set_fixed(0,{-0.2,0,0})==SoftbodyStatus::ok);
    CHECK(body.step(1.0/60.0,4,16)==SoftbodyStatus::ok);
    CHECK(body.positions()[0].x==-0.2);
    body.reset_to_rest();
    CHECK(body.positions()[0].x==-0.2); // current kinematic target remains
    CHECK(distance(body.positions()[1],rest[1])<1e-12);
}

void sphere_contact_and_overlap_stabilization() {
    const V rest[4]={{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
    const std::uint32_t tet[4]={0,1,2,3};
    SoftbodySolver<double> body;
    CHECK(body.prepare(rest,4,tet,4)==SoftbodyStatus::ok);
    CHECK(body.set_gravity({0,0,0})==SoftbodyStatus::ok);
    const SoftbodySphere<double> sphere{{0.25,0.25,-0.15},0.25,0};
    CHECK(body.set_spheres(&sphere,1)==SoftbodyStatus::ok);
    CHECK(body.step(1.0/60.0,4,8)==SoftbodyStatus::ok);

    // Initial penetration is resolved before deriving velocity. The surface
    // moves away from the sphere without gaining substep-sized fake momentum.
    double mean_z=0.0;
    for (int i=0;i<4;++i) mean_z+=body.positions()[i].z-rest[i].z;
    mean_z/=4.0;
    CHECK(mean_z>0.08 && mean_z<0.15);
    CHECK(signed_volume(body.positions())>0);
    for (V p:body.positions()) CHECK(deform3d::finite(p));
    CHECK(body.set_spheres(nullptr,0)==SoftbodyStatus::ok);
}

void topology_and_validation() {
    const V rest[5]={{0,0,0},{1,0,0},{0,1,0},{0,0,1},{0,0,-1}};
    const std::uint32_t two_tets[8]={0,1,2,3,0,2,1,4};
    SoftbodySolver<double> body;
    CHECK(body.step(0.01)==SoftbodyStatus::not_prepared);
    CHECK(body.prepare(rest,5,two_tets,8)==SoftbodyStatus::ok);
    CHECK(body.boundary_face_count()==6); // shared face is internal
    CHECK(body.edge_count()==9);
    CHECK(body.step(0.0)==SoftbodyStatus::invalid_input);
    CHECK(body.set_spheres(nullptr,1)==SoftbodyStatus::invalid_input);
    CHECK(body.set_compliance(-1,0)==SoftbodyStatus::invalid_input);
    CHECK(body.set_inverse_mass(0,-1)==SoftbodyStatus::invalid_input);
    CHECK(body.set_fixed(99,{0,0,0})==SoftbodyStatus::invalid_input);
    CHECK(body.set_gravity({std::numeric_limits<double>::infinity(),0,0})==
          SoftbodyStatus::invalid_input);
    const auto saved=body.positions();
    const std::uint32_t invalid[4]={0,1,2,2};
    CHECK(body.prepare(rest,5,invalid,4)==SoftbodyStatus::invalid_input);
    CHECK(body.vertex_count()==0); // failed preparation clears incomplete state
    CHECK(saved.size()==5);
    const V flat[4]={{0,0,0},{1,0,0},{0,1,0},{0.5,0.5,0}};
    const std::uint32_t tet[4]={0,1,2,3};
    CHECK(body.prepare(flat,4,tet,4)==SoftbodyStatus::invalid_input);
}

void float_instantiation() {
    using F=deform3d::Vec3<float>;
    const F rest[4]={{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
    const std::uint32_t tet[4]={0,1,2,3};
    const float masses[4]={0,1,1,1};
    SoftbodySolver<float> body;
    CHECK(body.prepare(rest,4,tet,4,masses)==SoftbodyStatus::ok);
    CHECK(body.set_gravity({0,0,0})==SoftbodyStatus::ok);
    CHECK(body.step(1.0f/60.0f,2,4)==SoftbodyStatus::ok);
    CHECK(body.positions()[0].x==0.0f);
}

void rejects_invalid_face_adjacency() {
    const V rest[6]={{0,0,0},{1,0,0},{0,1,0},{0,0,1},
                     {0,0,-1},{0.2,0.2,2}};
    SoftbodySolver<double> body;
    // Input order does not change whether these are the same tetrahedron.
    const std::uint32_t duplicate[8]={0,1,2,3,3,1,0,2};
    CHECK(body.prepare(rest,6,duplicate,8)==SoftbodyStatus::invalid_input);
    CHECK(body.vertex_count()==0);
    CHECK(body.step(0.01)==SoftbodyStatus::not_prepared);

    // Two distinct tets on the same side of the base overlap. Cancelling this
    // shared face would silently remove contact geometry from an invalid solid.
    const std::uint32_t overlap[8]={0,1,2,3,0,1,2,5};
    CHECK(body.prepare(rest,6,overlap,8)==SoftbodyStatus::invalid_input);
    CHECK(body.boundary_face_count()==0);

    const std::uint32_t nonmanifold[12]={0,1,2,3,0,2,1,4,0,1,2,5};
    CHECK(body.prepare(rest,6,nonmanifold,12)==SoftbodyStatus::invalid_input);
    CHECK(body.tetrahedron_count()==0);

    // Distinct tets on opposite sides still work with arbitrary input winding.
    const std::uint32_t valid[8]={3,1,0,2,2,1,4,0};
    CHECK(body.prepare(rest,6,valid,8)==SoftbodyStatus::ok);
    CHECK(body.boundary_face_count()==6);
}

} // namespace

int main() {
    rest_and_gravity();
    pins_and_elastic_recovery();
    sphere_contact_and_overlap_stabilization();
    topology_and_validation();
    rejects_invalid_face_adjacency();
    float_instantiation();
    if (failures) {
        std::cerr << failures << " soft-body checks failed\n";
        return 1;
    }
    std::cout << "soft-body checks passed\n";
}
