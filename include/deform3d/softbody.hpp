#pragma once

// Optional tetrahedral XPBD soft-body solver. The core header supplies only
// Vec3<T>; this extension owns its state, constraints, and reusable scratch.
// Macklin, Mueller and Chentanez, XPBD, Motion in Games 2016:
// https://doi.org/10.1145/2994258.2994272

#include "deform3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace deform3d {

enum class SoftbodyStatus {
    ok,
    invalid_input,
    not_prepared,
    numerical_failure
};

inline const char* softbody_status_string(SoftbodyStatus status) noexcept {
    switch (status) {
        case SoftbodyStatus::ok: return "ok";
        case SoftbodyStatus::invalid_input: return "invalid input";
        case SoftbodyStatus::not_prepared: return "solver not prepared";
        case SoftbodyStatus::numerical_failure: return "numerical failure";
    }
    return "unknown";
}

// Kinematic collider. Move center between steps to drive a soft enclosure.
// The soft body's reaction on the sphere is not computed by this class.
template <typename T>
struct SoftbodySphere {
    Vec3<T> center{};
    T radius{};
    T skin{};
};

namespace softbody_detail {

struct V {
    double x{}, y{}, z{};
};
inline V operator+(V a, V b) noexcept { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline V operator-(V a, V b) noexcept { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline V operator*(V a, double s) noexcept { return {a.x*s, a.y*s, a.z*s}; }
inline V operator/(V a, double s) noexcept { return {a.x/s, a.y/s, a.z/s}; }
inline double dot(V a, V b) noexcept { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline V cross(V a, V b) noexcept {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
inline double norm(V a) noexcept { return std::sqrt(dot(a,a)); }

template <typename T>
inline V vec(const Vec3<T>& v) noexcept {
    return {static_cast<double>(v.x),static_cast<double>(v.y),
            static_cast<double>(v.z)};
}
template <typename T>
inline Vec3<T> mesh(V v) noexcept {
    return {static_cast<T>(v.x),static_cast<T>(v.y),static_cast<T>(v.z)};
}
template <typename T>
inline bool finite(const Vec3<T>& p) noexcept {
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}

struct Barycentric {
    V point;
    double a{}, b{}, c{};
};

inline Barycentric segment_closest(V p, V a, V b, int edge) noexcept {
    const V ab = b-a;
    const double length2 = dot(ab,ab);
    const double t = length2 > 0 ?
        std::max(0.0,std::min(1.0,dot(p-a,ab)/length2)) : 0.0;
    if (edge == 0) return {a+ab*t,1.0-t,t,0.0};
    if (edge == 1) return {a+ab*t,0.0,1.0-t,t};
    return {a+ab*t,t,0.0,1.0-t};
}

// Closest point on a triangle including its edges/vertices. The degenerate
// fallback makes contact safe even when a tetra collapses under extreme load.
inline Barycentric closest(V p, V a, V b, V c) noexcept {
    const V ab=b-a, ac=c-a;
    const double scale=std::max({dot(ab,ab),dot(ac,ac),dot(c-b,c-b)});
    if (dot(cross(ab,ac),cross(ab,ac)) <= 1e-24*scale*scale) {
        const auto e0=segment_closest(p,a,b,0);
        const auto e1=segment_closest(p,b,c,1);
        const auto e2=segment_closest(p,c,a,2);
        const double d0=dot(e0.point-p,e0.point-p);
        const double d1=dot(e1.point-p,e1.point-p);
        const double d2=dot(e2.point-p,e2.point-p);
        return d0 <= d1 && d0 <= d2 ? e0 : (d1 <= d2 ? e1 : e2);
    }
    const V ap=p-a;
    const double d1=dot(ab,ap), d2=dot(ac,ap);
    if (d1<=0.0 && d2<=0.0) return {a,1,0,0};
    const V bp=p-b;
    const double d3=dot(ab,bp), d4=dot(ac,bp);
    if (d3>=0.0 && d4<=d3) return {b,0,1,0};
    const double vc=d1*d4-d3*d2;
    if (vc<=0.0 && d1>=0.0 && d3<=0.0) {
        const double v=d1/(d1-d3);
        return {a+ab*v,1.0-v,v,0.0};
    }
    const V cp=p-c;
    const double d5=dot(ab,cp), d6=dot(ac,cp);
    if (d6>=0.0 && d5<=d6) return {c,0,0,1};
    const double vb=d5*d2-d1*d6;
    if (vb<=0.0 && d2>=0.0 && d6<=0.0) {
        const double w=d2/(d2-d6);
        return {a+ac*w,1.0-w,0.0,w};
    }
    const double va=d3*d6-d5*d4;
    if (va<=0.0 && d4-d3>=0.0 && d5-d6>=0.0) {
        const double w=(d4-d3)/((d4-d3)+(d5-d6));
        return {b+(c-b)*w,0.0,1.0-w,w};
    }
    const double inv=1.0/(va+vb+vc);
    const double v=vb*inv,w=vc*inv;
    return {a+ab*v+ac*w,1.0-v-w,v,w};
}

} // namespace softbody_detail

// Sequential Gauss-Seidel XPBD with edge length and signed tetra volume
// constraints. No allocations occur in step() after prepare()/set_spheres().
// One instance is owned by one simulation thread; independent bodies can be
// stepped in parallel by an engine's job system. Coordinates are meters when
// gravity uses the default SI value. Material compliance is scale dependent.
template <typename T>
class SoftbodySolver {
    static_assert(std::is_floating_point<T>::value,
                  "SoftbodySolver requires float or double positions");
public:
    using Point=Vec3<T>;
    using Sphere=SoftbodySphere<T>;

    // index_count is a multiple of four: four vertex indices per tetrahedron.
    // Duplicate tetrahedra, faces with more than two incident tetrahedra, and
    // two tetrahedra on the same side of a shared face are rejected.
    // inverse_masses==nullptr gives every vertex unit inverse mass. A zero
    // inverse mass pins that vertex at its current target position.
    SoftbodyStatus prepare(const Point* rest, std::size_t vertex_count,
                           const std::uint32_t* tetra_indices,
                           std::size_t index_count,
                           const T* inverse_masses=nullptr) {
        clear();
        if (!rest || !vertex_count ||
            vertex_count>std::numeric_limits<std::uint32_t>::max() ||
            index_count%4 || (index_count && !tetra_indices) ||
            index_count/4>std::numeric_limits<std::size_t>::max()/6)
            return status_=SoftbodyStatus::invalid_input;
        for (std::size_t i=0;i<vertex_count;++i)
            if (!softbody_detail::finite(rest[i]) ||
                (inverse_masses && (!std::isfinite(inverse_masses[i]) ||
                                    inverse_masses[i]<T{})))
                return status_=SoftbodyStatus::invalid_input;
        rest_.assign(rest,rest+vertex_count);
        x_=rest_; x_prev_=rest_; x_saved_=rest_;
        v_.assign(vertex_count,Point{});
        v_saved_=v_;
        fixed_target_=rest_;
        inv_mass_.assign(vertex_count,T(1));
        if (inverse_masses)
            std::copy(inverse_masses,inverse_masses+vertex_count,
                      inv_mass_.begin());

        struct EdgeKey { std::uint32_t a,b; };
        struct FaceKey {
            std::array<std::uint32_t,3> key;
            std::array<std::uint32_t,3> face;
        };
        std::vector<EdgeKey> edge_keys;
        std::vector<FaceKey> face_keys;
        edge_keys.reserve(6*(index_count/4));
        face_keys.reserve(index_count);
        tets_.reserve(index_count/4);
        const auto add_edge=[&](std::uint32_t a,std::uint32_t b) {
            if (a>b) std::swap(a,b);
            edge_keys.push_back({a,b});
        };
        const auto add_face=[&](std::uint32_t a,std::uint32_t b,
                                std::uint32_t c,std::uint32_t opposite) {
            auto pa=softbody_detail::vec(rest_[a]);
            auto pb=softbody_detail::vec(rest_[b]);
            auto pc=softbody_detail::vec(rest_[c]);
            auto pd=softbody_detail::vec(rest_[opposite]);
            if (softbody_detail::dot(softbody_detail::cross(pb-pa,pc-pa),pd-pa)>0)
                std::swap(b,c); // outward normal of this tetra
            FaceKey f{{a,b,c},{a,b,c}};
            std::sort(f.key.begin(),f.key.end());
            face_keys.push_back(f);
        };
        for (std::size_t k=0;k<index_count;k+=4) {
            std::array<std::uint32_t,4> id{{tetra_indices[k],
                tetra_indices[k+1],tetra_indices[k+2],tetra_indices[k+3]}};
            for (auto a:id) if (a>=vertex_count) {
                clear(); return status_=SoftbodyStatus::invalid_input;
            }
            auto unique=id;
            std::sort(unique.begin(),unique.end());
            if (std::adjacent_find(unique.begin(),unique.end())!=unique.end()) {
                clear(); return status_=SoftbodyStatus::invalid_input;
            }
            const auto a=softbody_detail::vec(rest_[id[0]]);
            const auto b=softbody_detail::vec(rest_[id[1]]);
            const auto c=softbody_detail::vec(rest_[id[2]]);
            const auto d=softbody_detail::vec(rest_[id[3]]);
            double volume=softbody_detail::dot(
                softbody_detail::cross(b-a,c-a),d-a)/6.0;
            const double max_edge=std::max({softbody_detail::norm(b-a),
                softbody_detail::norm(c-a),softbody_detail::norm(d-a),
                softbody_detail::norm(c-b),softbody_detail::norm(d-b),
                softbody_detail::norm(d-c)});
            if (!std::isfinite(volume) ||
                std::abs(volume)<=1e-12*max_edge*max_edge*max_edge) {
                clear(); return status_=SoftbodyStatus::invalid_input;
            }
            if (volume<0) { std::swap(id[2],id[3]); volume=-volume; }
            tets_.push_back({id,volume,0.0});
            for (int p=0;p<4;++p)
                for (int q=p+1;q<4;++q) add_edge(id[p],id[q]);
            add_face(id[0],id[1],id[2],id[3]);
            add_face(id[0],id[3],id[1],id[2]);
            add_face(id[0],id[2],id[3],id[1]);
            add_face(id[1],id[3],id[2],id[0]);
        }
        std::sort(edge_keys.begin(),edge_keys.end(),[](EdgeKey a,EdgeKey b){
            return a.a<b.a || (a.a==b.a && a.b<b.b);
        });
        edge_keys.erase(std::unique(edge_keys.begin(),edge_keys.end(),
            [](EdgeKey a,EdgeKey b){return a.a==b.a && a.b==b.b;}),
            edge_keys.end());
        edges_.reserve(edge_keys.size());
        for (const auto& e:edge_keys) {
            const double length=softbody_detail::norm(
                softbody_detail::vec(rest_[e.a])-
                softbody_detail::vec(rest_[e.b]));
            if (!(length>0.0) || !std::isfinite(length)) {
                clear(); return status_=SoftbodyStatus::invalid_input;
            }
            edges_.push_back({e.a,e.b,length,0.0});
        }
        std::sort(face_keys.begin(),face_keys.end(),
            [](const FaceKey& a,const FaceKey& b){return a.key<b.key;});
        boundary_.reserve(face_keys.size());
        for (std::size_t i=0;i<face_keys.size();) {
            std::size_t j=i+1;
            while (j<face_keys.size() && face_keys[j].key==face_keys[i].key) ++j;
            if (j==i+1) {
                boundary_.push_back(face_keys[i].face);
            } else {
                if (j!=i+2) {
                    clear(); return status_=SoftbodyStatus::invalid_input;
                }
                // A valid interior face has opposite outward windings. Equal
                // windings identify overlapping neighbors, including duplicate
                // tetrahedra regardless of their input vertex permutations.
                const auto& a=face_keys[i].face;
                const auto& b=face_keys[i+1].face;
                std::size_t corner=0;
                while (b[corner]!=a[0]) ++corner;
                if (b[(corner+1)%3]==a[1]) {
                    clear(); return status_=SoftbodyStatus::invalid_input;
                }
            }
            i=j;
        }
        ready_=true;
        return status_=SoftbodyStatus::ok;
    }

    SoftbodyStatus set_compliance(T edge,T volume) noexcept {
        if (!std::isfinite(edge) || !std::isfinite(volume) ||
            edge<T{} || volume<T{}) return status_=SoftbodyStatus::invalid_input;
        edge_compliance_=edge;
        volume_compliance_=volume;
        return status_=SoftbodyStatus::ok;
    }
    SoftbodyStatus set_gravity(Point g) noexcept {
        if (!softbody_detail::finite(g)) return status_=SoftbodyStatus::invalid_input;
        gravity_=g;
        return status_=SoftbodyStatus::ok;
    }
    // Damping has units 1/s, applied as exp(-damping*substep_dt).
    SoftbodyStatus set_damping(T damping) noexcept {
        if (!std::isfinite(damping) || damping<T{})
            return status_=SoftbodyStatus::invalid_input;
        damping_=damping;
        return status_=SoftbodyStatus::ok;
    }
    SoftbodyStatus set_spheres(const Sphere* spheres,std::size_t count) {
        if (count && !spheres) return status_=SoftbodyStatus::invalid_input;
        for (std::size_t i=0;i<count;++i)
            if (!softbody_detail::finite(spheres[i].center) ||
                !std::isfinite(spheres[i].radius) ||
                !std::isfinite(spheres[i].skin) ||
                !(spheres[i].radius>T{}) || spheres[i].skin<T{} ||
                !std::isfinite(static_cast<double>(spheres[i].radius)+
                               static_cast<double>(spheres[i].skin)) ||
                static_cast<double>(spheres[i].radius)+
                    static_cast<double>(spheres[i].skin)>
                    std::sqrt(std::numeric_limits<double>::max()))
                return status_=SoftbodyStatus::invalid_input;
        if (count) spheres_.assign(spheres,spheres+count);
        else spheres_.clear();
        return status_=SoftbodyStatus::ok;
    }
    SoftbodyStatus set_spheres(const std::vector<Sphere>& spheres) {
        return set_spheres(spheres.data(),spheres.size());
    }
    SoftbodyStatus set_fixed(std::uint32_t index,Point target) {
        if (!ready_) return status_=SoftbodyStatus::not_prepared;
        if (index>=x_.size() || !softbody_detail::finite(target))
            return status_=SoftbodyStatus::invalid_input;
        inv_mass_[index]=T{};
        fixed_target_[index]=target;
        x_[index]=target;
        v_[index]=Point{};
        return status_=SoftbodyStatus::ok;
    }
    SoftbodyStatus set_inverse_mass(std::uint32_t index,T inverse_mass) {
        if (!ready_) return status_=SoftbodyStatus::not_prepared;
        if (index>=x_.size() || !std::isfinite(inverse_mass) ||
            inverse_mass<T{}) return status_=SoftbodyStatus::invalid_input;
        inv_mass_[index]=inverse_mass;
        if (inverse_mass==T{}) {
            fixed_target_[index]=x_[index];
            v_[index]=Point{};
        }
        return status_=SoftbodyStatus::ok;
    }
    SoftbodyStatus set_position(std::uint32_t index,Point position,
                                Point velocity=Point{}) {
        if (!ready_) return status_=SoftbodyStatus::not_prepared;
        if (index>=x_.size() || !softbody_detail::finite(position) ||
            !softbody_detail::finite(velocity))
            return status_=SoftbodyStatus::invalid_input;
        x_[index]=position;
        v_[index]=velocity;
        if (inv_mass_[index]==T{}) fixed_target_[index]=position;
        return status_=SoftbodyStatus::ok;
    }

    SoftbodyStatus step(T dt,unsigned substeps=4,unsigned iterations=8) {
        if (!ready_) return status_=SoftbodyStatus::not_prepared;
        if (!std::isfinite(dt) || !(dt>T{}) || !substeps || !iterations ||
            !std::isfinite(dt/static_cast<T>(substeps)) ||
            !(dt/static_cast<T>(substeps)>T{}))
            return status_=SoftbodyStatus::invalid_input;
        x_saved_=x_;
        v_saved_=v_;
        const double h=static_cast<double>(dt)/substeps;
        const double ae=static_cast<double>(edge_compliance_)/(h*h);
        const double av=static_cast<double>(volume_compliance_)/(h*h);
        const double damping=std::exp(-static_cast<double>(damping_)*h);
        if (!std::isfinite(ae) || !std::isfinite(av))
            return status_=SoftbodyStatus::invalid_input;
        // Resolve pre-existing penetration before capturing the substep's
        // previous positions. Otherwise overlap correction is interpreted as
        // a very large impact velocity (penetration / substep_dt).
        bool initial_contact=false;
        for (const auto& sphere:spheres_)
            initial_contact=project_sphere(sphere) || initial_contact;
        if (initial_contact) {
            for (auto& e:edges_) e.lambda=0.0;
            for (auto& t:tets_) t.lambda=0.0;
            for (unsigned iter=0;iter<iterations;++iter) {
                for (auto& e:edges_) project_edge(e,ae);
                for (auto& t:tets_) project_volume(t,av);
                for (const auto& sphere:spheres_) project_sphere(sphere);
            }
        }
        for (unsigned s=0;s<substeps;++s) {
            x_prev_=x_;
            for (std::size_t i=0;i<x_.size();++i) {
                if (inv_mass_[i]==T{}) {
                    x_[i]=fixed_target_[i];
                    continue;
                }
                const auto velocity=softbody_detail::vec(v_[i])+
                    softbody_detail::vec(gravity_)*h;
                v_[i]=softbody_detail::mesh<T>(velocity);
                x_[i]=softbody_detail::mesh<T>(
                    softbody_detail::vec(x_[i])+velocity*h);
            }
            for (auto& e:edges_) e.lambda=0.0;
            for (auto& t:tets_) t.lambda=0.0;
            for (unsigned iter=0;iter<iterations;++iter) {
                for (auto& e:edges_) project_edge(e,ae);
                for (auto& t:tets_) project_volume(t,av);
                for (const auto& sphere:spheres_) project_sphere(sphere);
            }
            for (std::size_t i=0;i<x_.size();++i) {
                const auto velocity=(softbody_detail::vec(x_[i])-
                                     softbody_detail::vec(x_prev_[i]))/h;
                v_[i]=softbody_detail::mesh<T>(
                    velocity*(inv_mass_[i]>T{} ? damping : 1.0));
                if (!softbody_detail::finite(x_[i]) ||
                    !softbody_detail::finite(v_[i])) {
                    x_=x_saved_; v_=v_saved_;
                    return status_=SoftbodyStatus::numerical_failure;
                }
            }
        }
        return status_=SoftbodyStatus::ok;
    }

    const std::vector<Point>& positions() const noexcept { return x_; }
    const std::vector<Point>& velocities() const noexcept { return v_; }
    const std::vector<T>& inverse_masses() const noexcept { return inv_mass_; }
    std::size_t vertex_count() const noexcept { return x_.size(); }
    std::size_t tetrahedron_count() const noexcept { return tets_.size(); }
    std::size_t edge_count() const noexcept { return edges_.size(); }
    std::size_t boundary_face_count() const noexcept { return boundary_.size(); }
    SoftbodyStatus status() const noexcept { return status_; }
    void reset_to_rest() noexcept {
        x_=rest_;
        std::fill(v_.begin(),v_.end(),Point{});
        for (std::size_t i=0;i<x_.size();++i)
            if (inv_mass_[i]==T{}) x_[i]=fixed_target_[i];
    }

private:
    struct Edge {
        std::uint32_t a,b;
        double rest_length,lambda;
    };
    struct Tet {
        std::array<std::uint32_t,4> id;
        double rest_volume,lambda;
    };
    void clear() {
        ready_=false;
        rest_.clear(); x_.clear(); x_prev_.clear(); x_saved_.clear();
        v_.clear(); v_saved_.clear(); fixed_target_.clear(); inv_mass_.clear();
        edges_.clear(); tets_.clear(); boundary_.clear(); spheres_.clear();
    }
    void project_edge(Edge& e,double alpha) noexcept {
        const double wa=inv_mass_[e.a],wb=inv_mass_[e.b];
        if (wa+wb==0.0) return;
        const auto a=softbody_detail::vec(x_[e.a]);
        const auto b=softbody_detail::vec(x_[e.b]);
        const auto delta=a-b;
        const double length=softbody_detail::norm(delta);
        if (!std::isfinite(length)) return;
        const auto direction=length>1e-30 ? delta/length :
            (softbody_detail::vec(rest_[e.a])-
             softbody_detail::vec(rest_[e.b]))/e.rest_length;
        const double dl=(-(length-e.rest_length)-alpha*e.lambda)/
                        (wa+wb+alpha);
        e.lambda+=dl;
        const auto correction=direction*dl;
        x_[e.a]=softbody_detail::mesh<T>(a+correction*wa);
        x_[e.b]=softbody_detail::mesh<T>(b-correction*wb);
    }
    void project_volume(Tet& t,double alpha) noexcept {
        using namespace softbody_detail;
        const auto& id=t.id;
        const V a=vec(x_[id[0]]),b=vec(x_[id[1]]);
        const V c=vec(x_[id[2]]),d=vec(x_[id[3]]);
        const V gb=cross(c-a,d-a)/6.0;
        const V gc=cross(d-a,b-a)/6.0;
        const V gd=cross(b-a,c-a)/6.0;
        const V ga=(gb+gc+gd)*-1.0;
        const V g[4]={ga,gb,gc,gd};
        const V p[4]={a,b,c,d};
        double denom=alpha;
        for (int i=0;i<4;++i)
            denom+=static_cast<double>(inv_mass_[id[i]])*dot(g[i],g[i]);
        if (!(denom>1e-30) || !std::isfinite(denom)) return;
        const double volume=dot(cross(b-a,c-a),d-a)/6.0;
        const double dl=(-(volume-t.rest_volume)-alpha*t.lambda)/denom;
        t.lambda+=dl;
        for (int i=0;i<4;++i)
            if (inv_mass_[id[i]]>T{})
                x_[id[i]]=mesh<T>(p[i]+g[i]*(dl*inv_mass_[id[i]]));
    }
    bool project_sphere(const Sphere& sphere) noexcept {
        using namespace softbody_detail;
        const V center=vec(sphere.center);
        const double radius=static_cast<double>(sphere.radius)+sphere.skin;
        const double radius2=radius*radius;
        bool touched=false;
        for (const auto& face:boundary_) {
            const auto a=vec(x_[face[0]]),b=vec(x_[face[1]]);
            const auto c=vec(x_[face[2]]);
            // Cheap face AABB rejection before closest-point evaluation.
            if (center.x< std::min({a.x,b.x,c.x})-radius ||
                center.x> std::max({a.x,b.x,c.x})+radius ||
                center.y< std::min({a.y,b.y,c.y})-radius ||
                center.y> std::max({a.y,b.y,c.y})+radius ||
                center.z< std::min({a.z,b.z,c.z})-radius ||
                center.z> std::max({a.z,b.z,c.z})+radius) continue;
            const auto q=closest(center,a,b,c);
            const V delta=q.point-center;
            const double distance2=dot(delta,delta);
            if (!(distance2<radius2)) continue;
            const double distance=std::sqrt(std::max(0.0,distance2));
            V normal;
            if (distance>1e-12) normal=delta/distance;
            else {
                normal=cross(b-a,c-a);
                const double n=norm(normal);
                normal=n>1e-20 ? normal/n : V{0,1,0};
            }
            const double bary[3]={q.a,q.b,q.c};
            double denom=0.0;
            for (int i=0;i<3;++i)
                denom+=static_cast<double>(inv_mass_[face[i]])*bary[i]*bary[i];
            if (!(denom>1e-30)) continue;
            touched=true;
            const double correction=(radius-distance)/denom;
            for (int i=0;i<3;++i) {
                const double w=static_cast<double>(inv_mass_[face[i]])*bary[i];
                if (w>0.0)
                    x_[face[i]]=mesh<T>(vec(x_[face[i]])+
                                          normal*(correction*w));
            }
        }
        return touched;
    }

    bool ready_=false;
    SoftbodyStatus status_=SoftbodyStatus::not_prepared;
    T edge_compliance_=T(0),volume_compliance_=T(0),damping_=T(0.1);
    Point gravity_{T(0),T(-9.81),T(0)};
    std::vector<Point> rest_,x_,x_prev_,x_saved_,v_,v_saved_,fixed_target_;
    std::vector<T> inv_mass_;
    std::vector<Edge> edges_;
    std::vector<Tet> tets_;
    std::vector<std::array<std::uint32_t,3>> boundary_;
    std::vector<Sphere> spheres_;
};

} // namespace deform3d
