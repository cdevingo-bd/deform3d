#include <deform3d/mujoco_contact.hpp>
#include <stdexcept>
#define CHECK(x) do { if (!(x)) throw std::runtime_error("check failed: " #x); } while(false)
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

int main() {
  const char* xml=R"(<mujoco>
    <option timestep=".001" gravity="0 0 -9.81" cone="elliptic"/>
    <worldbody>
      <geom name="floor" type="plane" size="1 1 .1" condim="6" friction=".7 .1 .01"/>
      <body name="object" pos="0 0 .049" euler="0 0 30">
        <freejoint/><geom name="box" type="box" size=".06 .04 .05" mass="1" condim="6" friction=".7 .1 .01"/>
      </body>
    </worldbody></mujoco>)";
  std::unique_ptr<mjSpec,decltype(&mj_deleteSpec)> spec(mj_parseXMLString(xml,nullptr,nullptr,0),mj_deleteSpec);
  CHECK(spec);
  std::unique_ptr<mjModel,decltype(&mj_deleteModel)> m(mj_compile(spec.get(),nullptr),mj_deleteModel);
  CHECK(m);
  std::unique_ptr<mjData,decltype(&mj_deleteData)> d(mj_makeData(m.get()),mj_deleteData);
  d->qvel[0]=.2;d->qvel[1]=-.1;d->qvel[5]=.5;
  mj_forward(m.get(),d.get());mj_rnePostConstraint(m.get(),d.get());
  using namespace deform3d;
  MujocoContactWrench floor,box;
  const Vec3<mjtNum> origin{.1,-.2,.3};
  CHECK(mujoco_contact_wrench(m.get(),d.get(),0,origin,floor)==Status::Ok);
  CHECK(mujoco_contact_wrench(m.get(),d.get(),1,origin,box)==Status::Ok);
  CHECK(box.contacts>0 && box.force.z>0 && box.force.x<0 && box.force.y>0);
  auto norm=[](Vec3<mjtNum> v){return std::sqrt(dot(v,v));};
  CHECK(norm(box.force+floor.force)<1e-10);
  CHECK(norm(box.torque+floor.torque)<1e-10);
  const int body=mj_name2id(m.get(),mjOBJ_BODY,"object");
  const mjtNum* raw=d->cfrc_ext+6*body;
  const mjtNum* com=d->subtree_com+3*m->body_rootid[body];
  Vec3<mjtNum> reference_force{raw[3],raw[4],raw[5]};
  Vec3<mjtNum> reference_torque{raw[0],raw[1],raw[2]};
  reference_torque+=cross(Vec3<mjtNum>{com[0],com[1],com[2]}-origin,reference_force);
  CHECK(norm(box.force-reference_force)<1e-9);
  CHECK(norm(box.torque-reference_torque)<1e-9);
  const auto saved=box;
  CHECK(mujoco_contact_wrench(m.get(),d.get(),-1,origin,box)==Status::InvalidArgument);
  CHECK(mujoco_contact_wrench(m.get(),d.get(),1,{std::numeric_limits<mjtNum>::quiet_NaN(),0,0},box)==Status::NonFinite);
  CHECK(norm(box.force-saved.force)==0 && norm(box.torque-saved.torque)==0);
  d->qpos[2]=.5;mj_forward(m.get(),d.get());
  CHECK(mujoco_contact_wrench(m.get(),d.get(),1,origin,box)==Status::Ok);
  CHECK(box.contacts==0 && norm(box.force)==0 && norm(box.torque)==0);
  std::cout<<"MuJoCo contact wrench agrees with postconstraint external wrench and Newton's third law\n";
}
