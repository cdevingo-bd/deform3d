#!/usr/bin/env python3
"""Install the opt-in element hook into a separate pristine MuJoCo 4.0.1 tree.

Never edits the installed Python wheel. Compile with MUJOCO_BUILD_EXAMPLES=OFF,
MUJOCO_BUILD_SIMULATE=OFF, MUJOCO_BUILD_TESTS=OFF. Both native and deform3d paths
use the resulting binary; thread-local enable selects the constitutive kernel.
"""
import argparse
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def prepare(source):
    engine=source/'src/engine/engine_passive.c'
    text=engine.read_text()
    if 'deform3d_simplex_response' in text:
        raise RuntimeError('source already patched; use an unmodified 4.0.1 tree')
    # Refuse arbitrary versions and moved insertion points.
    version=(source/'include/mujoco/mujoco.h').read_text()
    if '#define mjVERSION_HEADER 4000001' not in version:
        raise RuntimeError('requires MuJoCo 4.0.1')
    original_hash=hashlib.sha256(engine.read_bytes()).hexdigest()
    if original_hash!='f5f2d8366552a113856570ec65794eeaec66499445172dfe81dd13d8c5b7a6f0':
        raise RuntimeError('unreviewed MuJoCo passive implementation')
    marker='//----------------------------- passive forces'
    declarations='''// Optional deform3d element response, enabled per model and thread.
extern int deform3d_simplex_enabled(const mjModel* m);
extern void deform3d_simplex_response(int dim, const mjtNum* x,
    const mjtNum* length, const mjtNum* rest, const mjtNum* speed,
    const mjtNum* packed, mjtNum dt, mjtNum damping, int spring, int damper,
    mjtNum* fs, mjtNum* fd);

'''
    text=text.replace(marker,declarations+marker,1)
    start=text.index('static void mj_flexPassiveStretch(')
    loop='  // compute forces element-by-element\n'
    at=text.index(loop,start)
    text=text[:at]+'  const int deform3d_use_core=deform3d_simplex_enabled(m);\n'+text[at:]
    site='    const int* vert = elem + (dim+1) * t;\n'
    at=text.index(site,start)+len(site)
    insertion='''
    if (deform3d_use_core) {
      mjtNum local[12], lengths[6], rest[6], speeds[6], fs[12], fd[12];
      for (int v=0; v<dim+1; ++v)
        for (int c=0; c<3; ++c) local[3*v+c]=xpos[3*vert[v]+c];
      for (int e=0; e<nedge; ++e) {
        int id=edgeelem[t*nedge+e];
        lengths[e]=deformed[id]; rest[e]=reference[id]; speeds[e]=vel[id];
      }
      deform3d_simplex_response(dim, local, lengths, rest, speeds,
          k+(dim==3?24:21)*t, m->opt.timestep, m->flex_damping[f],
          enbl_spring, enbl_damper, fs, fd);
      for (int v=0; v<dim+1; ++v) for (int c=0; c<3; ++c) {
        if (enbl_spring) frc[3*vert[v]+c]+=fs[3*v+c];
        if (kD) dmp[3*vert[v]+c]+=fd[3*v+c];
      }
      continue;
    }
'''
    text=text[:at]+insertion+text[at:]
    engine.write_text(text)
    exports=source/'mujoco.lds'
    exports.write_text(exports.read_text().replace('global: mj*; _mj*;',
                                                  'global: mj*; _mj*; deform3d_simplex_*;'))
    cmake=source/'CMakeLists.txt'
    with cmake.open('a') as f:
        f.write(f'\n# Optional deform3d constitutive backend\ntarget_sources(mujoco PRIVATE "{ROOT}/adapters/mujoco/simplex_hook.cpp")\n'
                f'target_include_directories(mujoco PRIVATE "{ROOT}/include")\n')
    print('original_engine_passive_sha256='+original_hash)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path)
    prepare(parser.parse_args().source)
