"""Compare one dynamics step from identical saved CPU integration inputs.

The checkpoint collision audit must have reproduced a separately converged CPU
reference. Inputs include warm start, external wrenches, mocap and controls.
Each case starts over from the same input, without feedback from another case.
This is a local error diagnostic, not full-trajectory parity or a benchmark.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'examples')]
from mjwarp_embedding import refresh_flex_positions
from deform3d_warp.mujoco_contact import flex_contact_policy
from deform3d_warp.mujoco_contact_normals import stable_flex_normals_policy
from deform3d_warp.mujoco_metric import CooperativeMetricSolver
from deform3d_warp.mujoco_newton import ParallelNewtonSolver
from deform3d_warp.mujoco_ordering import CanonicalContactConstraints
from deform3d_warp.mujoco_assembly import OwnedContactAssembly
from deform3d_warp.mujoco_material import OwnedFlexOperators
from deform3d_warp.mujoco_wrench import OwnedWrenchProjection
from deform3d_warp.mujoco_effective import OwnedEffectiveMetric


def pad_forces_cpu(model, data, pads):
    result, local = np.zeros(2), np.empty(6)
    for i, contact in enumerate(data.contact):
        for j, pad in enumerate(pads):
            if pad in contact.geom:
                mujoco.mj_contactForce(model, data, i, local)
                result[j] += (-1 if contact.geom[0] == pad else 1)*(contact.frame.reshape(3, 3).T@local[:3])[1]
    return result


def run(args):
    audit = json.loads(args.checkpoints.read_text())
    scene = Path(audit['scene'])
    cm = mujoco.MjModel.from_xml_path(str(scene))
    cm.opt.ccd_tolerance = audit['cpu_ccd_tolerance']
    pads = [mujoco.mj_name2id(cm, mujoco.mjtObj.mjOBJ_GEOM, name+'_geom')
            for name in ('front_pad', 'back_pad')]
    if min(pads) < 0:
        raise ValueError('named gripper pads are required')
    snapshots = np.load(args.checkpoints.with_suffix('.npz'), allow_pickle=False)
    steps, states = snapshots['steps'], snapshots['integration_states']
    if len(steps) != len(audit['results']) or len(states) != len(steps):
        raise ValueError('different snapshot layout')
    selected = set(args.steps or steps.tolist())
    if not selected <= set(steps.tolist()):
        raise ValueError('requested checkpoint not found')
    specification = mujoco.mjtState(int(snapshots['specification']))
    result = []
    with wp.ScopedDevice('cuda:0'):
        model = mjw.put_model(cm)
        model.opt.ccd_tolerance.fill_(audit['gpu_ccd_tolerance'])
        model.opt.deterministic = mjw.DeterminismType.ALL
        for step, state, item in zip(steps, states, audit['results']):
            if step != item['step'] or hashlib.sha256(state.tobytes()).hexdigest() != item['state_sha256']:
                raise ValueError('checkpoint checksum mismatch')
            if step not in selected:
                continue
            cpu = mujoco.MjData(cm)
            common = state.astype(np.float32).astype(np.float64)
            mujoco.mj_setState(cm, cpu, common, specification)
            mujoco.mj_step(cm, cpu)
            cpu_force = pad_forces_cpu(cm, cpu, pads)
            mujoco.mj_kinematics(cm, cpu)
            mujoco.mj_flex(cm, cpu)
            if np.any(cpu.warning.number):
                raise RuntimeError('CPU warning at checkpoint')
            cases = {}
            for stable in (False, True):
                initial = mujoco.MjData(cm)
                mujoco.mj_setState(cm, initial, common, specification)
                data = mjw.put_data(cm, initial)
                owners = dict(metric=CooperativeMetricSolver(model, 1),
                    newton=ParallelNewtonSolver(model, 1, precision='float32', tile_size=32, diagonal='register'),
                    ordering=CanonicalContactConstraints(model, data),
                    assembly=OwnedContactAssembly(model, data, solver_reductions=True),
                    material=OwnedFlexOperators(model, data),
                    wrench=OwnedWrenchProjection(model, data),
                    effective=OwnedEffectiveMetric(model, data))
                with ExitStack() as policies:
                    policies.enter_context(flex_contact_policy(model))
                    if stable:
                        policies.enter_context(stable_flex_normals_policy(model))
                    for owner in owners.values():
                        policies.enter_context(owner.capture_policy())
                    mjw.step(model, data)
                    refresh_flex_positions(model, data)
                status = {name: owner.status.numpy().tolist() for name, owner in owners.items()}
                if any(any(v) for v in status.values()) or np.any(data.overflow.numpy()):
                    raise RuntimeError('GPU status failure at checkpoint: '+str(status))
                count = int(data.nacon.numpy()[0])
                ids = wp.array(np.arange(data.naconmax, dtype=np.int32))
                forces = wp.zeros(data.naconmax, dtype=wp.spatial_vector)
                mjw.contact_force(model, data, ids, True, forces)
                world_forces = forces.numpy()[:count, :3].astype(np.float64)
                geom = data.contact.geom.numpy()[:count]
                net = np.asarray([sum(world_forces[geom[:, 1] == pad, 1])-
                                  sum(world_forces[geom[:, 0] == pad, 1]) for pad in pads])
                position_delta = np.linalg.norm(data.flexvert_xpos.numpy()[0].astype(np.float64)-cpu.flexvert_xpos,axis=1)
                velocity_delta = data.qvel.numpy()[0].astype(np.float64)-cpu.qvel
                cases[str(stable)] = dict(stable_contact_normals=stable,
                    max_flex_position_delta_m=float(position_delta.max()),
                    rms_flex_position_delta_m=float(np.sqrt(np.mean(position_delta**2))),
                    max_generalized_velocity_delta=float(np.abs(velocity_delta).max()),
                    signed_pad_forces_N=net.tolist(), signed_pad_force_delta_N=(net-cpu_force).tolist(),
                    contacts=count, solver_iterations=data.solver_niter.numpy().tolist(),
                    newton_max_relative_residual=owners['newton'].max_relative_residual.numpy().tolist(),
                    status=status, overflow=data.overflow.numpy().tolist())
            row = dict(step=int(step), time_s=float(step*cm.opt.timestep), cpu_contacts=int(cpu.ncon),
                       cpu_signed_pad_forces_N=cpu_force.tolist(), cases=cases)
            result.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(dict(checkpoint_report=str(args.checkpoints),
        checkpoint_report_sha256=hashlib.sha256(args.checkpoints.read_bytes()).hexdigest(),
        scope='One dynamics step per independent float32-quantized integration state; all accepted 0.11 operators; '
              'controls and loads already applied identically; GPU geometry is refreshed after integration; '
              'generalized velocities mix translational and angular units; no performance or trajectory claim',
        results=result), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', type=Path)
    parser.add_argument('--steps', type=int, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
