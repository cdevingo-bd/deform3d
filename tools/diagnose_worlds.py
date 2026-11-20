"""Locate first disagreement between identical GPU worlds, stage by stage.

Each checkpoint starts afresh from one CPU state broadcast to all worlds.
This is a diagnostic with readbacks, not a throughput benchmark.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import forward
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'tools')]
from record_bag import drive
from deform3d_warp.mujoco_contact import flex_contact_policy
from deform3d_warp.mujoco_metric import CooperativeMetricSolver


def spread(value):
    value = np.asarray(value)
    delta = value[1:].astype(np.float64) - value[0].astype(np.float64)
    return {'different_entries': int(np.count_nonzero(delta)),
            'max_abs_delta': float(np.max(np.abs(delta), initial=0)),
            'rms_delta': float(np.sqrt(np.mean(delta*delta))) if delta.size else 0.,
            'reference_max_abs': float(np.max(np.abs(value[0]), initial=0))}


def run(args):
    cpu_model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    cpu_model.opt.ccd_tolerance = args.cpu_ccd_tolerance
    cpu = mujoco.MjData(cpu_model)
    mujoco.mj_forward(cpu_model, cpu)
    requested = sorted(set(round(t/cpu_model.opt.timestep) for t in args.times))
    report = {'scene': str(args.scene), 'worlds': args.worlds,
              'deterministic_atomics': args.deterministic_atomics,
              'cpu_ccd_tolerance': args.cpu_ccd_tolerance,
              'gpu_ccd_tolerance': args.gpu_ccd_tolerance,
              'parallel_newton': args.parallel_newton,
              'bounded_determinism': args.bounded_determinism,
              'canonical_constraints': args.canonical_constraints,
              'scope': 'independent CPU checkpoints, identical GPU worlds, uncaptured stage diagnostics',
              'constraint_row_note': 'efc arrays are compared in raw row order; different ordering is not itself a physical discrepancy',
              'checkpoints': []}
    with wp.ScopedDevice('cuda:0'):
        model = mjw.put_model(cpu_model)
        model.opt.ccd_tolerance.fill_(args.gpu_ccd_tolerance)
        model.opt.deterministic = (mjw.DeterminismType.ALL if args.deterministic_atomics else
                                  mjw.DeterminismType.CONTACTS | mjw.DeterminismType.CONSTRAINT)
        metric = CooperativeMetricSolver(model, args.worlds)
        newton, budget = None, None
        if args.parallel_newton:
            from deform3d_warp.mujoco_newton import ParallelNewtonSolver
            newton = ParallelNewtonSolver(model, args.worlds, precision='float32', tile_size=32)
        if args.bounded_determinism:
            from deform3d_warp.mujoco_determinism import SparseDeterminismBudget
            budget = SparseDeterminismBudget(model)
        for step in range(requested[-1]+1):
            drive(cpu_model, cpu)
            if step in requested:
                data = mjw.put_data(cpu_model, cpu, nworld=args.worlds)
                ordering = None
                if args.canonical_constraints:
                    from deform3d_warp.mujoco_ordering import CanonicalContactConstraints
                    ordering = CanonicalContactConstraints(model,data)
                sample = {'time_s': float(cpu.time), 'stages': {}}
                def snapshot(name, fields):
                    def get(field):
                        value = data
                        for part in field.split('.'):
                            value = getattr(value, part)
                        return value
                    sample['stages'][name] = {field: spread(get(field).numpy()) for field in fields}
                snapshot('input', ('qpos', 'qvel', 'qacc_warmstart'))
                with ExitStack() as setup:
                    setup.enter_context(flex_contact_policy(model))
                    setup.enter_context(metric.capture_policy())
                    if newton:
                        setup.enter_context(newton.capture_policy())
                    if budget:
                        setup.enter_context(budget.capture_policy())
                    if ordering:
                        setup.enter_context(ordering.capture_policy())
                    mjw.fwd_position(model, data, factorize=False)
                    snapshot('position', ('flexvert_xpos', 'M', 'efc.J', 'efc.D', 'efc.pos'))
                    counts, adr, cols, values = (getattr(data.efc,f).numpy() for f in
                                               ('J_rownnz','J_rowadr','J_colind','J'))
                    logical = np.zeros((args.worlds,data.njmax,model.nv),np.float32)
                    for w, n in enumerate(data.nefc.numpy()):
                        for row in range(min(n,data.njmax)):
                            start, count = adr[w,row], counts[w,row]
                            logical[w,row,cols[w,0,start:start+count]] = values[w,0,start:start+count]
                    sample['stages']['position']['logical_J'] = spread(logical)
                    mjw.fwd_velocity(model, data)
                    snapshot('velocity', ('qfrc_passive', 'qfrc_bias', 'efc.aref'))
                    mjw.fwd_actuation(model, data)
                    mjw.fwd_acceleration(model, data, factorize=True)
                    snapshot('acceleration', ('qfrc_smooth', 'efm_c', 'efm_K_val', 'efm_diag', 'qacc_smooth'))
                    mjw.solve(model, data)
                    snapshot('constraints', ('qacc', 'qfrc_constraint', 'solver_niter'))
                    forward._launch_discrete_free_gyro(model, data, data.qacc, data.qacc, False)
                    mjw.discrete(model, data)
                    snapshot('integrated', ('qpos', 'qvel'))
                sample['metric_status'] = metric.status.numpy().tolist()
                sample['newton_status'] = newton.status.numpy().tolist() if newton else None
                sample['ordering_status'] = ordering.status.numpy().tolist() if ordering else None
                sample['overflow'] = data.overflow.numpy().tolist()
                report['checkpoints'].append(sample)
                print(json.dumps(sample), flush=True)
            if step < requested[-1]:
                mujoco.mj_step(cpu_model, cpu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=ROOT/'examples/mujoco_bag_fast.xml')
    parser.add_argument('--times', type=float, nargs='+', default=[0, .2, 1.4])
    parser.add_argument('--worlds', type=int, default=2)
    parser.add_argument('--deterministic-atomics', action='store_true')
    parser.add_argument('--parallel-newton', action='store_true')
    parser.add_argument('--bounded-determinism', action='store_true')
    parser.add_argument('--canonical-constraints', action='store_true')
    parser.add_argument('--cpu-ccd-tolerance', type=float, default=1e-9)
    parser.add_argument('--gpu-ccd-tolerance', type=float, default=1e-8)
    parser.add_argument('--output', type=Path, default=ROOT/'results/world_stage_diagnostics.json')
    args = parser.parse_args()
    if args.worlds < 2 or not args.times or not np.isfinite(args.times).all() or min(args.times) < 0:
        parser.error('requires at least two worlds and finite nonnegative checkpoint times')
    if not np.isfinite([args.cpu_ccd_tolerance,args.gpu_ccd_tolerance]).all() or min(args.cpu_ccd_tolerance,args.gpu_ccd_tolerance) <= 0:
        parser.error('CCD tolerances must be finite and positive')
    run(args)
