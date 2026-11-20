"""Compare independent CPU/CUDA steps starting from identical CPU checkpoints.

This isolates local backend disagreement from accumulated trajectory drift. No
physics setting is changed. Diagnostic copies/compilation are not timed.
"""
import argparse
import json
from pathlib import Path
import sys
from contextlib import ExitStack

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'tools'), str(ROOT / 'examples')]
from record_bag import drive
from mjwarp_embedding import refresh_flex_positions


def run(args):
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    trajectory = mujoco.MjData(model)
    mujoco.mj_forward(model, trajectory)
    checkpoints = sorted(set(round(t / model.opt.timestep) for t in args.times))
    report = {'scene': str(args.scene), 'mujoco': mujoco.__version__,
              'kind': 'independent one-step comparisons at CPU checkpoints',
              'timestep_s': model.opt.timestep, 'samples': []}
    with wp.ScopedDevice(args.device):
        gpu_model = mjw.put_model(model)
        metric = None
        if args.cooperative_metric:
            from deform3d_warp.mujoco_metric import CooperativeMetricSolver
            metric = CooperativeMetricSolver(gpu_model)
        newton = None
        if args.parallel_newton:
            from deform3d_warp.mujoco_newton import ParallelNewtonSolver
            newton = ParallelNewtonSolver(gpu_model, precision=args.newton_precision, tile_size=args.newton_tile)
        for step in range(checkpoints[-1] + 1):
            drive(model, trajectory)
            if step in checkpoints:
                cpu = mujoco.MjData(model)
                rounded = mujoco.MjData(model)
                mujoco.mj_copyData(cpu, model, trajectory)
                mujoco.mj_copyData(rounded, model, trajectory)
                # A double-precision solve with float32-rounded input separates
                # storage quantization from different backend implementations.
                for name in ('qpos', 'qvel', 'qacc_warmstart', 'mocap_pos', 'mocap_quat'):
                    getattr(rounded, name)[:] = getattr(cpu, name).astype(np.float32)
                gpu = mjw.put_data(model, rounded)
                if args.contact_policy:
                    from deform3d_warp.mujoco_contact import flex_contact_policy
                with ExitStack() as stack:
                    if args.contact_policy:
                        stack.enter_context(flex_contact_policy(gpu_model))
                    if metric:
                        stack.enter_context(metric.capture_policy())
                    if newton:
                        newton.status.zero_()
                        newton.max_relative_residual.zero_()
                        stack.enter_context(newton.capture_policy())
                    mjw.step(gpu_model, gpu)
                refresh_flex_positions(gpu_model, gpu)
                mujoco.mj_step(model, cpu)
                mujoco.mj_step(model, rounded)
                for data in (cpu, rounded):
                    mujoco.mj_kinematics(model, data)
                    mujoco.mj_flex(model, data)
                xyz = gpu.flexvert_xpos.numpy()[0]
                point_error = np.linalg.norm(xyz - rounded.flexvert_xpos, axis=1)
                sample = {'time_s': float(trajectory.time),
                          'max_flex_delta_m': float(point_error.max()),
                          'worst_vertex': int(point_error.argmax()),
                          'rounding_only_flex_delta_m': float(np.linalg.norm(
                              rounded.flexvert_xpos - cpu.flexvert_xpos, axis=1).max()),
                          'cpu_contacts': int(rounded.ncon),
                          'gpu_contacts': int(gpu.nacon.numpy()[0]),
                          'cpu_iterations': int(rounded.solver_niter.sum()),
                          'gpu_iterations': int(gpu.solver_niter.numpy()[0]),
                          'overflow': gpu.overflow.numpy().tolist()}
                if newton:
                    sample['newton'] = {'status': newton.status.numpy().tolist(),
                                        'max_relative_residual': newton.max_relative_residual.numpy().tolist()}
                if metric:
                    sample['metric'] = {'status': metric.status.numpy().tolist(),
                                        'iterations': metric.iterations.numpy().tolist(),
                                        'relative_residual': metric.relative_residual.numpy().tolist()}
                    if args.check_linear_system:
                        matrix = np.zeros((model.nv, model.nv), np.float64)
                        row, col, adr = (getattr(gpu_model, name).numpy() for name in
                                         ('M_mulm_rowadr', 'M_mulm_col', 'M_mulm_madr'))
                        count, krow, kcol = (getattr(gpu_model, name).numpy() for name in
                                            ('efm_K_rownnz', 'efm_K_rowadr', 'efm_K_colind'))
                        mass, stiffness = gpu.M.numpy()[0], gpu.efm_K_val.numpy()[0]
                        for r in range(model.nv):
                            np.add.at(matrix[r], col[row[r]:row[r+1]], mass[adr[row[r]:row[r+1]]])
                            sl = slice(krow[r], krow[r] + count[r])
                            np.add.at(matrix[r], kcol[sl], stiffness[sl])
                        matrix[np.diag_indices(model.nv)] += gpu.efm_diag.numpy()[0]
                        rhs = sum(getattr(gpu, name).numpy()[0].astype(np.float64) for name in
                                  ('qfrc_smooth', 'efm_c', 'efm_ca'))
                        x = gpu.qacc_smooth.numpy()[0].astype(np.float64)
                        actual = float(np.linalg.norm(matrix @ x - rhs) / max(np.linalg.norm(rhs), 1e-30))
                        exact = np.linalg.solve(matrix, rhs)
                        sample['metric']['independent_cpu_matrix_relative_residual'] = actual
                        sample['metric']['relative_solution_delta_from_numpy'] = float(
                            np.linalg.norm(x - exact) / max(np.linalg.norm(exact), 1e-30))
                        if abs(actual - sample['metric']['relative_residual'][0]) > 5e-7:
                            raise AssertionError('device residual disagrees with independent CPU matrix assembly')
                for field in ('qvel', 'qacc', 'qacc_smooth', 'qfrc_passive', 'qfrc_smooth', 'qfrc_constraint'):
                    a, b = gpu.__getattribute__(field).numpy()[0], getattr(rounded, field)
                    sample[field] = {'max_abs_delta': float(np.abs(a - b).max()),
                                     'rms_delta': float(np.sqrt(np.mean((a - b)**2))),
                                     'cpu_rms': float(np.sqrt(np.mean(b*b)))}
                if args.contacts:
                    fields = ('geom', 'flex', 'elem', 'vert', 'pos', 'dist', 'frame', 'efc_address')
                    sample['cpu_contact_details'] = [{name: np.asarray(getattr(c, name)).tolist()
                        for name in fields} for c in rounded.contact]
                    count = sample['gpu_contacts']
                    arrays = {name: getattr(gpu.contact, name).numpy()[:count] for name in fields}
                    sample['gpu_contact_details'] = [{name: arrays[name][i].tolist()
                        for name in fields} for i in range(count)]
                report['samples'].append(sample)
                print(json.dumps(sample), flush=True)
                del gpu
            if step < checkpoints[-1]:
                mujoco.mj_step(model, trajectory)
    report['cpu_warnings'] = trajectory.warning.number.tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=ROOT / 'examples/mujoco_bag_fast.xml')
    parser.add_argument('--times', type=float, nargs='+', default=[0, .1, .2, .4, .7, 1, 1.4, 1.7, 2, 2.5])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--contacts', action='store_true')
    parser.add_argument('--contact-policy', action='store_true')
    parser.add_argument('--cooperative-metric', action='store_true')
    parser.add_argument('--check-linear-system', action='store_true')
    parser.add_argument('--parallel-newton', action='store_true')
    parser.add_argument('--newton-precision', choices=['float32', 'float64'], default='float64')
    parser.add_argument('--newton-tile', type=int, choices=[16, 32], default=16)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/mjwarp_checkpoint_diagnostics.json')
    args = parser.parse_args()
    if not args.times or not np.isfinite(args.times).all() or min(args.times) < 0:
        parser.error('checkpoint times must be finite and nonnegative')
    run(args)
