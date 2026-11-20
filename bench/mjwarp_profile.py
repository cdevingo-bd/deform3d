"""CUDA event stage profiling of captured MJWarp at an actual squeeze state."""
import argparse
import json
from pathlib import Path
import sys
from contextlib import ExitStack

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from mujoco_warp._src.warp_util import EventTracer, event_scope
from mujoco_warp._src import solver

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'python')]
from record_bag import drive
from deform3d_warp.mujoco_contact import flex_contact_policy


def run(args):
    m = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    m.opt.ccd_tolerance = args.cpu_ccd_tolerance
    cpu = mujoco.MjData(m)
    mujoco.mj_forward(m, cpu)
    for _ in range(round(args.time / m.opt.timestep)):
        drive(m, cpu)
        mujoco.mj_step(m, cpu)
    drive(m, cpu)
    with wp.ScopedDevice('cuda:0'):
        model = mjw.put_model(m)
        model.opt.ccd_tolerance.fill_(args.gpu_ccd_tolerance)
        if args.deterministic:
            model.opt.deterministic = mjw.DeterminismType.ALL
        if args.cholesky_threads:
            model.block_dim.update_gradient_cholesky_blocked = args.cholesky_threads
        if args.solver_iterations:
            model.opt.iterations = args.solver_iterations
        data = mjw.put_data(m, cpu,nworld=args.worlds)
        warm = mjw.put_data(m, cpu,nworld=args.worlds)
        metric = None
        newton = None
        assembly = None
        ordering = None
        material = None
        wrench = effective = None
        with ExitStack() as setup:
            if args.contact_policy:
                setup.enter_context(flex_contact_policy(model))
            if args.cooperative_metric:
                from deform3d_warp.mujoco_metric import CooperativeMetricSolver
                metric = CooperativeMetricSolver(model,args.worlds)
                setup.enter_context(metric.capture_policy())
            if args.parallel_newton:
                from deform3d_warp.mujoco_newton import ParallelNewtonSolver
                newton = ParallelNewtonSolver(model,args.worlds,precision=args.newton_precision,tile_size=args.newton_tile,
                                              diagonal=args.newton_diagonal,substitution=args.newton_substitution,
                                              panel=args.newton_panel,update_threads=args.newton_update_threads,
                                              conditional_factor=args.conditional_newton_factor,trailing=args.newton_trailing)
                setup.enter_context(newton.capture_policy())
            if args.canonical_constraints:
                from deform3d_warp.mujoco_ordering import CanonicalContactConstraints
                ordering = CanonicalContactConstraints(model,data)
                setup.enter_context(ordering.capture_policy())
            if args.owned_assembly:
                from deform3d_warp.mujoco_assembly import OwnedContactAssembly
                assembly = OwnedContactAssembly(model,data,solver_reductions=args.owned_solver_reductions)
                setup.enter_context(assembly.capture_policy())
            elif args.deterministic:
                from deform3d_warp.mujoco_determinism import SparseDeterminismBudget
                setup.enter_context(SparseDeterminismBudget(model).capture_policy())
            if args.owned_material:
                from deform3d_warp.mujoco_material import OwnedFlexOperators
                material = OwnedFlexOperators(model,data)
                setup.enter_context(material.capture_policy())
            if args.owned_wrench:
                from deform3d_warp.mujoco_wrench import OwnedWrenchProjection
                wrench=OwnedWrenchProjection(model,data)
                setup.enter_context(wrench.capture_policy())
            if args.owned_effective:
                from deform3d_warp.mujoco_effective import OwnedEffectiveMetric
                effective=OwnedEffectiveMetric(model,data)
                setup.enter_context(effective.capture_policy())
            if args.detail_solve:
                for name in ('_linesearch','_update_constraint','_update_gradient',
                             '_update_gradient_incremental','_cholesky_factorize_solve'):
                    original = getattr(solver,name)
                    setup.callback(setattr,solver,name,original)
                    setattr(solver,name,event_scope(original,name=name))
            mjw.step(model, warm)
            wp.synchronize_device('cuda:0')
            with EventTracer() as tracer:
                with wp.ScopedCapture() as capture:
                    mjw.step(model, data)
                traces = []
                iteration_counts = []
                for _ in range(args.samples):
                    wp.capture_launch(capture.graph)
                    wp.synchronize_device('cuda:0')
                    traces.append(tracer.trace())
                    iteration_counts.append(data.solver_niter.numpy().tolist())
    def flatten(tree, prefix=''):
        result = {}
        for name, (times, children) in tree.items():
            key = prefix + name
            result[key] = sum(times)
            result.update(flatten(children, key + '/'))
        return result
    flat = [flatten(t) for t in traces]
    report = {'scene': str(args.scene), 'start_s': args.time, 'steps': args.samples,
              'worlds': args.worlds, 'deterministic': args.deterministic,
              'cpu_ccd_tolerance': args.cpu_ccd_tolerance,'gpu_ccd_tolerance': args.gpu_ccd_tolerance,
              'owned_material': args.owned_material,
              'owned_wrench':bool(wrench),'owned_effective':bool(effective),
              'owned_wrench_status':wrench.status.numpy().tolist() if wrench else None,
              'owned_effective_status':effective.status.numpy().tolist() if effective else None,
              'owned_material_status': material.status.numpy().tolist() if material else None,
              'owned_contact_assembly': args.owned_assembly,
              'owned_solver_reductions': args.owned_solver_reductions,
              'owned_assembly_status': assembly.status.numpy().tolist() if assembly else None,
              'canonical_constraints': args.canonical_constraints,
              'constraint_ordering_status': ordering.status.numpy().tolist() if ordering else None,
              'scope': 'captured physics step, CUDA events per stage; includes event instrumentation overhead',
              'contact_policy': args.contact_policy, 'cooperative_metric': args.cooperative_metric,
              'cholesky_threads': model.block_dim.update_gradient_cholesky_blocked,
              'solver_iteration_counts': iteration_counts,
              'solver_iteration_limit': model.opt.iterations,
              'metric_status_last_step': metric.status.numpy().tolist() if metric else None,
              'parallel_newton': args.parallel_newton,
              'newton_precision': args.newton_precision,
              'newton_tile_size': args.newton_tile,
              'newton_diagonal': args.newton_diagonal,
              'newton_substitution': args.newton_substitution,
              'newton_panel': args.newton_panel,
              'newton_trailing': args.newton_trailing,
              'newton_update_threads': args.newton_update_threads,
              'conditional_newton_factor': args.conditional_newton_factor,
              'newton_status': newton.status.numpy().tolist() if newton else None,
              'newton_max_relative_residual': newton.max_relative_residual.numpy().tolist() if newton else None,
              'stages_ms': {
                  k: {'median': float(np.median([x[k] for x in flat])),
                      'p95': float(np.percentile([x[k] for x in flat], 95))} for k in flat[0]}}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owned-wrench',action='store_true')
    parser.add_argument('--owned-effective',action='store_true')
    parser.add_argument('--scene', type=Path, default=ROOT / 'examples/mujoco_bag_fast.xml')
    parser.add_argument('--time', type=float, default=1.4)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--contact-policy', action='store_true')
    parser.add_argument('--cooperative-metric', action='store_true')
    parser.add_argument('--cholesky-threads', type=int, choices=[32,64,128,256,512])
    parser.add_argument('--parallel-newton', action='store_true')
    parser.add_argument('--newton-precision', choices=['float32','float64'],default='float64')
    parser.add_argument('--newton-tile', type=int, choices=[16,32],default=16)
    parser.add_argument('--newton-diagonal',choices=['tiled','register'],default='tiled')
    parser.add_argument('--newton-substitution',choices=['tiled','ordered_warp'],default='tiled')
    parser.add_argument('--newton-panel',choices=['tiled','register'],default='tiled')
    parser.add_argument('--newton-trailing',choices=['tiled','register'],default='tiled')
    parser.add_argument('--newton-update-threads',type=int,choices=[64,128],default=64)
    parser.add_argument('--conditional-newton-factor',action='store_true')
    parser.add_argument('--worlds',type=int,default=1)
    parser.add_argument('--cpu-ccd-tolerance',type=float,default=1e-9)
    parser.add_argument('--gpu-ccd-tolerance',type=float,default=1e-8)
    parser.add_argument('--canonical-constraints',action='store_true')
    parser.add_argument('--owned-material',action='store_true')
    parser.add_argument('--owned-assembly',action='store_true')
    parser.add_argument('--owned-solver-reductions',action='store_true')
    parser.add_argument('--deterministic',action='store_true')
    parser.add_argument('--detail-solve', action='store_true')
    parser.add_argument('--solver-iterations', type=int)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/mjwarp_stage_profile.json')
    args=parser.parse_args()
    if args.owned_solver_reductions and not args.owned_assembly:
        parser.error('--owned-solver-reductions requires --owned-assembly')
    run(args)
