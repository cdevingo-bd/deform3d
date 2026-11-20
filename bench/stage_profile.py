"""Per-stage graph-replay timing of a native MJWarp step, next to deform3d's update cost.

Each stage of ``mjwarp.step`` is captured as its own CUDA graph and replayed, so timings match
how a graph-replayed rollout executes, unlike an eager trace, which is dominated by host
launch and sync gaps. Stages are replayed on the state left by real steps, so contacts and
constraints are live. Per-stage graphs add small inter-graph gaps, so the stage sum is slightly
above the full-step graph time; the shares are what matter.

Uses private MJWarp modules (forward/smooth/constraint/collision_driver/solver); it is a
diagnostic tool, tied to the MJWarp version it is run against.

  python bench/stage_profile.py examples/mujoco_bag.xml --binding package.bind --worlds 1 64
  python bench/stage_profile.py scene.xml --worlds 1 64      # no binding: skip deform3d lines
"""

import argparse
import statistics
import time

import mujoco
import mujoco_warp as mjw
import warp as wp
from mujoco_warp._src import collision_driver, constraint, forward, smooth, solver
from mujoco_warp._src.types import IntegratorType

from deform3d_warp.mujoco_bridge import FlexEmbedding


def capture(fn):
    fn()
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        fn()
    wp.synchronize()
    return cap.graph


def replay_ms(graph, launches, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(launches):
            wp.capture_launch(graph)
        wp.synchronize()
        samples.append(1e3 * (time.perf_counter() - start) / launches)
    return statistics.median(samples)


def profile(args, worlds):
    m = mujoco.MjModel.from_xml_path(args.scene)
    cpu = mujoco.MjData(m)
    mujoco.mj_forward(m, cpu)
    for _ in range(args.warm_steps):  # reach a state with live contacts
        mujoco.mj_step(m, cpu)
    with wp.ScopedDevice(args.device):
        model = mjw.put_model(m)
        d = mjw.put_data(m, cpu, nworld=worlds, nconmax=args.nconmax_per_world * worlds, njmax=args.njmax)
        dense = FlexEmbedding(m, d, args.binding) if args.binding else None
        for _ in range(3):
            mjw.step(model, d)
        wp.synchronize()

        def position_kinematics():
            forward.fwd_kinematics(model, d)

        def position_inertia():
            smooth.crb(model, d)
            smooth.tendon_armature(model, d)

        def collide():
            collision_driver.collision(model, d)

        def build_constraints():
            constraint.make_constraint(model, d)

        def velocity_actuation_accel():
            forward.fwd_velocity(model, d)
            forward.fwd_actuation(model, d)
            forward.fwd_acceleration(model, d, factorize=True)

        def solve():
            solver.solve(model, d)

        def integrate():
            integrator = model.opt.integrator
            if integrator == IntegratorType.EULER:
                forward.euler(model, d)
            elif integrator in (IntegratorType.IMPLICITFAST, IntegratorType.IMPLICIT):
                forward.implicit(model, d)
            else:
                forward.discrete(model, d)

        stages = [
            ("kinematics", position_kinematics), ("crb / armature", position_inertia),
            ("collision", collide), ("make_constraint", build_constraints),
            ("velocity+actuation+accel", velocity_actuation_accel),
            ("solver", solve), ("integrate", integrate),
        ]
        times = {name: replay_ms(capture(fn), args.launches, args.repeats) for name, fn in stages}
        full = replay_ms(capture(lambda: mjw.step(model, d)), args.launches, args.repeats)
        refresh = embed = None
        if dense is not None:
            refresh = replay_ms(capture(lambda: (mjw.kinematics(model, d), mjw.flex(model, d))),
                                args.launches, args.repeats)
            embed = replay_ms(capture(lambda: dense.update(model, d, refresh=False)),
                              args.launches, args.repeats)
        nefc = int(d.nefc.numpy().max())
        niter = int(d.solver_niter.numpy().max())
    return m, times, full, refresh, embed, nefc, niter


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene")
    p.add_argument("--binding", help="optional deform3d binding file")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--worlds", type=int, nargs="+", default=[1, 64])
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--launches", type=int, default=10)
    p.add_argument("--warm-steps", type=int, default=300)
    p.add_argument("--njmax", type=int, default=512)
    p.add_argument("--nconmax-per-world", type=int, default=64)
    args = p.parse_args()
    for worlds in args.worlds:
        m, times, full, refresh, embed, nefc, niter = profile(args, worlds)
        total = sum(times.values())
        print(f"\n== worlds={worlds}  nv={m.nv} nflexvert={m.nflexvert}  "
              f"full step graph {full:.3f} ms  (stage sum {total:.3f} ms)  "
              f"max nefc={nefc} max solver iters={niter}")
        print(f"  integrator={int(m.opt.integrator)} solver={int(m.opt.solver)} dt={m.opt.timestep}")
        for name, ms in times.items():
            print(f"  {name:<26s}{ms:8.3f} ms {100 * ms / total:5.1f}%")
        if refresh is not None:
            print(f"  {'deform3d refresh':<26s}{refresh:8.3f} ms {100 * refresh / full:5.1f}% of step")
            print(f"  {'deform3d embedding':<26s}{embed:8.3f} ms {100 * embed / full:5.1f}% of step")


if __name__ == "__main__":
    main()
