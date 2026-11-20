"""Dense direct vs native PCG solve of the flex effective metric: accuracy, cost, trajectory.

Three questions, in order:
  1. Accuracy: does the dense float32 solve match a float64 reference at least as well as the
     native PCG solve (at its configured tolerance) does?
  2. Cost: graph-replay time of assembly + factor + solve vs native ``eff_solve``.
  3. Trajectory: do full steps with the dense policy stay within a native-vs-native control?

Run in the 3.17 environment (MuJoCo 3.17 / Warp 1.16 / a MuJoCo Warp 3.17.0 source build):
  python bench/dense_effective_ab.py examples/mujoco_bag.xml --worlds 1 16 64 --json out.json
"""

import argparse
import json
import statistics
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from mujoco_warp._src import derivative, forward

from deform3d_warp.mujoco_effective_banded import BandedEffectiveSolve
from deform3d_warp.mujoco_effective_dense import DenseEffectiveSolve


def capture(fn):
    fn()
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        fn()
    wp.synchronize()
    return cap.graph


def replay_ms(graph, launches=10, repeats=15):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(launches):
            wp.capture_launch(graph)
        wp.synchronize()
        samples.append(1e3 * (time.perf_counter() - start) / launches)
    return statistics.median(samples)


def prepare_state(model, data):
    """Leave position/velocity/actuation-dependent terms valid so fwd_acceleration is meaningful."""
    for _ in range(3):
        mjw.step(model, data)
    forward.fwd_position(model, data, factorize=False)
    forward.fwd_velocity(model, data)
    forward.fwd_actuation(model, data)
    wp.synchronize()


SOLVERS = {"dense": DenseEffectiveSolve, "banded": BandedEffectiveSolve}


def run(args, worlds):
    m = mujoco.MjModel.from_xml_path(args.scene)
    cpu = mujoco.MjData(m)
    mujoco.mj_forward(m, cpu)
    for _ in range(args.warm_steps):
        mujoco.mj_step(m, cpu)
    out = {"worlds": worlds, "nv": int(m.nv), "solvers": {}}
    with wp.ScopedDevice("cuda:0"):
        model = mjw.put_model(m)
        data = mjw.put_data(m, cpu, nworld=worlds, nconmax=64 * worlds, njmax=args.njmax)
        prepare_state(model, data)

        def accel():
            data.qacc_smooth.zero_()
            forward.fwd_acceleration(model, data, factorize=True)

        accel()
        native = data.qacc_smooth.numpy().astype(np.float64)
        g_native = capture(accel)
        out["native_eff_solve_path_ms"] = replay_ms(g_native)
        for name in args.solvers:
            solver = SOLVERS[name](model, data)
            if name == "banded":
                out["banded_bandwidth_tiles"] = solver.bandwidth_tiles
            with solver.capture_policy():
                accel()
                got = data.qacc_smooth.numpy().astype(np.float64)
                g = capture(accel)
            # Accuracy against a float64 solve of the exact M_hat, built column-wise from the
            # native operator so neither solver's own assembly is trusted.
            ref = reference_solve(model, data, worlds)
            scale = np.abs(ref).max()
            r = {
                "native_pcg_rel_err_vs_float64": float(np.abs(native - ref).max() / scale),
                "rel_err_vs_float64": float(np.abs(got - ref).max() / scale),
                "workspace_status_max": int(solver.status.numpy().max()),
                "solve_path_ms": replay_ms(g),
            }
            r["solve_speedup_vs_native"] = out["native_eff_solve_path_ms"] / r["solve_path_ms"]
            out["solvers"][name] = r

        # Trajectory: native, control (second native), and each solver; same start and steps.
        def fresh():
            return mjw.put_data(m, cpu, nworld=worlds, nconmax=64 * worlds, njmax=args.njmax)

        datas = {"native": fresh(), "control": fresh()}
        graphs = {"native": capture(lambda: mjw.step(model, datas["native"])),
                  "control": capture(lambda: mjw.step(model, datas["control"]))}
        keep = []
        for name in args.solvers:
            datas[name] = fresh()
            solver = SOLVERS[name](model, datas[name])
            keep.append(solver)
            with solver.capture_policy():
                graphs[name] = capture(lambda n=name: mjw.step(model, datas[n]))
        for d in datas.values():  # captures advanced state: reset identically
            for field in ("qpos", "qvel"):
                getattr(d, field).assign(np.tile(getattr(cpu, field), (worlds, 1)).astype(np.float32))
            d.time.zero_()
        for _ in range(args.compare_steps):
            for g in graphs.values():
                wp.capture_launch(g)
        wp.synchronize()
        q = {k: d.qpos.numpy() for k, d in datas.items()}
        out["trajectory"] = {"steps": args.compare_steps,
                             "control_vs_native": float(np.abs(q["control"] - q["native"]).max())}
        out["step_ms"] = {"native": replay_ms(graphs["native"])}
        for name, solver in zip(args.solvers, keep):
            out["trajectory"][f"{name}_vs_native"] = float(np.abs(q[name] - q["native"]).max())
            out["trajectory"][f"{name}_status_max"] = int(solver.status.numpy().max())
            out["step_ms"][name] = replay_ms(graphs[name])
            out["step_ms"][f"{name}_speedup"] = out["step_ms"]["native"] / out["step_ms"][name]
    return out


def reference_solve(model, data, worlds):
    """float64 solve of the exact effective metric, one native matvec per unit vector."""
    from mujoco_warp._src import derivative

    nv = model.nv
    rhs = (data.qfrc_smooth.numpy() + data.efm_c.numpy() + data.efm_ca.numpy()).astype(np.float64)
    columns = np.zeros((worlds, nv, nv))
    unit = wp.zeros_like(data.qacc_smooth)
    res = wp.zeros_like(data.qacc_smooth)
    for j in range(nv):
        e = np.zeros((worlds, nv), dtype=np.float32)
        e[:, j] = 1.0
        unit.assign(e)
        derivative.eff_mul_m(model, data, res, unit)
        columns[:, :, j] = res.numpy()
    return np.stack([np.linalg.solve(columns[w], rhs[w]) for w in range(worlds)])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene")
    p.add_argument("--worlds", type=int, nargs="+", default=[1, 16, 64])
    p.add_argument("--warm-steps", type=int, default=300)
    p.add_argument("--compare-steps", type=int, default=100)
    p.add_argument("--njmax", type=int, default=512)
    p.add_argument("--solvers", nargs="+", default=["dense", "banded"], choices=list(SOLVERS))
    p.add_argument("--json")
    args = p.parse_args()
    report = {"mujoco": mujoco.__version__, "warp": wp.config.version, "runs": []}
    for worlds in args.worlds:
        try:
            r = run(args, worlds)
        except RuntimeError as exc:
            r = {"worlds": worlds, "error": str(exc).splitlines()[0]}
            print(f"worlds={worlds}: FAILED {r['error']}")
        else:
            t, s = r["trajectory"], r["step_ms"]
            print(f"worlds={worlds:3d} nv={r['nv']}  native eff_solve path {r['native_eff_solve_path_ms']:.3f} ms"
                  + (f"  (banded bandwidth {r['banded_bandwidth_tiles']} tiles)" if "banded_bandwidth_tiles" in r else ""))
            for name, v in r["solvers"].items():
                print(f"  {name:<7s} solve {v['solve_path_ms']:.3f} ms  x{v['solve_speedup_vs_native']:.2f}  "
                      f"rel err vs float64: native PCG {v['native_pcg_rel_err_vs_float64']:.1e}  "
                      f"{name} {v['rel_err_vs_float64']:.1e}  status {v['workspace_status_max']}")
            print(f"  full step: native {s['native']:.3f} ms" + "".join(
                f"  {n} {s[n]:.3f} ms (x{s[n + '_speedup']:.2f})" for n in r["solvers"]))
            print(f"  trajectory ({t['steps']} steps) qpos vs native: control {t['control_vs_native']:.2e}" + "".join(
                f"  {n} {t[n + '_vs_native']:.2e}" for n in r["solvers"]))
        report["runs"].append(r)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
