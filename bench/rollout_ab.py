"""A/B timing and result check: MJWarp rollout with and without deform3d.

Mirrors a typical batched rollout loop:
``put_model`` -> ``put_data(nworld=N)`` -> capture one ``mjwarp.step`` CUDA graph ->
launch it ``n_steps`` times -> ``wp.synchronize``. Two variants are compared:

  native    graph = step
  deform3d  graph = step + FlexEmbedding.update (kinematics/flex refresh + embedding)

Variants run interleaved so thermal or clock drift cannot favour one. Results are
compared against an A-vs-A control, because MJWarp is not guaranteed to be bitwise
repeatable and a deform3d-vs-native difference only means something relative to that.

Run in the environment that matches the stack under test, e.g. MuJoCo 3.17.0 /
Warp 1.16.0 with a MuJoCo Warp 3.17.0 source build on PYTHONPATH:

  python bench/rollout_ab.py examples/mujoco_bag.xml package.bind \
      --worlds 1 16 64 --n-steps 5 --repeats 15 --json out.json
"""

import argparse
import json
import statistics
import sys
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from deform3d_warp.bindings import read_bindings
from deform3d_warp.mujoco_bridge import FlexEmbedding


def build(mj_model, mj_data, nworld, nconmax, njmax):
    model = mjw.put_model(mj_model)
    data = mjw.put_data(mj_model, mj_data, nworld=nworld, nconmax=nconmax, njmax=njmax)
    return model, data


def capture(fn, device):
    fn()  # warm up: compile kernels outside capture
    wp.synchronize()
    with wp.ScopedCapture(device=device) as cap:
        fn()
    wp.synchronize()
    return cap.graph


def timed(graph, n_steps):
    """Wall time for n_steps graph launches plus sync, as a rollout loop measures it."""
    start = time.perf_counter()
    for _ in range(n_steps):
        wp.capture_launch(graph)
    wp.synchronize()
    return time.perf_counter() - start


def reset(model_cpu, data_cpu, data, nworld):
    mujoco.mj_resetData(model_cpu, data_cpu)
    mujoco.mj_forward(model_cpu, data_cpu)
    for name in ("qpos", "qvel"):
        host = np.tile(getattr(data_cpu, name), (nworld, 1)).astype(np.float32)
        getattr(data, name).assign(host)
    data.time.zero_()


def summarize(values):
    values = sorted(values)
    q = statistics.quantiles(values, n=4) if len(values) >= 4 else [values[0], values[0], values[-1]]
    return {"median_ms": 1e3 * statistics.median(values), "q1_ms": 1e3 * q[0],
            "q3_ms": 1e3 * q[2], "min_ms": 1e3 * values[0], "n": len(values)}


def dense_reference(flexvert_xpos, records, offset):
    """float64 barycentric reconstruction of the dense mesh from host flex vertices."""
    ids = records["indices"].astype(np.int64) + offset
    w = records["weights"].astype(np.float64)
    pts = flexvert_xpos.astype(np.float64)[:, ids, :]  # [world, point, 4, 3]
    return np.einsum("pk,wpkc->wpc", w, pts)


def run_scene(args, worlds):
    mj_model = mujoco.MjModel.from_xml_path(args.scene)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    device = wp.get_device(args.device)
    nconmax = args.nconmax_per_world * worlds
    out = {"worlds": worlds}

    with wp.ScopedDevice(device):
        # Two independent simulations from identical initial state: A (native),
        # B (deform3d), and A2 (control, same as A) to measure run-to-run noise.
        model_a, data_a = build(mj_model, mj_data, worlds, nconmax, args.njmax)
        model_b, data_b = build(mj_model, mj_data, worlds, nconmax, args.njmax)
        model_c, data_c = build(mj_model, mj_data, worlds, nconmax, args.njmax)
        dense = FlexEmbedding(mj_model, data_b, args.binding)

        g_native = capture(lambda: mjw.step(model_a, data_a), device)
        g_control = capture(lambda: mjw.step(model_c, data_c), device)

        def step_and_embed():
            mjw.step(model_b, data_b)
            dense.update(model_b, data_b)

        g_deform = capture(step_and_embed, device)
        # Marginal pieces, each as its own graph.
        g_refresh = capture(lambda: (mjw.kinematics(model_b, data_b), mjw.flex(model_b, data_b)), device)
        g_embed = capture(lambda: dense.update(model_b, data_b, refresh=False), device)

        # Timing: interleave native / deform3d / refresh / embed each repeat.
        samples = {"native": [], "deform3d": [], "refresh_only": [], "embed_only": []}
        for _ in range(args.repeats):
            samples["native"].append(timed(g_native, args.n_steps))
            samples["deform3d"].append(timed(g_deform, args.n_steps))
            samples["refresh_only"].append(timed(g_refresh, args.n_steps))
            samples["embed_only"].append(timed(g_embed, args.n_steps))
        out["timing"] = {k: summarize(v) for k, v in samples.items()}
        base = out["timing"]["native"]["median_ms"]
        out["overhead_pct"] = 100.0 * (out["timing"]["deform3d"]["median_ms"] - base) / base
        out["per_world_step_us"] = {
            k: 1e3 * v["median_ms"] / (args.n_steps * worlds) for k, v in out["timing"].items()
        }

        # Results: identical initial state, same steps, then compare.
        for data in (data_a, data_b, data_c):
            reset(mj_model, mj_data, data, worlds)
        for _ in range(args.compare_steps):
            wp.capture_launch(g_native)
            wp.capture_launch(g_deform)
            wp.capture_launch(g_control)
        wp.synchronize()
        qa, qb, qc = (d.qpos.numpy() for d in (data_a, data_b, data_c))
        fa, fb = data_a.flexvert_xpos.numpy(), data_b.flexvert_xpos.numpy()
        out["results"] = {
            "steps": args.compare_steps,
            "qpos_max_abs_diff_deform3d_vs_native": float(np.abs(qb - qa).max()),
            "qpos_max_abs_diff_control_vs_native": float(np.abs(qc - qa).max()),
            "qpos_bitwise_equal_deform3d_vs_native": bool(np.array_equal(qb, qa)),
            "qpos_bitwise_equal_control_vs_native": bool(np.array_equal(qc, qa)),
            "flexvert_max_abs_diff_deform3d_vs_native": float(np.abs(fb - fa).max()),
        }

        # Dense mesh vs float64 reference computed from the very same flex state.
        _, offset, records = read_bindings(args.binding, mj_model)
        # The embedding was evaluated after the final step's refresh; refresh once more
        # without stepping so dense and flexvert_xpos describe the same instant.
        dense.update(model_b, data_b)
        wp.synchronize()
        ref = dense_reference(data_b.flexvert_xpos.numpy(), records, offset)
        got = dense.positions.numpy().astype(np.float64)
        err = np.abs(got - ref)
        out["dense_vs_float64_reference"] = {
            "points": int(got.shape[1]), "max_abs_err_m": float(err.max()),
            "mean_abs_err_m": float(err.mean()),
        }
        out["nflexvert"] = int(mj_model.nflexvert)
        out["dense_points"] = int(got.shape[1])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene")
    p.add_argument("binding")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--worlds", type=int, nargs="+", default=[1, 16, 64])
    p.add_argument("--n-steps", type=int, default=5, help="graph launches per timed rollout")
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--compare-steps", type=int, default=10)
    p.add_argument("--nconmax-per-world", type=int, default=300)
    p.add_argument("--njmax", type=int, default=4000)
    p.add_argument("--json", help="write full results here")
    args = p.parse_args()

    if not wp.get_device(args.device).is_cuda:
        sys.exit("CUDA graphs are required to mirror a rollout loop; pass a CUDA device")
    wp.config.quiet = True
    report = {
        "scene": args.scene, "binding": args.binding, "device": args.device,
        "mujoco": mujoco.__version__, "warp": wp.config.version,
        "mujoco_warp": getattr(mjw, "__file__", ""),
        "gpu": wp.get_device(args.device).name,
        "n_steps": args.n_steps, "repeats": args.repeats, "runs": [],
    }
    for worlds in args.worlds:
        try:
            run = run_scene(args, worlds)
        except RuntimeError as exc:  # e.g. device out of memory at large world counts
            message = str(exc).splitlines()[0]
            print(f"worlds={worlds:4d}  FAILED: {message}")
            report["runs"].append({"worlds": worlds, "error": message})
            write_report(args, report)
            continue
        report["runs"].append(run)
        write_report(args, report)
        t = run["timing"]
        print(f"worlds={worlds:4d}  native {t['native']['median_ms']:8.2f} ms  "
              f"deform3d {t['deform3d']['median_ms']:8.2f} ms  overhead {run['overhead_pct']:+6.2f}%  "
              f"(refresh {t['refresh_only']['median_ms']:.2f}, embed {t['embed_only']['median_ms']:.3f} ms)  "
              f"qpos d3d-vs-native {run['results']['qpos_max_abs_diff_deform3d_vs_native']:.2e} "
              f"control {run['results']['qpos_max_abs_diff_control_vs_native']:.2e}  "
              f"dense err {run['dense_vs_float64_reference']['max_abs_err_m']:.2e} m")


def write_report(args, report):
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
