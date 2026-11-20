"""Dense device path benchmark, with separate resident and transfer timings.

Example: python bench/warp_dense.py --device cuda:0 --side 1000 --worlds 2
The synthetic curved surface spans a regular tetrahedral cage. Timings exclude
setup, compilation, and validation. CUDA timings use events around warmed CUDA
graph batches; host timings include graph submission and final synchronization.
Transfers use reusable pinned buffers and are reported separately. This is a
geometry benchmark, not a contact/physics or sim-to-real accuracy benchmark.
"""

from pathlib import Path
import argparse
import json
import platform
import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from deform3d_warp import DeviceBindings, SurfaceTopology, enqueue_embedding, enqueue_normals


def fixture(side, cage_side, worlds):
    uv = np.linspace(0.0, 1.0, side, dtype=np.float32)
    x, y = np.meshgrid(uv, uv)
    dense = np.stack([x, y, 0.5 + 0.1 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)], axis=-1).reshape(-1, 3)
    cell = np.minimum(np.floor(dense * (cage_side - 1)).astype(np.int32), cage_side - 2)
    local = dense * (cage_side - 1) - cell
    order = np.argsort(-local, axis=1, kind="stable")
    sorted_local = np.take_along_axis(local, order, axis=1)
    strides = np.array([1, cage_side, cage_side * cage_side], dtype=np.int32)
    base = cell @ strides
    ids = np.column_stack([base, base + strides[order[:, 0]],
                           base + strides[order[:, 0]] + strides[order[:, 1]], base + strides.sum()])
    weights = np.column_stack([1 - sorted_local[:, 0], sorted_local[:, 0] - sorted_local[:, 1],
                               sorted_local[:, 1] - sorted_local[:, 2], sorted_local[:, 2]]).astype(np.float32)
    axis = np.linspace(0.0, 1.0, cage_side, dtype=np.float32)
    zc, yc, xc = np.meshgrid(axis, axis, axis, indexing="ij")
    cage = np.stack([xc, yc, zc], axis=-1).reshape(-1, 3)
    sources = []
    for world in range(worlds):
        deformed = cage.copy()
        deformed[:, 0] += 0.08 * np.sin(3.0 * cage[:, 2] + 0.3 * world) * cage[:, 1]
        deformed[:, 2] += 0.12 * np.sin(np.pi * cage[:, 0]) * np.sin(np.pi * cage[:, 1])
        deformed[:, 1] *= 1.0 + 0.03 * world
        sources.append(deformed)
    row, col = np.meshgrid(np.arange(side - 1), np.arange(side - 1), indexing="ij")
    a = (row * side + col).reshape(-1)
    faces = np.stack([np.column_stack([a, a + 1, a + side]),
                      np.column_stack([a + 1, a + side + 1, a + side])], axis=1).reshape(-1, 3).astype(np.int32)
    return np.stack(sources), ids, weights, faces


def statistics(values):
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
            "min_ms": float(np.min(values))}


def measure(fn, device, *, samples, repeat, warmup):
    for _ in range(warmup):
        fn()
    wp.synchronize_device(device)
    gpu_times, wall_times = [], []
    if device.is_cuda:
        # Many iterations in one graph amortize Python dispatch and event cost.
        wp.capture_begin(device=device)
        try:
            for _ in range(repeat):
                fn()
        finally:
            graph = wp.capture_end(device=device)
        start = wp.Event(device=device, enable_timing=True)
        end = wp.Event(device=device, enable_timing=True)
        wp.capture_launch(graph)
        wp.synchronize_device(device)
        for _ in range(samples):
            t0 = time.perf_counter()
            wp.record_event(start)
            wp.capture_launch(graph)
            wp.record_event(end)
            gpu_times.append(wp.get_event_elapsed_time(start, end) / repeat)
            wall_times.append((time.perf_counter() - t0) * 1000 / repeat)
    else:
        for _ in range(samples):
            t0 = time.perf_counter()
            for _ in range(repeat):
                fn()
            wp.synchronize_device(device)
            wall_times.append((time.perf_counter() - t0) * 1000 / repeat)
    return {"device_event": statistics(gpu_times) if gpu_times else None,
            "host_wall": statistics(wall_times)}


def validate(source, ids, weights, faces, side, output, normals):
    # Readback occurs only once after measurements. Test sample points include
    # all corners and a seeded random selection from the full dense mesh.
    selection = np.unique(np.concatenate(([0, side - 1, side * (side - 1), side * side - 1],
                                          np.random.default_rng(72).integers(0, side * side, 128))))
    actual, actual_normals = output.numpy(), normals.numpy()
    reference = np.sum(source[:, ids[selection]].astype(np.float64) * weights[None, selection, :, None], axis=2)
    position_error = float(np.max(np.linalg.norm(actual[:, selection] - reference, axis=-1)))
    reference_normals = []
    for vertex in selection:
        # Grid topology identifies the only four cells that can touch a vertex.
        row, col = divmod(int(vertex), side)
        face_ids = []
        for r in (row - 1, row):
            for c in (col - 1, col):
                if 0 <= r < side - 1 and 0 <= c < side - 1:
                    first = 2 * (r * (side - 1) + c)
                    face_ids.extend([first, first + 1])
        incident = faces[face_ids]
        incident = incident[np.any(incident == vertex, axis=1)]
        # Compute reference face positions from the cage, independently of the
        # GPU positions, to catch either gather or normal mistakes.
        points = np.sum(source[:, ids[incident]].astype(np.float64)
                        * weights[None, incident, :, None], axis=3)
        total = np.cross(points[:, :, 1] - points[:, :, 0],
                         points[:, :, 2] - points[:, :, 0]).sum(axis=1)
        total /= np.linalg.norm(total, axis=-1, keepdims=True)
        reference_normals.append(total)
    reference_normals = np.stack(reference_normals, axis=1)
    normal_error = float(np.max(np.linalg.norm(actual_normals[:, selection] - reference_normals, axis=-1)))
    if not np.isfinite(actual).all() or not np.isfinite(actual_normals).all():
        raise AssertionError("nonfinite output")
    # With a dense grid, subtraction of adjacent float32 positions amplifies
    # rounding in face directions. The tolerance scales with vertex spacing.
    if position_error > 2e-6 or normal_error > max(2e-5, side * 5e-7):
        raise AssertionError(f"reference mismatch: positions={position_error}, normals={normal_error}")
    return {"sample_count": len(selection), "max_position_error": position_error,
            "max_unit_normal_error": normal_error}


def run(args):
    wp.init()
    device = wp.get_device(args.device)
    source, ids, weights, faces = fixture(args.side, args.cage_side, args.worlds)
    count = args.side * args.side
    with wp.ScopedDevice(device):
        bindings = DeviceBindings(ids, weights, source_vertex_count=source.shape[1], device=device)
        topology = SurfaceTopology(faces, vertex_count=count, device=device)
        cage = wp.array(source, dtype=wp.vec3f, device=device)
        output = wp.empty((args.worlds, count), dtype=wp.vec3f, device=device)
        normals = wp.empty_like(output)
        scratch = wp.empty((args.worlds, len(faces)), dtype=wp.vec3f, device=device)

        def gather():
            enqueue_embedding(cage, bindings, output)

        def normals_only():
            enqueue_normals(output, topology, scratch, normals)

        def resident():
            gather()
            normals_only()

        resident()
        timings = {}
        for name, fn in (("gather", gather), ("normals", normals_only), ("resident_gather_normals", resident)):
            timings[name] = measure(fn, device, samples=args.samples, repeat=args.repeat, warmup=args.warmup)
        if device.is_cuda:
            pinned_cage = wp.array(source, dtype=wp.vec3f, device="cpu", pinned=True)
            pinned_output = wp.empty((args.worlds, count), dtype=wp.vec3f, device="cpu", pinned=True)
            pinned_normals = wp.empty_like(pinned_output)
            for name, fn in (("h2d_coarse_positions", lambda: wp.copy(cage, pinned_cage)),
                             ("d2h_dense_positions", lambda: wp.copy(pinned_output, output)),
                             ("d2h_dense_normals", lambda: wp.copy(pinned_normals, normals))):
                timings[name] = measure(fn, device, samples=args.samples, repeat=args.repeat, warmup=args.warmup)
        validation = validate(source, ids, weights, faces, args.side, output, normals)
    total = timings["resident_gather_normals"]["device_event" if device.is_cuda else "host_wall"]["median_ms"]
    result = {"benchmark": "deform3d_warp_dense", "python": platform.python_version(),
              "warp": wp.__version__, "device": str(device), "device_name": device.name,
              "worlds": args.worlds, "vertices_per_world": count,
              "triangles_per_world": len(faces), "cage_vertices_per_world": source.shape[1],
              "samples": args.samples, "iterations_per_sample": args.repeat, "warmup": args.warmup,
              "timing_method": "CUDA events around repeated graph; pinned transfers separate" if device.is_cuda else "host wall clock repeated direct calls",
              "total_dense_mvertices_per_second": args.worlds * count / (total * 1000),
              "h2d_bytes_per_frame": int(source.nbytes),
              "dense_position_bytes_per_frame": args.worlds * count * 12,
              "timings": timings, "validation": validation,
              "scope": "Synthetic geometry only; excludes physics, collision, rendering, and material calibration."}
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--side", type=int, default=1000, help="dense square-grid side; vertices=side squared")
    parser.add_argument("--cage-side", type=int, default=17)
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.side < 2 or args.cage_side < 2 or min(args.worlds, args.samples, args.repeat, args.warmup) < 1:
        parser.error("side/cage-side must be >=2; worlds/samples/repeat/warmup must be positive")
    run(args)
