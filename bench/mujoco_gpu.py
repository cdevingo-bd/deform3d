"""Hybrid frame benchmark: CPU MuJoCo + coarse upload + dense GPU mesh/normals.

Uses the actual C++ BVH binding exporter on a tessellated tetra boundary. Includes
multiple physics substeps per visual frame, float conversion, H2D transfer and
completion synchronization. No dense readback occurs in the measured frame loop.
Rendering and graphics interop are excluded. Setup/compilation are untimed.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import tempfile
import time

import mujoco
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "examples")]
from deform3d_warp import DeviceBindings, SurfaceTopology, enqueue_embedding, enqueue_normals
from mujoco_bindings import read_bindings
from mujoco_surface import dense_surface


def stats(values):
    return {"p50_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}


def run(args):
    if min(args.subdivisions, args.frames, args.substeps) < 1 or args.warmup < 0:
        raise ValueError("invalid workload size")
    if args.reset_frames < 0:
        raise ValueError("reset_frames must be nonnegative")
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise ValueError("this benchmark measures the hybrid CUDA path")
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    outer = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, args.flex)
    if outer < 0:
        raise ValueError(f"scene has no flex named {args.flex}")
    rest, triangles = dense_surface(model, data, outer, args.subdivisions)
    with tempfile.TemporaryDirectory(prefix="deform3d-") as directory:
        directory = Path(directory)
        mesh = directory / "surface.xyz"
        binding_file = directory / "surface.bind"
        np.savetxt(mesh, rest, fmt="%.9g")
        subprocess.run([str(args.exporter.resolve()), str(args.xml.resolve()), str(binding_file),
                        "--mesh", str(mesh), "--flex", args.flex], check=True)
        _, offset, records = read_bindings(binding_file, model)
    with wp.ScopedDevice(device):
        binding = DeviceBindings(records["indices"], records["weights"],
                                 source_vertex_count=int(model.flex_vertnum[outer]), device=device)
        topology = SurfaceTopology(triangles, vertex_count=len(rest), device=device)
        host = wp.empty((1, binding.source_vertex_count), dtype=wp.vec3f, device="cpu", pinned=True)
        host_view = host.numpy()[0]
        coarse = wp.empty(host.shape, dtype=wp.vec3f, device=device)
        positions = wp.empty((1, len(rest)), dtype=wp.vec3f, device=device)
        normals = wp.empty_like(positions)
        scratch = wp.empty((1, len(triangles)), dtype=wp.vec3f, device=device)

        def upload():
            np.copyto(host_view, data.flexvert_xpos[offset:offset + binding.source_vertex_count],
                      casting="unsafe")
            wp.copy(coarse, host)

        def deform():
            enqueue_embedding(coarse, binding, positions)
            enqueue_normals(positions, topology, scratch, normals)

        upload()
        deform()
        wp.synchronize_device(device)
        with wp.ScopedCapture(device=device) as capture:
            deform()
        times = {key: [] for key in ("physics_and_refresh", "stage_upload_deform_sync", "frame")}
        contact_observations = 0
        for frame in range(args.frames + args.warmup):
            # Repeat the contact transient in the original unanchored smoke scene.
            # Closed-shell scenes can use --reset-frames 0 for sustained motion.
            if args.reset_frames and frame % args.reset_frames == 0:
                mujoco.mj_resetData(model, data)
            begin = time.perf_counter_ns()
            for _ in range(args.substeps):
                mujoco.mj_step(model, data)
                contact_observations += data.ncon
            mujoco.mj_kinematics(model, data)
            mujoco.mj_flex(model, data)
            physics_end = time.perf_counter_ns()
            upload()
            wp.capture_launch(capture.graph)
            wp.synchronize_device(device)
            end = time.perf_counter_ns()
            if frame >= args.warmup:
                times["physics_and_refresh"].append((physics_end - begin) * 1e-6)
                times["stage_upload_deform_sync"].append((end - physics_end) * 1e-6)
                times["frame"].append((end - begin) * 1e-6)

        # Readback is validation only, outside the measured loop.
        warnings = [int(w.number) for w in data.warning]
        if any(warnings) or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise AssertionError(f"MuJoCo state or solver warning failure: {warnings}")
        dense = positions.numpy()[0]
        normal_values = normals.numpy()[0]
        expected = np.sum(host_view[records["indices"]] * records["weights"][..., None], axis=1)
        error = float(np.max(np.linalg.norm(dense - expected, axis=1)))
        length = np.linalg.norm(normal_values, axis=1)
        if not np.isfinite(dense).all() or not np.isfinite(normal_values).all() or error > 3e-5:
            raise AssertionError("dense GPU output failed reference validation")
        if np.min(length) < 0.99 or np.max(length) > 1.01:
            raise AssertionError("surface normals are not unit length")
        # Check orientation as well as length against float64 area sums at a
        # deterministic sample of vertices, outside the measured frame loop.
        offsets = topology.offsets.numpy()
        incident = topology.incident_faces.numpy()
        normal_error = 0.0
        sample = np.unique(np.linspace(0, len(rest) - 1, min(128, len(rest)), dtype=np.int64))
        for vertex in sample:
            fan = triangles[incident[offsets[vertex]:offsets[vertex + 1]]]
            xyz = dense[fan].astype(np.float64)
            summed = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]).sum(axis=0)
            norm = np.linalg.norm(summed)
            reference = summed / norm if norm else np.zeros(3)
            normal_error = max(normal_error, float(np.linalg.norm(normal_values[vertex] - reference)))
        if normal_error > 2e-3:
            raise AssertionError(f"GPU normals differ from float64 reference: {normal_error}")
        result = {"device": device.name, "warp": wp.__version__, "mujoco": mujoco.__version__,
                  "scene": args.xml.name, "render_vertices": len(rest), "render_triangles": len(triangles),
                  "physics_tetrahedra": int(np.sum(model.flex_elemnum[model.flex_dim == 3])), "frames": args.frames,
                  "physics_substeps_per_frame": args.substeps,
                  "simulated_ms_per_frame": float(model.opt.timestep * args.substeps * 1000),
                  "h2d_bytes_per_frame": binding.source_vertex_count * 12,
                  "reset_interval_frames": args.reset_frames,
                  "contact_observations_including_warmup": int(contact_observations),
                  "max_embedding_error_m": error, "normal_length_min": float(np.min(length)),
                  "normal_length_max": float(np.max(length)),
                  "max_sampled_normal_error": normal_error, "normal_reference_samples": len(sample),
                  "dense_render_flex": args.flex, "validation_scope": "final frame",
                  "mujoco_warning_counts": warnings,
                  "timings": {key: stats(value) for key, value in times.items()},
                  "excludes": ["setup", "compilation", "periodic scene reset", "dense validation readback", "rendering", "graphics interop"]}
        print(json.dumps(result, indent=2))
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(result, indent=2) + "\n")
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=ROOT / "examples/mujoco_cavity.xml")
    parser.add_argument("--exporter", type=Path, default=ROOT / "build/deform3d_export_bindings")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--flex", default="cavity")
    parser.add_argument("--subdivisions", type=int, default=64)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--reset-frames", type=int, default=8)
    parser.add_argument("--json", type=Path)
    run(parser.parse_args())
