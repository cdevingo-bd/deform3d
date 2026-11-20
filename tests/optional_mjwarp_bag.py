"""Full CUDA bag/contents contact + dense surface deformation smoke and timing.

Run with the explicit Pascal copy on older GPUs:
  python tools/mjwarp_pascal_compat.py --run tests/optional_mjwarp_bag.py \
    --duration 3 --output results/mjwarp_bag_cuda.json

Uses a tessellation of the actual simulated shell as dense test geometry. This
is a numerical integration test, not a measured-material or rendered asset test.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from contextlib import ExitStack
from importlib.metadata import version

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "examples"), str(ROOT / "tools")]
from deform3d_warp import (DeviceSurfaceBindings, SurfaceTopology,
                          enqueue_surface_embedding, enqueue_normals)
from mjwarp_embedding import refresh_flex_positions
from record_bag import drive


@wp.kernel
def drive_pads(times: wp.array(dtype=wp.float32), front: int, back: int,
               start: float, travel: float,
               mocap: wp.array2d(dtype=wp.vec3f)):
    world = wp.tid()
    t = times[world]
    closure = float(0.0)
    if t >= 0.7 and t < 1.4:
        x = (t - 0.7) / 0.7
        closure = x * x * (3.0 - 2.0 * x)
    elif t >= 1.4 and t < 1.7:
        closure = 1.0
    elif t >= 1.7 and t < 2.5:
        x = (t - 1.7) / 0.8
        closure = 1.0 - x * x * (3.0 - 2.0 * x)
    center = start - travel * closure
    f, b = mocap[world, front], mocap[world, back]
    f[1] = -center
    b[1] = center
    mocap[world, front] = f
    mocap[world, back] = b


@wp.func
def contents_load_profile(t: float):
    # A second, independent load during compression: smooth opposed shear on
    # the rigid fruit and deformable package, with a small fruit torque.
    value=float(0.0)
    if t>=0.9 and t<1.2:
        x=(t-0.9)/0.3
        value=x*x*(3.0-2.0*x)
    elif t>=1.2 and t<1.6:
        value=1.0
    elif t>=1.6 and t<2.2:
        x=(t-1.6)/0.6
        value=1.0-x*x*(3.0-2.0*x)
    return value


@wp.kernel
def drive_contents(times: wp.array[float], fruit: int, package_bodies: wp.array[int],
                   peak_force: float, wrenches: wp.array2d[wp.spatial_vector]):
    w=wp.tid()
    force=peak_force*contents_load_profile(times[w])
    wrenches[w,fruit]=wp.spatial_vector(force,0.,0.,0.,0.,0.015*force)
    for i in range(package_bodies.shape[0]):
        wrenches[w,package_bodies[i]]=wp.spatial_vector(-force/float(package_bodies.shape[0]),0.,0.,0.,0.,0.)


def drive_contents_cpu(data,fruit,package_bodies,peak_force):
    t=float(data.time)
    value=0.
    if .9<=t<1.2:
        x=(t-.9)/.3;value=x*x*(3-2*x)
    elif 1.2<=t<1.6:value=1.
    elif 1.6<=t<2.2:
        x=(t-1.6)/.6;value=1-x*x*(3-2*x)
    force=peak_force*value
    data.xfrc_applied[fruit]=[force,0,0,0,0,.015*force]
    data.xfrc_applied[package_bodies]=[-force/len(package_bodies),0,0,0,0,0]


@wp.kernel(enable_backward=False)
def drive_precise(times: wp.array[wp.float64], front: int, back: int,
                  start: wp.float64, travel: wp.float64, fruit: int,
                  package_bodies: wp.array[int], peak_force: wp.float64,
                  mocap: wp.array2d[wp.vec3], wrenches: wp.array2d[wp.spatial_vector]):
    w = wp.tid()
    t = times[w]
    closure = wp.float64(0.0)
    if t >= wp.float64(0.7) and t < wp.float64(1.4):
        x = (t-wp.float64(0.7))/wp.float64(0.7)
        closure = x*x*(wp.float64(3.0)-wp.float64(2.0)*x)
    elif t >= wp.float64(1.4) and t < wp.float64(1.7):
        closure = wp.float64(1.0)
    elif t >= wp.float64(1.7) and t < wp.float64(2.5):
        x = (t-wp.float64(1.7))/wp.float64(0.8)
        closure = wp.float64(1.0)-x*x*(wp.float64(3.0)-wp.float64(2.0)*x)
    center = start-travel*closure
    f, b = mocap[w, front], mocap[w, back]
    f[1] = -wp.float32(center)
    b[1] = wp.float32(center)
    mocap[w, front] = f
    mocap[w, back] = b
    if peak_force > wp.float64(0.0):
        value = wp.float64(0.0)
        if t >= wp.float64(0.9) and t < wp.float64(1.2):
            x = (t-wp.float64(0.9))/wp.float64(0.3)
            value = x*x*(wp.float64(3.0)-wp.float64(2.0)*x)
        elif t >= wp.float64(1.2) and t < wp.float64(1.6):
            value = wp.float64(1.0)
        elif t >= wp.float64(1.6) and t < wp.float64(2.2):
            x = (t-wp.float64(1.6))/wp.float64(0.6)
            value = wp.float64(1.0)-x*x*(wp.float64(3.0)-wp.float64(2.0)*x)
        force = peak_force*value
        wrenches[w, fruit] = wp.spatial_vector(wp.float32(force),0.,0.,0.,0.,wp.float32(wp.float64(0.015)*force))
        for i in range(package_bodies.shape[0]):
            wrenches[w, package_bodies[i]] = wp.spatial_vector(
                -wp.float32(force/wp.float64(package_bodies.shape[0])),0.,0.,0.,0.,0.)


@wp.kernel
def accumulate_overflow(current: wp.array(dtype=wp.int32), seen: wp.array(dtype=wp.int32)):
    world = wp.tid()
    seen[world] = seen[world] | current[world]


@wp.kernel
def accumulate_solver_iterations(current: wp.array(dtype=wp.int32), maximum: wp.array(dtype=wp.int32)):
    world = wp.tid()
    maximum[world] = wp.max(maximum[world], current[world])


@wp.kernel
def accumulate_metric(status: wp.array(dtype=wp.int32), iterations: wp.array(dtype=wp.int32),
                      residual: wp.array(dtype=wp.float32), seen: wp.array(dtype=wp.int32),
                      max_iterations: wp.array(dtype=wp.int32), max_residual: wp.array(dtype=wp.float32)):
    world = wp.tid()
    if status[world] != 0:
        seen[world] = seen[world] | (1 << status[world])
    max_iterations[world] = wp.max(max_iterations[world], iterations[world])
    max_residual[world] = wp.max(max_residual[world], residual[world])


def tessellate(faces, subdivisions):
    """Triangle-local barycentric grid; split coarse edges preserve creases."""
    weights, triangles = [], []
    lookup = {}
    for i in range(subdivisions + 1):
        for j in range(subdivisions + 1 - i):
            lookup[i, j] = len(weights)
            weights.append([1.0 - (i + j) / subdivisions, i / subdivisions, j / subdivisions])
    for i in range(subdivisions):
        for j in range(subdivisions - i):
            triangles.append([lookup[i, j], lookup[i + 1, j], lookup[i, j + 1]])
            if i + j < subdivisions - 1:
                triangles.append([lookup[i + 1, j], lookup[i + 1, j + 1], lookup[i, j + 1]])
    count = len(weights)
    return (np.repeat(faces, count, axis=0), np.tile(np.asarray(weights, np.float32), (len(faces), 1)),
            (np.asarray(triangles)[None] + count * np.arange(len(faces))[:, None, None]).reshape(-1, 3))


def run(args):
    m = mujoco.MjModel.from_xml_path(str(args.scene))
    if args.cpu_ccd_tolerance is not None:
        m.opt.ccd_tolerance = args.cpu_ccd_tolerance
    cpu = mujoco.MjData(m)
    mujoco.mj_forward(m, cpu)
    bag = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, "bag")
    package = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, "package")
    fruit = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fruit_collision")
    pads = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name + "_geom")
            for name in ("front_pad", "back_pad")]
    if min(bag, package, fruit, *pads) < 0:
        raise ValueError('bag fixture requires named bag/package flexes, fruit_collision and both pad geoms')
    if m.flex_dim[bag] != 2 or m.flex_dim[package] != 3:
        raise ValueError('bag must be a triangle flex and package a tetrahedral flex')
    mocap_ids = [int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)])
                 for name in ("front_pad", "back_pad")]
    motion_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, 'deform3d_pad_motion')
    pad_start, pad_travel = m.numeric_data[m.numeric_adr[motion_id]:m.numeric_adr[motion_id]+2] if motion_id >= 0 else (.12, .073)
    start = int(m.flex_elemdataadr[bag])
    faces = m.flex_elem[start:start + 3 * m.flex_elemnum[bag]].reshape(-1, 3).copy()
    offset = int(m.flex_vertadr[bag])
    start = int(m.flex_elemdataadr[package])
    tets = (m.flex_elem[start:start + 4 * m.flex_elemnum[package]].reshape(-1, 4)
            + int(m.flex_vertadr[package]))
    ids, weights, dense_faces = tessellate(faces, args.subdivisions)
    global_faces = faces + offset
    def volumes(xyz):
        q = xyz[tets]
        return np.einsum("ij,ij->i", np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]), q[:, 3] - q[:, 0])
    def areas(xyz):
        q = xyz[global_faces]
        return np.linalg.norm(np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]), axis=1)
    rest_volumes, rest_areas = volumes(cpu.flexvert_xpos), areas(cpu.flexvert_xpos)
    recorded = {key: [] for key in ('points', 'qpos', 'mocap', 'fruit_pose', 'diagnostics',
                                   'package_thickness', 'force_time')}
    cpu_recorded = {key: [] for key in recorded}
    cpu_ms = []
    package_ids = np.arange(m.flex_vertadr[package], m.flex_vertadr[package] + m.flex_vertnum[package])
    package_rest = cpu.flexvert_xpos[package_ids]
    front_face = np.isclose(package_rest[:, 1], package_rest[:, 1].min())
    back_face = np.isclose(package_rest[:, 1], package_rest[:, 1].max())
    rest_thickness = float(package_rest[back_face, 1].mean() - package_rest[front_face, 1].mean())
    fruit_body = int(m.geom_bodyid[fruit])
    package_bodies=np.unique(m.flex_vertbodyid[package_ids]).astype(np.int32)
    if args.contents_shear_force and (fruit_body in package_bodies or np.any(package_bodies==0)):
        raise ValueError('contents shear requires independent dynamic contents bodies')
    steps = int(round(args.duration / m.opt.timestep))
    if steps < 1:
        raise ValueError("duration must contain at least one physics step")
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise ValueError("this full-device integration smoke requires CUDA")
    report = {"scene": str(args.scene), "device": str(device), "duration_s": args.duration,
              "mujoco_version": mujoco.__version__, "mujoco_warp_version": version('mujoco-warp'),
              "warp_version": wp.__version__, "gpu_name": device.name,
              "steps": steps, "timestep_s": float(m.opt.timestep), "worlds": args.worlds,
              "dense_vertices_per_world": len(ids), "dense_triangles_per_world": len(dense_faces),
              "dense_geometry": "uniform barycentric tessellation of the actual shell triangles",
              "contact_policy": bool(args.contact_policy),
              "local_flex_contacts": bool(args.local_contacts),
              "stable_contact_normals": bool(args.stable_contact_normals),
              "precise_controls": bool(args.precise_controls),
              "gpu_epa_horizon_capacity": int(args.epa_horizon),
              "contents_load": (dict(case='opposed_contents_shear_v1',peak_force_N=args.contents_shear_force,
                                     fruit_torque_lever_m=.015,frame='world',application='body centers of mass',
                                     ramp_s=[.9,1.2],hold_s=[1.2,1.6],release_s=[1.6,2.2])
                                if args.contents_shear_force else None),
              "graph": True, "timing_scope": "CUDA event and host wall per evolving physics step, averaged in batches; controller, MJWarp physics, integrated flex refresh, dense surface gather and normals; excludes compilation, setup, rendering, CPU reference and diagnostic readbacks"}
    with wp.ScopedDevice(device):
        model = mjw.put_model(m)
        if args.gpu_ccd_tolerance is not None:
            model.opt.ccd_tolerance.fill_(args.gpu_ccd_tolerance)
        report['cpu_ccd_tolerance'] = float(m.opt.ccd_tolerance)
        report['gpu_ccd_tolerance'] = float(model.opt.ccd_tolerance.numpy()[0])
        if args.gpu_iterations is not None:
            model.opt.iterations = args.gpu_iterations
        if args.deterministic:
            model.opt.deterministic = mjw.DeterminismType.ALL
        elif args.stable_ordering:
            model.opt.deterministic = mjw.DeterminismType.CONTACTS | mjw.DeterminismType.CONSTRAINT
        report['gpu_solver_iteration_limit'] = model.opt.iterations
        report['gpu_conditional_graph'] = bool(model.opt.graph_conditional)
        report['gpu_determinism'] = int(model.opt.deterministic)
        report['upstream_sorting_note'] = 'MJWarp 4.0.1 CONTACTS and CONSTRAINT sort hooks are stubs; canonical_constraints reports the separate deform3d ordering policy'
        data = mjw.make_data(m, nworld=args.worlds)
        warm = mjw.make_data(m, nworld=args.worlds)
        bindings = DeviceSurfaceBindings(ids, weights, source_vertex_count=int(m.flex_vertnum[bag]), device=device)
        topology = SurfaceTopology(dense_faces, vertex_count=len(ids), device=device)
        output = wp.zeros((args.worlds, len(ids)), dtype=wp.vec3f, device=device)
        valid = wp.zeros((args.worlds, len(ids)), dtype=wp.int32, device=device)
        normals = wp.zeros_like(output)
        scratch = wp.empty((args.worlds, len(dense_faces)), dtype=wp.vec3f, device=device)
        overflow_seen = wp.zeros(args.worlds, dtype=wp.int32, device=device)
        max_iterations = wp.zeros(args.worlds, dtype=wp.int32, device=device)
        metric = None
        if args.cooperative_metric:
            from deform3d_warp.mujoco_metric import CooperativeMetricSolver
            metric = CooperativeMetricSolver(model, args.worlds)
        newton = None
        if args.parallel_newton:
            from deform3d_warp.mujoco_newton import ParallelNewtonSolver
            newton = ParallelNewtonSolver(model, args.worlds,precision=args.newton_precision,tile_size=args.newton_tile,
                                          diagonal=args.newton_diagonal,substitution=args.newton_substitution,
                                              panel=args.newton_panel,update_threads=args.newton_update_threads,
                                              conditional_factor=args.conditional_newton_factor,trailing=args.newton_trailing)
        ordering = None
        if args.canonical_constraints:
            from deform3d_warp.mujoco_ordering import CanonicalContactConstraints
            ordering = CanonicalContactConstraints(model,data)
        deterministic_budget = None
        if args.bounded_determinism:
            from deform3d_warp.mujoco_determinism import SparseDeterminismBudget
            deterministic_budget = SparseDeterminismBudget(model)
        assembly = None
        if args.owned_assembly:
            from deform3d_warp.mujoco_assembly import OwnedContactAssembly
            assembly = OwnedContactAssembly(model,data,solver_reductions=args.owned_solver_reductions)
        material = None
        if args.owned_material:
            if args.shared_simplex:
                from deform3d_warp.mujoco_simplex import SimplexFlexOperators
                material = SimplexFlexOperators(model,data)
            else:
                from deform3d_warp.mujoco_material import OwnedFlexOperators
                material = OwnedFlexOperators(model,data)
        wrench = effective = None
        if args.owned_wrench:
            from deform3d_warp.mujoco_wrench import OwnedWrenchProjection
            wrench=OwnedWrenchProjection(model,data)
        if args.owned_effective:
            from deform3d_warp.mujoco_effective import OwnedEffectiveMetric
            effective=OwnedEffectiveMetric(model,data)
        load_bodies=wp.array(package_bodies,dtype=int,device=device)
        clocks = {}
        if args.precise_controls:
            from deform3d_warp.clock import DeviceSimulationClock
            clocks = {id(state): DeviceSimulationClock(args.worlds, device=device) for state in (data, warm)}
        def controls(state):
            if args.precise_controls:
                clock = clocks[id(state)]
                clock.enqueue_write(state.time)
                wp.launch(drive_precise, args.worlds, inputs=[clock.elapsed, *mocap_ids,
                    float(pad_start), float(pad_travel), fruit_body, load_bodies,
                    args.contents_shear_force, state.mocap_pos, state.xfrc_applied], device=device)
            else:
                wp.launch(drive_pads, dim=args.worlds, inputs=[state.time, *mocap_ids,
                    float(pad_start), float(pad_travel), state.mocap_pos], device=device)
                drive_load(state)
        def advance_clock(state):
            if args.precise_controls:
                clocks[id(state)].enqueue_advance(model.opt.timestep, state.time)
        def drive_load(state):
            if args.contents_shear_force:
                wp.launch(drive_contents,args.worlds,inputs=[state.time,fruit_body,load_bodies,
                          args.contents_shear_force,state.xfrc_applied],device=device)
        report['deterministic_max_row_dofs'] = deterministic_budget.max_row_dofs if deterministic_budget else None
        metric_status = wp.zeros(args.worlds, dtype=wp.int32, device=device)
        metric_iterations = wp.zeros(args.worlds, dtype=wp.int32, device=device)
        metric_residual = wp.zeros(args.worlds, dtype=wp.float32, device=device)
        contact_ids = wp.array(np.arange(data.naconmax, dtype=np.int32), dtype=wp.int32, device=device)
        contact_force = wp.zeros(data.naconmax, dtype=wp.spatial_vector, device=device)
        def frame(state):
            controls(state)
            mjw.step(model, state)
            advance_clock(state)
            if metric:
                wp.launch(accumulate_metric, dim=args.worlds,
                          inputs=[metric.status, metric.iterations, metric.relative_residual,
                                  metric_status, metric_iterations, metric_residual], device=device)
            wp.launch(accumulate_overflow, dim=args.worlds, inputs=[state.overflow, overflow_seen], device=device)
            wp.launch(accumulate_solver_iterations, dim=args.worlds, inputs=[state.solver_niter, max_iterations], device=device)
            refresh_flex_positions(model, state)
            enqueue_surface_embedding(state.flexvert_xpos, bindings, output, valid, source_vertex_offset=offset)
            enqueue_normals(output, topology, scratch, normals)
        def physics_frame(state):
            # Common comparison scope for native CPU, native MJWarp and the
            # composed adapter. Dense asset display and diagnostics are outside.
            controls(state)
            mjw.step(model, state)
            advance_clock(state)
            refresh_flex_positions(model, state)
        def comparison_diagnostics(state):
            if metric:
                wp.launch(accumulate_metric, dim=args.worlds,
                          inputs=[metric.status, metric.iterations, metric.relative_residual,
                                  metric_status, metric_iterations, metric_residual], device=device)
            wp.launch(accumulate_overflow, dim=args.worlds, inputs=[state.overflow, overflow_seen], device=device)
            wp.launch(accumulate_solver_iterations, dim=args.worlds, inputs=[state.solver_niter, max_iterations], device=device)
            enqueue_surface_embedding(state.flexvert_xpos, bindings, output, valid, source_vertex_offset=offset)
            enqueue_normals(output, topology, scratch, normals)
        print("Compiling/warming full bag CUDA frame on disposable state", flush=True)
        if args.contact_policy:
            from deform3d_warp.mujoco_contact import flex_contact_policy
        with ExitStack() as stack:
            if args.contact_policy:
                stack.enter_context(flex_contact_policy(model))
            if args.local_contacts:
                from deform3d_warp.mujoco_local_contact import local_flex_contact_policy
                stack.enter_context(local_flex_contact_policy(model))
            if args.stable_contact_normals:
                from deform3d_warp.mujoco_contact_normals import stable_flex_normals_policy
                stack.enter_context(stable_flex_normals_policy(model))
            if metric:
                stack.enter_context(metric.capture_policy())
            if newton:
                stack.enter_context(newton.capture_policy())
            if deterministic_budget:
                stack.enter_context(deterministic_budget.capture_policy())
            if ordering:
                stack.enter_context(ordering.capture_policy())
            if assembly:
                stack.enter_context(assembly.capture_policy())
            if material:
                stack.enter_context(material.capture_policy())
            if wrench:stack.enter_context(wrench.capture_policy())
            if effective:stack.enter_context(effective.capture_policy())
            frame(warm)
            wp.synchronize_device(device)
            overflow_seen.zero_()
            max_iterations.zero_()
            metric_status.zero_()
            metric_iterations.zero_()
            metric_residual.zero_()
            if newton:
                newton.status.zero_()
                newton.max_relative_residual.zero_()
            if ordering:
                ordering.status.zero_()
            if assembly:
                assembly.status.zero_()
            if material:
                material.status.zero_()
            if wrench:wrench.status.zero_()
            if effective:effective.status.zero_()
            with wp.ScopedCapture(device=device) as capture:
                (physics_frame if args.physics_only_timing else frame)(data)
        print("Captured full bag CUDA frame; beginning evolving trajectory", flush=True)
        event_start, event_end = wp.Event(device=device, enable_timing=True), wp.Event(device=device, enable_timing=True)
        device_ms, wall_ms = [], []
        contacts = np.zeros(4, np.int64)
        flex_error, qpos_error, dense_error, normal_unit_error = 0.0, 0.0, 0.0, 0.0
        min_volume, min_area, max_area = 1.0, 1.0, 1.0
        valid_all = True
        finite = True
        diagnostic_samples = 0
        checkpoints = []
        force_samples, rigid_translation_error = [], 0.0
        per_flex_error = np.zeros(m.nflex)
        time_error = 0.0
        identical_world_spread = 0.0
        maximum_row_dofs = 0
        for first in range(0, steps, args.batch):
            count = min(args.batch, steps - first)
            began = time.perf_counter()
            wp.record_event(event_start)
            for _ in range(count):
                wp.capture_launch(capture.graph)
            wp.record_event(event_end)
            elapsed = wp.get_event_elapsed_time(event_start, event_end)
            device_ms.append(elapsed / count)
            wall_ms.append((time.perf_counter() - began) * 1000 / count)
            if args.physics_only_timing:
                comparison_diagnostics(data)
                # Keep asynchronous diagnostics out of the CPU timing too.
                wp.synchronize_device(device)
            cpu_began = time.perf_counter_ns()
            for _ in range(count):
                drive(m, cpu)
                if args.contents_shear_force:
                    drive_contents_cpu(cpu,fruit_body,package_bodies,args.contents_shear_force)
                mujoco.mj_step(m, cpu)
            mujoco.mj_kinematics(m, cpu)
            mujoco.mj_flex(m, cpu)
            cpu_ms.append((time.perf_counter_ns() - cpu_began) * 1e-6 / count)
            # Test-only diagnostic readbacks outside the measured graph. The
            # production frame above contains no host synchronization/transfers.
            xyz = data.flexvert_xpos.numpy()
            identical_world_spread = max(identical_world_spread, float(np.max(np.linalg.norm(
                xyz.astype(np.float64) - xyz[0].astype(np.float64), axis=2))))
            qpos = data.qpos.numpy()
            body_positions, body_quaternions = data.xpos.numpy(), data.xquat.numpy()
            dense = output.numpy()
            current_normals = normals.numpy()
            finite &= bool(np.isfinite(xyz).all() and np.isfinite(qpos).all()
                           and np.isfinite(data.qvel.numpy()).all() and np.isfinite(dense).all()
                           and np.isfinite(current_normals).all()
                           and np.isfinite(body_positions).all() and np.isfinite(body_quaternions).all())
            normal_unit_error = max(normal_unit_error, float(np.max(np.abs(np.linalg.norm(current_normals, axis=2) - 1.0))))
            valid_all &= bool(np.all(valid.numpy() == 1))
            flex_error = max(flex_error, float(np.max(np.linalg.norm(xyz - cpu.flexvert_xpos[None], axis=2))))
            for flex_id in range(m.nflex):
                start = int(m.flex_vertadr[flex_id])
                end = start + int(m.flex_vertnum[flex_id])
                per_flex_error[flex_id] = max(per_flex_error[flex_id], float(np.max(np.linalg.norm(
                    xyz[:, start:end] - cpu.flexvert_xpos[None, start:end], axis=2))))
            time_error = max(time_error, float(np.max(np.abs(data.time.numpy().astype(np.float64) - cpu.time))))
            qpos_error = max(qpos_error, float(np.max(np.abs(qpos - cpu.qpos[None]))))
            rigid_translation_error = max(rigid_translation_error, float(np.max(np.linalg.norm(body_positions - cpu.xpos[None], axis=2))))
            reference = np.sum(xyz[:, offset + ids] * weights[None, :, :, None], axis=2)
            dense_error = max(dense_error, float(np.max(np.linalg.norm(dense - reference, axis=2))))
            for world in range(args.worlds):
                min_volume = min(min_volume, float(np.min(volumes(xyz[world]) / rest_volumes)))
                ratio = areas(xyz[world]) / rest_areas
                min_area, max_area = min(min_area, float(np.min(ratio))), max(max_area, float(np.max(ratio)))
            active = int(data.nacon.numpy()[0])
            row_counts, active_rows = data.efc.J_rownnz.numpy(), data.nefc.numpy()
            maximum_row_dofs = max(maximum_row_dofs, max(
                int(row_counts[w, :active_rows[w]].max(initial=0)) for w in range(args.worlds)))
            world_mask = data.contact.worldid.numpy()[:active] == 0
            cf = data.contact.flex.numpy()[:active][world_mask]
            cg = data.contact.geom.numpy()[:active][world_mask]
            has_bag = np.any(cf == bag, axis=1)
            current_contacts = np.array([
                np.sum(has_bag & np.any(cf == package, axis=1)),
                np.sum(has_bag & np.any(cg == fruit, axis=1)),
                np.sum(has_bag & np.any(np.isin(cg, pads), axis=1)),
                np.sum(np.all(cf == bag, axis=1))])
            contacts += current_contacts
            # Both force arrays describe the last pre-integration state, even
            # after the geometry-only refresh. Keep their timestamp explicit.
            mjw.contact_force(model, data, contact_ids, True, contact_force)
            gpu_force = contact_force.numpy()[:active, :3].astype(np.float64)
            gpu_geom = data.contact.geom.numpy()[:active]
            gpu_world = data.contact.worldid.numpy()[:active]
            net = np.zeros((args.worlds, 2))
            for world in range(args.worlds):
                for j, pad in enumerate(pads):
                    mask = gpu_world == world
                    signs = (gpu_geom[:, 1] == pad).astype(np.int32) - (gpu_geom[:, 0] == pad).astype(np.int32)
                    net[world, j] = np.sum(gpu_force[mask, 1] * signs[mask])
            cpu_net, local_force = np.zeros(2), np.zeros(6)
            for contact_index, contact in enumerate(cpu.contact):
                for j, pad in enumerate(pads):
                    if pad in contact.geom:
                        mujoco.mj_contactForce(m, cpu, contact_index, local_force)
                        cpu_net[j] += (-1 if contact.geom[0] == pad else 1) * (contact.frame.reshape(3, 3).T @ local_force[:3])[1]
            force_samples.append((cpu_net, net))
            if args.record_cpu_states:
                p = cpu.flexvert_xpos[package_ids]
                ratio = areas(cpu.flexvert_xpos) / rest_areas
                cpu_recorded['points'].append(cpu.flexvert_xpos.copy())
                cpu_recorded['qpos'].append(cpu.qpos.copy())
                cpu_recorded['mocap'].append(cpu.mocap_pos.copy())
                cpu_recorded['fruit_pose'].append(np.r_[cpu.xpos[fruit_body], cpu.xquat[fruit_body]])
                cpu_recorded['package_thickness'].append(float(p[back_face, 1].mean() - p[front_face, 1].mean()))
                cpu_recorded['force_time'].append(float(cpu.time - m.opt.timestep))
                cpu_contacts = [0, 0, 0, 0]
                for c in cpu.contact:
                    cpu_contacts[0] += int(bag in c.flex and package in c.flex)
                    cpu_contacts[1] += int(bag in c.flex and fruit in c.geom)
                    cpu_contacts[2] += int(bag in c.flex and any(pad in c.geom for pad in pads))
                    cpu_contacts[3] += int(c.flex[0] == bag and c.flex[1] == bag)
                cpu_recorded['diagnostics'].append([float(cpu.time), *cpu_contacts, *np.abs(cpu_net),
                    max((max(0., -float(c.dist)) for c in cpu.contact), default=0.),
                    float(np.min(volumes(cpu.flexvert_xpos) / rest_volumes)), float(ratio.min()), float(ratio.max())])
            if args.record_states:
                # World zero is the actual integrated GPU trajectory. No CPU
                # geometry, interpolation or corrective displacement is saved.
                gpu_time = float(data.time.numpy()[0])
                distance = data.contact.dist.numpy()[:active][world_mask]
                package_points = xyz[0, package_ids]
                ratio = areas(xyz[0]) / rest_areas
                recorded['points'].append(xyz[0].copy())
                recorded['qpos'].append(qpos[0].copy())
                recorded['mocap'].append(data.mocap_pos.numpy()[0].copy())
                recorded['fruit_pose'].append(np.r_[body_positions[0, fruit_body],
                                                   body_quaternions[0, fruit_body]])
                recorded['package_thickness'].append(float(
                    package_points[back_face, 1].mean() - package_points[front_face, 1].mean()))
                recorded['force_time'].append(gpu_time - float(m.opt.timestep))
                recorded['diagnostics'].append([gpu_time, *current_contacts, *np.abs(net[0]),
                    max(0., -float(distance.min())) if distance.size else 0.,
                    float(np.min(volumes(xyz[0]) / rest_volumes)), float(ratio.min()), float(ratio.max())])
            checkpoints.append({'time_s': float(cpu.time),
                                'force_time_s': float(cpu.time - m.opt.timestep),
                                'cpu_pad_forces_N': cpu_net.tolist(), 'gpu_pad_forces_N': net.tolist(),
                                'max_flex_delta_m': float(np.max(np.linalg.norm(xyz - cpu.flexvert_xpos[None], axis=2))),
                                'cpu_contacts': int(cpu.ncon), 'gpu_contacts_world0': int(np.sum(world_mask)),
                                'gpu_solver_iterations_world0': int(data.solver_niter.numpy()[0])})
            diagnostic_samples += 1
            if (first + count) // 500 > first // 500:
                print(f'steps={first+count}/{steps} max_flex_cpu_delta_m={flex_error:.6g}', flush=True)
        report.update({"finite": finite, "all_surface_bindings_valid": valid_all,
                       "max_flex_cpu_delta_m": flex_error, "max_qpos_cpu_delta": qpos_error,
                       "max_dense_reference_error_m": dense_error,
                       "max_normal_unit_length_error": normal_unit_error,
                       "package_min_volume_ratio": min_volume, "bag_area_ratio_range": [min_area, max_area],
                       "cpu_warning_counts": cpu.warning.number.tolist(),
                       "gpu_overflow_flags_every_step": overflow_seen.numpy().tolist(),
                       "gpu_max_solver_iterations_every_step": max_iterations.numpy().tolist(),
                       "capacities": {"naconmax": data.naconmax, "naccdmax": data.naccdmax, "njmax": data.njmax},
                       "diagnostic_samples": diagnostic_samples,
                       "checkpoints": checkpoints,
                       "contact_observations_world0": dict(zip(("bag_package", "bag_fruit", "pad_bag", "bag_self"), contacts.tolist())),
                       "cuda_step_ms": {"p50": float(np.median(device_ms)), "p95": float(np.percentile(device_ms, 95))},
                       "host_step_ms": {"p50": float(np.median(wall_ms)), "p95": float(np.percentile(wall_ms, 95))}})
        report["realtime_factor_median"] = float(m.opt.timestep * 1000 / np.median(wall_ms))
        if args.physics_only_timing:
            report['timing_scope'] = ('Controller + physics + integrated kinematics/flex refresh, one world; '
                'host wall includes synchronous completion, CUDA events also reported. '
                'Excludes diagnostics, dense display geometry, rendering, setup and compilation.')
            report['physics_only_timing'] = True
            report['host_physics_total_s'] = float(sum(wall_ms) / 1000)
            report['cuda_physics_total_s'] = float(sum(device_ms) / 1000)
            report['realtime_factor_full_cycle'] = steps * float(m.opt.timestep) / report['host_physics_total_s']
            report['cpu_physics_total_s'] = float(sum(cpu_ms) / 1000)
            report['cpu_realtime_factor_full_cycle'] = steps * float(m.opt.timestep) / report['cpu_physics_total_s']
            report['cpu_step_ms'] = dict(p50=float(np.median(cpu_ms)), p95=float(np.percentile(cpu_ms, 95)))
            report['step_timings_ms'] = dict(host=wall_ms, cuda=device_ms, cpu=cpu_ms)
        report['cpu_parity_tolerance_m'] = args.max_cpu_delta
        report['cpu_parity_passed'] = bool(flex_error <= args.max_cpu_delta)
        ref_forces = np.array([x[0] for x in force_samples])[:, None, :]
        device_forces = np.array([x[1] for x in force_samples])
        force_rms_error = float(np.sqrt(np.mean((device_forces - ref_forces)**2)))
        force_reference_rms = float(np.sqrt(np.mean(ref_forces**2)))
        report['pad_force_rms_delta_N'] = force_rms_error
        report['pad_force_reference_rms_N'] = force_reference_rms
        report['pad_force_relative_rms_delta'] = force_rms_error / max(force_reference_rms, 1e-12)
        report['pad_force_relative_tolerance'] = args.max_force_delta
        report['pad_force_parity_passed'] = report['pad_force_relative_rms_delta'] <= args.max_force_delta
        report['max_body_translation_cpu_delta_m'] = rigid_translation_error
        report['max_per_flex_cpu_delta_m'] = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, i) or str(i):
                                            float(v) for i, v in enumerate(per_flex_error)}
        report['max_simulation_time_delta_s'] = time_error
        report['control_clock_status'] = clocks[id(data)].status.numpy().tolist() if clocks else None
        report['control_clock_time_s'] = clocks[id(data)].elapsed.numpy().tolist() if clocks else None
        report['max_identical_world_flex_spread_m'] = identical_world_spread
        report['max_constraint_row_dofs_observed'] = maximum_row_dofs
        report['solver_budget_exhausted'] = bool(np.any(max_iterations.numpy() >= model.opt.iterations))
        report['cooperative_metric'] = bool(metric)
        report['metric_status_bits_every_step'] = metric_status.numpy().tolist() if metric else None
        report['metric_max_iterations_every_step'] = metric_iterations.numpy().tolist() if metric else None
        report['metric_max_relative_residual_every_step'] = metric_residual.numpy().tolist() if metric else None
        report['parallel_newton'] = bool(newton)
        report['newton_precision'] = args.newton_precision if newton else None
        report['newton_tile_size'] = args.newton_tile if newton else None
        report['newton_diagonal'] = args.newton_diagonal if newton else None
        report['newton_substitution'] = args.newton_substitution if newton else None
        report['newton_panel'] = args.newton_panel if newton else None
        report['newton_trailing'] = args.newton_trailing if newton else None
        report['newton_update_threads'] = args.newton_update_threads if newton else None
        report['conditional_newton_factor'] = args.conditional_newton_factor if newton else False
        report['canonical_constraints'] = bool(ordering)
        report['constraint_ordering_status'] = ordering.status.numpy().tolist() if ordering else None
        report['owned_contact_assembly'] = bool(assembly)
        report['owned_solver_reductions'] = bool(assembly and assembly.solver_ops)
        report['owned_assembly_status'] = assembly.status.numpy().tolist() if assembly else None
        report['owned_material'] = bool(material)
        report['shared_simplex'] = bool(args.shared_simplex)
        report['shared_simplex_core_sha256'] = material.core_sha256 if args.shared_simplex else None
        report['owned_material_reduction'] = 'warp_binned_float_run_to_run' if material else None
        report['owned_material_status'] = material.status.numpy().tolist() if material else None
        report['owned_material_workspace_bytes'] = material.workspace_bytes if material else None
        for name,owner in (('owned_wrench',wrench),('owned_effective',effective)):
            report[name]=bool(owner)
            report[name+'_status']=owner.status.numpy().tolist() if owner else None
            report[name+'_workspace_bytes']=owner.workspace_bytes if owner else None
        report['owned_assembly_workspace_bytes'] = assembly.workspace_bytes if assembly else None
        report['newton_status_bits_every_solve'] = newton.status.numpy().tolist() if newton else None
        report['newton_max_relative_residual_every_solve'] = newton.max_relative_residual.numpy().tolist() if newton else None
        report['unvalidated_metric_budget_override'] = bool(not metric and model.opt.iterations < m.opt.iterations)
        metric_valid = (not metric or not np.any(metric_status.numpy())) and not report['unvalidated_metric_budget_override']
        report['deterministic_support_bound_passed'] = (not deterministic_budget
            or maximum_row_dofs <= deterministic_budget.max_row_dofs)
        report['validation_passed'] = bool((not clocks or not np.any(clocks[id(data)].status.numpy()))
            and finite and valid_all and min_volume > 0 and min_area > 1e-8
            and dense_error < 3e-5 and normal_unit_error < 1e-4 and not np.any(cpu.warning.number)
            and not any(report['gpu_overflow_flags_every_step']) and report['cpu_parity_passed']
            and not report['solver_budget_exhausted']
            and metric_valid
            and report['deterministic_support_bound_passed']
            and (not newton or not np.any(newton.status.numpy()))
            and (not ordering or not np.any(ordering.status.numpy()))
            and (not assembly or not np.any(assembly.status.numpy()))
            and (not material or not np.any(material.status.numpy()))
            and (not wrench or not np.any(wrench.status.numpy()))
            and (not effective or not np.any(effective.status.numpy()))
            and report['pad_force_parity_passed'] and rigid_translation_error <= args.max_cpu_delta
            and (args.duration < 2.5 or np.all(contacts[:3] > 0)))
        if args.record_states:
            import hashlib
            args.record_states.parent.mkdir(parents=True, exist_ok=True)
            # Passing a handle prevents NumPy silently changing the filename.
            with args.record_states.open('wb') as file:
                np.savez_compressed(file, **recorded)
            report['recording'] = {
                'path': str(args.record_states.resolve()),
                'sha256': hashlib.sha256(args.record_states.read_bytes()).hexdigest(),
                'world': 0, 'samples': len(recorded['points']),
                'physics': 'MJWarp CUDA integrated state',
                'state_scope': 'post-integration geometry; contact counts and forces are from the preceding solve',
                'force_scope': 'magnitude of signed net world-Y contact force per pad, including friction; explicit force_time array',
                'package_rest_thickness_m': rest_thickness,
                'package_max_compression_fraction': float(1 - min(recorded['package_thickness']) / rest_thickness)}
        if args.record_cpu_states:
            import hashlib
            args.record_cpu_states.parent.mkdir(parents=True, exist_ok=True)
            with args.record_cpu_states.open('wb') as file:
                np.savez_compressed(file, **cpu_recorded)
            report['cpu_recording'] = {
                'path': str(args.record_cpu_states.resolve()),
                'sha256': hashlib.sha256(args.record_cpu_states.read_bytes()).hexdigest(),
                'samples': len(cpu_recorded['points']), 'physics': 'Native MuJoCo CPU integrated state',
                'package_rest_thickness_m': rest_thickness,
                'package_max_compression_fraction': float(1 - min(cpu_recorded['package_thickness']) / rest_thickness)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        # A validation failure must still exit nonzero under python -O.
        if not report['validation_passed']:
            raise RuntimeError('simulation acceptance failed; inspect the written JSON for shape, force, quality and solver results')
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=ROOT / "examples/mujoco_bag_fast.xml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument('--contact-policy', action='store_true', help='preserve per-element contacts and fuse FPS during CUDA graph construction')
    parser.add_argument('--gpu-iterations', type=int, help='explicit GPU solve budget; every-step exhaustion is a validation failure')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--bounded-determinism', action='store_true', help='topology-bounded deterministic sparse Newton scratch')
    parser.add_argument('--stable-ordering', action='store_true', help='request upstream sorting flags; MJWarp 4.0.1 sorting hooks are stubs')
    parser.add_argument('--cooperative-metric', action='store_true')
    parser.add_argument('--parallel-newton', action='store_true')
    parser.add_argument('--canonical-constraints', action='store_true', help='canonical contact-only sparse Newton rows; independent of upstream sorting flags')
    parser.add_argument('--owned-material', action='store_true', help='experimental topology-owned elastic forces and stiffness reductions')
    parser.add_argument('--shared-simplex', action='store_true', help='execute the shared C++ quadratic simplex law inside owned material kernels')
    parser.add_argument('--owned-assembly', action='store_true', help='experimental output-owned pyramidal contact Hessian and force accumulation')
    parser.add_argument('--owned-solver-reductions', action='store_true', help='unique CSR metric insertion and reproducible warp gradient norm; requires owned assembly')
    parser.add_argument('--owned-wrench',action='store_true',help='owned deterministic sparse body-wrench projection')
    parser.add_argument('--owned-effective',action='store_true',help='owned deterministic implicit flex metric assembly')
    parser.add_argument('--contents-shear-force',type=float,default=0.,help='peak opposed world-X force in newtons on fruit/package; adds fruit world-Z torque of 0.015 m times this force')
    parser.add_argument('--local-contacts', action='store_true', help='experimental local-coordinate flex-pair GJK/EPA')
    parser.add_argument('--stable-contact-normals', action='store_true',
                        help='experimental pre-inflation normal transport for flex-pair GJK')
    parser.add_argument('--precise-controls', action='store_true',
                        help='compensated device clock and float64 control curve evaluation')
    parser.add_argument('--cpu-ccd-tolerance', type=float, help='explicit CPU collision-accuracy experiment; does not change acceptance thresholds')
    parser.add_argument('--gpu-ccd-tolerance', type=float, help='explicit GPU collision-accuracy experiment')
    parser.add_argument('--newton-precision',choices=['float32','float64'],default='float64')
    parser.add_argument('--newton-tile',type=int,choices=[16,32],default=16)
    parser.add_argument('--newton-diagonal',choices=['tiled','register'],default='tiled')
    parser.add_argument('--newton-substitution',choices=['tiled','ordered_warp'],default='tiled')
    parser.add_argument('--newton-panel',choices=['tiled','register'],default='tiled')
    parser.add_argument('--newton-trailing',choices=['tiled','register'],default='tiled')
    parser.add_argument('--newton-update-threads',type=int,choices=[64,128],default=64)
    parser.add_argument('--conditional-newton-factor',action='store_true')
    parser.add_argument('--epa-horizon',type=int,default=24,
                        help='EPA horizon scratch slots for both native flex and convex allocators; math and overflow gates unchanged')
    parser.add_argument("--duration", type=float, default=0.24)
    parser.add_argument('--max-cpu-delta', type=float, default=.005)
    parser.add_argument('--max-force-delta', type=float, default=.1)
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--subdivisions", type=int, default=16)
    parser.add_argument("--output", type=Path, default=ROOT / "results/mjwarp_bag_cuda.json")
    parser.add_argument('--record-states', type=Path, help='record every actual GPU state in world zero for offline rendering; requires --batch 1')
    parser.add_argument('--physics-only-timing', action='store_true', help='common one-world physics timing for side-by-side comparisons; diagnostics and dense geometry remain validated outside the timed graph')
    parser.add_argument('--record-cpu-states', type=Path, help='also record the independently integrated native CPU reference; requires physics-only timing')
    args = parser.parse_args()
    if args.shared_simplex and not args.owned_material:
        parser.error('--shared-simplex requires --owned-material')
    if not np.isfinite(args.contents_shear_force) or args.contents_shear_force<0:
        parser.error('contents shear force must be finite and nonnegative')
    if args.local_contacts and args.stable_contact_normals:
        parser.error('local contacts and stable contact normals are independent experiments')
    if (args.owned_wrench or args.owned_effective) and not args.deterministic:
        parser.error('owned wrench/effective policies require --deterministic')
    if args.owned_assembly and args.bounded_determinism:
        parser.error('--owned-assembly replaces the assembly that --bounded-determinism sizes; choose one')
    if args.owned_solver_reductions and not args.owned_assembly:
        parser.error('--owned-solver-reductions requires --owned-assembly')
    if args.record_states and args.batch != 1:
        parser.error('state recording requires --batch 1')
    if args.physics_only_timing and (args.batch != 1 or args.worlds != 1):
        parser.error('physics-only comparison timing requires --worlds 1 --batch 1')
    if args.record_cpu_states and not args.physics_only_timing:
        parser.error('CPU comparison recording requires --physics-only-timing')
    paths = [p.resolve() for p in (args.output, args.record_states, args.record_cpu_states) if p]
    if len(paths) != len(set(paths)):
        parser.error('output paths must be distinct')
    if args.record_states and args.record_states.resolve() == args.output.resolve():
        parser.error('state and report paths must differ')
    for value in (args.cpu_ccd_tolerance, args.gpu_ccd_tolerance):
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error('CCD tolerances must be finite and positive')
    if args.gpu_iterations is not None and args.gpu_iterations < 1:
        parser.error('gpu-iterations must be positive')
    if not np.isfinite(args.max_force_delta) or args.max_force_delta < 0:
        parser.error('max-force-delta must be finite and nonnegative')
    if args.worlds < 1 or args.batch < 1 or args.subdivisions < 1 or not np.isfinite(args.duration) or args.duration <= 0 or not np.isfinite(args.max_cpu_delta) or args.max_cpu_delta < 0:
        parser.error("worlds, batch, subdivisions and duration must be positive")
    from mjwarp_capacity import epa_horizon_capacity
    with epa_horizon_capacity(args.epa_horizon):
        run(args)
