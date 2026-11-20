"""Build an opt-in, source-guarded MJWarp 4.0.1 compatibility copy for Pascal.

The installed package is never edited. Kernel argument transport
changes: immutable model array descriptors are loaded from a device struct array
instead of occupying CUDA's 4 KiB kernel parameter space. Contact math stays
unchanged. Generated source, unified diff, and provenance manifest are retained.

Usage:
  python tools/mjwarp_pascal_compat.py --output build/mjwarp_pascal
  python tools/mjwarp_pascal_compat.py --output build/mjwarp_pascal --run \
      tests/optional_mjwarp.py examples/mujoco_cavity.xml build/cavity.bind \
      --device cuda:0 --graph
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

EXPECTED_VERSION = "4.0.1"
EXPECTED_SOURCE_SHA256 = "9f9c62ec5bf99418f3cf3d698bf9eaa09f11f065859f7f35d214b498e1f3e10d"
RELATIVE_SOURCE = Path("mujoco_warp/_src/collision_flex.py")
FORWARD_SOURCE = Path("mujoco_warp/_src/forward.py")
FORWARD_SHA256 = "7a3a345f5ee23143bbf31d734ddf0fb7a9db4b20fd4693c7665ece1f101150c6"
GYRO_FIELDS = ("body_dofadr", "dof_treeid", "ten_J_rownnz", "ten_J_rowadr",
               "ten_J_colind", "M_rownnz", "M_rowadr", "M_colind",
               "efm_K_rownnz", "body_freeadr")
CONVEX_SOURCE = Path("mujoco_warp/_src/collision_convex.py")
CONVEX_SHA256 = "07d12f9ebc3ea64ac29f62d88578c86d21db403dfcf992d70059717e6629190b"
CONVEX_FIELDS = tuple((name, 'wp.vec3' if name in ('mesh_vert', 'mesh_polynormal') else 'int') for name in (
    'mesh_vertadr','mesh_vertnum','mesh_graphadr','mesh_vert','mesh_graph',
    'mesh_polynum','mesh_polyadr','mesh_polynormal','mesh_polyvertadr','mesh_polyvertnum',
    'mesh_polyvert','mesh_polymapadr','mesh_polymapnum','mesh_polymap'))
MESH_FIELDS = (
    ("mesh_vertadr", "int"), ("mesh_vertnum", "int"), ("mesh_graphadr", "int"),
    ("mesh_vert", "wp.vec3"), ("mesh_graph", "int"), ("mesh_pos", "wp.vec3"),
    ("mesh_polynormal", "wp.vec3"), ("mesh_polyvertadr", "int"),
    ("mesh_polyvert", "int"), ("mesh_polymapadr", "int"),
    ("mesh_polymapnum", "int"), ("mesh_polymap", "int"),
)


def configure_pascal_warp():
    """Use Warp's supported scalar CUDA tile math instead of unsupported MathDx.

    Call before loading/compiling simulation kernels. Newton and the contact
    equations remain on CUDA; only their tile linear algebra implementation
    changes. These flags apply to the Python process and affect kernel caches.
    """
    import warp as wp
    if wp.__version__ != "1.19.0":
        raise ValueError(f"Pascal compatibility is tested with Warp 1.19.0 +cu12, got {wp.__version__}")
    wp.config.enable_mathdx_solver = False
    wp.config.enable_mathdx_gemm = False
    print("Pascal Warp configuration: cooperative CUDA tile solvers and scalar CUDA GEMM; MathDx disabled", flush=True)


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"source layout mismatch: expected one occurrence of {old!r}")
    return source.replace(old, new, 1)


def transform(source):
    """Apply a surgical source transform; exact upstream bytes are required."""
    if hashlib.sha256(source.encode()).hexdigest() != EXPECTED_SOURCE_SHA256:
        raise ValueError("MJWarp collision_flex.py hash mismatch; refusing an unreviewed upstream source")
    tree = ast.parse(source)
    kernel_factory = next(node for node in tree.body
                          if isinstance(node, ast.FunctionDef) and node.name == "_flex_narrowphase_elem_detect")
    launcher = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_detect_elem_geom_candidates")
    lines = source.splitlines(keepends=True)
    factory_source = "".join(lines[kernel_factory.lineno - 1:kernel_factory.end_lineno])
    launcher_source = "".join(lines[launcher.lineno - 1:launcher.end_lineno])
    transformed_factory = factory_source
    transformed_launcher = launcher_source
    for index, (name, dtype) in enumerate(MESH_FIELDS):
        transformed_factory = _replace_once(
            transformed_factory, f"    {name}: wp.array[{dtype}],\n",
            "    mesh_descriptors: wp.array[_Deform3dMeshDescriptors],\n" if index == 0 else "")
        transformed_launcher = _replace_once(
            transformed_launcher, f"      m.{name},\n",
            "      _deform3d_mesh_descriptors(m),\n" if index == 0 else "")
    aliases = "    mesh_tables = mesh_descriptors[0]\n" + "".join(
        f"    {name} = mesh_tables.{name}\n" for name, _ in MESH_FIELDS)
    transformed_factory = _replace_once(transformed_factory,
                                         "    worldid, elemid = wp.tid()\n",
                                         "    worldid, elemid = wp.tid()\n" + aliases)
    struct_source = '''# deform3d opt-in Pascal transport compatibility; contact math is unchanged.
DEFORM3D_PASCAL_COMPAT = True


@wp.struct
class _Deform3dMeshDescriptors:
'''
    struct_source += "".join(f"  {name}: wp.array[{dtype}]\n" for name, dtype in MESH_FIELDS)
    struct_source += '''

def _deform3d_mesh_descriptors(m):
  """Prepare once outside capture; retain every referenced source array."""
  fields = ''' + repr(tuple(name for name, _ in MESH_FIELDS)) + '''
  arrays = tuple(getattr(m, name) for name in fields)
  signature = tuple((id(a), a.ptr, a.shape, a.strides, a.dtype) for a in arrays)
  cached = getattr(m, "_deform3d_mesh_descriptor_cache", None)
  if cached is not None:
    if cached[0] != signature:
      raise RuntimeError("mesh descriptor storage changed; create a new MJWarp model and recapture graphs")
    return cached[1]
  device = arrays[0].device
  if device.is_capturing:
    raise RuntimeError("prepare MJWarp Pascal mesh descriptors with a warmup step before graph capture")
  if any(a.device != device for a in arrays):
    raise ValueError("MJWarp model mesh arrays must share one device")
  descriptor = _Deform3dMeshDescriptors()
  for name, array in zip(fields, arrays):
    setattr(descriptor, name, array)
  packed = wp.array([descriptor], dtype=_Deform3dMeshDescriptors, device=device)
  m._deform3d_mesh_descriptor_cache = (signature, packed, arrays)
  return packed


'''
    source = _replace_once(source, factory_source, transformed_factory)
    source = _replace_once(source, launcher_source, transformed_launcher)
    source = _replace_once(source, "@cache_kernel\ndef _flex_narrowphase_elem_detect", struct_source +
                           "@cache_kernel\ndef _flex_narrowphase_elem_detect")
    ast.parse(source)
    return source


def transform_forward(source):
    """Pack integer model descriptors; preserve both forward and adjoint math.

    Warp emits an adjoint signature even for forward-only execution. The free
    gyro kernel's adjoint exceeded Pascal's 4096-byte parameter limit. These
    integer topology tables are not differentiable values and can be indirected
    without changing the differentiation path of any floating-point input.
    """
    if hashlib.sha256(source.encode()).hexdigest() != FORWARD_SHA256:
        raise ValueError("MJWarp forward.py hash mismatch; refusing unreviewed source")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "_discrete_free_gyro_kernel")
    launcher = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_launch_discrete_free_gyro")
    old_factory = "".join(lines[factory.lineno - 1:factory.end_lineno])
    old_launcher = "".join(lines[launcher.lineno - 1:launcher.end_lineno])
    new_factory, new_launcher = old_factory, old_launcher
    for index, name in enumerate(GYRO_FIELDS):
        new_factory = _replace_once(new_factory, f"    {name}: wp.array[int],\n",
            "    gyro_descriptors: wp.array[_Deform3dGyroDescriptors],\n" if index == 0 else "")
        new_launcher = _replace_once(new_launcher, f"      m.{name},\n",
            "      _deform3d_gyro_descriptors(m),\n" if index == 0 else "")
    new_factory = _replace_once(new_factory, "    worldid, freeid = wp.tid()\n",
        "    worldid, freeid = wp.tid()\n    tables = gyro_descriptors[0]\n" +
        "".join(f"    {name} = tables.{name}\n" for name in GYRO_FIELDS))
    helper = "@wp.struct\nclass _Deform3dGyroDescriptors:\n" + "".join(
        f"  {name}: wp.array[int]\n" for name in GYRO_FIELDS)
    helper += '''

def _deform3d_gyro_descriptors(m):
  fields = ''' + repr(GYRO_FIELDS) + '''
  arrays = tuple(getattr(m, name) for name in fields)
  signature = tuple((id(a), a.ptr, a.shape, a.strides, a.dtype) for a in arrays)
  cached = getattr(m, "_deform3d_gyro_descriptor_cache", None)
  if cached is not None:
    if cached[0] != signature:
      raise RuntimeError("gyro descriptor storage changed; rebuild model and recapture")
    return cached[1]
  device = arrays[0].device
  if device.is_cuda and wp.get_stream(device).is_capturing:
    raise RuntimeError("gyro descriptors require warmup before graph capture")
  descriptor = _Deform3dGyroDescriptors()
  for name, array in zip(fields, arrays):
    setattr(descriptor, name, array)
  packed = wp.array([descriptor], dtype=_Deform3dGyroDescriptors, device=device)
  m._deform3d_gyro_descriptor_cache = (signature, packed, arrays)
  return packed


'''
    source = _replace_once(source, old_factory, new_factory)
    source = _replace_once(source, old_launcher, new_launcher)
    source = _replace_once(source, "@cache_kernel\ndef _discrete_free_gyro_kernel",
                           helper + "@cache_kernel\ndef _discrete_free_gyro_kernel")
    ast.parse(source)
    return source


def transform_convex(source):
    """Keep convex CCD equations intact while reducing its launch signature."""
    if hashlib.sha256(source.encode()).hexdigest() != CONVEX_SHA256:
        raise ValueError("MJWarp collision_convex.py hash mismatch; refusing unreviewed source")
    tree = ast.parse(source); lines = source.splitlines(keepends=True)
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'ccd_kernel_builder')
    kernel = next(n for n in factory.body if isinstance(n, ast.FunctionDef) and n.name == 'ccd_kernel')
    launcher = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'convex_narrowphase')
    old_kernel = ''.join(lines[kernel.lineno - 1:kernel.end_lineno])
    old_launcher = ''.join(lines[launcher.lineno - 1:launcher.end_lineno])
    marker = '  # Launch non-heightfield collision kernels (no hfield args, 78 args total)'
    split = old_launcher.index(marker)
    prefix, new_launcher = old_launcher[:split], old_launcher[split:]
    new_kernel = old_kernel
    for index, (name, dtype) in enumerate(CONVEX_FIELDS):
        new_kernel = _replace_once(new_kernel, f'    {name}: wp.array[{dtype}],\n',
            '    mesh_descriptors: wp.array[_Deform3dConvexDescriptors],\n' if index == 0 else '')
        new_launcher = _replace_once(new_launcher, f'          m.{name},\n',
            '          _deform3d_convex_descriptors(m),\n' if index == 0 else '')
    new_kernel = _replace_once(new_kernel, '    tid = wp.tid()\n',
        '    tid = wp.tid()\n    tables = mesh_descriptors[0]\n' +
        ''.join(f'    {name} = tables.{name}\n' for name, _ in CONVEX_FIELDS))
    helper = '@wp.struct\nclass _Deform3dConvexDescriptors:\n' + ''.join(
        f'  {name}: wp.array[{dtype}]\n' for name, dtype in CONVEX_FIELDS)
    helper += '''

def _deform3d_convex_descriptors(m):
  fields = ''' + repr(tuple(n for n,_ in CONVEX_FIELDS)) + '''
  arrays = tuple(getattr(m, name) for name in fields)
  signature = tuple((id(a), a.ptr, a.shape, a.strides, a.dtype) for a in arrays)
  cached = getattr(m, "_deform3d_convex_descriptor_cache", None)
  if cached is not None:
    if cached[0] != signature:
      raise RuntimeError("convex descriptor storage changed; rebuild model and recapture")
    return cached[1]
  device = arrays[0].device
  if device.is_cuda and wp.get_stream(device).is_capturing:
    raise RuntimeError("convex descriptors require warmup before graph capture")
  descriptor = _Deform3dConvexDescriptors()
  for name, array in zip(fields, arrays):
    setattr(descriptor, name, array)
  packed = wp.array([descriptor], dtype=_Deform3dConvexDescriptors, device=device)
  m._deform3d_convex_descriptor_cache = (signature, packed, arrays)
  return packed


'''
    source = _replace_once(source, old_kernel, new_kernel)
    source = _replace_once(source, old_launcher, prefix + new_launcher)
    source = _replace_once(source, '@cache_kernel\ndef ccd_kernel_builder',
                           helper + '@cache_kernel\ndef ccd_kernel_builder')
    ast.parse(source)
    return source


def build(output: Path):
    dist = metadata.distribution("mujoco-warp")
    if dist.version != EXPECTED_VERSION:
        raise ValueError(f"requires mujoco-warp=={EXPECTED_VERSION}; installed {dist.version}")
    installed = Path(dist.locate_file("mujoco_warp")).resolve()
    original_path = installed / "_src/collision_flex.py"
    original = original_path.read_text()
    patched = transform(original)
    forward_original = (installed / "_src/forward.py").read_text()
    forward_patched = transform_forward(forward_original)
    forward_hash = hashlib.sha256(forward_patched.encode()).hexdigest()
    convex_original = (installed / '_src/collision_convex.py').read_text()
    convex_patched = transform_convex(convex_original)
    convex_hash = hashlib.sha256(convex_patched.encode()).hexdigest()
    output = output.resolve()
    if output == installed or installed in output.parents or output in installed.parents:
        raise ValueError("output must be separate from the installed MJWarp package")
    manifest_path = output / "deform3d_pascal_compat.json"
    patched_hash = hashlib.sha256(patched.encode()).hexdigest()
    if manifest_path.is_file() and (output / RELATIVE_SOURCE).is_file():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("kind") == "deform3d-opt-in-mjwarp-pascal-compat"
                and manifest.get("mujoco_warp_version") == dist.version
                and manifest.get("original_sha256") == EXPECTED_SOURCE_SHA256
                and manifest.get("patched_sha256") == patched_hash
                and manifest.get("forward_patched_sha256") == forward_hash
                and manifest.get('convex_patched_sha256') == convex_hash
                and (output / CONVEX_SOURCE).is_file()
                and hashlib.sha256((output / CONVEX_SOURCE).read_bytes()).hexdigest() == convex_hash
                and (output / FORWARD_SOURCE).is_file()
                and hashlib.sha256((output / FORWARD_SOURCE).read_bytes()).hexdigest() == forward_hash
                and hashlib.sha256((output / RELATIVE_SOURCE).read_bytes()).hexdigest() == patched_hash):
            print(f"Using verified opt-in MJWarp copy: {output}", flush=True)
            return output
    if (output / "mujoco_warp").exists() and not manifest_path.is_file():
        raise ValueError("refusing to overwrite an output package without a deform3d compatibility manifest")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(installed, output / "mujoco_warp", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (output / RELATIVE_SOURCE).write_text(patched)
    (output / FORWARD_SOURCE).write_text(forward_patched)
    (output / CONVEX_SOURCE).write_text(convex_patched)
    patch = "".join(difflib.unified_diff(original.splitlines(keepends=True), patched.splitlines(keepends=True),
                                        fromfile="a/" + str(RELATIVE_SOURCE), tofile="b/" + str(RELATIVE_SOURCE)))
    patch += "".join(difflib.unified_diff(forward_original.splitlines(keepends=True),
                     forward_patched.splitlines(keepends=True),
                     fromfile="a/" + str(FORWARD_SOURCE), tofile="b/" + str(FORWARD_SOURCE)))
    patch += ''.join(difflib.unified_diff(convex_original.splitlines(keepends=True),
                     convex_patched.splitlines(keepends=True),
                     fromfile='a/' + str(CONVEX_SOURCE), tofile='b/' + str(CONVEX_SOURCE)))
    (output / "mjwarp_pascal.patch").write_text(patch)
    manifest = {"kind": "deform3d-opt-in-mjwarp-pascal-compat", "mujoco_warp_version": dist.version,
                "installed_source": str(original_path), "original_sha256": EXPECTED_SOURCE_SHA256,
                "patched_sha256": patched_hash,
                "forward_original_sha256": FORWARD_SHA256,
                "forward_patched_sha256": forward_hash,
                "gyro_packed_integer_fields": list(GYRO_FIELDS),
                'convex_original_sha256': CONVEX_SHA256,
                'convex_patched_sha256': convex_hash,
                "changed_kernel": "_flex_narrowphase_elem_detect", "packed_fields": [name for name, _ in MESH_FIELDS],
                "method": "one device-resident descriptor struct replaces twelve by-value array descriptors",
                "installed_package_modified": False}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared opt-in MJWarp copy: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/mjwarp_pascal"))
    parser.add_argument("--run", nargs=argparse.REMAINDER, help="Python script and arguments to run with the compatibility copy")
    parser.add_argument("--child-run", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_run:
        configure_pascal_warp()
        sys.argv = args.child_run
        sys.path.insert(0, str(Path(args.child_run[0]).resolve().parent))
        runpy.run_path(args.child_run[0], run_name="__main__")
        return 0
    output = build(args.output)
    if args.run:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(output) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return subprocess.call([sys.executable, "-u", str(Path(__file__).resolve()), "--child-run", *args.run], env=env)
    print(f"Use PYTHONPATH={output} to select this copy explicitly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
