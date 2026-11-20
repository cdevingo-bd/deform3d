"""Configure C++ integration tests against the active Python MuJoCo install."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    args = parser.parse_args()
    import mujoco

    root = Path(__file__).resolve().parents[1]
    package = Path(mujoco.__file__).resolve().parent
    patterns = {"linux": "libmujoco.so.*", "darwin": "libmujoco*.dylib", "win32": "mujoco.lib"}
    if sys.platform not in patterns:
        parser.error(f"unsupported platform: {sys.platform}")
    libraries = sorted({p.resolve() for p in package.glob(patterns[sys.platform])})
    if len(libraries) != 1:
        parser.error(f"expected one MuJoCo library in {package}; found {libraries}")
    subprocess.run([
        "cmake", "-S", str(root), "-B", str(args.build_dir), "-DCMAKE_BUILD_TYPE=Release",
        "-DDEFORM3D_BUILD_BENCH=OFF", "-DDEFORM3D_BUILD_MUJOCO_TEST=ON",
        f"-DMUJOCO_INCLUDE_DIR={package / 'include'}", f"-DMUJOCO_LIBRARY={libraries[0]}",
    ], check=True)


if __name__ == "__main__":
    main()
