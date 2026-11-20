"""Check wheel contents and import its installed package outside the checkout.

The current interpreter must already have numpy and warp-lang. No installer or
network access is used: the wheel is unpacked into a temporary site directory.
"""

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(args.wheel) as wheel, tempfile.TemporaryDirectory() as directory:
        header = "deform3d_warp/_include/deform3d/simplex_material.hpp"
        assert hashlib.sha256(wheel.read(header)).digest() == hashlib.sha256(
            (root / "include/deform3d/simplex_material.hpp").read_bytes()
        ).digest(), "wheel header differs from the C++ source"
        assert any(n.endswith("/LICENSE") for n in wheel.namelist()), "MIT license missing"
        assert any("reproducible_floating_sums-LICENSE.txt" in n for n in wheel.namelist())
        wheel.extractall(directory)
        # Carry dependency import roots, but exclude this checkout and its source tree.
        paths = [p for p in sys.path if p and not Path(p).resolve().is_relative_to(root)]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([directory, *paths]))
        code = '''from pathlib import Path
import deform3d_warp
from deform3d_warp.resources import simplex_material_header
assert Path.cwd() in Path(deform3d_warp.__file__).parents
assert "_include" in simplex_material_header().parts
from deform3d_warp import DeviceBindings
b = DeviceBindings([[0,1,2,3]], [[.25,.25,.25,.25]], source_vertex_count=4)
assert b.count == 1
print("PASS: installed wheel imports and finds the shared C++ header outside the source checkout")
'''
        subprocess.run([sys.executable, "-c", code], cwd=directory, env=env, check=True)


if __name__ == "__main__":
    main()
