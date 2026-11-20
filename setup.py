"""Include the shared material header in wheels without duplicating its source."""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        source = Path(__file__).parent / "include/deform3d/simplex_material.hpp"
        target = Path(self.build_lib) / "deform3d_warp/_include/deform3d"
        target.mkdir(parents=True, exist_ok=True)
        self.copy_file(str(source), str(target / source.name))


setup(cmdclass={"build_py": BuildPy})
