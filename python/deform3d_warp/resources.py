"""Locate headers in a source checkout or an installed wheel."""

from pathlib import Path


def simplex_material_header() -> Path:
    """Return the shared material header; fail before compiling if it is missing."""
    package = Path(__file__).resolve().parent
    candidates = (
        package / "_include/deform3d/simplex_material.hpp",
        package.parents[1] / "include/deform3d/simplex_material.hpp",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("deform3d installation is missing simplex_material.hpp; reinstall the wheel")
