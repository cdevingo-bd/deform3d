"""Locate bundled evidence after moving a source checkout to another machine."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def recorded_file(recorded, *, folder):
    """Prefer the transferred copy; the caller must still verify its checksum."""
    path = Path(recorded)
    bundled = Path(folder) / path.name
    if bundled.is_file():
        return bundled
    if not path.is_absolute():
        path = ROOT / path
    if path.is_file():
        return path
    raise FileNotFoundError(f"Missing recorded input: {bundled} (original: {recorded})")


def comparison_font(size, *, bold=False):
    """Use the original font when available, then an installed portable fallback."""
    from PIL import ImageFont
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = ["/usr/share/fonts/truetype/dejavu/" + name, name,
                  "arialbd.ttf" if bold else "arial.ttf",
                  "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
                  "/System/Library/Fonts/Supplemental/Arial.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)
