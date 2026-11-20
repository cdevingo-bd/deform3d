from deform3d_warp.resources import simplex_material_header


def test_shared_header_available():
    header = simplex_material_header()
    assert "SimplexMaterial" in header.read_text()
