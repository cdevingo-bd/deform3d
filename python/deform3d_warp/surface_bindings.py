"""Validated portable triangle-shell bindings exported by the C++ surface tool.

Header: 8-byte D3SUR001, four little-endian uint32 fields (record count, flex ID,
vertex offset, vertex count), then uint64 topology and rest-float32 FNV-1a hashes.
Records: three local uint32 indices, three float32 barycentrics, one signed
float32 normal offset in scene length units; exactly 28 bytes without padding.
"""
from pathlib import Path
import struct

import mujoco
import numpy as np

from .bindings import topology_hash, rest_hash

MAGIC = b"D3SUR001"
RECORD = np.dtype([("indices", "<u4", (3,)), ("weights", "<f4", (3,)),
                   ("normal_offset", "<f4")], align=False)
assert RECORD.itemsize == 28


def read_surface_bindings(path, model):
    """Return (flex_id, model-global vertex offset, immutable prepared records).

    Validation requires the model's initial reference pose, ordered topology and
    vertex indexing to match the export. Hashes detect accidental mismatches;
    they are not cryptographic authenticity checks. Setup may allocate/read back.
    """
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(40)
        if len(header) != 40 or header[:8] != MAGIC:
            raise ValueError("invalid surface binding header")
        count, flex_id, offset, vertex_count, expected_topology, expected_rest = \
            struct.unpack("<IIIIQQ", header[8:])
        if count == 0 or count > 100_000_000 or flex_id >= model.nflex or model.flex_dim[flex_id] != 2:
            raise ValueError("surface binding flex is missing or is not triangular")
        if offset != model.flex_vertadr[flex_id] or vertex_count != model.flex_vertnum[flex_id]:
            raise ValueError("surface binding vertex range mismatch")
        if path.stat().st_size != 40 + count * RECORD.itemsize:
            raise ValueError("surface binding file length mismatch")
        address = int(model.flex_elemdataadr[flex_id])
        triangle_count = int(model.flex_elemnum[flex_id])
        triangles = model.flex_elem[address:address + 3 * triangle_count].reshape(-1, 3)
        if topology_hash(triangles.reshape(-1)) != expected_topology:
            raise ValueError("surface binding topology mismatch")
        reference = mujoco.MjData(model)
        mujoco.mj_forward(model, reference)
        if rest_hash(reference.flexvert_xpos[offset:offset + vertex_count]) != expected_rest:
            raise ValueError("surface binding reference pose mismatch")
        records = np.fromfile(stream, dtype=RECORD, count=count)
        if len(records) != count:
            raise ValueError("surface binding file truncated during read")
    weights = records["weights"]
    if np.any(records["indices"] >= vertex_count) or not np.isfinite(weights).all() or \
            not np.isfinite(records["normal_offset"]).all():
        raise ValueError("invalid surface binding indices or numeric fields")
    if np.any(weights < 0) or np.any(weights > 1) or \
            not np.allclose(weights.sum(axis=1, dtype=np.float64), 1., rtol=0., atol=2e-5):
        raise ValueError("invalid surface barycentric weights")
    # Winding is meaningful for the signed offset: require an ordered source
    # triangle, rather than merely three individually in-range vertex IDs.
    known = {tuple(map(int, row)) for row in triangles}
    if any(tuple(map(int, row)) not in known for row in np.unique(records["indices"], axis=0)):
        raise ValueError("binding indices do not name an oriented source triangle")
    records.flags.writeable = False
    return int(flex_id), int(offset), records
