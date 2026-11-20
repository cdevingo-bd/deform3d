"""Validated C++ binding-file reader shared by CPU/GPU MuJoCo examples."""
from pathlib import Path
import struct
import mujoco
import numpy as np

MAGIC = b"D3EMB003"
RECORD = np.dtype([("indices", "<u4", (4,)), ("weights", "<f4", (4,))])


def topology_hash(indices):
    value = 14695981039346656037
    for item in indices:
        for shift in (0, 8, 16, 24):
            value ^= (int(item) >> shift) & 255
            value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def rest_hash(rest_float32):
    value = 14695981039346656037
    for byte in np.ascontiguousarray(rest_float32, dtype="<f4").tobytes():
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def read_bindings(path, model: mujoco.MjModel):
    path = Path(path)
    with path.open("rb") as f:
        header = f.read(40)
        if len(header) != 40 or header[:8] != MAGIC:
            raise ValueError("invalid binding header")
        count, flex_id, vertadr, vertnum, expected_topology, expected_rest = \
            struct.unpack("<IIIIQQ", header[8:])
        if count == 0 or flex_id >= model.nflex or model.flex_dim[flex_id] != 3:
            raise ValueError("binding flex is missing or is not tetrahedral")
        if vertadr != model.flex_vertadr[flex_id] or vertnum != model.flex_vertnum[flex_id]:
            raise ValueError("binding/model flex vertex range mismatch")
        adr = int(model.flex_elemdataadr[flex_id])
        elem_count = int(model.flex_elemnum[flex_id])
        if topology_hash(model.flex_elem[adr:adr + 4 * elem_count]) != expected_topology:
            raise ValueError("binding/model tetra topology mismatch")
        reference = mujoco.MjData(model)
        mujoco.mj_forward(model, reference)
        if rest_hash(reference.flexvert_xpos[vertadr:vertadr + vertnum]) != expected_rest:
            raise ValueError("binding/model rest positions mismatch")
        if count > 100_000_000 or count * RECORD.itemsize != path.stat().st_size - 40:
            raise ValueError("binding file length mismatch")
        records = np.fromfile(f, dtype=RECORD, count=count)
    if np.any(records["indices"] >= vertnum) or not np.isfinite(records["weights"]).all():
        raise ValueError("invalid binding index or weight")
    if not np.allclose(records["weights"].sum(axis=1), 1.0, atol=2e-5):
        raise ValueError("binding weights do not sum to one")
    return flex_id, int(vertadr), records
