"""Extract oriented tetra boundary faces and tessellate a dense visual surface.

Setup utility only. Vertices are shared within each original face and duplicated
across original face seams. Returned geometry uses the current world-space pose.
"""
import numpy as np


def dense_surface(model, data, flex_id, subdivisions):
    if subdivisions < 1:
        raise ValueError("subdivisions must be positive")
    if model.flex_dim[flex_id] != 3:
        raise ValueError("a tetrahedral flex is required")
    offset = int(model.flex_vertadr[flex_id])
    count = int(model.flex_vertnum[flex_id])
    rest = data.flexvert_xpos[offset:offset + count]
    start = int(model.flex_elemdataadr[flex_id])
    tetra = model.flex_elem[start:start + 4 * int(model.flex_elemnum[flex_id])].reshape(-1, 4)
    faces = {}
    for tet in tetra:
        for opposite in range(4):
            ids = [int(tet[j]) for j in range(4) if j != opposite]
            a, b, c = rest[ids]
            if np.dot(np.cross(b - a, c - a), rest[tet[opposite]] - a) > 0:
                ids[1], ids[2] = ids[2], ids[1]
            faces.setdefault(tuple(sorted(ids)), []).append(ids)
    if any(len(group) > 2 for group in faces.values()):
        raise ValueError("nonmanifold tetrahedral boundary")
    boundary = [group[0] for group in faces.values() if len(group) == 1]
    n = subdivisions
    uv = np.asarray([(i / n, j / n) for i in range(n + 1) for j in range(n + 1 - i)])
    def index(i, j):
        return i * (n + 1) - i * (i - 1) // 2 + j
    local_faces = []
    for i in range(n):
        for j in range(n - i):
            local_faces.append((index(i, j), index(i + 1, j), index(i, j + 1)))
            if i + j + 2 <= n:
                local_faces.append((index(i + 1, j), index(i + 1, j + 1), index(i, j + 1)))
    local_faces = np.asarray(local_faces, dtype=np.int32)
    points, triangles = [], []
    for face_index, ids in enumerate(boundary):
        a, b, c = rest[ids]
        points.append(a + uv[:, :1] * (b - a) + uv[:, 1:] * (c - a))
        triangles.append(local_faces + face_index * len(uv))
    return np.concatenate(points).astype(np.float32), np.concatenate(triangles)
