"""Experimental banded direct solve of the discrete flex effective metric.

Same scope and contract as ``mujoco_effective_dense`` (``M + diag + K`` only, DISCRETE
integrator, CUDA), but exploits the fixed sparsity pattern: dofs are reordered once with reverse
Cuthill-McKee so M_hat is narrow-banded, the tile Cholesky only visits tiles inside that band, and
the residual gate is an O(nnz) sparse product instead of an O(n^2) dense one.

The reordering depends only on model topology (``M_mulm_*`` and ``efm_K_*`` index arrays), so it
is computed once at construction. ``bandwidth_tiles`` and ``speedup_hint`` report what it found;
``worthwhile(model)`` says whether banding is expected to beat the dense path.

Reviewed against MuJoCo Warp 3.17.0 and Warp 1.16 using the 'tiled' kernels (the same ones as the
dense path). Float32 only; ``status`` bit 1 = invalid factor/output, bit 2 = residual > 1e-4.
"""

from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import warp as wp

from .mujoco_effective_dense import _EffectiveSolvePolicy, _rhs, supported
from .mujoco_newton import _kernels


def reverse_cuthill_mckee(adjacency):
    """Reverse Cuthill-McKee ordering of an undirected graph given as adjacency sets."""
    n = len(adjacency)
    degree = np.array([len(a) for a in adjacency])
    visited = np.zeros(n, dtype=bool)
    order = []
    for start in np.argsort(degree, kind="stable"):
        if visited[start]:
            continue
        visited[start] = True
        queue = [int(start)]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            order.append(node)
            for nb in sorted(adjacency[node], key=lambda v: (degree[v], v)):
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
    return np.array(order[::-1], dtype=np.int32)


def _structure(model):
    """Host copies of the fixed sparsity: adjacency over dofs from M gather and K CSR."""
    nv = model.nv
    m_rowadr = model.M_mulm_rowadr.numpy()
    m_col = model.M_mulm_col.numpy()
    k_rownnz = model.efm_K_rownnz.numpy()
    k_rowadr = model.efm_K_rowadr.numpy()
    k_colind = model.efm_K_colind.numpy()
    adjacency = [set() for _ in range(nv)]
    for row in range(nv):
        for c in m_col[m_rowadr[row]:m_rowadr[row + 1]]:
            if c != row:
                adjacency[row].add(int(c)); adjacency[int(c)].add(row)
        for c in k_colind[k_rowadr[row]:k_rowadr[row] + k_rownnz[row]]:
            if c != row:
                adjacency[row].add(int(c)); adjacency[int(c)].add(row)
    return adjacency


def plan(model, tile_size):
    """Return (perm, invperm, bandwidth_in_tiles) for a model's effective-metric pattern."""
    adjacency = _structure(model)
    perm = reverse_cuthill_mckee(adjacency)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(model.nv, dtype=np.int32)
    band = 0
    for row, neighbours in enumerate(adjacency):
        for c in neighbours:
            band = max(band, abs(int(inv[row]) // tile_size - int(inv[c]) // tile_size))
    return perm, inv, band


def worthwhile(model, tile_size=16):
    """True when the banded path should visit clearly fewer tiles than the dense one."""
    ok, _ = supported(model)
    if not ok:
        return False
    _, _, band = plan(model, tile_size)
    blocks = (model.nv + tile_size - 1) // tile_size
    return 2 * band + 1 < blocks


@wp.kernel(enable_backward=False)
def _assemble_banded(
    perm: wp.array(dtype=wp.int32), invperm: wp.array(dtype=wp.int32), nv: int, tile: int,
    band: int, size: int,
    m_rowadr: wp.array(dtype=wp.int32), m_col: wp.array(dtype=wp.int32),
    m_madr: wp.array(dtype=wp.int32), k_rownnz: wp.array(dtype=wp.int32),
    k_rowadr: wp.array(dtype=wp.int32), k_colind: wp.array(dtype=wp.int32),
    M: wp.array2d(dtype=wp.float32), diag: wp.array2d(dtype=wp.float32),
    kval: wp.array2d(dtype=wp.float32), out: wp.array3d(dtype=wp.float32),
):
    """One thread per (world, permuted row): write the lower band window, identity on padding."""
    w, rp = wp.tid()
    row_tile = rp // tile
    lo = wp.max(0, (row_tile - band) * tile)
    hi = (row_tile + 1) * tile
    for j in range(lo, hi):
        out[w, rp, j] = 0.0
    if rp >= nv:
        out[w, rp, rp] = 1.0
        return
    r = perm[rp]
    for k in range(m_rowadr[r], m_rowadr[r + 1]):
        cp = invperm[m_col[k]]
        if cp // tile <= row_tile:
            out[w, rp, cp] += M[w, m_madr[k]]
    out[w, rp, rp] += diag[w, r]
    adr = k_rowadr[r]
    for t in range(k_rownnz[r]):
        cp = invperm[k_colind[adr + t]]
        if cp // tile <= row_tile:
            out[w, rp, cp] += kval[w, adr + t]


@wp.kernel(enable_backward=False)
def _permute_rhs(perm: wp.array(dtype=wp.int32), nv: int, rhs: wp.array2d(dtype=wp.float32),
                 x: wp.array2d(dtype=wp.float32)):
    w, rp = wp.tid()
    value = float(0.0)
    if rp < nv:
        value = rhs[w, perm[rp]]
    x[w, rp] = value


@lru_cache(None)
def _banded_substitute(size, tile, band):
    @wp.kernel(module="unique", enable_backward=False,
               module_options={"enable_mathdx_solver": False, "enable_mathdx_gemm": False})
    def substitute(a: wp.array3d(dtype=wp.float32), solution: wp.array3d(dtype=wp.float32)):
        w = wp.tid()
        x = wp.tile_load(solution[w], shape=(size, 1), offset=(0, 0), storage="shared", bounds_check=False)
        blocks = int(size // tile)
        for block in range(blocks):
            k = block * tile
            part = wp.tile_view(x, shape=(tile, 1), offset=(k, 0))
            for previous in range(wp.max(0, block - band), block):
                j = previous * tile
                l = wp.tile_load(a[w], shape=(tile, tile), offset=(k, j), storage="shared", bounds_check=False)
                y = wp.tile_view(x, shape=(tile, 1), offset=(j, 0))
                wp.tile_matmul(l, y, part, alpha=wp.float32(-1.0))
            l = wp.tile_load(a[w], shape=(tile, tile), offset=(k, k), storage="shared", bounds_check=False)
            wp.tile_lower_solve_inplace(l, part)
        for reverse in range(blocks):
            block = blocks - 1 - reverse
            k = block * tile
            part = wp.tile_view(x, shape=(tile, 1), offset=(k, 0))
            for following in range(block + 1, wp.min(blocks, block + 1 + band)):
                j = following * tile
                l = wp.tile_load(a[w], shape=(tile, tile), offset=(j, k), storage="shared", bounds_check=False)
                y = wp.tile_view(x, shape=(tile, 1), offset=(j, 0))
                wp.tile_matmul(wp.tile_transpose(l), y, part, alpha=wp.float32(-1.0))
            l = wp.tile_load(a[w], shape=(tile, tile), offset=(k, k), storage="shared", bounds_check=False)
            wp.tile_upper_solve_inplace(wp.tile_transpose(l), part)
        wp.tile_store(solution[w], x, bounds_check=False)
    return substitute


@wp.kernel(enable_backward=False)
def _unpermute(perm: wp.array(dtype=wp.int32), nv: int, x: wp.array2d(dtype=wp.float32),
               qacc: wp.array2d(dtype=wp.float32)):
    w, rp = wp.tid()
    if rp < nv:
        qacc[w, perm[rp]] = x[w, rp]


@wp.kernel(enable_backward=False)
def _residual_gate(
    nv: int, invperm: wp.array(dtype=wp.int32),
    m_rowadr: wp.array(dtype=wp.int32), m_col: wp.array(dtype=wp.int32),
    m_madr: wp.array(dtype=wp.int32), k_rownnz: wp.array(dtype=wp.int32),
    k_rowadr: wp.array(dtype=wp.int32), k_colind: wp.array(dtype=wp.int32),
    M: wp.array2d(dtype=wp.float32), diag: wp.array2d(dtype=wp.float32),
    kval: wp.array2d(dtype=wp.float32), rhs: wp.array2d(dtype=wp.float32),
    factor: wp.array3d(dtype=wp.float32), qacc: wp.array2d(dtype=wp.float32),
    status: wp.array(dtype=wp.int32), max_residual: wp.array(dtype=wp.float32),
):
    """float64 ||b - M_hat x|| / ||b|| from the sparse structure, one block per world."""
    w, r = wp.tid()
    value = wp.float64(0.0)
    b = wp.float64(0.0)
    ax = wp.float64(0.0)
    bad = int(0)
    if r < nv:
        value = wp.float64(qacc[w, r])
        b = wp.float64(rhs[w, r])
        for k in range(m_rowadr[r], m_rowadr[r + 1]):
            ax += wp.float64(M[w, m_madr[k]]) * wp.float64(qacc[w, m_col[k]])
        ax += wp.float64(diag[w, r]) * value
        adr = k_rowadr[r]
        for t in range(k_rownnz[r]):
            ax += wp.float64(kval[w, adr + t]) * wp.float64(qacc[w, k_colind[adr + t]])
        rp = invperm[r]
        pivot = factor[w, rp, rp]
        if not wp.isfinite(wp.float32(value)) or not wp.isfinite(pivot) or pivot <= 0.0:
            bad = 1
    error = b - ax
    rr = wp.tile_reduce(wp.add, wp.tile(error * error))[0]
    bb = wp.tile_reduce(wp.add, wp.tile(b * b))[0]
    invalid = wp.tile_reduce(wp.add, wp.tile(bad))[0]
    if r == 0:
        relative = wp.float64(0.0)
        if bb > 0.0:
            relative = wp.sqrt(rr / bb)
        elif rr > 0.0:
            relative = wp.float64(wp.inf)
        code = int(0)
        if invalid > 0 or not wp.isfinite(relative):
            code = 1
            relative = wp.float64(wp.inf)
        elif relative > wp.float64(1.0e-4):
            code = 2
        status[w] = status[w] | code
        max_residual[w] = wp.max(max_residual[w], wp.float32(relative))


class BandedEffectiveSolve(_EffectiveSolvePolicy):
    """Own the permuted band matrix and solver buffers for one model/data pair."""

    def __init__(self, model, data, *, tile_size=16):
        ok, reason = supported(model)
        if not ok:
            raise ValueError(f"unsupported model for banded effective solve: {reason}")
        device = model.body_mass.device
        if not device.is_cuda:
            raise ValueError("requires a CUDA device")
        if tile_size not in (16, 32):
            raise ValueError("tile_size must be 16 or 32")
        self.nv, self.nworld, self.device, self.tile = model.nv, data.nworld, device, tile_size
        self.size = (model.nv + tile_size - 1) // tile_size * tile_size
        perm, inv, self.bandwidth_tiles = plan(model, tile_size)
        padded = np.concatenate([perm, np.zeros(self.size - model.nv, dtype=np.int32)])
        self._perm = wp.array(padded, dtype=wp.int32, device=device)
        self._inv = wp.array(inv, dtype=wp.int32, device=device)
        self.work = wp.zeros((data.nworld, self.size, self.size), dtype=wp.float32, device=device)
        self.rhs = wp.zeros((data.nworld, model.nv), dtype=wp.float32, device=device)
        self._x = wp.zeros((data.nworld, self.size), dtype=wp.float32, device=device)
        self._x_tiles = self._x.reshape((data.nworld, self.size, 1))
        self.status = wp.zeros(data.nworld, dtype=wp.int32, device=device)
        self.max_relative_residual = wp.zeros(data.nworld, dtype=wp.float32, device=device)
        self._model = model
        self._kernels = _kernels(model.nv, "float32", tile_size)
        self._substitute = _banded_substitute(self.size, tile_size, self.bandwidth_tiles)
        self._gate_block = max(32, 1 << (model.nv - 1).bit_length())
        self._active = wp.ones(data.nworld, dtype=wp.int32, device=device)

    def assemble(self, model, data):
        wp.launch(_assemble_banded, dim=(data.nworld, self.size), device=self.device,
                  inputs=[self._perm, self._inv, model.nv, self.tile, self.bandwidth_tiles, self.size,
                          model.M_mulm_rowadr, model.M_mulm_col, model.M_mulm_madr,
                          model.efm_K_rownnz, model.efm_K_rowadr, model.efm_K_colind,
                          data.M, data.efm_diag, data.efm_K_val], outputs=[self.work])

    def _factor(self):
        diagonal, panel, trailing, _, _ = self._kernels
        tile, band = self.tile, self.bandwidth_tiles
        for k in range(0, self.size, tile):
            wp.launch_tiled(diagonal, dim=self.nworld, inputs=[k, self._active, self.work],
                            block_dim=64, device=self.device)
            remaining = min((self.size - k) // tile - 1, band)
            if remaining:
                wp.launch_tiled(panel, dim=(self.nworld, remaining), inputs=[k, self._active, self.work],
                                block_dim=64, device=self.device)
                wp.launch_tiled(trailing, dim=(self.nworld, remaining, remaining),
                                inputs=[k, self._active, self.work], block_dim=64, device=self.device)

    def solve(self, model, data, qacc, qfrc=None):
        """Drop-in for ``derivative.eff_solve(m, d, qacc, qfrc=None)`` on supported models."""
        dev = self.device
        if qfrc is None:
            wp.launch(_rhs, dim=(data.nworld, model.nv), device=dev,
                      inputs=[data.qfrc_smooth, data.efm_c, data.efm_ca], outputs=[self.rhs])
        else:
            wp.copy(self.rhs, qfrc)
        self.assemble(model, data)
        self._factor()
        wp.launch(_permute_rhs, dim=(data.nworld, self.size), device=dev,
                  inputs=[self._perm, model.nv, self.rhs], outputs=[self._x])
        wp.launch_tiled(self._substitute, dim=data.nworld, inputs=[self.work, self._x_tiles],
                        block_dim=64, device=dev)
        wp.launch(_unpermute, dim=(data.nworld, self.size), device=dev,
                  inputs=[self._perm, model.nv, self._x], outputs=[qacc])
        wp.launch_tiled(_residual_gate, dim=data.nworld, device=dev, block_dim=self._gate_block,
                        inputs=[model.nv, self._inv, model.M_mulm_rowadr, model.M_mulm_col,
                                model.M_mulm_madr, model.efm_K_rownnz, model.efm_K_rowadr,
                                model.efm_K_colind, data.M, data.efm_diag, data.efm_K_val,
                                self.rhs, self.work, qacc, self.status, self.max_relative_residual])
