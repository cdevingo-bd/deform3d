"""Coupled-block Newton Cholesky: factor only the DOFs that the Hessian actually couples.

MJWarp's Newton solver hands the hook a dense ``nv x nv`` Hessian ``H = M + J^T D J`` per world.
In scenes dominated by many independent DOFs (flex vertices, free bodies) most rows have *no*
off-diagonal entry, so ``H`` is a diagonal plus a small dense block over the "coupled" DOFs. This
workspace exploits that exactly:

* the coupled DOFs are found either by scanning ``H`` for off-diagonal nonzeros (always exact) or
  from the constraint Jacobian and mass-matrix values (``mass_pairs``; see below);
* the flagged DOFs are compacted into a dense ``c x c`` matrix and factored with deform3d's tile
  Cholesky, launching work only for the tiles each world needs (``c`` is per world);
* every other DOF is solved by ``x_i = g_i / H_ii``.

This is exact, not an approximation. The cost per world scales with ``c^3`` instead of ``nv^3``;
on a flex scene with ``nv = 414`` the coupled set stays near 40-60 DOFs, so the work per world is
two orders of magnitude below a dense factorization. The speedup over MJWarp's own hook is largest
at one world and shrinks as the world count grows, because both paths then become throughput-bound.

``max_coupled`` bounds the compacted matrix. A world whose coupled set exceeds it is not solved
here: it is handed to the host's own hook through a shadow context in which only those worlds
are active, so the result stays correct (and the world only loses the speedup). ``fallbacks``
counts how often each world did that. Scenes whose Hessian is dense (large robots with dense mass
blocks and many contacts) will fall back constantly; measure ``coupled`` before enabling.

Structural marking (``mass_pairs`` given, and ``data`` passed to ``enqueue``) replaces the scan,
which reads half of ``H`` per solve and is the largest cost of this workspace at high world counts.
It flags the nonzero Jacobian columns of every quadratic constraint row with two or more of them
(accumulated over a solve), plus both DOFs of every
stored off-diagonal mass-matrix entry that is nonzero. That is a superset of the nonzero pattern of
``M + J^T D J`` for a sparse, non-compact, non-DISCRETE MJWarp, so the result stays exact, but unlike
the scan it relies on how MJWarp assembles ``H``; ``mujoco_newton_factor`` guards that with a source
hash. ``verify_structure=True`` also runs the scan and sets ``status`` bit 4 when the structure
missed a coupled DOF. The structure can also be too loose (friction rows that cancel exactly), so a
world whose structural set exceeds ``rescan_above`` DOFs (default: one tile) is rescanned for its
exact set; ``rescans`` counts how often.

The float64 residual gate (``status`` bit 2) and finiteness gate (bit 1) match
``mujoco_newton`` but only evaluate the coupled rows, so they cost ``O(c)`` per row.
"""

import copy
import dataclasses
from functools import lru_cache

import numpy as np
import warp as wp

from .mujoco_newton import _masks


@wp.kernel(enable_backward=False)
def _clear_flags(refactor: wp.array(dtype=wp.int32), flag: wp.array2d(dtype=wp.int32)):
    w, i = wp.tid()
    if refactor[w] != 0:
        flag[w, i] = 0


@wp.kernel(enable_backward=False)
def _mark(nv: int, chunk: int, refactor: wp.array(dtype=wp.int32), h: wp.array3d(dtype=wp.float32),
          flag: wp.array2d(dtype=wp.int32)):
    # H is upper-triangular in the host's buffer. Both endpoints of an off-diagonal entry are
    # coupled; concurrent writers all store 1, so the race is benign. A NaN compares != 0 and is
    # flagged, which the finiteness gate then reports. Consecutive threads take consecutive
    # column chunks of one row, so a warp reads contiguous memory.
    w, i, block = wp.tid()
    if refactor[w] == 0:
        return
    first = block * chunk
    if first + chunk <= i + 1:
        return
    for q in range(chunk):
        j = first + q
        if j > i and j < nv:
            if h[w, i, j] != 0.0:
                flag[w, i] = 1
                flag[w, j] = 1


@wp.kernel(enable_backward=False)
def _mark_jacobian(refactor: wp.array(dtype=wp.int32), nefc: wp.array(dtype=wp.int32),
                   state: wp.array2d(dtype=wp.int32),
                   rownnz: wp.array2d(dtype=wp.int32), rowadr: wp.array2d(dtype=wp.int32),
                   colind: wp.array3d(dtype=wp.int32), jac: wp.array3d(dtype=wp.float32),
                   flag: wp.array2d(dtype=wp.int32)):
    # With pyramidal cones, J^T D J is a sum over quadratic rows (state 1) of D * outer(J_r, J_r),
    # so a row couples the columns where its Jacobian is nonzero, provided there are at least two
    # (a floor contact along an axis stores explicit zeros and couples nothing). The flags are NOT
    # cleared between the calls of one solve: MJWarp updates H incrementally, adding and
    # subtracting a row's contribution as it changes state, and the subtraction can leave rounding
    # residue on that row's support. Accumulating over the solve therefore covers every entry H can
    # hold; the first call of a solve clears.
    w, r = wp.tid()
    if refactor[w] == 0 or r >= nefc[w]:
        return
    if state[w, r] != 1:
        return
    address = rowadr[w, r]
    count = rownnz[w, r]
    nonzero = int(0)
    for k in range(count):
        if jac[w, 0, address + k] != 0.0:
            nonzero += 1
    if nonzero < 2:
        return
    for k in range(count):
        if jac[w, 0, address + k] != 0.0:
            flag[w, colind[w, 0, address + k]] = 1


@wp.kernel(enable_backward=False)
def _mark_mass(refactor: wp.array(dtype=wp.int32), rows: wp.array(dtype=wp.int32),
               cols: wp.array(dtype=wp.int32), element: wp.array(dtype=wp.int32),
               mass: wp.array2d(dtype=wp.float32), flag: wp.array2d(dtype=wp.int32)):
    # The mass matrix couples two DOFs only where a stored off-diagonal entry is nonzero. A
    # flex vertex's 3x3 block is structurally present but numerically diagonal, so test values.
    w, e = wp.tid()
    if refactor[w] == 0:
        return
    if mass[w, element[e]] != 0.0:
        flag[w, rows[e]] = 1
        flag[w, cols[e]] = 1


@wp.kernel(enable_backward=False)
def _compare_structure(refactor: wp.array(dtype=wp.int32), flag: wp.array2d(dtype=wp.int32),
                       scanned: wp.array2d(dtype=wp.int32), status: wp.array(dtype=wp.int32)):
    # A DOF the Hessian scan found coupled but the structure did not: the structural set is wrong.
    w, i = wp.tid()
    if refactor[w] != 0 and scanned[w, i] != 0 and flag[w, i] == 0:
        status[w] = status[w] | 4


LANES = 32


@wp.kernel(enable_backward=False)
def _count_chunks(nv: int, lanes: int, refactor: wp.array(dtype=wp.int32), flag: wp.array2d(dtype=wp.int32),
                  counts: wp.array2d(dtype=wp.int32)):
    # Lane g owns the contiguous DOFs [g * chunk, (g + 1) * chunk).
    w, g = wp.tid()
    if refactor[w] == 0:
        return
    chunk = (nv + lanes - 1) // lanes
    n = int(0)
    for q in range(chunk):
        i = g * chunk + q
        if i < nv:
            if flag[w, i] != 0:
                n += 1
    counts[w, g] = n


@wp.kernel(enable_backward=False)
def _decide_rescan(nv: int, lanes: int, limit: int, refactor: wp.array(dtype=wp.int32),
                   counts: wp.array2d(dtype=wp.int32), flag: wp.array2d(dtype=wp.int32),
                   rescan: wp.array(dtype=wp.int32), rescans: wp.array(dtype=wp.int32)):
    # A structural set above ``limit`` is probably loose (friction rows that cancel exactly are
    # flagged), so such a world discards it and takes the exact scan instead. Each lane clears its
    # own chunk of the flags so the scan starts from nothing.
    w, g = wp.tid()
    total = int(0)
    if refactor[w] != 0:
        for k in range(lanes):
            total += counts[w, k]
    over = total > limit
    if over:
        chunk = (nv + lanes - 1) // lanes
        for q in range(chunk):
            i = g * chunk + q
            if i < nv:
                flag[w, i] = 0
    if g == 0:
        rescan[w] = int(over)
        if over:
            rescans[w] = rescans[w] + 1


@wp.kernel(enable_backward=False)
def _compact(nv: int, lanes: int, cmax: int, tile: int, active: wp.array(dtype=wp.int32),
             refactor: wp.array(dtype=wp.int32), flag: wp.array2d(dtype=wp.int32),
             counts: wp.array2d(dtype=wp.int32),
             idx: wp.array2d(dtype=wp.int32), pos: wp.array2d(dtype=wp.int32),
             count: wp.array(dtype=wp.int32), cpad: wp.array(dtype=wp.int32),
             overflow: wp.array(dtype=wp.int32), initialized: wp.array(dtype=wp.int32),
             fallbacks: wp.array(dtype=wp.int32), fac: wp.array(dtype=wp.int32),
             ok: wp.array(dtype=wp.int32), shadow_done: wp.array(dtype=wp.bool)):
    # Lane g scatters its chunk to the positions after the lower lanes' counts; lane 0 also does
    # the per-world bookkeeping. Every lane derives the same total from the chunk counts.
    w, g = wp.tid()
    total = int(0)
    first = int(0)
    if refactor[w] != 0:
        for k in range(lanes):
            n = counts[w, k]
            if k < g:
                first += n
            total += n
        chunk = (nv + lanes - 1) // lanes
        c = first
        for q in range(chunk):
            i = g * chunk + q
            if i < nv:
                if flag[w, i] != 0:
                    if c < cmax:
                        idx[w, c] = i
                        pos[w, i] = c
                    c += 1
    if g != 0:
        return
    fac[w] = 0
    if refactor[w] != 0:
        count[w] = total
        cpad[w] = (total + tile - 1) // tile * tile
        if total > cmax:
            overflow[w] = 1
            initialized[w] = 0  # the host path solves this world, so no factor of ours is valid
            fallbacks[w] = fallbacks[w] + 1
        else:
            overflow[w] = 0
            fac[w] = 1
    solved_here = active[w] != 0 and overflow[w] == 0
    ok[w] = int(solved_here)
    shadow_done[w] = not (active[w] != 0 and overflow[w] != 0)


@wp.kernel(enable_backward=False)
def _pack(fac: wp.array(dtype=wp.int32), cpad: wp.array(dtype=wp.int32),
          count: wp.array(dtype=wp.int32), idx: wp.array2d(dtype=wp.int32),
          h: wp.array3d(dtype=wp.float32), work: wp.array3d(dtype=wp.float32)):
    # One thread per row of the compacted lower triangle; the factorization never reads the rest.
    w, a = wp.tid()
    if fac[w] == 0 or a >= cpad[w]:
        return
    c = count[w]
    for b in range(a + 1):
        value = float(0.0)
        if a < c and b < c:
            ia = idx[w, a]
            ib = idx[w, b]
            value = h[w, wp.min(ia, ib), wp.max(ia, ib)]
        elif a == b:
            value = 1.0  # identity padding keeps the last tile positive definite
        work[w, a, b] = value


@wp.kernel(enable_backward=False)
def _gather_rhs(ok: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32),
                idx: wp.array2d(dtype=wp.int32), g: wp.array2d(dtype=wp.float32),
                x: wp.array2d(dtype=wp.float32)):
    w, a = wp.tid()
    if ok[w] == 0:
        return
    value = float(0.0)
    if a < count[w]:
        value = g[w, idx[w, a]]
    x[w, a] = value


@lru_cache(None)
def _kernels(nv, cmax, tile):
    options = {'enable_mathdx_solver': False, 'enable_mathdx_gemm': False}

    @wp.kernel(module='unique', enable_backward=False, module_options=options)
    def solve(fac: wp.array(dtype=wp.int32), ok: wp.array(dtype=wp.int32), cpad: wp.array(dtype=wp.int32),
              a: wp.array3d(dtype=wp.float32), solution: wp.array3d(dtype=wp.float32)):
        # One block per world factors (when ``fac``) and then substitutes (when ``ok``), looping
        # over only the ``cpad / tile`` tiles that world needs. A world with a cached factor has
        # ``ok`` without ``fac``. Tile loads and stores synchronize the block, so each step reads
        # what the previous one wrote.
        w = wp.tid()
        blocks = cpad[w] // tile
        if blocks == 0 or (fac[w] == 0 and ok[w] == 0):
            return
        if fac[w] != 0:
            for kb in range(blocks):
                k = kb * tile
                lower = wp.tile_load(a[w], shape=(tile, tile), offset=(k, k), storage='shared', bounds_check=False)
                wp.tile_cholesky_inplace(lower, fill_mode='lower')
                wp.tile_store(a[w], lower, offset=(k, k), bounds_check=False)
                for bi in range(kb + 1, blocks):
                    i = bi * tile
                    aik = wp.tile_load(a[w], shape=(tile, tile), offset=(i, k), storage='shared',
                                       bounds_check=False)
                    wp.tile_lower_solve_inplace(lower, wp.tile_transpose(aik))
                    wp.tile_store(a[w], aik, offset=(i, k), bounds_check=False)
                for bi in range(kb + 1, blocks):
                    i = bi * tile
                    li = wp.tile_load(a[w], shape=(tile, tile), offset=(i, k), storage='shared',
                                      bounds_check=False)
                    for bj in range(kb + 1, bi + 1):
                        j = bj * tile
                        lj = wp.tile_load(a[w], shape=(tile, tile), offset=(j, k), storage='shared',
                                          bounds_check=False)
                        target = wp.tile_load(a[w], shape=(tile, tile), offset=(i, j), storage='shared',
                                              bounds_check=False)
                        wp.tile_matmul(li, wp.tile_transpose(lj), target, alpha=wp.float32(-1.0))
                        wp.tile_store(a[w], target, offset=(i, j), bounds_check=False)
        if ok[w] != 0:
            x = wp.tile_load(solution[w], shape=(cmax, 1), offset=(0, 0), storage='shared', bounds_check=False)
            for block in range(blocks):
                k = block * tile
                part = wp.tile_view(x, shape=(tile, 1), offset=(k, 0))
                for previous in range(block):
                    j = previous * tile
                    lower = wp.tile_load(a[w], shape=(tile, tile), offset=(k, j), storage='shared',
                                         bounds_check=False)
                    y = wp.tile_view(x, shape=(tile, 1), offset=(j, 0))
                    wp.tile_matmul(lower, y, part, alpha=wp.float32(-1.0))
                lower = wp.tile_load(a[w], shape=(tile, tile), offset=(k, k), storage='shared',
                                     bounds_check=False)
                wp.tile_lower_solve_inplace(lower, part)
            for reverse in range(blocks):
                block = blocks - 1 - reverse
                k = block * tile
                part = wp.tile_view(x, shape=(tile, 1), offset=(k, 0))
                for following in range(block + 1, blocks):
                    j = following * tile
                    lower = wp.tile_load(a[w], shape=(tile, tile), offset=(j, k), storage='shared',
                                         bounds_check=False)
                    y = wp.tile_view(x, shape=(tile, 1), offset=(j, 0))
                    wp.tile_matmul(wp.tile_transpose(lower), y, part, alpha=wp.float32(-1.0))
                lower = wp.tile_load(a[w], shape=(tile, tile), offset=(k, k), storage='shared',
                                     bounds_check=False)
                wp.tile_upper_solve_inplace(wp.tile_transpose(lower), part)
            wp.tile_store(solution[w], x, bounds_check=False)

    @wp.kernel(module='unique', enable_backward=False)
    def finish(ok: wp.array(dtype=wp.int32), h: wp.array3d(dtype=wp.float32),
               g: wp.array2d(dtype=wp.float32), flag: wp.array2d(dtype=wp.int32),
               pos: wp.array2d(dtype=wp.int32), idx: wp.array2d(dtype=wp.int32),
               count: wp.array(dtype=wp.int32), x: wp.array2d(dtype=wp.float32),
               factor: wp.array3d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32),
               norm: wp.array(dtype=wp.float32), decrement: wp.array(dtype=wp.float32),
               status: wp.array(dtype=wp.int32), max_residual: wp.array(dtype=wp.float32)):
        w, row = wp.tid()
        if ok[w] == 0:
            return
        value, b, ax = wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
        bad = int(0)
        if row < nv:
            b = wp.float64(g[w, row])
            diagonal_entry = h[w, row, row]
            if flag[w, row] == 0:
                # No off-diagonal entry anywhere in this row/column: exact scalar solve.
                solved = g[w, row] / diagonal_entry
                value = wp.float64(wp.float32(solved))
                ax = wp.float64(diagonal_entry) * value
                if not wp.isfinite(value) or not diagonal_entry > 0.0:
                    bad = 1
            else:
                p = pos[w, row]
                value = wp.float64(wp.float32(x[w, p]))
                for column in range(count[w]):
                    other = idx[w, column]
                    ax += wp.float64(h[w, wp.min(row, other), wp.max(row, other)]) * wp.float64(
                        wp.float32(x[w, column]))
                pivot = factor[w, p, p]
                if not wp.isfinite(value) or not wp.isfinite(pivot) or pivot <= 0.0:
                    bad = 1
            out[w, row] = -wp.float32(value)
        error = b - ax
        rr = wp.tile_reduce(wp.add, wp.tile(error * error))[0]
        bb = wp.tile_reduce(wp.add, wp.tile(b * b))[0]
        xx = wp.tile_reduce(wp.add, wp.tile(value * value))[0]
        bx = wp.tile_reduce(wp.add, wp.tile(b * value))[0]
        invalid = wp.tile_reduce(wp.add, wp.tile(bad))[0]
        relative = wp.float64(0.0)
        if bb > 0.0:
            relative = wp.sqrt(rr / bb)
        elif rr > 0.0:
            relative = wp.float64(wp.inf)
        if row == 0:
            code = int(0)
            if invalid > 0 or not wp.isfinite(relative) or not wp.isfinite(wp.float32(xx)) or not wp.isfinite(
                    wp.float32(bx)):
                code = 1
                relative = wp.float64(wp.inf)
            elif relative > wp.float64(1.0e-4):
                code = 2
            norm[w] = wp.float32(xx)
            decrement[w] = wp.float32(bx)
            status[w] = status[w] | code
            max_residual[w] = wp.max(max_residual[w], wp.float32(relative))

    return solve, finish


class CoupledWorkspace:
    """Per-world coupled-block Cholesky. Keep alive for as long as captured graphs are replayed."""

    def __init__(self, nv, nworld, device, *, tile_size=32, max_coupled=128, mark_chunk=8,
                 mass_pairs=None, verify_structure=False, rescan_above=None):
        if tile_size not in (16, 32):
            raise ValueError('tile_size must be 16 or 32')
        if isinstance(max_coupled, bool) or int(max_coupled) < 1:
            raise ValueError('max_coupled must be a positive integer')
        self.nv, self.nworld, self.device = int(nv), int(nworld), wp.get_device(device)
        self.tile_size = tile_size
        self.mark_chunk = int(mark_chunk)
        size = (self.nv + tile_size - 1) // tile_size * tile_size
        self.max_coupled = min(size, (int(max_coupled) + tile_size - 1) // tile_size * tile_size)
        n, c = self.nworld, self.max_coupled
        i32 = wp.int32
        self.flag = wp.zeros((n, self.nv), dtype=i32, device=device)
        self.idx = wp.zeros((n, c), dtype=i32, device=device)
        self.pos = wp.zeros((n, self.nv), dtype=i32, device=device)
        self.counts = wp.zeros((n, LANES), dtype=i32, device=device)
        # Structural marking: ``mass_pairs`` = (rows, cols, element ids) of the stored off-diagonal
        # mass-matrix entries. None means every solve scans H for its coupled set instead.
        self.structural = mass_pairs is not None
        self.verify_structure = bool(verify_structure)
        # Rescan worlds whose structural set exceeds this many DOFs (default: one tile).
        self.rescan_above = tile_size if rescan_above is None else int(rescan_above)
        self.rescan = wp.zeros(n, dtype=i32, device=device)
        self.rescans = wp.zeros(n, dtype=i32, device=device)
        if self.verify_structure and not self.structural:
            raise ValueError('verify_structure needs structural marking (mass_pairs)')
        if self.structural:
            self.mass_pairs = tuple(wp.array(np.asarray(a, np.int32), dtype=i32, device=device) for a in mass_pairs)
        if self.verify_structure:
            self.scanned = wp.zeros((n, self.nv), dtype=i32, device=device)
        self.coupled = wp.zeros(n, dtype=i32, device=device)
        self.cpad = wp.zeros(n, dtype=i32, device=device)
        self.overflow = wp.zeros(n, dtype=i32, device=device)
        self.fallbacks = wp.zeros(n, dtype=i32, device=device)
        self.active = wp.zeros(n, dtype=i32, device=device)
        self.refactor = wp.zeros(n, dtype=i32, device=device)
        self.initialized = wp.zeros(n, dtype=i32, device=device)
        self.fac = wp.zeros(n, dtype=i32, device=device)
        self.ok = wp.zeros(n, dtype=i32, device=device)
        self.shadow_done = wp.ones(n, dtype=wp.bool, device=device)
        self.work = wp.zeros((n, c, c), dtype=wp.float32, device=device)
        self.x = wp.zeros((n, c), dtype=wp.float32, device=device)
        self.x_tiles = self.x.reshape((n, c, 1))
        self.status = wp.zeros(n, dtype=i32, device=device)
        self.max_relative_residual = wp.zeros(n, dtype=wp.float32, device=device)

    def _check(self, ctx, quad, state):
        arrays = ((ctx.h, 'h', wp.float32, 3), (ctx.grad, 'grad', wp.float32, 2),
                  (ctx.done, 'done', wp.bool, 1), (quad, 'quad', wp.int32, 1),
                  (state, 'state', wp.int32, 1), (ctx.search, 'search', wp.float32, 2),
                  (ctx.search_dot, 'search_dot', wp.float32, 1),
                  (ctx.newton_decrement, 'newton_decrement', wp.float32, 1))
        for array, name, dtype, ndim in arrays:
            if (not isinstance(array, wp.array) or array.dtype != dtype or array.ndim != ndim
                    or array.device != self.device or not array.is_contiguous
                    or array.shape[0] != self.nworld):
                raise ValueError(f'{name} must have the expected dtype, rank, worlds and device, '
                                 f'with contiguous storage')
        if (min(ctx.h.shape[1:]) < self.nv or ctx.grad.shape[1] < self.nv
                or ctx.search.shape[1] != self.nv):
            raise ValueError('Newton matrix/vector dimensions do not match the model')

    def _shadow(self, ctx):
        """A copy of ``ctx`` where only worlds that overflowed ``max_coupled`` are still active."""
        shadow = dataclasses.replace(ctx) if dataclasses.is_dataclass(ctx) else copy.copy(ctx)
        shadow.done = self.shadow_done
        return shadow

    def enqueue(self, ctx, quad, state, skip_unchanged, skip_noflip, fallback, data=None):
        """Launch the solve. ``fallback(shadow_ctx)`` must run the host hook without skipping.

        ``data`` supplies the constraint Jacobian and mass matrix for structural marking; without it
        (or with a workspace built without ``mass_pairs``) the coupled set comes from scanning H.
        """
        self._check(ctx, quad, state)
        dev = self.device
        n, nv, c, t = self.nworld, self.nv, self.max_coupled, self.tile_size
        wp.launch(_masks, dim=n, inputs=[ctx.done, quad, state, skip_unchanged, skip_noflip],
                  outputs=[self.active, self.refactor, self.initialized], device=dev)
        structural = self.structural and data is not None
        if not (structural and skip_unchanged):  # structural flags accumulate over one solve
            wp.launch(_clear_flags, dim=(n, nv), inputs=[self.refactor, self.flag], device=dev)
        if structural:
            efc = data.efc
            wp.launch(_mark_jacobian, dim=(n, efc.J_rownnz.shape[1]),
                      inputs=[self.refactor, data.nefc, efc.state, efc.J_rownnz, efc.J_rowadr, efc.J_colind, efc.J],
                      outputs=[self.flag], device=dev)
            rows, cols, element = self.mass_pairs
            if rows.shape[0]:
                wp.launch(_mark_mass, dim=(n, rows.shape[0]),
                          inputs=[self.refactor, rows, cols, element, data.M], outputs=[self.flag], device=dev)
            if self.verify_structure:
                wp.launch(_clear_flags, dim=(n, nv), inputs=[self.refactor, self.scanned], device=dev)
                wp.launch(_mark, dim=(n, nv, (nv + self.mark_chunk - 1) // self.mark_chunk),
                          inputs=[nv, self.mark_chunk, self.refactor, ctx.h, self.scanned], device=dev)
                wp.launch(_compare_structure, dim=(n, nv),
                          inputs=[self.refactor, self.flag, self.scanned], outputs=[self.status], device=dev)
            if self.rescan_above < nv:
                wp.launch(_count_chunks, dim=(n, LANES), inputs=[nv, LANES, self.refactor, self.flag],
                          outputs=[self.counts], device=dev)
                wp.launch(_decide_rescan, dim=(n, LANES),
                          inputs=[nv, LANES, self.rescan_above, self.refactor, self.counts, self.flag],
                          outputs=[self.rescan, self.rescans], device=dev)
                wp.launch(_mark, dim=(n, nv, (nv + self.mark_chunk - 1) // self.mark_chunk),
                          inputs=[nv, self.mark_chunk, self.rescan, ctx.h, self.flag], device=dev)
        else:
            wp.launch(_mark, dim=(n, nv, (nv + self.mark_chunk - 1) // self.mark_chunk),
                      inputs=[nv, self.mark_chunk, self.refactor, ctx.h, self.flag], device=dev)
        wp.launch(_count_chunks, dim=(n, LANES), inputs=[nv, LANES, self.refactor, self.flag],
                  outputs=[self.counts], device=dev)
        wp.launch(_compact, dim=(n, LANES),
                  inputs=[nv, LANES, c, t, self.active, self.refactor, self.flag, self.counts],
                  outputs=[self.idx, self.pos, self.coupled, self.cpad, self.overflow,
                           self.initialized, self.fallbacks, self.fac, self.ok, self.shadow_done],
                  device=dev)
        wp.launch(_pack, dim=(n, c), inputs=[self.fac, self.cpad, self.coupled, self.idx, ctx.h],
                  outputs=[self.work], device=dev)
        solve, finish = _kernels(nv, c, t)
        wp.launch(_gather_rhs, dim=(n, c), inputs=[self.ok, self.coupled, self.idx, ctx.grad],
                  outputs=[self.x], device=dev)
        wp.launch_tiled(solve, dim=n, inputs=[self.fac, self.ok, self.cpad, self.work, self.x_tiles],
                        block_dim=64, device=dev)
        wp.launch_tiled(
            finish, dim=n,
            inputs=[self.ok, ctx.h, ctx.grad, self.flag, self.pos, self.idx, self.coupled, self.x,
                    self.work, ctx.search, ctx.search_dot, ctx.newton_decrement, self.status,
                    self.max_relative_residual],
            block_dim=max(32, 1 << (nv - 1).bit_length()), device=dev)
        # Worlds whose coupled set did not fit (and only those) are solved by the host hook.
        fallback(self._shadow(ctx))
