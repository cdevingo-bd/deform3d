"""Coupled-block Newton Cholesky: exactness, fallback, caching and failure gates."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from deform3d_warp.mujoco_newton_coupled import CoupledWorkspace

pytestmark = pytest.mark.integration
cuda = pytest.mark.skipif(not wp.is_cuda_available(), reason="requires CUDA")

NV, PAD = 94, 96  # 94 DOFs: all 32 scan lanes (3 DOFs each) own a real DOF
SENTINEL = 12345.0


def spd(rng, coupled, triples=0):
    """Diagonal SPD matrix with a dense SPD block over ``coupled`` random DOFs and 3x3 blocks."""
    h = np.diag(rng.uniform(2.0, 6.0, NV))
    if coupled:
        ids = rng.choice(NV, size=coupled, replace=False)
        a = rng.standard_normal((coupled, coupled)) * 0.2
        h[np.ix_(ids, ids)] += a @ a.T
    for _ in range(triples):
        ids = rng.choice(NV, size=3, replace=False)
        a = rng.standard_normal((3, 3)) * 0.3
        h[np.ix_(ids, ids)] += a @ a.T
    return h


def coupled_set(h):
    off = np.triu(h, 1) != 0
    return off.any(0) | off.any(1)


class Harness:
    """Builds the arrays the host hook would hand over and runs the workspace on them."""

    def __init__(self, matrices, grads, *, tile=16, max_coupled=64, **workspace_kwargs):
        self.n = len(matrices)
        self.matrices = [m.copy() for m in matrices]
        self.g = np.asarray(grads, np.float32)
        self.rng = np.random.default_rng(11)
        self.ctx = SimpleNamespace(
            h=wp.array(self._upper(matrices), dtype=wp.float32), grad=wp.array(self._pad(self.g), dtype=wp.float32),
            done=wp.zeros(self.n, dtype=wp.bool),
            search=wp.array(np.full((self.n, NV), SENTINEL, np.float32), dtype=wp.float32),
            search_dot=wp.zeros(self.n, dtype=wp.float32), newton_decrement=wp.zeros(self.n, dtype=wp.float32))
        self.nefc = wp.array(np.full(self.n, 5, np.int32), dtype=wp.int32)
        self.zero = wp.zeros(self.n, dtype=wp.int32)
        self.ws = CoupledWorkspace(NV, self.n, "cuda:0", tile_size=tile, max_coupled=max_coupled, **workspace_kwargs)
        self.shadows = []

    def _pad(self, g):
        out = np.zeros((self.n, PAD), np.float32)
        out[:, :NV] = g
        return out

    def _upper(self, matrices):
        out = np.zeros((self.n, PAD, PAD), np.float32)
        for w, h in enumerate(matrices):
            out[w, :NV, :NV] = np.triu(h)
            # Garbage strictly below the diagonal: the host only fills the upper triangle.
            out[w] += np.tril(self.rng.standard_normal((PAD, PAD)).astype(np.float32) * 1e3, -1)
            for i in range(NV, PAD):
                out[w, i, i] = 1.0
        return out

    def set_matrices(self, matrices):
        self.matrices = [m.copy() for m in matrices]
        self.ctx.h.assign(self._upper(matrices))

    def set_grad(self, grads):
        self.g = np.asarray(grads, np.float32)
        self.ctx.grad.assign(self._pad(self.g))

    def run(self, quad=None, state=None, skip_unchanged=False, skip_noflip=False, data=None):
        self.shadows = []

        def fallback(shadow):
            self.shadows.append(shadow.done.numpy().copy())

        wp.synchronize()
        self.ws.enqueue(self.ctx, quad if quad is not None else self.nefc,
                        state if state is not None else self.nefc, skip_unchanged, skip_noflip, fallback, data)
        wp.synchronize()
        assert len(self.shadows) == 1, "the host hook must be offered exactly once per call"
        return self.ctx.search.numpy()

    def reference(self, w, h=None):
        h = self.matrices[w] if h is None else h
        return -np.linalg.solve(h, self.g[w].astype(np.float64))


@cuda
@pytest.mark.parametrize("tile", [16, 32])
def test_matches_float64_reference_and_finds_the_exact_coupled_set(tile):
    rng = np.random.default_rng(1)
    mats = [spd(rng, 0), spd(rng, 0, triples=3), spd(rng, 20, 2), spd(rng, 55)]
    g = rng.standard_normal((len(mats), NV))
    with wp.ScopedDevice("cuda:0"):
        harness = Harness(mats, g, tile=tile, max_coupled=64)
        out = harness.run()
        for w in range(len(mats)):
            ref = harness.reference(w)
            assert np.abs(out[w] - ref).max() <= 1e-5 * np.abs(ref).max()
            # search_dot = |x|^2 and newton_decrement = g.x, both of the *solution* x = -search
            assert harness.ctx.search_dot.numpy()[w] == pytest.approx(np.sum(ref**2), rel=1e-4)
            assert harness.ctx.newton_decrement.numpy()[w] == pytest.approx(-np.dot(g[w], ref), rel=1e-4)
            assert harness.ws.coupled.numpy()[w] == coupled_set(mats[w]).sum()
        np.testing.assert_array_equal(harness.ws.status.numpy(), 0)
        np.testing.assert_array_equal(harness.ws.fallbacks.numpy(), 0)
        assert harness.shadows[0].all(), "no world should be left for the host hook"


@cuda
def test_world_beyond_max_coupled_is_left_to_the_host_hook_untouched():
    rng = np.random.default_rng(2)
    mats = [spd(rng, 20), spd(rng, 50), spd(rng, 10)]
    g = rng.standard_normal((3, NV))
    with wp.ScopedDevice("cuda:0"):
        harness = Harness(mats, g, max_coupled=32)
        out = harness.run()
        np.testing.assert_array_equal(harness.shadows[0], [True, False, True])  # done in shadow = solved here
        np.testing.assert_array_equal(harness.ws.fallbacks.numpy(), [0, 1, 0])
        assert (out[1] == SENTINEL).all(), "an overflowing world must not be written by this solver"
        for w in (0, 2):
            ref = harness.reference(w)
            assert np.abs(out[w] - ref).max() <= 1e-5 * np.abs(ref).max()


@cuda
def test_cached_factor_is_reused_only_when_nothing_changed():
    rng = np.random.default_rng(3)
    mats = [spd(rng, 30), spd(rng, 12)]
    g1 = rng.standard_normal((2, NV))
    with wp.ScopedDevice("cuda:0"):
        harness = Harness(mats, g1)
        harness.run()
        # The coupled block changes but the host says no constraint changed: the cached factor
        # must still be used. (Uncoupled diagonals are left alone, as they are when H is unchanged.)
        new_mats = []
        for m in mats:
            changed = m.copy()
            ids = np.where(coupled_set(m))[0]
            changed[np.ix_(ids, ids)] = m[np.ix_(ids, ids)] * 1.7 + np.eye(len(ids))
            new_mats.append(changed)
        g2 = rng.standard_normal((2, NV))
        harness.set_matrices(new_mats)
        harness.set_grad(g2)
        unchanged = wp.zeros(2, dtype=wp.int32)
        out = harness.run(quad=unchanged, skip_unchanged=True)
        for w in range(2):
            ref = -np.linalg.solve(mats[w], g2[w])
            assert np.abs(out[w] - ref).max() <= 1e-5 * np.abs(ref).max(), "stale factor must be reused"
        # Now the host reports a change in world 0 only: world 0 refactors, world 1 still reuses.
        quad = wp.array(np.array([1, 0], np.int32), dtype=wp.int32)
        out = harness.run(quad=quad, skip_unchanged=True)
        ref0 = -np.linalg.solve(new_mats[0], g2[0])
        ref1 = -np.linalg.solve(mats[1], g2[1])
        assert np.abs(out[0] - ref0).max() <= 1e-5 * np.abs(ref0).max()
        assert np.abs(out[1] - ref1).max() <= 1e-5 * np.abs(ref1).max()


@cuda
def test_first_call_factors_even_when_the_host_claims_nothing_changed():
    rng = np.random.default_rng(4)
    mats = [spd(rng, 25)]
    g = rng.standard_normal((1, NV))
    with wp.ScopedDevice("cuda:0"):
        harness = Harness(mats, g)
        out = harness.run(quad=wp.zeros(1, dtype=wp.int32), skip_unchanged=True)
        ref = harness.reference(0)
        assert np.abs(out[0] - ref).max() <= 1e-5 * np.abs(ref).max()


@cuda
def test_an_overflowed_world_never_reuses_a_factor_it_does_not_have():
    rng = np.random.default_rng(5)
    mats = [spd(rng, 50)]
    g = rng.standard_normal((1, NV))
    with wp.ScopedDevice("cuda:0"):
        harness = Harness(mats, g, max_coupled=32)
        harness.run()
        np.testing.assert_array_equal(harness.shadows[0], [False])
        harness.run(quad=wp.zeros(1, dtype=wp.int32), skip_unchanged=True)
        np.testing.assert_array_equal(harness.shadows[0], [False])  # still handed to the host hook
        np.testing.assert_array_equal(harness.ws.fallbacks.numpy(), [2])
        # Once the matrix fits again the world is solved here from a fresh factorization.
        small = [spd(rng, 10)]
        harness.set_matrices(small)
        out = harness.run(quad=wp.zeros(1, dtype=wp.int32), skip_unchanged=True)
        np.testing.assert_array_equal(harness.shadows[0], [True])
        ref = harness.reference(0)
        assert np.abs(out[0] - ref).max() <= 1e-5 * np.abs(ref).max()


@cuda
def test_done_and_unflipped_worlds_are_left_alone():
    rng = np.random.default_rng(6)
    mats = [spd(rng, 20), spd(rng, 20), spd(rng, 20)]
    g = rng.standard_normal((3, NV))
    with wp.ScopedDevice("cuda:0"):
        harness = Harness(mats, g)
        harness.ctx.done.assign(np.array([True, False, False]))
        state = wp.array(np.array([1, 1, 0], np.int32), dtype=wp.int32)
        out = harness.run(state=state, skip_noflip=True)
        assert (out[0] == SENTINEL).all(), "a finished world must not be written"
        assert (out[2] == SENTINEL).all(), "a world whose active set did not flip keeps its direction"
        ref = harness.reference(1)
        assert np.abs(out[1] - ref).max() <= 1e-5 * np.abs(ref).max()
        assert harness.shadows[0].all(), "none of these worlds overflowed"


@cuda
@pytest.mark.parametrize("breakage", ["negative_diagonal", "nan_entry", "indefinite_block"])
def test_invalid_systems_set_the_status_bits(breakage):
    rng = np.random.default_rng(7)
    h = spd(rng, 20)
    if breakage == "negative_diagonal":  # an uncoupled row with a non-positive pivot
        free = np.where(~coupled_set(h))[0][0]
        h[free, free] = -1.0
    elif breakage == "nan_entry":
        ids = np.where(coupled_set(h))[0]
        h[ids[0], ids[1]] = h[ids[1], ids[0]] = np.nan
    else:  # a coupled block that is not positive definite
        ids = np.where(coupled_set(h))[0]
        h[ids[0], ids[0]] = -50.0
    with wp.ScopedDevice("cuda:0"):
        harness = Harness([h], rng.standard_normal((1, NV)))
        harness.run()
        assert harness.ws.status.numpy()[0] & 1, "must report an invalid factor or output"


@cuda
def test_unsupported_arguments_are_rejected():
    with wp.ScopedDevice("cuda:0"):
        with pytest.raises(ValueError, match="tile_size"):
            CoupledWorkspace(NV, 1, "cuda:0", tile_size=24)
        with pytest.raises(ValueError, match="max_coupled"):
            CoupledWorkspace(NV, 1, "cuda:0", max_coupled=0)
        rng = np.random.default_rng(8)
        harness = Harness([spd(rng, 10)], rng.standard_normal((1, NV)))
        harness.ctx.h = wp.zeros((1, 40, 40), dtype=wp.float32)
        with pytest.raises(ValueError, match="dimensions"):
            harness.run()


@cuda
@pytest.mark.parametrize("pair", [(6, 7), (7, 8), (14, 15), (0, 1), (NV - 2, NV - 1)])
def test_a_single_adjacent_coupling_is_found_at_every_scan_chunk_boundary(pair):
    rng = np.random.default_rng(9)
    h = np.diag(rng.uniform(2.0, 6.0, NV))
    h[pair] = h[pair[::-1]] = 0.7
    g = rng.standard_normal((1, NV))
    with wp.ScopedDevice("cuda:0"):
        harness = Harness([h], g)
        out = harness.run()
        assert harness.ws.coupled.numpy()[0] == 2
        ref = harness.reference(0)
        assert np.abs(out[0] - ref).max() <= 1e-5 * np.abs(ref).max()


@cuda
def test_ill_conditioned_block_trips_the_residual_gate():
    rng = np.random.default_rng(10)
    h = np.diag(rng.uniform(2.0, 6.0, NV))
    ids = rng.choice(NV, size=20, replace=False)
    q, _ = np.linalg.qr(rng.standard_normal((20, 20)))
    h[np.ix_(ids, ids)] = q @ np.diag(np.logspace(0, 8, 20)) @ q.T
    h = (h + h.T) / 2
    with wp.ScopedDevice("cuda:0"):
        harness = Harness([h], rng.standard_normal((1, NV)))
        harness.run()
        assert harness.ws.status.numpy()[0] & 2, "float32 cannot solve a 1e8-conditioned block to 1e-4"
        assert harness.ws.max_relative_residual.numpy()[0] > 1e-4


# --- structural marking -------------------------------------------------------------------------

D_ROW = 2.0


def jacobian_data(worlds, mass=None, stale=0):
    """Data stand-in holding sparse constraint rows per world.

    ``worlds`` is a list (one per world) of rows ``(state, {dof: value})``; ``mass`` is
    ``(rows, cols, values-per-world)`` for stored off-diagonal mass entries.
    """
    n = len(worlds)
    njmax = max(8, max(len(rows) for rows in worlds))
    nnz = max(64, max(sum(len(e) for _, e in rows) for rows in worlds))
    state = np.zeros((n, njmax), np.int32)
    rownnz = np.zeros((n, njmax), np.int32)
    rowadr = np.zeros((n, njmax), np.int32)
    colind = np.zeros((n, 1, nnz), np.int32)
    jac = np.zeros((n, 1, nnz), np.float32)
    nefc = np.zeros(n, np.int32)
    for w, rows in enumerate(worlds):
        adr = 0
        nefc[w] = len(rows) - stale  # the last ``stale`` rows are leftovers beyond nefc
        for r, (st, entries) in enumerate(rows):
            state[w, r], rownnz[w, r], rowadr[w, r] = st, len(entries), adr
            for dof, value in entries.items():
                colind[w, 0, adr], jac[w, 0, adr] = dof, value
                adr += 1
    pairs = mass if mass is not None else ([], [], np.zeros((n, 0)))
    values = np.asarray(pairs[2], np.float32).reshape(n, -1)
    data = SimpleNamespace(
        nefc=wp.array(nefc, dtype=wp.int32),
        efc=SimpleNamespace(state=wp.array(state, dtype=wp.int32), J_rownnz=wp.array(rownnz, dtype=wp.int32),
                            J_rowadr=wp.array(rowadr, dtype=wp.int32), J_colind=wp.array(colind, dtype=wp.int32),
                            J=wp.array(jac, dtype=wp.float32)),
        M=wp.array(values, dtype=wp.float32))
    return data, (np.asarray(pairs[0]), np.asarray(pairs[1]), np.arange(len(pairs[0])))


def hessian(rows, mass_entries=(), extra=()):
    """diag + D * J_r J_r^T over quadratic rows + off-diagonal mass entries + any ``extra`` entries."""
    h = np.diag(np.full(NV, 4.0))
    for st, entries in rows:
        if st == 1:
            for a, va in entries.items():
                for b, vb in entries.items():
                    h[a, b] += D_ROW * va * vb
    for a, b, v in list(mass_entries) + list(extra):
        h[a, b] += v
        h[b, a] += v
    return h


ROWS = [
    (1, {3: 1.0, 7: 0.5, 20: -0.3}),   # quadratic: couples 3, 7, 20
    (0, {40: 1.0, 41: 1.0}),           # satisfied: contributes nothing
    (1, {50: 1.0, 51: 0.0}),           # quadratic but only one nonzero entry: couples nothing
    (1, {60: 1.0}),                    # single entry: diagonal only
    (4, {70: 1.0, 71: 1.0}),           # not pyramidal-quadratic: ignored by the marker
]


@cuda
def test_structural_marking_finds_jacobian_and_mass_coupling_and_solves_exactly():
    rng = np.random.default_rng(12)
    mats = [hessian(ROWS, [(10, 11, 0.7)]), hessian(ROWS, [(10, 11, 0.0)])]
    g = rng.standard_normal((2, NV))
    with wp.ScopedDevice("cuda:0"):
        data, pairs = jacobian_data([ROWS, ROWS], ([10], [11], [[0.7], [0.0]]))
        harness = Harness(mats, g, mass_pairs=pairs)
        out = harness.run(data=data)
        np.testing.assert_array_equal(harness.ws.coupled.numpy(), [5, 3])  # {3,7,20,10,11} and {3,7,20}
        for w in range(2):
            ref = harness.reference(w)
            assert np.abs(out[w] - ref).max() <= 1e-5 * np.abs(ref).max()
        np.testing.assert_array_equal(harness.ws.status.numpy(), 0)


@cuda
def test_rows_beyond_nefc_are_ignored():
    rng = np.random.default_rng(16)
    rows = [(1, {3: 1.0, 7: 0.5}), (1, {90: 1.0, 91: 1.0})]  # the second row is stale leftover data
    g = rng.standard_normal((1, NV))
    with wp.ScopedDevice("cuda:0"):
        data, pairs = jacobian_data([rows], ([10], [11], [[0.0]]), stale=1)
        harness = Harness([hessian(rows[:1])], g, mass_pairs=pairs)
        out = harness.run(data=data)
        assert harness.ws.coupled.numpy()[0] == 2
        ref = harness.reference(0)
        assert np.abs(out[0] - ref).max() <= 1e-5 * np.abs(ref).max()


@cuda
def test_verification_flags_a_coupling_the_structure_does_not_explain():
    rng = np.random.default_rng(13)
    g = rng.standard_normal((2, NV))
    # World 1's Hessian holds an entry (80, 81) that no row or mass entry explains.
    mats = [hessian(ROWS), hessian(ROWS, extra=[(80, 81, 0.9)])]
    with wp.ScopedDevice("cuda:0"):
        data, pairs = jacobian_data([ROWS, ROWS], ([10], [11], [[0.0], [0.0]]))
        harness = Harness(mats, g, mass_pairs=pairs, verify_structure=True)
        harness.run(data=data)
        status = harness.ws.status.numpy()
        assert status[0] == 0 and status[1] & 4, status
        # The world that fits under the rescan limit is not rescanned: it keeps the structural set.
        np.testing.assert_array_equal(harness.ws.coupled.numpy(), [3, 3])
        np.testing.assert_array_equal(harness.ws.rescans.numpy(), [0, 0])
        # Without verification the same mismatch goes unreported: this is why verify exists.
        silent = Harness(mats, g, mass_pairs=pairs)
        silent.run(data=data)
        assert silent.ws.status.numpy()[1] & 4 == 0


@cuda
def test_structural_flags_accumulate_over_a_solve_and_reset_on_the_next():
    rng = np.random.default_rng(14)
    g = rng.standard_normal((1, NV))
    active = [(1, {40: 1.0, 41: 1.0})]
    inactive = [(0, {40: 1.0, 41: 1.0})]
    with wp.ScopedDevice("cuda:0"):
        data1, pairs = jacobian_data([active], ([10], [11], [[0.0]]))
        harness = Harness([hessian(active)], g, mass_pairs=pairs)
        out = harness.run(data=data1)  # first call of a solve: the row is quadratic
        assert harness.ws.coupled.numpy()[0] == 2
        # Second call (an incremental iteration): the row flipped to satisfied and the subtraction left
        # rounding residue on its support. The solve must still see 40 and 41 as coupled.
        data2, _ = jacobian_data([inactive], ([10], [11], [[0.0]]))
        residue = hessian(inactive, extra=[(40, 41, 3e-7)])
        harness.set_matrices([residue])
        changed = wp.array(np.array([1], np.int32), dtype=wp.int32)
        out = harness.run(quad=changed, skip_unchanged=True, data=data2)
        assert harness.ws.coupled.numpy()[0] == 2, "flags must persist within a solve"
        ref = harness.reference(0)
        assert np.abs(out[0] - ref).max() <= 1e-5 * np.abs(ref).max()
        # The next solve's first call clears them.
        harness.set_matrices([hessian(inactive)])
        harness.run(data=data2)
        assert harness.ws.coupled.numpy()[0] == 0


@cuda
def test_structural_workspace_without_data_falls_back_to_scanning():
    rng = np.random.default_rng(15)
    mats = [spd(rng, 20)]
    g = rng.standard_normal((1, NV))
    with wp.ScopedDevice("cuda:0"):
        _, pairs = jacobian_data([ROWS], ([10], [11], [[0.0]]))
        harness = Harness(mats, g, mass_pairs=pairs)
        out = harness.run(data=None)  # e.g. a compact solve: scan H instead
        assert harness.ws.coupled.numpy()[0] == coupled_set(mats[0]).sum()
        ref = harness.reference(0)
        assert np.abs(out[0] - ref).max() <= 1e-5 * np.abs(ref).max()


@cuda
def test_verify_structure_requires_structural_marking():
    with wp.ScopedDevice("cuda:0"):
        with pytest.raises(ValueError, match="verify_structure"):
            CoupledWorkspace(NV, 1, "cuda:0", verify_structure=True)


def cancelling_rows(pairs):
    """Quadratic friction-style row pairs (+1, +1) and (+1, -1) whose off-diagonal sum is exactly 0."""
    rows = []
    for k in range(pairs):
        a, b = 2 * k, 2 * k + 1
        rows += [(1, {a: 1.0, b: 1.0}), (1, {a: 1.0, b: -1.0})]
    return rows


@cuda
def test_loose_structural_set_is_rescanned_for_the_exact_one():
    rng = np.random.default_rng(17)
    g = rng.standard_normal((2, NV))
    loose = cancelling_rows(12) + [(1, {80: 1.0, 81: 0.5})]  # 24 loosely flagged DOFs + one real pair
    tight = [(1, {80: 1.0, 81: 0.5})]
    with wp.ScopedDevice("cuda:0"):
        data, pairs = jacobian_data([loose, tight], ([10], [11], [[0.0], [0.0]]))
        harness = Harness([hessian(loose), hessian(tight)], g, mass_pairs=pairs, tile=16, rescan_above=16,
                          max_coupled=32)
        out = harness.run(data=data)
        # World 0: 26 structural DOFs (> 16) -> rescanned: only {80, 81} are really coupled.
        # World 1: 2 structural DOFs -> kept as is.
        np.testing.assert_array_equal(harness.ws.coupled.numpy(), [2, 2])
        np.testing.assert_array_equal(harness.ws.rescans.numpy(), [1, 0])
        np.testing.assert_array_equal(harness.ws.fallbacks.numpy(), [0, 0])
        for w in range(2):
            ref = harness.reference(w)
            assert np.abs(out[w] - ref).max() <= 1e-5 * np.abs(ref).max()
        np.testing.assert_array_equal(harness.ws.status.numpy(), 0)


@cuda
def test_rescan_keeps_a_world_that_would_overflow_out_of_the_host_fallback():
    rng = np.random.default_rng(18)
    g = rng.standard_normal((1, NV))
    loose = cancelling_rows(12)  # 24 structural DOFs, actually uncoupled
    with wp.ScopedDevice("cuda:0"):
        data, pairs = jacobian_data([loose], ([10], [11], [[0.0]]))
        # max_coupled=16 would overflow on the structural set (24) but not on the exact one (0)
        harness = Harness([hessian(loose)], g, mass_pairs=pairs, tile=16, rescan_above=16, max_coupled=16)
        out = harness.run(data=data)
        np.testing.assert_array_equal(harness.ws.fallbacks.numpy(), [0])
        np.testing.assert_array_equal(harness.shadows[0], [True])
        ref = harness.reference(0)
        assert np.abs(out[0] - ref).max() <= 1e-5 * np.abs(ref).max()


@cuda
def test_rescan_is_off_when_the_limit_covers_every_dof():
    rng = np.random.default_rng(19)
    g = rng.standard_normal((1, NV))
    loose = cancelling_rows(12)
    with wp.ScopedDevice("cuda:0"):
        data, pairs = jacobian_data([loose], ([10], [11], [[0.0]]))
        harness = Harness([hessian(loose)], g, mass_pairs=pairs, tile=16, rescan_above=NV, max_coupled=64)
        harness.run(data=data)
        assert harness.ws.coupled.numpy()[0] == 24 and harness.ws.rescans.numpy()[0] == 0


@cuda
def test_rescan_threshold_is_strictly_above_the_limit():
    rng = np.random.default_rng(20)
    g = rng.standard_normal((2, NV))
    at_limit = cancelling_rows(8)       # 16 structural DOFs == limit: kept
    just_over = cancelling_rows(8) + [(1, {80: 1.0, 81: 1.0})]  # 18 > 16: rescanned
    with wp.ScopedDevice("cuda:0"):
        data, pairs = jacobian_data([at_limit, just_over], ([10], [11], [[0.0], [0.0]]))
        harness = Harness([hessian(at_limit), hessian(just_over)], g, mass_pairs=pairs, tile=16,
                          rescan_above=16, max_coupled=32)
        harness.run(data=data)
        np.testing.assert_array_equal(harness.ws.rescans.numpy(), [0, 1])
        np.testing.assert_array_equal(harness.ws.coupled.numpy(), [16, 2])


@cuda
def test_a_cached_call_does_not_rescan_again():
    rng = np.random.default_rng(21)
    g = rng.standard_normal((1, NV))
    # 20 truly coupled DOFs (one dense block) > limit 16: the exact set is still large.
    rows = [(1, {d: 1.0 + 0.1 * d for d in range(20)})]
    with wp.ScopedDevice("cuda:0"):
        data, pairs = jacobian_data([rows], ([10], [11], [[0.0]]))
        harness = Harness([hessian(rows)], g, mass_pairs=pairs, tile=16, rescan_above=16, max_coupled=32)
        harness.run(data=data)
        assert harness.ws.rescans.numpy()[0] == 1 and harness.ws.coupled.numpy()[0] == 20
        out = harness.run(quad=wp.zeros(1, dtype=wp.int32), skip_unchanged=True, data=data)
        assert harness.ws.rescans.numpy()[0] == 1, "no constraint changed: nothing to rescan"
        assert harness.ws.coupled.numpy()[0] == 20
        ref = harness.reference(0)
        assert np.abs(out[0] - ref).max() <= 1e-5 * np.abs(ref).max()
