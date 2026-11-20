"""Experimental dense direct solve of the discrete flex effective metric.

MuJoCo Warp's ``derivative.eff_solve`` solves ``M_hat qacc = rhs`` for flex models with
preconditioned CG: a data-dependent device loop whose cost is dominated by launch latency at
small world counts. For models whose effective metric is only ``M + diag + K`` (CSR flex
stiffness/damping), this module assembles M_hat densely once per step and factors it with
deform3d's blocked Cholesky (``mujoco_newton._Workspace``), replacing the iteration count by a
fixed-cost factorization.

Scope and safety:
  * Supported only when every other effective-metric term is absent (no tendon/actuator/
    contact/fluid/non-simple-flex terms), the integrator is DISCRETE, and sleeping is off.
    ``supported(model)`` reports why not; unsupported models fall back to the native solve.
  * Dense storage is nworld x nv x nv float32; use it for modest nv (the workspace limit is 512).
  * Results are checked by the workspace's float32 relative residual (``status`` bit 2).
  * ``capture_policy`` replaces ``derivative.eff_solve`` process-wide while active. Construct
    and capture on one thread, retain this object through graph replay, and do not build
    unrelated model graphs concurrently. Reviewed against MuJoCo Warp 3.17.0 (source build) and
    Warp 1.16 using the 'tiled' Newton kernels; the 'register/ordered' kernels need Warp 1.19.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import warp as wp

from .mujoco_newton import _Workspace


@wp.kernel(enable_backward=False)
def _assemble(
    m_rowadr: wp.array(dtype=wp.int32), m_col: wp.array(dtype=wp.int32),
    m_madr: wp.array(dtype=wp.int32), k_rownnz: wp.array(dtype=wp.int32),
    k_rowadr: wp.array(dtype=wp.int32), k_colind: wp.array(dtype=wp.int32),
    nv: int, M: wp.array2d(dtype=wp.float32), diag: wp.array2d(dtype=wp.float32),
    kval: wp.array2d(dtype=wp.float32), out: wp.array3d(dtype=wp.float32),
):
    """One thread per (world, row): gather M, add diagonal, scatter K. No atomics."""
    w, row = wp.tid()
    for j in range(nv):
        out[w, row, j] = 0.0
    for k in range(m_rowadr[row], m_rowadr[row + 1]):
        out[w, row, m_col[k]] += M[w, m_madr[k]]
    out[w, row, row] += diag[w, row]
    nnz = k_rownnz[row]
    adr = k_rowadr[row]
    for t in range(nnz):
        out[w, row, k_colind[adr + t]] += kval[w, adr + t]


@wp.kernel(enable_backward=False)
def _rhs(qfrc_smooth: wp.array2d(dtype=wp.float32), efm_c: wp.array2d(dtype=wp.float32),
         efm_ca: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
    w, i = wp.tid()
    out[w, i] = qfrc_smooth[w, i] + efm_c[w, i] + efm_ca[w, i]


@wp.kernel(enable_backward=False)
def _store(search: wp.array2d(dtype=wp.float32), qacc: wp.array2d(dtype=wp.float32)):
    w, i = wp.tid()
    qacc[w, i] = -search[w, i]  # workspace returns -H^-1 grad


def supported(model):
    """Return (ok, reason). Every term outside M + diag + K must be absent."""
    from mujoco_warp._src.types import EnableBit, IntegratorType

    checks = [
        (model.opt.integrator == IntegratorType.DISCRETE, "integrator is not DISCRETE"),
        (model.nefmK > 0, "no CSR flex stiffness (nothing to solve iteratively)"),
        (not model.has_non_simple_flex, "non-simple flex terms are matrix-free"),
        (not model.has_flex_passive, "flex contact term"),
        (not model.has_fluid, "fluid term"),
        (not model.has_efm_actuator, "actuator term"),
        (not model.has_tendon_stiffness, "tendon stiffness term"),
        (not model.has_tendon_damping, "tendon damping term"),
        (not model.efm0_active, "efm0 backbone path"),
        (model.flex_interp_assemblable, "flex interp is not assemblable"),
        (not (model.opt.enableflags & EnableBit.SLEEP), "sleeping enabled"),
        (model.nv <= 512, "nv exceeds the 512-dof workspace limit"),
    ]
    for ok, reason in checks:
        if not ok:
            return False, reason
    return True, ""


class _EffectiveSolvePolicy:
    """Shared ``derivative.eff_solve`` replacement plumbing."""

    _model = None

    @contextmanager
    def capture_policy(self):
        """Replace ``derivative.eff_solve`` while warming up and capturing graphs."""
        from mujoco_warp._src import derivative

        original = derivative.eff_solve
        owner = self

        def dispatch(m, d, qacc, qfrc=None):
            if m is owner._model:
                return owner.solve(m, d, qacc, qfrc)
            return original(m, d, qacc, qfrc)

        derivative.eff_solve = dispatch
        try:
            yield self
        finally:
            derivative.eff_solve = original


class DenseEffectiveSolve(_EffectiveSolvePolicy):
    """Own the dense M_hat buffer and Cholesky workspace for one model/data pair."""

    def __init__(self, model, data, *, tile_size=32, update_threads=64):
        ok, reason = supported(model)
        if not ok:
            raise ValueError(f"unsupported model for dense effective solve: {reason}")
        device = model.body_mass.device
        if not device.is_cuda:
            raise ValueError("requires a CUDA device")
        self.nv, self.nworld, self.device = model.nv, data.nworld, device
        self.workspace = _Workspace(model.nv, data.nworld, device, "float32", tile_size,
                                    update_threads=update_threads)
        self.matrix = wp.zeros((data.nworld, model.nv, model.nv), dtype=wp.float32, device=device)
        self.rhs = wp.zeros((data.nworld, model.nv), dtype=wp.float32, device=device)
        self._ctx = SimpleNamespace(
            h=self.matrix, grad=self.rhs, done=wp.zeros(data.nworld, dtype=wp.bool, device=device),
            search=wp.zeros((data.nworld, model.nv), dtype=wp.float32, device=device),
            search_dot=wp.zeros(data.nworld, dtype=wp.float32, device=device),
            newton_decrement=wp.zeros(data.nworld, dtype=wp.float32, device=device),
        )
        self._count = wp.ones(data.nworld, dtype=wp.int32, device=device)
        self._model = model

    @property
    def status(self):
        """Sticky workspace status per world: 0 ok, bit 1 invalid factor, bit 2 residual > 1e-4."""
        return self.workspace.status

    def assemble(self, model, data):
        wp.launch(_assemble, dim=(data.nworld, model.nv), device=self.device,
                  inputs=[model.M_mulm_rowadr, model.M_mulm_col, model.M_mulm_madr,
                          model.efm_K_rownnz, model.efm_K_rowadr, model.efm_K_colind, model.nv,
                          data.M, data.efm_diag, data.efm_K_val], outputs=[self.matrix])

    def solve(self, model, data, qacc, qfrc=None):
        """Drop-in for ``derivative.eff_solve(m, d, qacc, qfrc=None)`` on supported models."""
        if qfrc is None:
            wp.launch(_rhs, dim=(data.nworld, model.nv), device=self.device,
                      inputs=[data.qfrc_smooth, data.efm_c, data.efm_ca], outputs=[self.rhs])
        else:
            wp.copy(self.rhs, qfrc)
        self.assemble(model, data)
        self.workspace.enqueue(self._ctx, self._count, self._count, True, False)
        wp.launch(_store, dim=(data.nworld, model.nv), device=self.device,
                  inputs=[self._ctx.search], outputs=[qacc])
