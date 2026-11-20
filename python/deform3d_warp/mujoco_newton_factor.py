"""Experimental drop-in Newton Cholesky for MuJoCo Warp builds that lack a version pin.

``mujoco_newton.ParallelNewtonSolver`` is reviewed only for the PyPI mujoco-warp 4.0.1 and checks
``importlib.metadata`` for it. A MuJoCo Warp built from a patched source archive has no
distribution metadata, so that check cannot pass there. This policy applies the same replacement
(deform3d's blocked tile Cholesky for ``solver._cholesky_factorize_solve``) but guards on the
*source text of the hook it replaces* instead: it only installs over a hook whose SHA-256 is in
``REVIEWED_HOOKS``. Add a hash only after re-reviewing that implementation.

Reviewed: mujoco_warp 3.17.0 source build (upstream commit f34c6dd5 plus local patches), Warp 1.16,
'tiled' kernels.

The host still assembles the Newton Hessian, picks active constraints and runs its line search.
Only the SPD factor + solve is replaced. Outputs are checked by the workspace's float64 residual
(``status`` bit 2) and finiteness (bit 1); a nonzero status invalidates the trajectory.

Two methods are available:

``method="coupled"`` (default) factors only the DOFs that the Hessian actually couples and solves
the rest by a scalar division. It is exact, and its cost per world scales with the size of the
coupled set rather than ``nv``, so it keeps winning as the world count grows. A world whose coupled
set exceeds ``max_coupled`` is solved by the host's own hook instead (see
``mujoco_newton_coupled``); ``fallbacks`` and ``coupled`` expose how often that happens and how
large the sets are. Measure ``coupled`` on your scene: a Hessian that is dense (large robots with
dense mass blocks and many contacts) falls back constantly and gains nothing.

``method="dense"`` factors the full matrix with the blocked tile Cholesky. It wins only when the
solve is latency-bound (few worlds) and loses at large world counts.
"""

from contextlib import contextmanager
import hashlib
import inspect
import threading

import warp as wp

from .mujoco_newton import _Workspace
from .mujoco_newton_coupled import CoupledWorkspace

_LOCK = threading.RLock()

REVIEWED_HOOKS = {
    "5f04c3acf13da87ed1d913c53da048c384e7f1312ae5bb8bf74f8ff0213d0433": "mujoco_warp 3.17.0 source build (f34c6dd5 + local patches)",
}


# Sources that assemble the Newton Hessian; structural coupled-set marking depends on them.
_ASSEMBLY = ("_update_gradient", "_update_gradient_incremental", "_JTDACJ_sparse",
             "_update_gradient_h_incremental_sparse", "_update_gradient_init_h_sparse")
REVIEWED_ASSEMBLIES = {
    "fdc952e6c5fc771d60cc43e6618bb955f7147d55eb7de811057d92d4a657602a": "mujoco_warp 3.17.0 source build (f34c6dd5 + local patches)",
}


def assembly_is_reviewed():
    """Return (ok, description) for the installed Hessian assembly code."""
    from mujoco_warp._src import solver

    try:
        text = "".join(inspect.getsource(getattr(solver, name)) for name in _ASSEMBLY)
    except (AttributeError, OSError, TypeError) as error:
        return False, f"cannot read the Hessian assembly source: {error}"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return digest in REVIEWED_ASSEMBLIES, REVIEWED_ASSEMBLIES.get(digest, f"unreviewed sha256 {digest[:12]}")


def hook_is_reviewed(hook=None):
    """Return (ok, description). ``hook`` defaults to the installed solver's factor hook."""
    if hook is None:
        from mujoco_warp._src import solver

        hook = solver._cholesky_factorize_solve
    digest = hashlib.sha256(inspect.getsource(hook).encode()).hexdigest()
    return digest in REVIEWED_HOOKS, REVIEWED_HOOKS.get(digest, f"unreviewed sha256 {digest[:12]}")


def structure_unsupported(model):
    """Why structural coupled-set marking cannot be used for this model, or None."""
    from mujoco_warp._src import types

    if not model.is_sparse:
        return "the constraint Jacobian is dense"
    if model.opt.integrator == types.IntegratorType.DISCRETE:
        return "the DISCRETE integrator adds tendon, actuator and flex terms to the Hessian"
    if model.opt.cone == types.ConeType.ELLIPTIC:
        return "elliptic cones couple the rows of a contact, which the per-row structure does not model"
    return None


def _mass_pairs(model):
    """Stored off-diagonal mass-matrix entries as (row dof, column dof, element id) arrays."""
    import numpy as np

    elements = model.M_elemid.numpy()
    rows, cols = np.nonzero(np.tril(elements >= 0, -1))
    return rows, cols, elements[rows, cols]


class NewtonFactorPolicy:
    """Own a Cholesky workspace and install it over MJWarp's Newton factor hook."""

    def __init__(self, model, nworld, *, method="coupled", tile_size=32, update_threads=64,
                 max_coupled=128, structure="scan", verify_structure=False, rescan_above=None):
        from mujoco_warp._src import types

        device = model.body_mass.device
        if not device.is_cuda:
            raise ValueError("requires a CUDA device")
        if model.opt.solver != types.SolverType.NEWTON:
            raise ValueError("requires the Newton solver")
        if not 1 <= model.nv <= 512:
            raise ValueError("requires 1..512 DOFs")
        if model.opt.enableflags & types.EnableBit.SLEEP:
            raise ValueError("sleeping is not supported")
        if isinstance(nworld, bool) or int(nworld) < 1:
            raise ValueError("nworld must be a positive integer")
        if method not in ("coupled", "dense"):
            raise ValueError("method must be 'coupled' or 'dense'")
        if structure not in ("scan", "jacobian"):
            raise ValueError("structure must be 'scan' or 'jacobian'")
        if structure == "jacobian" and method != "coupled":
            raise ValueError("structure='jacobian' needs method='coupled'")
        if verify_structure and structure != "jacobian":
            raise ValueError("verify_structure needs structure='jacobian'")
        ok, why = hook_is_reviewed()
        if not ok:
            raise RuntimeError(f"Newton factor hook not reviewed for this build: {why}")
        mass_pairs = None
        if structure == "jacobian":
            why = structure_unsupported(model)
            if why:
                raise ValueError(f"structure='jacobian' is not supported here: {why}")
            ok, why = assembly_is_reviewed()
            if not ok:
                raise RuntimeError(f"Hessian assembly not reviewed for structure='jacobian': {why}")
            mass_pairs = _mass_pairs(model)
        self._model, self.nworld, self.device, self.method = model, int(nworld), device, method
        self.structure = structure
        if method == "coupled":
            self.workspace = CoupledWorkspace(model.nv, self.nworld, device, tile_size=tile_size,
                                              max_coupled=max_coupled, mass_pairs=mass_pairs,
                                              verify_structure=verify_structure, rescan_above=rescan_above)
        else:
            self.workspace = _Workspace(model.nv, self.nworld, device, "float32", tile_size,
                                        update_threads=update_threads)

    @property
    def status(self):
        """Sticky per-world status: 0 ok, bit 1 invalid factor/output, bit 2 residual > 1e-4, bit 4 structural
        coupled set missed a DOF (only with ``verify_structure``)."""
        return self.workspace.status

    @property
    def fallbacks(self):
        """Per-world count of solves handed back to the host hook (coupled method only)."""
        return getattr(self.workspace, "fallbacks", None)

    @property
    def rescans(self):
        """Per-world count of structural sets replaced by an exact scan (structure='jacobian')."""
        return getattr(self.workspace, "rescans", None)

    @property
    def coupled(self):
        """Per-world size of the most recent coupled set (coupled method only)."""
        return getattr(self.workspace, "coupled", None)

    @contextmanager
    def capture_policy(self):
        """Replace the Newton factor hook while warming up and capturing graphs."""
        from mujoco_warp._src import solver

        with _LOCK:
            original = solver._cholesky_factorize_solve
            ok, why = hook_is_reviewed(original)
            if not ok:
                raise RuntimeError(f"unreviewed Newton implementation or nested policy: {why}")
            owner = self

            def dispatch(m, d, ctx, skip_unchanged=False, skip_noflip=False):
                if m is not owner._model:
                    return original(m, d, ctx, skip_unchanged, skip_noflip)
                if d.nworld != owner.nworld or d.qpos.device != owner.device:
                    raise ValueError("data world count or device mismatch")
                quad = ctx.quad_changed_count if skip_unchanged else d.nefc
                state = ctx.state_changed_count if skip_noflip else d.nefc
                if owner.method == "coupled":
                    # Overflowing worlds are re-solved by the host hook, always refactoring: the
                    # host's cached factor may be stale because this policy solved earlier calls.
                    # A compact (sub-model) solve has a different Hessian layout: scan it.
                    structural = d if owner.structure == "jacobian" and ctx.compact_d_full is None else None
                    owner.workspace.enqueue(
                        ctx, quad, state, skip_unchanged, skip_noflip,
                        lambda shadow: original(m, d, shadow, False, False), structural)
                else:
                    owner.workspace.enqueue(ctx, quad, state, skip_unchanged, skip_noflip)

            solver._cholesky_factorize_solve = dispatch
            try:
                yield self
            finally:
                solver._cholesky_factorize_solve = original
