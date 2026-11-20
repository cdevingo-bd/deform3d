"""Cooperative CUDA PCG for small, assembled MuJoCo flex material systems.

One block owns each world's solve. Sparse matrix products, reductions and vector
updates stay in the same launch. The matrix is exactly M + diag(efm_diag) + efm_K;
unsupported additional metric terms are rejected. Double precision recurrence
and periodic true-residual replacement limit cancellation in stiff thin shells.
"""
from contextlib import contextmanager
from functools import lru_cache
import hashlib
from importlib.metadata import version
import inspect
import threading
import numbers

import warp as wp
from . import _buffer, _no_overlap

_LOCK = threading.RLock()
_SOURCE_SHA = 'c16d1967bd68653e59e503afc9471336b966493dab5e8963a81b28b34c22039f'


@lru_cache(None)
def _kernel(block_size):
    @wp.kernel(module='unique', enable_backward=False)
    def solve(nv: int, max_iterations: int, supplied_rhs: bool,
              m_row: wp.array(dtype=wp.int32), m_col: wp.array(dtype=wp.int32),
              m_address: wp.array(dtype=wp.int32), mass: wp.array2d(dtype=wp.float32),
              k_count: wp.array(dtype=wp.int32), k_row: wp.array(dtype=wp.int32),
              k_col: wp.array(dtype=wp.int32), stiffness: wp.array2d(dtype=wp.float32),
              diagonal: wp.array2d(dtype=wp.float32),
              force: wp.array2d(dtype=wp.float32), shift: wp.array2d(dtype=wp.float32),
              actuator_shift: wp.array2d(dtype=wp.float32), tolerance: wp.array(dtype=wp.float32),
              output: wp.array2d(dtype=wp.float32), iterations: wp.array(dtype=wp.int32),
              residual: wp.array(dtype=wp.float32), status: wp.array(dtype=wp.int32)):
        world, row = wp.tid()
        b, diag = wp.float64(0.0), wp.float64(1.0)
        if row < nv:
            b = wp.float64(force[world, row])
            if not supplied_rhs:
                b += wp.float64(shift[world, row]) + wp.float64(actuator_shift[world, row])
            diag = wp.float64(diagonal[world, row])
            for j in range(m_row[row], m_row[row + 1]):
                if m_col[j] == row:
                    diag += wp.float64(mass[world, m_address[j]])
            for j in range(k_row[row], k_row[row] + k_count[row]):
                if k_col[j] == row:
                    diag += wp.float64(stiffness[world, j])
        bad = int(0)
        if not wp.isfinite(b) or not wp.isfinite(diag) or diag <= 0.0:
            bad = 1
        bad_sum = wp.tile_reduce(wp.add, wp.tile(bad))[0]
        if bad_sum > 0:
            if row == 0:
                status[world] = 2
                iterations[world] = 0
                residual[world] = wp.inf
            return
        shared = wp.tile_zeros(shape=(block_size,), dtype=wp.float64, storage='shared')
        x = wp.float64(0.0)
        # Explicit value construction is required here: aliasing r = b lets
        # Warp's loop-carried variable update also overwrite the immutable RHS.
        r = wp.float64(b)
        p = r / diag
        rho = wp.tile_reduce(wp.add, wp.tile(r * p))[0]
        bn = wp.tile_reduce(wp.add, wp.tile(b * b))[0]
        tol = wp.float64(wp.max(tolerance[world % tolerance.shape[0]], 1.0e-5))
        target = tol * tol * bn
        done = bn == 0.0
        steps, code = int(0), int(0)
        for it in range(max_iterations):
            if done:
                break
            wp.tile_assign(shared, wp.tile(p))
            ap = wp.float64(0.0)
            if row < nv:
                ap = wp.float64(diagonal[world, row]) * p
                for j in range(m_row[row], m_row[row + 1]):
                    ap += wp.float64(mass[world, m_address[j]]) * shared[m_col[j]]
                for j in range(k_row[row], k_row[row] + k_count[row]):
                    ap += wp.float64(stiffness[world, j]) * shared[k_col[j]]
            pap = wp.tile_reduce(wp.add, wp.tile(p * ap))[0]
            if not wp.isfinite(pap) or pap <= 0.0:
                code = 2
                break
            alpha = rho / pap
            x += alpha * p
            r -= alpha * ap
            steps = it + 1
            rr = wp.tile_reduce(wp.add, wp.tile(r * r))[0]
            restart = rr <= target or steps % 32 == 0
            if restart:
                wp.tile_assign(shared, wp.tile(x))
                ax = wp.float64(0.0)
                if row < nv:
                    ax = wp.float64(diagonal[world, row]) * x
                    for j in range(m_row[row], m_row[row + 1]):
                        ax += wp.float64(mass[world, m_address[j]]) * shared[m_col[j]]
                    for j in range(k_row[row], k_row[row] + k_count[row]):
                        ax += wp.float64(stiffness[world, j]) * shared[k_col[j]]
                r = b - ax
                rr = wp.tile_reduce(wp.add, wp.tile(r * r))[0]
            done = rr <= target
            z = r / diag
            next_rho = wp.tile_reduce(wp.add, wp.tile(r * z))[0]
            beta = wp.float64(0.0)
            if not restart:
                beta = next_rho / rho
            p = z + beta * p
            rho = next_rho
        if not done and code == 0:
            code = 1
        # Measure the actual float32 output against the matrix, rather than the
        # recursively updated PCG residual. This catches loss on conversion too.
        rounded = wp.float64(wp.float32(x))
        wp.tile_assign(shared, wp.tile(rounded))
        ax = wp.float64(0.0)
        if row < nv:
            ax = wp.float64(diagonal[world, row]) * rounded
            for j in range(m_row[row], m_row[row + 1]):
                ax += wp.float64(mass[world, m_address[j]]) * shared[m_col[j]]
            for j in range(k_row[row], k_row[row] + k_count[row]):
                ax += wp.float64(stiffness[world, j]) * shared[k_col[j]]
            output[world, row] = wp.float32(x)
        err = b - ax
        norm2 = wp.tile_reduce(wp.add, wp.tile(err * err))[0]
        relative = wp.float64(0.0)
        if bn > 0.0:
            relative = wp.sqrt(norm2 / bn)
        if not wp.isfinite(relative):
            code = 2
        elif relative > wp.float64(5.0) * tol and code == 0:
            code = 3
        if row == 0:
            iterations[world] = steps
            residual[world] = wp.float32(relative)
            status[world] = code
    return solve


class CooperativeMetricSolver:
    """Opt-in solver for 1..512 DOF assembled flex systems on CUDA.

    Allocate once per model/world-count/stream. No allocations or readbacks in
    enqueue. Status per world: 0 success, 1 iteration budget, 2 invalid/non-SPD,
    3 float32 output residual above 5*max(model tolerance, 1e-5). Check status
    after stepping; any nonzero status invalidates that step. Diagnostics are
    overwritten each call, so accumulate them on device for long graph runs.
    """
    def __init__(self, model, nworld=1, *, max_iterations=None):
        if version('mujoco-warp') != '4.0.1':
            raise RuntimeError('reviewed only for mujoco-warp 4.0.1')
        from mujoco_warp._src import types
        device = model.body_mass.device
        if (not device.is_cuda or not 1 <= model.nv <= 512
                or isinstance(nworld, bool) or not isinstance(nworld, numbers.Integral)
                or not 1 <= nworld <= 2147483647):
            raise ValueError('requires CUDA, 1..512 DOFs and positive world count')
        if (model.opt.integrator != types.IntegratorType.DISCRETE or model.nefmK <= 0
                or model.ntendon or model.nactuator or model.has_non_simple_flex
                or model.nflexrodedge or model.efm0_active or model.has_flex_passive or model.has_fluid
                or not model.flex_interp_assemblable
                or model.opt.enableflags & types.EnableBit.SLEEP
                or int(model.opt.enableflags) & (1 << 6)):  # CPU mjENBL_IPC; not exposed by MJWarp 4.0.1
            raise ValueError('model has unsupported terms outside the assembled M + diag + K metric')
        self.model, self.nworld, self.device = model, nworld, device
        self.max_iterations = max_iterations if max_iterations is not None else 4 * model.nv
        if (isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, numbers.Integral)
                or self.max_iterations < 1 or self.max_iterations > 2147483647):
            raise ValueError('max_iterations must be positive')
        self.block_size = max(32, 1 << (model.nv - 1).bit_length())
        self.iterations = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.relative_residual = wp.zeros(nworld, dtype=wp.float32, device=device)
        self.status = wp.zeros(nworld, dtype=wp.int32, device=device)

    def enqueue(self, data, output, force=None):
        m = self.model
        if data.nworld != self.nworld or data.qpos.device != self.device or data.M.ndim != 2:
            raise ValueError('data world count/device/inertia layout mismatch')
        rhs = data.qfrc_smooth if force is None else force
        for name, array in [('output', output), ('force', rhs)]:
            _buffer(array, name, self.device, wp.float32)
            if array.shape != (self.nworld, m.nv):
                raise ValueError(f'{name} must be float32 [world, nv] on the model device')
        # Each block reads its matrix throughout the solve. An output view into
        # any input (or overlapping world rows) would introduce a device race.
        for source in (rhs, data.efm_c, data.efm_ca, data.M, data.efm_diag, data.efm_K_val,
                       m.M_mulm_rowadr, m.M_mulm_col, m.M_mulm_madr,
                       m.efm_K_rownnz, m.efm_K_rowadr, m.efm_K_colind, m.opt.tolerance,
                       self.iterations, self.relative_residual, self.status):
            _no_overlap(output, source, 'output must not overlap solve inputs or diagnostics')
        wp.launch_tiled(_kernel(self.block_size), dim=self.nworld,
                        inputs=[m.nv, self.max_iterations, force is not None,
                                m.M_mulm_rowadr, m.M_mulm_col, m.M_mulm_madr, data.M,
                                m.efm_K_rownnz, m.efm_K_rowadr, m.efm_K_colind, data.efm_K_val,
                                data.efm_diag, data.qfrc_smooth if force is None else force,
                                data.efm_c, data.efm_ca, m.opt.tolerance],
                        outputs=[output, self.iterations, self.relative_residual, self.status],
                        block_dim=self.block_size, device=self.device)

    @contextmanager
    def capture_policy(self):
        """Use this solver for the selected model during warmup/graph capture."""
        from mujoco_warp._src import derivative
        if version('mujoco-warp') != '4.0.1':
            raise RuntimeError('reviewed only for mujoco-warp 4.0.1')
        with _LOCK:
            original = derivative.eff_solve
            if hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != _SOURCE_SHA:
                raise RuntimeError('unsupported MJWarp effective solve or nested policy')
            def dispatch(m, d, qacc, qfrc=None):
                if m is self.model:
                    return self.enqueue(d, qacc, qfrc)
                return original(m, d, qacc, qfrc)
            derivative.eff_solve = dispatch
            try:
                yield self
            finally:
                derivative.eff_solve = original
