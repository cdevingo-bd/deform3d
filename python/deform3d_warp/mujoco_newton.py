"""Experimental parallel blocked Newton factorization for MJWarp 4.0.1.

The host still assembles the Newton Hessian, chooses active constraints, and
performs its line search. Only the SPD linear solve is replaced. Independent
matrix tiles update in separate CUDA blocks. Factor precision is explicit;
the returned float32 search direction is checked with a double residual.
"""
from contextlib import contextmanager
from functools import lru_cache
import hashlib
from importlib.metadata import version
import inspect
import numbers
import threading

import warp as wp

_LOCK = threading.RLock()
_SOURCE_SHA = 'bf51395162c4b0880183f28994538836580f9da91a44a6be507c80cb9d2d302f'


@wp.kernel(enable_backward=False)
def _masks(done: wp.array(dtype=wp.bool), quad: wp.array(dtype=wp.int32),
           state: wp.array(dtype=wp.int32), skip_unchanged: bool, skip_noflip: bool,
           solve: wp.array(dtype=wp.int32), factor: wp.array(dtype=wp.int32),
           initialized: wp.array(dtype=wp.int32)):
    w = wp.tid()
    active = not done[w] and (not skip_noflip or state[w] > 0)
    solve[w] = int(active)
    factor[w] = int(active and (not skip_unchanged or quad[w] > 0 or initialized[w] == 0))
    if active:
        initialized[w] = 1


@wp.kernel(enable_backward=False)
def _any_refactor(refactor:wp.array(dtype=wp.int32), condition:wp.array(dtype=wp.int32)):
    # One writer, no reset/atomic race. Stop as soon as one world needs work.
    value=int(0)
    for world in range(refactor.shape[0]):
        if refactor[world]!=0:
            value=1
            break
    condition[0]=value


@lru_cache(None)
def _transport_kernels(precision):
    real = wp.float64 if precision == 'float64' else wp.float32
    @wp.kernel(enable_backward=False)
    def _copy_upper(nv: int, matrix: wp.array3d(dtype=wp.float32),
                    active: wp.array(dtype=wp.int32), work: wp.array3d(dtype=real)):
        w, i, j = wp.tid()
        if active[w] == 0:
            return
        value = real(0.0)
        if i < nv and j < nv:
            value = real(matrix[w, wp.min(i,j), wp.max(i,j)])
        elif i == j:
            value = real(1.0)
        work[w,i,j] = value


    @wp.kernel(enable_backward=False)
    def _rhs(nv: int, active: wp.array(dtype=wp.int32), rhs: wp.array2d(dtype=wp.float32),
             x: wp.array2d(dtype=real)):
        w,row = wp.tid()
        if active[w] != 0:
            value = real(0.0)
            if row < nv:
                value = real(rhs[w,row])
            x[w,row] = value
    return _copy_upper, _rhs


@lru_cache(None)
def _register_diagonal(precision, tile_size):
    real = wp.float64 if precision == 'float64' else wp.float32
    scalar = 'double' if precision == 'float64' else 'float'
    snippet = r'''
    #if defined(__CUDA_ARCH__)
    using Real = SCALAR;
    constexpr int T = TILE;
    const int lane = threadIdx.x;
    Real* matrix = reinterpret_cast<Real*>(
        reinterpret_cast<char*>(a.data) + world*a.strides[0]);
    const int stride = a.strides[1]/sizeof(Real);
    Real row[T];
    #pragma unroll
    for (int col = 0; col < T; ++col)
        row[col] = lane < T ? matrix[(k+lane)*stride+k+col] : Real(0);
    #pragma unroll
    for (int pivot = 0; pivot < T; ++pivot) {
        Real diag = wp::sqrt(__shfl_sync(0xffffffffu, row[pivot], pivot));
        Real value = lane >= pivot && lane < T ? row[pivot]/diag : Real(0);
        row[pivot] = value;
        #pragma unroll
        for (int col = 0; col < T; ++col) {
            if (col > pivot) {
                Real other = __shfl_sync(0xffffffffu, value, col);
                if (lane >= col && lane < T) row[col] -= value*other;
            }
        }
    }
    #pragma unroll
    for (int col = 0; col < T; ++col)
        if (lane < T) matrix[(k+lane)*stride+k+col] = col <= lane ? row[col] : Real(0);
    #endif
    '''.replace('SCALAR', scalar).replace('TILE', str(tile_size))
    @wp.func_native(snippet)
    def run(world: int, k: int, a: wp.array3d(dtype=real)):
        pass
    @wp.kernel(module='unique', enable_backward=False)
    def diagonal(k: int, active: wp.array(dtype=wp.int32), a: wp.array3d(dtype=real)):
        world, thread = wp.tid()
        if active[world] != 0:
            run(world, k, a)
    return diagonal


@lru_cache(None)
def _kernels(nv, precision, tile_size):
    real = wp.float64 if precision == 'float64' else wp.float32
    size = (nv + tile_size-1)//tile_size*tile_size
    options = {'enable_mathdx_solver': False, 'enable_mathdx_gemm': False}

    @wp.kernel(module='unique', enable_backward=False, module_options=options)
    def diagonal(k: int, active: wp.array(dtype=wp.int32), a: wp.array3d(dtype=real)):
        w = wp.tid()
        if active[w] == 0:
            return
        tile = wp.tile_load(a[w], shape=(tile_size,tile_size), offset=(k,k), storage='shared', bounds_check=False)
        wp.tile_cholesky_inplace(tile, fill_mode='lower')
        wp.tile_store(a[w], tile, offset=(k,k), bounds_check=False)

    @wp.kernel(module='unique', enable_backward=False, module_options=options)
    def panel(k: int, active: wp.array(dtype=wp.int32), a: wp.array3d(dtype=real)):
        w, block = wp.tid()
        if active[w] == 0:
            return
        i = k + tile_size*(block+1)
        l = wp.tile_load(a[w], shape=(tile_size,tile_size), offset=(k,k), storage='shared', bounds_check=False)
        aik = wp.tile_load(a[w], shape=(tile_size,tile_size), offset=(i,k), storage='shared', bounds_check=False)
        wp.tile_lower_solve_inplace(l, wp.tile_transpose(aik))
        wp.tile_store(a[w], aik, offset=(i,k), bounds_check=False)

    @wp.kernel(module='unique', enable_backward=False, module_options=options)
    def trailing(k: int, active: wp.array(dtype=wp.int32), a: wp.array3d(dtype=real)):
        w, bi, bj = wp.tid()
        if active[w] == 0 or bi < bj:
            return
        i, j = k + tile_size*(bi+1), k + tile_size*(bj+1)
        li = wp.tile_load(a[w], shape=(tile_size,tile_size), offset=(i,k), storage='shared', bounds_check=False)
        lj = wp.tile_load(a[w], shape=(tile_size,tile_size), offset=(j,k), storage='shared', bounds_check=False)
        target = wp.tile_load(a[w], shape=(tile_size,tile_size), offset=(i,j), storage='shared', bounds_check=False)
        wp.tile_matmul(li, wp.tile_transpose(lj), target, alpha=real(-1.0))
        wp.tile_store(a[w], target, offset=(i,j), bounds_check=False)

    @wp.kernel(module='unique', enable_backward=False, module_options=options)
    def substitute(active: wp.array(dtype=wp.int32), a: wp.array3d(dtype=real),
                   solution: wp.array3d(dtype=real)):
        w = wp.tid()
        if active[w] == 0:
            return
        x = wp.tile_load(solution[w],shape=(size,1),offset=(0,0),storage='shared',bounds_check=False)
        # Runtime loops prevent the compiler from expanding every tile pair.
        blocks = int(size//tile_size)
        for block in range(blocks):
            k = block*tile_size
            part = wp.tile_view(x, shape=(tile_size,1), offset=(k,0))
            for previous in range(block):
                j = previous*tile_size
                l = wp.tile_load(a[w], shape=(tile_size,tile_size), offset=(k,j), storage='shared', bounds_check=False)
                y = wp.tile_view(x, shape=(tile_size,1), offset=(j,0))
                wp.tile_matmul(l,y,part,alpha=real(-1.0))
            l = wp.tile_load(a[w], shape=(tile_size,tile_size), offset=(k,k), storage='shared', bounds_check=False)
            wp.tile_lower_solve_inplace(l,part)
        for reverse in range(blocks):
            block = blocks-1-reverse
            k = block*tile_size
            part = wp.tile_view(x, shape=(tile_size,1), offset=(k,0))
            for following in range(block+1,blocks):
                j = following*tile_size
                l = wp.tile_load(a[w], shape=(tile_size,tile_size), offset=(j,k), storage='shared', bounds_check=False)
                y = wp.tile_view(x, shape=(tile_size,1), offset=(j,0))
                wp.tile_matmul(wp.tile_transpose(l),y,part,alpha=real(-1.0))
            l = wp.tile_load(a[w], shape=(tile_size,tile_size), offset=(k,k), storage='shared', bounds_check=False)
            wp.tile_upper_solve_inplace(wp.tile_transpose(l),part)
        wp.tile_store(solution[w], x, bounds_check=False)

    @wp.kernel(module='unique', enable_backward=False)
    def finish(active: wp.array(dtype=wp.int32), h: wp.array3d(dtype=wp.float32),
               g: wp.array2d(dtype=wp.float32), x: wp.array2d(dtype=real),
               factor: wp.array3d(dtype=real), out: wp.array2d(dtype=wp.float32),
               norm: wp.array(dtype=wp.float32), decrement: wp.array(dtype=wp.float32),
               status: wp.array(dtype=wp.int32), max_residual: wp.array(dtype=wp.float32)):
        w,row = wp.tid()
        if active[w] == 0:
            return
        value, b, ax = wp.float64(0.0),wp.float64(0.0),wp.float64(0.0)
        bad = int(0)
        if row < nv:
            value = wp.float64(wp.float32(x[w,row]))
            b = wp.float64(g[w,row])
            for col in range(nv):
                ax += wp.float64(h[w,wp.min(row,col),wp.max(row,col)]) * wp.float64(wp.float32(x[w,col]))
            pivot = factor[w,row,row]
            if not wp.isfinite(value) or not wp.isfinite(pivot) or pivot <= 0.0:
                bad = 1
            out[w,row] = -wp.float32(value)
        error = b-ax
        rr = wp.tile_reduce(wp.add,wp.tile(error*error))[0]
        bb = wp.tile_reduce(wp.add,wp.tile(b*b))[0]
        xx = wp.tile_reduce(wp.add,wp.tile(value*value))[0]
        bx = wp.tile_reduce(wp.add,wp.tile(b*value))[0]
        invalid = wp.tile_reduce(wp.add,wp.tile(bad))[0]
        relative = wp.float64(0.0)
        if bb > 0.0:
            relative = wp.sqrt(rr/bb)
        elif rr > 0.0:
            relative = wp.float64(wp.inf)
        if row == 0:
            code = int(0)
            if invalid > 0 or not wp.isfinite(relative) or not wp.isfinite(wp.float32(xx)) or not wp.isfinite(wp.float32(bx)):
                code = 1
                relative = wp.float64(wp.inf)
            elif relative > wp.float64(1.0e-4):
                code = 2
            norm[w] = wp.float32(xx)
            decrement[w] = wp.float32(bx)
            status[w] = status[w] | code
            max_residual[w] = wp.max(max_residual[w],wp.float32(relative))
    return diagonal,panel,trailing,substitute,finish


class _Workspace:
    def __init__(self,nv,nworld,device,precision='float64',tile_size=16,diagonal='tiled',substitution='tiled',
                 panel='tiled',update_threads=64,conditional_factor=False,trailing='tiled'):
        if precision not in ('float32','float64'):
            raise ValueError('precision must be float32 or float64')
        if tile_size not in (16,32):
            raise ValueError('tile_size must be 16 or 32')
        if diagonal not in ('tiled','register'):
            raise ValueError('diagonal must be tiled or register')
        if substitution not in ('tiled','ordered_warp'):
            raise ValueError('substitution must be tiled or ordered_warp')
        if panel not in ('tiled','register'):
            raise ValueError('panel must be tiled or register')
        if trailing not in ('tiled','register'):
            raise ValueError('trailing must be tiled or register')
        if isinstance(update_threads,bool) or not isinstance(update_threads,numbers.Integral) or update_threads not in (64,128):
            raise ValueError('update_threads must be 64 or 128')
        if not isinstance(conditional_factor,bool):
            raise ValueError('conditional_factor must be bool')
        if conditional_factor and not wp.get_device(device).is_cuda:
            raise ValueError('conditional factorization requires CUDA')
        if substitution == 'ordered_warp' or panel == 'register' or trailing == 'register':
            if not wp.get_device(device).is_cuda:
                raise ValueError('ordered Newton kernels require CUDA')
            from ._newton_substitution import _check_tile_sources
            _check_tile_sources()
        self.conditional_factor = conditional_factor
        self.trailing = trailing
        self.panel = panel
        self.update_threads = int(update_threads)
        self.substitution = substitution
        self.diagonal = diagonal
        self.tile_size = tile_size
        self.precision = precision
        real = wp.float64 if precision == 'float64' else wp.float32
        self.nv,self.nworld,self.device = nv,nworld,wp.get_device(device)
        self.size = (nv+tile_size-1)//tile_size*tile_size
        self.work = wp.empty((nworld,self.size,self.size),dtype=real,device=device)
        self.x = wp.empty((nworld,self.size),dtype=real,device=device)
        self.x_tiles = self.x.reshape((nworld,self.size,1))
        self.active = wp.empty(nworld,dtype=wp.int32,device=device)
        self.refactor = wp.empty(nworld,dtype=wp.int32,device=device)
        self.initialized = wp.zeros(nworld,dtype=wp.int32,device=device)
        self.factor_condition = wp.zeros(1,dtype=wp.int32,device=device) if conditional_factor else None
        self.status = wp.zeros(nworld,dtype=wp.int32,device=device)
        self.max_relative_residual = wp.zeros(nworld,dtype=wp.float32,device=device)

    def _validate_context(self, ctx, quad, state):
        """Reject malformed or overlapping views before any device writes."""
        arrays = ((ctx.h, 'h', wp.float32, 3), (ctx.grad, 'grad', wp.float32, 2),
                  (ctx.done, 'done', wp.bool, 1), (quad, 'quad', wp.int32, 1),
                  (state, 'state', wp.int32, 1), (ctx.search, 'search', wp.float32, 2),
                  (ctx.search_dot, 'search_dot', wp.float32, 1),
                  (ctx.newton_decrement, 'newton_decrement', wp.float32, 1))
        for array, name, dtype, ndim in arrays:
            if (not isinstance(array, wp.array) or array.dtype != dtype or array.ndim != ndim
                or array.device != self.device or not array.is_contiguous
                or array.shape[0] != self.nworld):
                raise ValueError(f'{name} must have the expected dtype, rank, worlds and device, with contiguous storage')
            alignment = wp.types.type_size_in_bytes(dtype)
            if array.ptr % alignment:
                raise ValueError(f'{name} must be naturally aligned')
        if (min(ctx.h.shape[1:]) < self.nv or ctx.grad.shape[1] < self.nv
            or ctx.search.shape[1] != self.nv):
            raise ValueError('Newton matrix/vector dimensions do not match the model')
        # All writes are disjoint from each other and from every input. Check
        # owned workspace too, since Python callers can construct aliasing views.
        sources = [entry[0] for entry in arrays[:5]]
        writes = [entry[0] for entry in arrays[5:]] + [
            self.work, self.x, self.active, self.refactor, self.initialized,
            self.status, self.max_relative_residual]
        if self.factor_condition is not None:
            writes.append(self.factor_condition)
        def span(array):
            return array.ptr, array.ptr + array.size * wp.types.type_size_in_bytes(array.dtype)
        for index, output in enumerate(writes):
            lo, hi = span(output)
            for other in sources + writes[:index]:
                start, end = span(other)
                if lo < end and start < hi:
                    raise ValueError('Newton outputs/workspace must not overlap each other or inputs')

    def _enqueue_factor(self,ctx):
        _copy_upper, _ = _transport_kernels(self.precision)
        wp.launch(_copy_upper,dim=(self.nworld,self.size,self.size),inputs=[self.nv,ctx.h,self.refactor],
                  outputs=[self.work],device=self.device)
        diagonal,panel,trailing,_,_ = _kernels(self.nv,self.precision,self.tile_size)
        if self.diagonal == 'register':
            diagonal = _register_diagonal(self.precision,self.tile_size)
        if self.panel == 'register':
            from ._newton_panel import ordered_panel
            panel = ordered_panel(self.precision,self.tile_size)
        if self.trailing == 'register':
            from ._newton_trailing import ordered_trailing
            trailing = ordered_trailing(self.precision,self.tile_size,self.update_threads)
        for k in range(0,self.size,self.tile_size):
            wp.launch_tiled(diagonal,dim=self.nworld,inputs=[k,self.refactor,self.work],
                            block_dim=32 if self.diagonal == 'register' else 64,device=self.device)
            remaining = (self.size-k)//self.tile_size-1
            if remaining:
                wp.launch_tiled(panel,dim=(self.nworld,remaining),inputs=[k,self.refactor,self.work],block_dim=32 if self.panel == 'register' else 64,device=self.device)
                wp.launch_tiled(trailing,dim=(self.nworld,remaining,remaining),inputs=[k,self.refactor,self.work],block_dim=self.update_threads,device=self.device)

    def enqueue(self,ctx,quad,state,skip_unchanged=False,skip_noflip=False):
        self._validate_context(ctx,quad,state)
        wp.launch(_masks,dim=self.nworld,inputs=[ctx.done,quad,state,skip_unchanged,skip_noflip],
                  outputs=[self.active,self.refactor,self.initialized],device=self.device)
        if self.conditional_factor:
            wp.launch(_any_refactor,dim=1,inputs=[self.refactor,self.factor_condition],device=self.device)
        if self.conditional_factor and wp.get_stream(self.device).is_capturing:
            wp.capture_if(self.factor_condition,on_true=lambda:self._enqueue_factor(ctx))
        else:
            # Uncaptured enqueue keeps device masks and never reads a condition
            # back to the host. This also warms every branch before graph capture.
            self._enqueue_factor(ctx)
        _,_rhs = _transport_kernels(self.precision)
        _,_,_,substitute,finish = _kernels(self.nv,self.precision,self.tile_size)
        wp.launch(_rhs,dim=(self.nworld,self.size),inputs=[self.nv,self.active,ctx.grad],outputs=[self.x],device=self.device)
        if self.substitution == 'ordered_warp':
            from ._newton_substitution import ordered_substitution
            wp.launch_tiled(ordered_substitution(self.nv,self.precision,self.tile_size),dim=self.nworld,
                            inputs=[self.active,self.work,self.x],block_dim=32,device=self.device)
        else:
            wp.launch_tiled(substitute,dim=self.nworld,inputs=[self.active,self.work,self.x_tiles],block_dim=64,device=self.device)
        wp.launch_tiled(finish,dim=self.nworld,
            inputs=[self.active,ctx.h,ctx.grad,self.x,self.work,ctx.search,ctx.search_dot,ctx.newton_decrement,
                    self.status,self.max_relative_residual],block_dim=max(32,1<<(self.nv-1).bit_length()),device=self.device)


class ParallelNewtonSolver(_Workspace):
    """Opt-in 1..512 DOF CUDA Newton factorization. Keep alive during graph use.

    Status is sticky until explicitly cleared: bit 1 invalid factor/output, bit
    2 float32 output relative residual > 1e-4. A nonzero status invalidates the
    trajectory; there is no state rollback. Buffers are owned by one stream.
    """
    def __init__(self,model,nworld=1,*,precision='float64',tile_size=16,diagonal='tiled',substitution='tiled',
                 panel='tiled',update_threads=64,conditional_factor=False,trailing='tiled'):
        from mujoco_warp._src import types
        if version('mujoco-warp') != '4.0.1':
            raise RuntimeError('reviewed only for mujoco-warp 4.0.1')
        device = model.body_mass.device
        if (not device.is_cuda or not 1<=model.nv<=512 or model.opt.solver != types.SolverType.NEWTON
            or model.opt.enableflags & types.EnableBit.SLEEP or isinstance(nworld,bool)
            or not isinstance(nworld,numbers.Integral) or not 1<=nworld<=2147483647):
            raise ValueError('requires CUDA Newton, 1..512 DOFs, positive world count, and no sleeping')
        super().__init__(model.nv,nworld,device,precision,tile_size,diagonal,substitution,panel,update_threads,conditional_factor,trailing)
        self.model = model

    @contextmanager
    def capture_policy(self):
        from mujoco_warp._src import solver
        from mujoco_warp._src import types
        if (version('mujoco-warp')!='4.0.1' or self.model.nv!=self.nv
            or self.model.opt.solver!=types.SolverType.NEWTON
            or self.model.opt.enableflags & types.EnableBit.SLEEP):
            raise RuntimeError('solver/model configuration changed; rebuild the policy')
        with _LOCK:
            original = solver._cholesky_factorize_solve
            if hashlib.sha256(inspect.getsource(original).encode()).hexdigest()!=_SOURCE_SHA:
                raise RuntimeError('unreviewed Newton implementation or nested policy')
            def dispatch(m,d,ctx,skip_unchanged=False,skip_noflip=False):
                if m is not self.model:
                    return original(m,d,ctx,skip_unchanged,skip_noflip)
                if d.nworld != self.nworld or d.qpos.device != self.device:
                    raise ValueError('data world count or device mismatch')
                self.enqueue(ctx,ctx.quad_changed_count if skip_unchanged else d.nefc,
                             ctx.state_changed_count if skip_noflip else d.nefc,skip_unchanged,skip_noflip)
            solver._cholesky_factorize_solve = dispatch
            try:
                yield self
            finally:
                solver._cholesky_factorize_solve = original
