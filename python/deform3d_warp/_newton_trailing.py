"""Register-owned Schur updates with the reviewed scalar GEMM arithmetic.

Each warp owns complete output rows. Padded shared operands avoid bank
conflicts when reading transposed rows, and each lane accumulates its output
columns in registers. Products use ascending-pivot explicit muladd, followed
by subtraction of the completed sum, exactly as the scalar tile GEMM does.

The caller supplies complete padded tiles and exactly `threads` block threads.
There is no persistent scratch or communication between blocks in this kernel.
"""
from functools import lru_cache
import warp as wp


@lru_cache(None)
def ordered_trailing(precision, tile, threads):
    real = wp.float64 if precision == 'float64' else wp.float32
    code = r'''
    #if defined(__CUDA_ARCH__)
    using Real = SCALAR;
    constexpr int T = TILE, W = THREADS/32, R = T/W;
    const int tid = threadIdx.x, lane = tid%32, warp = tid/32;
    const int i = k+(bi+1)*T, j = k+(bj+1)*T;
    Real* matrix = reinterpret_cast<Real*>(
        reinterpret_cast<char*>(a.data)+world*a.strides[0]);
    const int stride = a.strides[1]/sizeof(Real);
    __shared__ Real left[T*(T+1)], right[T*(T+1)];
    // Coalesced loads, padded rows for the transposed operand reads below.
    for (int index=tid; index<T*T; index+=THREADS) {
        const int r=index/T, c=index%T;
        left[r*(T+1)+c] = matrix[(i+r)*stride+k+c];
        right[r*(T+1)+c] = matrix[(j+r)*stride+k+c];
    }
    __syncthreads();
    if (lane<T) {
        Real sum[R];
        #pragma unroll
        for (int r=0; r<R; ++r) sum[r] = Real(0);
        #pragma unroll
        for (int p=0; p<T; ++p) {
            const Real b = right[lane*(T+1)+p];
            #pragma unroll
            for (int r=0; r<R; ++r)
                sum[r] = wp::muladd(left[(warp+r*W)*(T+1)+p], b, sum[r]);
        }
        #pragma unroll
        for (int r=0; r<R; ++r) {
            const int index = (i+warp+r*W)*stride+j+lane;
            matrix[index] = -sum[r]+matrix[index];
        }
    }
    #endif
    '''.replace('SCALAR', 'double' if precision == 'float64' else 'float').replace(
        'TILE', str(tile)).replace('THREADS', str(threads))

    @wp.func_native(code)
    def run(world: int, bi: int, bj: int, k: int, a: wp.array3d(dtype=real)):
        pass

    @wp.kernel(module='unique', enable_backward=False)
    def trailing(k: int, active: wp.array(dtype=wp.int32), a: wp.array3d(dtype=real)):
        world, bi, bj, lane = wp.tid()
        # Uniform for every thread in this block, before its shared barrier.
        if active[world] != 0 and bi >= bj:
            run(world, bi, bj, k, a)

    return trailing
