"""One-warp triangular substitution with the reviewed tile arithmetic order.

For each off-diagonal block, accumulate products in ascending column order
with explicit muladd, then subtract the completed sum from the RHS. Inside a
diagonal block, forward substitution subtracts individual products in ascending
pivot order. Back substitution also subtracts in ascending column order, even
though rows are solved in descending order. Reversing that inner sum changes
rounding and can change a long contact trajectory.

This implementation changes ownership and transport: coalesced tile loads,
padded shared storage, one row per lane, and warp broadcasts of solved values.
The factor, padded identity, output precision and residual check are unchanged.
It is intentionally independent of MathDx. One CUDA block is exactly one warp.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import warp as wp

_TILE_SOURCES = {
    'tile_solve.h': '7802c93f757081ba25bf1d5ea82611c3ed58d0ae2effa292885604a5b6334fd7',
    'tile_matmul.h': 'eb47fff57a0ed694dfbd4c02e2763f1416e9bd910994c124397a2c15f9fafed2',
}


def _check_tile_sources():
    # The equality claim is against this reviewed scalar tile implementation.
    for name, digest in _TILE_SOURCES.items():
        path = Path(wp.__file__).resolve().parent / 'native' / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('ordered Newton kernels require reviewed Warp tile sources: ' + name)


@lru_cache(None)
def ordered_substitution(nv, precision, tile):
    real = wp.float64 if precision == 'float64' else wp.float32
    size = (nv + tile - 1) // tile * tile
    code = r'''
    #if defined(__CUDA_ARCH__)
    using Real = SCALAR;
    constexpr int N = SIZE, T = TILE;
    const int lane = threadIdx.x;
    const Real* matrix = reinterpret_cast<const Real*>(
        reinterpret_cast<const char*>(a.data) + world*a.strides[0]);
    Real* output = reinterpret_cast<Real*>(
        reinterpret_cast<char*>(solution.data) + world*solution.strides[0]);
    const int stride = a.strides[1]/sizeof(Real);
    __shared__ Real x[N], block[T*(T+1)];
    for (int i=lane; i<N; i+=32) x[i] = output[i];
    __syncwarp();
    for (int k=0; k<N; k+=T) {
        Real value = lane<T ? x[k+lane] : Real(0);
        for (int j=0; j<k; j+=T) {
            #pragma unroll
            for (int r=0; r<T; ++r)
                if (lane<T) block[r*(T+1)+lane] = matrix[(k+r)*stride+j+lane];
            __syncwarp();
            if (lane<T) {
                Real sum = Real(0);
                #pragma unroll
                for (int c=0; c<T; ++c)
                    sum = wp::muladd(block[lane*(T+1)+c], x[j+c], sum);
                value = -sum + value;
            }
            __syncwarp();
        }
        #pragma unroll
        for (int r=0; r<T; ++r)
            if (lane<T) block[r*(T+1)+lane] = matrix[(k+r)*stride+k+lane];
        __syncwarp();
        // Distributing rows keeps each row's ascending product-subtraction
        // sequence intact, while eliminating the serial shared-memory chain.
        Real row[T];
        #pragma unroll
        for (int c=0; c<T; ++c) row[c] = lane<T ? block[lane*(T+1)+c] : Real(0);
        #pragma unroll
        for (int pivot=0; pivot<T; ++pivot) {
            Real d = row[pivot];
            Real solved = d != Real(0) ? value/d : value;
            solved = __shfl_sync(0xffffffffu,solved,pivot);
            if (lane>pivot && lane<T) value -= row[pivot]*solved;
            if (lane==pivot) value = solved;
        }
        if (lane<T) x[k+lane] = value;
        __syncwarp();
    }
    for (int k=N-T; k>=0; k-=T) {
        Real value = lane<T ? x[k+lane] : Real(0);
        for (int j=k+T; j<N; j+=T) {
            #pragma unroll
            for (int r=0; r<T; ++r)
                if (lane<T) block[r*(T+1)+lane] = matrix[(j+r)*stride+k+lane];
            __syncwarp();
            if (lane<T) {
                Real sum = Real(0);
                #pragma unroll
                for (int c=0; c<T; ++c)
                    sum = wp::muladd(block[c*(T+1)+lane], x[j+c], sum);
                value = -sum + value;
            }
            __syncwarp();
        }
        #pragma unroll
        for (int r=0; r<T; ++r)
            if (lane<T) block[r*(T+1)+lane] = matrix[(k+r)*stride+k+lane];
        __syncwarp();
        // Do not use descending rank-one updates here: the reference upper
        // solve subtracts higher columns in ASCENDING order for each row.
        Real row[T];
        #pragma unroll
        for (int c=0; c<T; ++c) row[c] = lane<T ? block[c*(T+1)+lane] : Real(0);
        #pragma unroll
        for (int pivot=T-1; pivot>=0; --pivot) {
            Real s = value;
            #pragma unroll
            for (int c=pivot+1; c<T; ++c) {
                Real solved = __shfl_sync(0xffffffffu,value,c);
                s -= row[c]*solved;
            }
            if (lane==pivot) value = row[pivot] != Real(0) ? s/row[pivot] : s;
        }
        if (lane<T) x[k+lane] = value;
        __syncwarp();
    }
    for (int i=lane; i<N; i+=32) output[i] = x[i];
    #endif
    '''.replace('SCALAR', 'double' if precision=='float64' else 'float').replace('SIZE',str(size)).replace('TILE',str(tile))
    @wp.func_native(code)
    def run(world:int, a:wp.array3d(dtype=real), solution:wp.array2d(dtype=real)):
        pass
    @wp.kernel(module='unique', enable_backward=False)
    def substitute(active:wp.array(dtype=wp.int32), a:wp.array3d(dtype=real), solution:wp.array2d(dtype=real)):
        world, lane = wp.tid()
        if active[world] != 0:
            run(world, a, solution)
    return substitute
