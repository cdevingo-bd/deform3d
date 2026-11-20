"""Register-owned triangular panel solves for blocked Newton Cholesky.

One warp owns a complete off-diagonal tile. Coalesced transfers use padded
shared memory; each lane then owns one independent RHS in registers. Updating
later columns after each pivot preserves the scalar tile solve's ascending
product-subtraction order for every output, including its zero-pivot behavior.

The caller must launch exactly 32 threads per block, with complete padded tiles.
This is not a different factorization or a changed convergence criterion.
"""
from functools import lru_cache
import warp as wp

@lru_cache(None)
def ordered_panel(precision,tile):
    real=wp.float64 if precision=='float64' else wp.float32
    code=r'''
    #if defined(__CUDA_ARCH__)
    using Real=SCALAR;
    constexpr int T=TILE;
    const int lane=threadIdx.x,i=k+(block+1)*T;
    Real* matrix=reinterpret_cast<Real*>(reinterpret_cast<char*>(a.data)+world*a.strides[0]);
    const int stride=a.strides[1]/sizeof(Real);
    // Padding avoids bank conflicts when coalesced loads become register rows.
    __shared__ Real diagonal[T*(T+1)], target[T*(T+1)];
    #pragma unroll
    for(int r=0;r<T;++r) if(lane<T) {
        diagonal[r*(T+1)+lane]=matrix[(k+r)*stride+k+lane];
        target[r*(T+1)+lane]=matrix[(i+r)*stride+k+lane];
    }
    __syncwarp();
    Real values[T];
    #pragma unroll
    for(int c=0;c<T;++c) values[c]=lane<T ? target[lane*(T+1)+c] : Real(0);
    // Each output subtracts pivot contributions in the original order.
    #pragma unroll
    for(int pivot=0;pivot<T;++pivot) {
        Real d=diagonal[pivot*(T+1)+pivot];
        Real solved=d!=Real(0) ? values[pivot]/d : values[pivot];
        values[pivot]=solved;
        #pragma unroll
        for(int c=0;c<T;++c) if(c>pivot) values[c]-=diagonal[c*(T+1)+pivot]*solved;
    }
    #pragma unroll
    for(int c=0;c<T;++c) if(lane<T) target[lane*(T+1)+c]=values[c];
    __syncwarp();
    #pragma unroll
    for(int r=0;r<T;++r) if(lane<T) matrix[(i+r)*stride+k+lane]=target[r*(T+1)+lane];
    #endif
    '''.replace('SCALAR','double' if precision=='float64' else 'float').replace('TILE',str(tile))
    @wp.func_native(code)
    def run(world:int,block:int,k:int,a:wp.array3d(dtype=real)):
        pass
    @wp.kernel(module='unique',enable_backward=False)
    def panel(k:int,active:wp.array(dtype=wp.int32),a:wp.array3d(dtype=real)):
        world,block,lane=wp.tid()
        if active[world]!=0:run(world,block,k,a)
    return panel
