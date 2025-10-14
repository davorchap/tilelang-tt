"""
Example GEMM using TileLang DSL with Tenstorrent backend.

This demonstrates how to use the @tilelang.jit decorator with the TT backend,
showing that all TileLang DSL features work with Tenstorrent hardware.
"""

import tilelang
import tilelang.language as T
from tilelang.utils.target import TENSTORRENT_TARGET


@tilelang.jit(target=TENSTORRENT_TARGET, out_idx=[-1])
def matmul_tt(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    """
    TileLang GEMM for Tenstorrent backend.

    All DSL features map to TT concepts:
    - T.alloc_shared → L1 circular buffers
    - T.alloc_fragment → tile registers
    - T.Pipelined → persistent loop with double buffering
    - T.gemm → TT matmul_tiles intrinsic

    Note: threads parameter is ignored (TT uses tile-level parallelism)
    """

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Grid-level parallelism (threads param ignored by TT)
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # Shared memory → L1 circular buffers
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)

            # Fragment memory → tile registers
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Initialize accumulator
            T.clear(C_local)

            # Pipelined loop → persistent kernel with double buffering
            # Note: TT supports max 2 stages (num_stages > 2 ignored)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                # Copy tiles to L1 buffers (NOC DMA operations)
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)

                # Tile-level GEMM → TT matmul intrinsics
                T.gemm(A_shared, B_shared, C_local)

            # Write result back
            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def main():
    # Create kernel with TT-friendly tile sizes (32x32)
    kernel = matmul_tt(128, 128, 128, 32, 32, 32)

    import torch
    import json

    # TT backend uses CPU tensors for host-side management
    a = torch.randn(128, 128).half()
    b = torch.randn(128, 128).half()

    # Note: Actual execution requires TT hardware or simulator
    # This runs in simulation mode
    c = kernel(a, b)

    # Display generated TT artifacts
    print("TT Artifacts generated:")
    source = kernel.get_kernel_source()
    artifacts = json.loads(source)
    for name in sorted(artifacts.keys()):
        print(f"  - {name}")

    print("\nFirst 500 chars of compute.cpp:")
    print(artifacts.get("compute.cpp", "")[:500])

    print("\nExecution completed (simulation mode).")
    print("To run on hardware, compile artifacts with TT-Metalium SDK.")


if __name__ == "__main__":
    main()