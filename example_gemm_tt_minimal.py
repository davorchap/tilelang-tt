#!/usr/bin/env python3
"""
Minimal changes to run examples/gemm/example_gemm.py with TT backend.

This is the EXACT same code as the original, with only TWO changes:
1. Import TENSTORRENT_TARGET
2. Add target parameter to @tilelang.jit
"""

import tilelang
import tilelang.language as T
from tilelang.utils.target import TENSTORRENT_TARGET  # CHANGE 1: Import TT target


@tilelang.jit(target=TENSTORRENT_TARGET, out_idx=[-1])  # CHANGE 2: Add target=TENSTORRENT_TARGET
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

    @T.prim_func
    def gemm(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def main():
    # Create kernel - using TT-compatible tile sizes
    # Original used 1024x1024x1024 with 128x128x32 blocks
    # For TT, we use sizes that work well with 32x32 tiles
    kernel = matmul(256, 256, 256, 32, 32, 32)  # TT works best with 32x32 tiles

    print("✓ Kernel created with TT backend")

    # Since we're in simulation mode, we can't actually execute
    # but we can inspect the generated artifacts
    import json
    source = kernel.get_kernel_source()
    artifacts = json.loads(source)

    print(f"\n✓ Generated TT artifacts:")
    for artifact_name in artifacts:
        print(f"  - {artifact_name}")

    # Show the runtime plan
    plan = json.loads(artifacts["tt.plan.json"])
    print(f"\n✓ Execution plan:")
    print(f"  - Core grid: {plan.get('core_grid', 'N/A')}")
    print(f"  - Work partitions: {len(plan.get('work_partition', {}))} cores")

    print("\n✓ Artifacts ready for TT-Metalium SDK compilation!")

    # Note: Actual execution would look like this (requires TT hardware):
    # import torch
    # a = torch.randn(256, 256).half()  # No .cuda() for TT
    # b = torch.randn(256, 256).half()
    # c = kernel(a, b)  # Would execute on TT hardware


if __name__ == "__main__":
    main()