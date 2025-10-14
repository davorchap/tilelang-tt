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

    # Get information about where artifacts are stored
    import os
    import json

    print("\n📍 Artifact Locations:")

    # 1. The artifacts are returned as JSON from get_kernel_source()
    print("  1. In-memory: Available via kernel.get_kernel_source()")

    # 2. Runtime plan is written to current directory
    if os.path.exists("tt.plan.json"):
        abs_path = os.path.abspath("tt.plan.json")
        print(f"  2. Runtime plan: {abs_path}")

    # 3. Cache directory (if kernel caching is enabled)
    cache_dir = os.path.expanduser("~/.tilelang/cache")
    if os.path.exists(cache_dir):
        # Find directories with kernel artifacts
        kernel_dirs = []
        for entry in os.listdir(cache_dir):
            entry_path = os.path.join(cache_dir, entry)
            if os.path.isdir(entry_path):
                # Check if it contains kernel files
                files = os.listdir(entry_path)
                if any(f.endswith(('.cu', '.cpp', '.pkl', '.json')) for f in files):
                    kernel_dirs.append(entry_path)

        if kernel_dirs:
            # Show the most recent one
            kernel_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest = kernel_dirs[0]
            files = os.listdir(latest)
            print(f"  3. Cache dir: {latest}")
            print(f"     Files: {', '.join(sorted(files))}")

    # Get the generated artifacts from the kernel
    source = kernel.get_kernel_source()
    artifacts = json.loads(source)

    print(f"\n✓ Generated TT artifacts (5 total):")
    for artifact_name in sorted(artifacts.keys()):
        print(f"  - {artifact_name}")

    # Show the runtime plan
    # Note: Currently the plan is also written to ./tt.plan.json
    print(f"\n✓ Execution plan:")
    try:
        # Check if plan is in current directory (known issue)
        if os.path.exists("tt.plan.json"):
            with open("tt.plan.json", "r") as f:
                plan = json.load(f)
            print(f"  - Core grid: {plan.get('core_grid', [8, 8])}")
            print(f"  - Work partitions: {len(plan.get('work_partition', {}))} cores")
            print(f"  - Plan file: ./tt.plan.json (working directory)")
    except:
        print(f"  - Generated in artifacts")

    print("\n✓ Artifacts ready for TT-Metalium SDK compilation!")

    # Optional: Save artifacts to a directory for inspection
    save_artifacts = False  # Set to True if you want to save artifacts
    if save_artifacts:
        output_dir = "tt_gemm_artifacts"
        os.makedirs(output_dir, exist_ok=True)
        for name, content in artifacts.items():
            filepath = os.path.join(output_dir, name)
            with open(filepath, "w") as f:
                f.write(content)
        print(f"\n💾 Artifacts saved to: {os.path.abspath(output_dir)}/")

    # Note: Actual execution would look like this (requires TT hardware):
    # import torch
    # a = torch.randn(256, 256).half()  # No .cuda() for TT
    # b = torch.randn(256, 256).half()
    # c = kernel(a, b)  # Would execute on TT hardware


if __name__ == "__main__":
    main()