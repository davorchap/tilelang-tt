#!/usr/bin/env python3
"""
Demonstrates how to run the original example_gemm.py with the Tenstorrent backend.

This shows two approaches:
1. Minimal changes to the original (just add target parameter)
2. Full TT-optimized version with proper tile sizes
"""

import json
import tilelang
import tilelang.language as T
from tilelang.utils.target import TENSTORRENT_TARGET


def run_original_with_tt_backend():
    """Run the original example_gemm with minimal changes for TT backend."""

    print("=" * 60)
    print("Running Original GEMM Example with TT Backend")
    print("=" * 60)

    # Original example with just target parameter added
    @tilelang.jit(target=TENSTORRENT_TARGET, out_idx=[-1])
    def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

        @T.prim_func
        def gemm(
                A: T.Tensor((M, K), dtype),
                B: T.Tensor((K, N), dtype),
                C: T.Tensor((M, N), dtype),
        ):
            # Note: threads=128 is ignored by TT backend (no intra-tile parallelism)
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                T.clear(C_local)
                # Note: num_stages=3 becomes double buffering in TT
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

                T.copy(C_local, C[by * block_M, bx * block_N])

        return gemm

    # Create kernel - using TT-friendly tile sizes (32x32)
    # Original used 128x128x32, we'll use 128x128x32 but TT will internally tile to 32x32
    kernel = matmul(256, 256, 256, 32, 32, 32)  # TT works best with 32x32 tiles

    print("\n✓ Kernel created successfully with TT backend")

    # Get the generated TT artifacts
    source = kernel.get_kernel_source()
    artifacts = json.loads(source)

    print(f"\n✓ Generated {len(artifacts)} TT artifacts:")
    for name in artifacts.keys():
        print(f"  - {name}")

    # Show the runtime plan
    plan = json.loads(artifacts["tt.plan.json"])
    print("\n✓ Runtime Plan:")
    print(f"  - Core grid: {plan['core_grid'][0]}x{plan['core_grid'][1]}")
    print(f"  - Core ranges: {plan['core_ranges']}")
    print(f"  - Work partitions: {len(plan['work_partition'])} cores assigned")

    # Show a snippet of the compute kernel
    print("\n✓ Compute Kernel (snippet):")
    compute_lines = artifacts["compute.cpp"].split('\n')
    for line in compute_lines[30:45]:  # Show lines 30-45
        print(f"  {line}")

    return kernel, artifacts


def run_tt_optimized_version():
    """Run an optimized version specifically for TT architecture."""

    print("\n" + "=" * 60)
    print("TT-Optimized GEMM (Tile-Aware)")
    print("=" * 60)

    @tilelang.jit(target=TENSTORRENT_TARGET, out_idx=[-1])
    def tt_optimized_matmul(M, N, K, dtype="float16"):
        """TT-optimized matmul using fixed 32x32 tiles."""

        # TT always uses 32x32 tiles
        TILE_SIZE = 32

        @T.prim_func
        def gemm(
                A: T.Tensor((M, K), dtype),
                B: T.Tensor((K, N), dtype),
                C: T.Tensor((M, N), dtype),
        ):
            # Grid dimensions based on 32x32 tiles
            with T.Kernel(T.ceildiv(N, TILE_SIZE), T.ceildiv(M, TILE_SIZE)) as (bx, by):
                # L1 buffers for tiles
                A_l1 = T.alloc_shared((TILE_SIZE, TILE_SIZE), dtype)
                B_l1 = T.alloc_shared((TILE_SIZE, TILE_SIZE), dtype)
                # Accumulator in tile registers
                C_reg = T.alloc_fragment((TILE_SIZE, TILE_SIZE), "float")

                T.clear(C_reg)

                # Process K dimension tile by tile
                for kt in T.Pipelined(T.ceildiv(K, TILE_SIZE), num_stages=2):
                    # NOC DMA to load tiles
                    T.copy(A[by * TILE_SIZE, kt * TILE_SIZE], A_l1)
                    T.copy(B[kt * TILE_SIZE, bx * TILE_SIZE], B_l1)
                    # Tile-level matmul
                    T.gemm(A_l1, B_l1, C_reg)

                # Write result tile back
                T.copy(C_reg, C[by * TILE_SIZE, bx * TILE_SIZE])

        return gemm

    # Create kernels for different sizes (all multiples of 32)
    sizes = [
        (64, 64, 64),  # 2x2x2 tiles
        (128, 128, 128),  # 4x4x4 tiles
        (256, 256, 256),  # 8x8x8 tiles
    ]

    for M, N, K in sizes:
        kernel = tt_optimized_matmul(M, N, K)
        source = kernel.get_kernel_source()
        artifacts = json.loads(source)
        plan = json.loads(artifacts["tt.plan.json"])

        print(f"\n✓ {M}x{N}x{K} kernel:")
        print(f"  - Grid: {plan['core_grid'][0]}x{plan['core_grid'][1]} cores")
        print(f"  - Total tiles: {(M//32) * (N//32)} output tiles")
        print(f"  - K tiles: {K//32} per output")


def show_artifact_details():
    """Show detailed information about TT artifacts."""

    print("\n" + "=" * 60)
    print("TT Artifact Details")
    print("=" * 60)

    @tilelang.jit(target=TENSTORRENT_TARGET, out_idx=[-1])
    def simple_gemm(M, N, K):

        @T.prim_func
        def gemm(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"), C: T.Tensor(
            (M, N), "float16")):
            with T.Kernel(T.ceildiv(N, 32), T.ceildiv(M, 32)) as (bx, by):
                A_shared = T.alloc_shared((32, 32), "float16")
                B_shared = T.alloc_shared((32, 32), "float16")
                C_local = T.alloc_fragment((32, 32), "float")

                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, 32), num_stages=2):
                    T.copy(A[by * 32, k * 32], A_shared)
                    T.copy(B[k * 32, bx * 32], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C[by * 32, bx * 32])

        return gemm

    kernel = simple_gemm(128, 128, 128)
    source = kernel.get_kernel_source()
    artifacts = json.loads(source)

    print("\n1. Reader Kernel (loads tiles from DRAM to L1):")
    print("-" * 40)
    reader_lines = artifacts["reader.cpp"].split('\n')
    for line in reader_lines[25:40]:  # Show key parts
        if line.strip():
            print(f"  {line}")

    print("\n2. Compute Kernel (performs matmul):")
    print("-" * 40)
    compute_lines = artifacts["compute.cpp"].split('\n')
    for line in compute_lines[35:50]:  # Show key parts
        if line.strip():
            print(f"  {line}")

    print("\n3. Writer Kernel (stores results to DRAM):")
    print("-" * 40)
    writer_lines = artifacts["writer.cpp"].split('\n')
    for line in writer_lines[20:35]:  # Show key parts
        if line.strip():
            print(f"  {line}")

    print("\n4. Host Program (main.cpp):")
    print("-" * 40)
    main_lines = artifacts["main.cpp"].split('\n')
    for line in main_lines[55:70]:  # Show metadata section
        if line.strip():
            print(f"  {line}")


def main():
    """Main demonstration."""

    print("\n" + "=" * 60)
    print("HOW TO RUN ORIGINAL example_gemm.py WITH TT BACKEND")
    print("=" * 60)

    print("""
The original example_gemm.py needs just ONE change to run with TT backend:

    @tilelang.jit(target=TENSTORRENT_TARGET, out_idx=[-1])
    #             ^^^^^^^^^^^^^^^^^^^^^^^^
    #             Add this parameter!

Key differences when running with TT:
1. No CUDA execution - generates TT artifacts for hardware compilation
2. threads=128 parameter is ignored (TT uses tile-level parallelism)
3. Best performance with 32x32 tile sizes (TT's native tile size)
4. num_stages in Pipelined becomes double-buffering in TT

Let's see it in action...
""")

    # Run the demonstrations
    kernel, artifacts = run_original_with_tt_backend()
    run_tt_optimized_version()
    show_artifact_details()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
To run ANY TileLang kernel with TT backend:

1. Import the target:
   from tilelang.utils.target import TENSTORRENT_TARGET

2. Add target parameter to @tilelang.jit:
   @tilelang.jit(target=TENSTORRENT_TARGET, out_idx=[-1])

3. Use TT-friendly tile sizes (32x32 for best performance)

4. The kernel generates 5 artifacts:
   - reader.cpp   : Loads tiles from DRAM
   - compute.cpp  : Performs computation
   - writer.cpp   : Stores results
   - main.cpp     : Host coordination
   - tt.plan.json : Runtime execution plan

5. These artifacts are ready for TT-Metalium SDK compilation

That's it! Your TileLang code now runs on Tenstorrent hardware.
""")


if __name__ == "__main__":
    main()
