#!/usr/bin/env python3
"""
Tenstorrent Backend Example: GEMM (Matrix Multiplication)
==========================================================

Matrix-Matrix Multiplication: C = A @ B

This example demonstrates how to:
1. Write a TileLang GEMM kernel
2. Apply the TT compilation pipeline
3. Generate Tenstorrent Metalium C++ artifacts
"""

import tvm
import tilelang.language as T
import tilelang.tenstorrent as tt


def matmul(M, N, K, block_M, block_N, block_K, dtype="float16"):
    """
    Create a GEMM kernel for Tenstorrent backend.

    Args:
        M, N, K: Matrix dimensions (C: M×N = A: M×K @ B: K×N)
        block_M, block_N, block_K: Tile/block dimensions
        dtype: Data type

    Returns:
        TVM IRModule ready for TT pipeline
    """

    @T.prim_func
    def gemm(
            A: T.Buffer((M, K), dtype),
            B: T.Buffer((K, N), dtype),
            C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M)) as (bx, by):
            A_tile = T.alloc_fragment((block_M, block_K), dtype)
            B_tile = T.alloc_fragment((block_K, block_N), dtype)
            C_tile = T.alloc_fragment((block_M, block_N), dtype)

            # Initialize accumulator
            T.clear(C_tile)

            # K-loop for matrix multiplication
            for k in T.serial(T.ceildiv(K, block_K)):
                T.copy(A[by * block_M:(by + 1) * block_M, k * block_K:(k + 1) * block_K], A_tile)
                T.copy(B[k * block_K:(k + 1) * block_K, bx * block_N:(bx + 1) * block_N], B_tile)
                T.gemm(A_tile, B_tile, C_tile)

            T.copy(C_tile, C[by * block_M:(by + 1) * block_M, bx * block_N:(bx + 1) * block_N])

    return tvm.IRModule({"main": gemm})


def main():
    print("=" * 70)
    print("Tenstorrent GEMM: Matrix-Matrix Multiplication")
    print("=" * 70)
    print()

    # Create kernel for 1024×1024 matrix multiplication
    M, N, K = 1024, 1024, 1024
    block_M, block_N, block_K = 128, 128, 32

    print("GEMM Configuration:")
    print(f"- Matrix A: {M}×{K}")
    print(f"- Matrix B: {K}×{N}")
    print(f"- Matrix C: {M}×{N}")
    print(f"- Block size: {block_M}×{block_N}×{block_K}")
    print(f"- Grid: {M//block_M}×{N//block_N} = {(M//block_M)*(N//block_N)} tiles")
    print()

    # Create module
    mod = matmul(M, N, K, block_M, block_N, block_K)

    print("Applying TT compilation pipeline...")
    print()

    # Apply TT pipeline
    mod = tt.apply_tt_defaults(mod)
    print("✓ Applied TT defaults")

    # Run the TT pipeline with default settings
    mod = tt.run_pipeline(mod, verbose=True)
    print("✓ Applied TT pipeline transforms")

    artifacts = tt.emit_tt_artifacts(mod)
    print("✓ Generated TT artifacts")
    print()

    # Display generated artifacts
    print("=" * 70)
    print("Generated Artifacts:")
    print("=" * 70)
    print()

    for name in sorted(artifacts.keys()):
        print(f"✓ {name} ({len(artifacts[name])} bytes)")
    print()

    # Show snippet of compute kernel
    print("=" * 70)
    print("Compute Kernel Snippet (compute.cpp):")
    print("=" * 70)
    compute = artifacts.get("compute.cpp", "")
    lines = compute.split('\n')

    # Show first 30 lines
    for i, line in enumerate(lines[:30], 1):
        print(f"{i:>3}: {line}")

    if len(lines) > 30:
        print(f"... ({len(lines) - 30} more lines)")
    print()

    # Show execution plan
    print("=" * 70)
    print("Execution Plan (tt.plan.json):")
    print("=" * 70)
    plan = artifacts.get("tt.plan.json", "")
    import json
    plan_data = json.loads(plan)
    print(json.dumps(plan_data, indent=2))
    print()

    # Optionally write to disk
    output_dir = "tt_gemm_artifacts"
    print("=" * 70)
    print(f"Writing artifacts to disk: {output_dir}/")
    print("=" * 70)
    tt.write_artifacts_to_disk(artifacts, output_dir=output_dir)

    import os
    for name in sorted(artifacts.keys()):
        path = os.path.join(output_dir, name)
        print(f"✓ {path}")
    print()

    # Validation
    print("=" * 70)
    print("Validation:")
    print("=" * 70)

    checks = []

    # Check for key TT operations in compute kernel
    has_matmul = "matmul_tiles" in compute or "mm_init" in compute
    has_cb_ops = "cb_wait_front" in compute or "cb_pop_front" in compute
    has_dst = "tile_regs_acquire" in compute or "tile_regs_commit" in compute

    checks.append(("Matmul operations present", has_matmul))
    checks.append(("Circular buffer operations", has_cb_ops))
    checks.append(("DST register lifecycle", has_dst))
    checks.append(("Plan JSON generated", "version" in plan_data))
    checks.append(("Grid metadata present", "grid" in plan_data))

    passed = sum(1 for _, result in checks if result)
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}")

    print()
    print(f"Result: {passed}/{len(checks)} checks passed")
    print()

    if passed == len(checks):
        print("=" * 70)
        print("✅ SUCCESS: GEMM compilation complete!")
        print("=" * 70)
        print()
        print("Next steps:")
        print(f"  1. Review generated files in {output_dir}/")
        print("  2. Integrate with TT-Metalium SDK for hardware execution")
        print("  3. See docs/tenstorrent/METALIUM_SETUP_GUIDE.md for SDK setup")
    else:
        print("⚠ Some validation checks failed")
    print()


if __name__ == "__main__":
    main()
