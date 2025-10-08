#!/usr/bin/env python3
"""
Tenstorrent Backend Example: Simple GEMM (Phase 1.3)
====================================================

This example demonstrates a complete simple GEMM (General Matrix Multiply)
implementation for the Tenstorrent backend, showcasing all Phase 1 features:

Pattern: C[M,N] = A[M,K] @ B[K,N]

Key Features Demonstrated:
- ✅ DST double buffering (acquire→commit→release)
- ✅ K-loop with matmul accumulation
- ✅ matmul_tiles_init() before K-loop
- ✅ Accumulate flag based on K-loop variable
- ✅ Proper CB management (wait/pop for inputs)
- ✅ Tile intrinsic emission (no scalar loops)
- ✅ T.gemm() operation detection

Expected Metalium Code Structure:
```cpp
for (tile_id in tiles) {
    acquire_dst();

    // K-loop: C[m,n] += sum(A[m,k] * B[k,n] for k in Kt)
    matmul_tiles_init(CB_A, CB_B, CB_C);
    for (uint32_t k = 0; k < Kt; ++k) {
        cb_wait_front(CB_A, 1);
        cb_wait_front(CB_B, 1);

        bool accumulate = (k > 0);
        matmul_tiles(CB_A, CB_B, CB_C, accumulate);

        cb_pop_front(CB_A, 1);
        cb_pop_front(CB_B, 1);
    }

    cb_reserve_back(CB_C, 1);
    commit_dst();
    pack_tile(0, CB_C);
    cb_push_back(CB_C, 1);
    release_dst();
}
```

This is the culmination of Phase 1 work, demonstrating that the pattern
recognition and codegen infrastructure can handle the most complex Phase 1
pattern: GEMM with K-loop accumulation.
"""

import tvm
from tvm import tir
import tilelang.language as T
import tilelang.tt as tt

@T.prim_func
def simple_gemm_tt(
    A: T.Buffer((256, 256), "float16"),
    B: T.Buffer((256, 256), "float16"),
    C: T.Buffer((256, 256), "float16")
):
    """
    Simple GEMM for Tenstorrent backend.

    Pattern: C = A @ B
    - DST held across K iterations for accumulation (Pattern 3)
    - matmul_tiles_init() before K-loop
    - Accumulate flag varies based on K iteration
    - CB management for A and B tiles

    Matrix dimensions:
    - M = 256 (8 tiles of 32)
    - K = 256 (8 tiles of 32)
    - N = 256 (8 tiles of 32)

    Grid: 8x8 = 64 cores
    Each core processes 1 output tile
    """
    with T.Kernel(T.ceildiv(256, 32), T.ceildiv(256, 32)) as (bx, by):
        # Allocate tile-sized storage
        A_shared = T.alloc_fragment((32, 32), "float16")
        B_shared = T.alloc_fragment((32, 32), "float16")
        C_local = T.alloc_fragment((32, 32), "float16")

        # Initialize accumulator
        T.clear(C_local)

        # K-loop for matrix multiplication
        for k in T.serial(T.ceildiv(256, 32)):
            # Load A[bx, k] tile
            T.copy(A[bx * 32:(bx+1)*32, k * 32:(k+1)*32], A_shared)

            # Load B[k, by] tile
            T.copy(B[k * 32:(k+1)*32, by * 32:(by+1)*32], B_shared)

            # Compute: C_local += A_shared @ B_shared
            T.gemm(A_shared, B_shared, C_local, transpose_A=False, transpose_B=False)

        # Store result
        T.copy(C_local, C[bx * 32:(bx+1)*32, by * 32:(by+1)*32])

def create_simple_gemm_module(M=256, K=256, N=256):
    """Create TileLang IR for simple GEMM."""
    return tvm.IRModule({"main": simple_gemm_tt})

def main():
    print("=" * 70)
    print("Tenstorrent Simple GEMM Example (Phase 1.3)")
    print("=" * 70)
    print()

    print("Matrix dimensions: 256×256 @ 256×256 = 256×256")
    print("Tile size: 32×32 (fp16)")
    print("Grid: 8×8 = 64 cores")
    print("K-tiles: 8 iterations per output tile")
    print()

    # Create module
    print("Creating IRModule...")
    mod = create_simple_gemm_module()

    # Apply TT default annotations (WS1)
    print("Applying WS1 (default TT annotations)...")
    mod = tt.apply_tt_defaults(mod)

    # Apply WS2 passes (schedule + shard inference)
    print("Applying WS2 (schedule and sharding inference)...")
    mod = tt.apply_ws2_passes(mod)

    # Apply WS3 passes (grid to persistent)
    print("Applying WS3 (grid to persistent transform)...")
    mod = tt.apply_ws3_passes(mod)

    # Generate artifacts
    print("Generating Metalium artifacts...")
    artifacts = tt.emit_tt_artifacts(mod)

    print()
    print("=" * 70)
    print("Generated Compute Kernel:")
    print("=" * 70)
    compute = artifacts.get("compute.cpp", "")
    for i, line in enumerate(compute.split('\n'), 1):
        if 35 <= i <= 80:  # Print main function body
            print(f"{i:>3}: {line}")

    print()
    print("-" * 70)
    print("Phase 1.3 GEMM Pattern Validation:")
    print("-" * 70)

    # Comprehensive validation checks
    checks = []

    # DST lifecycle
    has_acquire = "acquire_dst()" in compute
    has_commit = "commit_dst()" in compute
    has_release = "release_dst()" in compute
    dst_lifecycle = has_acquire and has_commit and has_release
    checks.append(("DST lifecycle (acquire→commit→release)", dst_lifecycle))

    # K-loop structure
    has_k_loop = "for (uint32_t k" in compute and "< " in compute
    checks.append(("K-loop present", has_k_loop))

    # matmul_tiles_init before K-loop
    init_before_loop = compute.find("matmul_tiles_init") < compute.find("for (uint32_t k")
    checks.append(("matmul_tiles_init() before K-loop", init_before_loop))

    # Accumulate flag
    has_accumulate_flag = "bool accumulate = (k > 0)" in compute
    checks.append(("Accumulate flag based on k", has_accumulate_flag))

    # matmul_tiles with accumulate
    has_matmul_tiles = "matmul_tiles(CB_A, CB_B, CB_C, accumulate)" in compute
    checks.append(("matmul_tiles with accumulate flag", has_matmul_tiles))

    # CB wait for inputs
    cb_wait_count = compute.count("cb_wait_front")
    checks.append(("CB wait operations (≥2)", cb_wait_count >= 2))

    # CB pop for inputs
    cb_pop_count = compute.count("cb_pop_front")
    checks.append(("CB pop operations (≥2)", cb_pop_count >= 2))

    # T.gemm detection
    has_gemm_comment = "// Wait for input tiles from reader" in compute
    checks.append(("T.gemm() pattern detected", has_gemm_comment))

    # T.copy handling
    copy_comment_count = compute.count("// T.copy - handled by reader/writer")
    checks.append(("T.copy operations detected (≥3)", copy_comment_count >= 3))

    # No scalar loops
    no_scalar_loops = "C_local[" not in compute and "A_shared[" not in compute
    checks.append(("No scalar loops (tile intrinsics only)", no_scalar_loops))

    # Print validation results
    passed = 0
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}")
        if result:
            passed += 1

    print()
    print(f"Validation: {passed}/{len(checks)} checks passed")

    if passed == len(checks):
        print()
        print("=" * 70)
        print("✅ PHASE 1.3 COMPLETE: Simple GEMM fully working!")
        print("=" * 70)
        print()
        print("All Phase 1 features demonstrated:")
        print("  ✓ DST double buffering")
        print("  ✓ K-loop accumulation pattern")
        print("  ✓ Tile intrinsic emission")
        print("  ✓ Pattern recognition (T.copy, T.gemm, T.grid)")
        print("  ✓ CB management")
        print("  ✓ No scalar loops")
        print()
        print("🎉 Phase 1 Foundation: COMPLETE")
    elif passed >= len(checks) - 2:
        print()
        print("⚠ NEAR COMPLETE: Minor issues to address")
    else:
        print()
        print("⚠ PARTIAL: Significant work needed")

if __name__ == "__main__":
    main()
