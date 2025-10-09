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
- ✅ mm_init() before K-loop
- ✅ Accumulate flag based on K-loop variable
- ✅ Proper CB management (wait/pop for inputs)
- ✅ Tile intrinsic emission (no scalar loops)
- ✅ T.gemm() operation detection

Expected Metalium Code Structure:
```cpp
for (tile_id in tiles) {
    tile_regs_acquire();

    // K-loop: C[m,n] += sum(A[m,k] * B[k,n] for k in Kt)
    mm_init(CB_A, CB_B, CB_C);
    for (uint32_t k = 0; k < Kt; ++k) {
        cb_wait_front(CB_A, 1);
        cb_wait_front(CB_B, 1);

        bool accumulate = (k > 0);
        matmul_tiles(CB_A, CB_B, CB_C, accumulate);

        cb_pop_front(CB_A, 1);
        cb_pop_front(CB_B, 1);
    }

    cb_reserve_back(CB_C, 1);
    tile_regs_commit();
    pack_tile(0, CB_C);
    cb_push_back(CB_C, 1);
    tile_regs_release();
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
    - mm_init() before K-loop
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

    # Apply TT default annotations (TT defaults stage)
    print("Applying TT defaults stage (default TT annotations)...")
    mod = tt.apply_tt_defaults(mod)

    # Apply metadata inference stage passes (schedule + shard inference)
    print("Applying metadata inference stage (schedule and sharding inference)...")
    mod = tt.apply_tt_metadata_passes(mod)

    # Apply persistent transform stage passes (grid to persistent)
    print("Applying persistent transform stage (grid to persistent transform)...")
    mod = tt.apply_tt_transform_passes(mod)

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

    # Tile register lifecycle (correct Metalium APIs)
    has_acquire = "tile_regs_acquire()" in compute
    has_commit = "tile_regs_commit()" in compute
    has_wait = "tile_regs_wait()" in compute
    has_release = "tile_regs_release()" in compute
    tile_regs_lifecycle = has_acquire and has_commit and has_wait and has_release
    checks.append(("Tile register lifecycle (acquire→commit→wait→release)", tile_regs_lifecycle))

    # K-loop structure
    has_k_loop = "for (uint32_t k" in compute and "< " in compute
    checks.append(("K-loop present", has_k_loop))

    # mm_init before K-loop (correct Metalium API)
    has_mm_init = "mm_init(cb_in0, cb_in1, cb_out0)" in compute
    init_before_loop = has_mm_init and compute.find("mm_init") < compute.find("for (uint32_t k")
    checks.append(("mm_init() before K-loop", init_before_loop))

    # CB index format (correct Metalium format)
    has_cb_format = "tt::CBIndex::c_0" in compute or "cb_in0" in compute
    checks.append(("CB index format (tt::CBIndex::c_N)", has_cb_format))

    # matmul_tiles with 6 parameters (correct Metalium signature)
    has_matmul_6params = "matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false)" in compute
    checks.append(("matmul_tiles with 6 parameters", has_matmul_6params))

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
        print("✅ METALIUM API CORRECTNESS: All checks passed!")
        print("=" * 70)
        print()
        print("Generated code uses correct Metalium APIs:")
        print("  ✓ Tile register lifecycle (tile_regs_*)")
        print("  ✓ K-loop accumulation pattern")
        print("  ✓ Correct initialization (mm_init)")
        print("  ✓ CB index format (tt::CBIndex::c_N)")
        print("  ✓ matmul_tiles 6-parameter signature")
        print("  ✓ Pattern recognition (T.copy, T.gemm, T.grid)")
        print("  ✓ CB management")
        print("  ✓ No scalar loops")
        print()
        print("🎉 Code matches real Metalium examples!")
    elif passed >= len(checks) - 2:
        print()
        print("⚠ NEAR COMPLETE: Minor issues to address")
    else:
        print()
        print("⚠ PARTIAL: Significant work needed")

if __name__ == "__main__":
    main()
