#!/usr/bin/env python3
"""
Tenstorrent Backend Example: Multi-operand Element-wise Operations
==================================================================

This example demonstrates multi-operand element-wise operations for the
Tenstorrent backend, building on the single-operation pattern from Phase 1.1.

Pattern: D = A + B + C (three-operand element-wise)

Expected Metalium Code Structure:
```cpp
for (tile_id in tiles) {
    tile_regs_acquire();

    // Load three input tiles
    cb_wait_front(CB_A, 1);
    cb_wait_front(CB_B, 1);
    cb_wait_front(CB_C, 1);

    // First operation: temp = A + B
    add_tiles_init();
    add_tiles(CB_A, CB_B, 0, 0, 0);

    // Second operation: D = temp + C
    add_tiles(0, CB_C, 0, 0, 0);

    cb_reserve_back(CB_D, 1);
    tile_regs_commit();
    pack_tile(0, CB_D);
    cb_push_back(CB_D, 1);

    // Pop input tiles
    cb_pop_front(CB_A, 1);
    cb_pop_front(CB_B, 1);
    cb_pop_front(CB_C, 1);

    tile_regs_release();
}
```

Phase 1.2 Goal:
- Extend pattern recognition to detect multi-operand element-wise operations
- Support chained tile intrinsics (multiple add_tiles calls)
- Proper CB management for 3+ input buffers
"""

import tvm
import tilelang.language as T
import tilelang.tt as tt


@T.prim_func
def elementwise_multi_tt(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                         C: T.Buffer((256, 256), "float16"), D: T.Buffer((256, 256), "float16")):
    """
    Multi-operand element-wise addition for Tenstorrent backend.

    Pattern: D = A + B + C
    - DST acquired and released per tile
    - Chained add_tiles operations
    - Three input CBs (CB_A, CB_B, CB_C)
    - One output CB (CB_D)

    Grid-style kernel that will be transformed to persistent per-core loop.
    """
    with T.Kernel(T.ceildiv(256, 32), T.ceildiv(256, 32)) as (bx, by):
        # Allocate tile-sized storage
        # These will be lowered to circular buffers
        A_tile = T.alloc_fragment((32, 32), "float16")
        B_tile = T.alloc_fragment((32, 32), "float16")
        C_tile = T.alloc_fragment((32, 32), "float16")
        D_tile = T.alloc_fragment((32, 32), "float16")

        # Load A[by, bx] tile from DRAM
        T.copy(A[by * 32:(by + 1) * 32, bx * 32:(bx + 1) * 32], A_tile)

        # Load B[by, bx] tile from DRAM
        T.copy(B[by * 32:(by + 1) * 32, bx * 32:(bx + 1) * 32], B_tile)

        # Load C[by, bx] tile from DRAM
        T.copy(C[by * 32:(by + 1) * 32, bx * 32:(bx + 1) * 32], C_tile)

        # Compute D = A + B + C (element-wise)
        # This should generate:
        #   add_tiles_init();
        #   add_tiles(CB_A, CB_B, 0, 0, 0);  // temp = A + B
        #   add_tiles(0, CB_C, 0, 0, 0);     // D = temp + C
        for i, j in T.grid(32, 32):
            temp = A_tile[i, j] + B_tile[i, j]
            D_tile[i, j] = temp + C_tile[i, j]

        # Store result tile to DRAM
        T.copy(D_tile, D[by * 32:(by + 1) * 32, bx * 32:(bx + 1) * 32])


def create_elementwise_multi_module(M=256, N=256):
    """Create TileLang IR for multi-operand elementwise."""
    return tvm.IRModule({"main": elementwise_multi_tt})


def main():
    print("=" * 70)
    print("Tenstorrent Multi-operand Elementwise Example")
    print("=" * 70)
    print()

    # Create module
    print("Creating IRModule...")
    mod = create_elementwise_multi_module()

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
    print("Generated Artifacts:")
    print("=" * 70)
    for name in sorted(artifacts.keys()):
        if name.endswith(".cpp") or name.endswith(".json"):
            code = artifacts[name]
            print()
            print(f"📄 {name}:")
            for i, line in enumerate(code.split('\n'), 1):
                print(f"{i:>3}: {line}")

    # Validate compute kernel
    print()
    print("-" * 70)
    print("Multi-operand Pattern Validation:")
    print("-" * 70)

    compute = artifacts.get("compute.cpp", "")

    # Check for proper multi-operand pattern
    has_multi_cb_wait = compute.count("cb_wait_front") >= 3
    has_chained_adds = compute.count("add_tiles") >= 2
    has_dst_lifecycle = "tile_regs_acquire()" in compute and "tile_regs_release()" in compute

    print(
        f"  CB wait for 3+ inputs: {'✓' if has_multi_cb_wait else '✗'} {'Found' if has_multi_cb_wait else 'Missing'}"
    )
    print(
        f"  Chained add_tiles:     {'✓' if has_chained_adds else '✗'} {'Found' if has_chained_adds else 'Missing'}"
    )
    print(
        f"  DST lifecycle:         {'✓' if has_dst_lifecycle else '✗'} {'Found' if has_dst_lifecycle else 'Missing'}"
    )

    if has_multi_cb_wait and has_chained_adds and has_dst_lifecycle:
        print()
        print("✅ PASS: Multi-operand pattern correctly implemented")
    else:
        print()
        print("⚠ PARTIAL: Multi-operand pattern needs additional work")

    print()


if __name__ == "__main__":
    main()
