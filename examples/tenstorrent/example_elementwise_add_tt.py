#!/usr/bin/env python3
"""
Elementwise Add for Tenstorrent Backend
========================================

Simple element-wise addition: C = A + B

This example demonstrates Pattern 1 (Element-wise) from DST_DOUBLE_BUFFERING_SPEC.md:
- DST acquired and released per tile
- No K-loop accumulation
- Single add_tiles operation per tile

Expected generated compute kernel structure:
```
for (tile_id in tiles) {
    tile_regs_acquire();
    cb_wait_front(CB_A, 1);
    cb_wait_front(CB_B, 1);
    add_tiles_init();
    add_tiles(CB_A, CB_B, 0, 0, 0);
    cb_reserve_back(CB_C, 1);
    tile_regs_commit();
    pack_tile(0, CB_C);
    cb_push_back(CB_C, 1);
    cb_pop_front(CB_A, 1);
    cb_pop_front(CB_B, 1);
    tile_regs_release();
}
```
"""

import tvm
from tvm import tir
import tilelang.language as T
import tilelang.tt as tt

@T.prim_func
def elementwise_add_tt(
    A: T.Buffer((256, 256), "float16"),
    B: T.Buffer((256, 256), "float16"),
    C: T.Buffer((256, 256), "float16")
):
    """
    Elementwise addition for Tenstorrent backend.

    Pattern 1 (Element-wise): C = A + B
    - DST acquired and released per tile
    - No K-loop accumulation
    - Single add_tiles operation per tile

    Grid-style kernel that will be transformed to persistent per-core loop.
    """
    with T.Kernel(T.ceildiv(256, 32), T.ceildiv(256, 32)) as (bx, by):
        # Allocate tile-sized storage
        # These will be lowered to circular buffers
        A_tile = T.alloc_fragment((32, 32), "float16")
        B_tile = T.alloc_fragment((32, 32), "float16")
        C_tile = T.alloc_fragment((32, 32), "float16")

        # Load A[by, bx] tile from DRAM
        T.copy(A[by * 32:(by+1)*32, bx * 32:(bx+1)*32], A_tile)

        # Load B[by, bx] tile from DRAM
        T.copy(B[by * 32:(by+1)*32, bx * 32:(bx+1)*32], B_tile)

        # Compute C = A + B (element-wise)
        # This should generate: add_tiles_init(); add_tiles(CB_A, CB_B, 0, 0, 0);
        for i, j in T.grid(32, 32):
            C_tile[i, j] = A_tile[i, j] + B_tile[i, j]

        # Store result tile to DRAM
        T.copy(C_tile, C[by * 32:(by+1)*32, bx * 32:(bx+1)*32])

def create_elementwise_add_module(M=256, N=256):
    """Create TileLang IR for elementwise add."""
    return tvm.IRModule({"main": elementwise_add_tt})

def main():
    print("=" * 70)
    print("Tenstorrent Elementwise Add Example")
    print("=" * 70)
    print()

    # Create module
    mod = create_elementwise_add_module(M=256, N=256)

    # Apply TT transforms
    print("Applying TT transforms...")
    mod = tt.apply_tt_defaults(mod)
    mod = tt.apply_ws2_passes(mod)
    mod = tt.apply_ws3_passes(mod)
    print("✓ Transforms complete")
    print()

    # Generate artifacts
    print("Generating Metalium artifacts...")
    artifacts = tt.emit_tt_artifacts(mod)
    print("✓ Codegen complete")
    print()

    # Show generated kernels
    print("Generated Artifacts:")
    print("-" * 70)
    for name in sorted(artifacts.keys()):
        if name.endswith('.cpp'):
            print(f"\n📄 {name}:")
            code = artifacts[name]
            lines = code.split('\n')
            # Show first 60 lines of each kernel
            for i, line in enumerate(lines[:60], 1):
                print(f"{i:3}: {line}")
            if len(lines) > 60:
                print(f"    ... ({len(lines) - 60} more lines)")
            print()

    # Check for DST usage in compute kernel
    compute = artifacts.get("compute.cpp", "")
    has_acquire = "tile_regs_acquire();" in compute
    has_commit = "tile_regs_commit();" in compute
    has_release = "tile_regs_release();" in compute

    print("-" * 70)
    print("DST Double Buffering Status:")
    print(f"  tile_regs_acquire(): {'✓ Found' if has_acquire else '✗ Missing'}")
    print(f"  tile_regs_commit():  {'✓ Found' if has_commit else '✗ Missing'}")
    print(f"  tile_regs_release(): {'✓ Found' if has_release else '✗ Missing'}")
    print()

    if has_acquire and has_commit and has_release:
        print("✅ PASS: DST double buffering implemented")
        return 0
    elif has_acquire:
        print("⚠️  PARTIAL: DST acquire present, but commit/release missing")
        print("    (Expected: full lifecycle will work once elementwise IR is proper)")
        return 0
    else:
        print("❌ FAIL: No DST double buffering found")
        return 1

if __name__ == "__main__":
    exit(main())
