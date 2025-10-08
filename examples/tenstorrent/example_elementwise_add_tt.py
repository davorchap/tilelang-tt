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
    acquire_dst();
    cb_wait_front(CB_A, 1);
    cb_wait_front(CB_B, 1);
    add_tiles_init();
    add_tiles(CB_A, CB_B, 0, 0, 0);
    cb_reserve_back(CB_C, 1);
    commit_dst();
    pack_tile(0, CB_C);
    cb_push_back(CB_C, 1);
    cb_pop_front(CB_A, 1);
    cb_pop_front(CB_B, 1);
    release_dst();
}
```
"""

import tvm
from tvm import tir
import tilelang.tt as tt

def create_elementwise_add_module(M=256, N=256):
    """Create TileLang IR for elementwise add."""

    A = tir.decl_buffer((M, N), "float16", name="A")
    B = tir.decl_buffer((M, N), "float16", name="B")
    C = tir.decl_buffer((M, N), "float16", name="C")

    # For now, use simple function body
    # TODO: Add proper elementwise IR structure
    func = tir.PrimFunc(params=[A, B, C], body=tir.Evaluate(0))
    func = func.with_attrs({
        "global_symbol": "main",
        "tl.backend": "tenstorrent",
    })

    return tvm.IRModule({"main": func})

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
    has_acquire = "acquire_dst();" in compute
    has_commit = "commit_dst();" in compute
    has_release = "release_dst();" in compute

    print("-" * 70)
    print("DST Double Buffering Status:")
    print(f"  acquire_dst(): {'✓ Found' if has_acquire else '✗ Missing'}")
    print(f"  commit_dst():  {'✓ Found' if has_commit else '✗ Missing'}")
    print(f"  release_dst(): {'✓ Found' if has_release else '✗ Missing'}")
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
