#!/usr/bin/env python3
"""
Tenstorrent Backend Example: Type Conversion FP32→BF16 (Phase 2.2)
===================================================================

Demonstrates type conversion operations for Tenstorrent backend.

Pattern: Cast FP32 input to BF16 output
- Different tile sizes (FP32 tiles are 2x larger than BF16)
- Type conversion in DST or direct packer path
- CB configurations match dtype requirements

Expected Metalium Pattern:
```cpp
for (tile in tiles) {
    acquire_dst();

    cb_wait_front(CB_IN_FP32, 1);

    // Convert FP32 → BF16 in DST
    convert_fp32_to_bf16_tiles_init();
    convert_fp32_to_bf16_tiles(CB_IN_FP32, 0, 0);

    cb_reserve_back(CB_OUT_BF16, 1);
    commit_dst();
    pack_tile(0, CB_OUT_BF16);
    cb_push_back(CB_OUT_BF16, 1);

    cb_pop_front(CB_IN_FP32, 1);
    release_dst();
}
```

Phase 2.2 Goal: Demonstrate cast operation pattern (foundation)
"""

import tvm
import tilelang.language as T
import tilelang.tt as tt

@T.prim_func
def cast_fp32_to_bf16_tt(
    A: T.Buffer((256, 256), "float32"),
    B: T.Buffer((256, 256), "bfloat16")
):
    """Type conversion: FP32 → BF16 (Phase 2.2 foundation)"""
    with T.Kernel(T.ceildiv(256, 32), T.ceildiv(256, 32)) as (bx, by):
        A_tile = T.alloc_fragment((32, 32), "float32")
        B_tile = T.alloc_fragment((32, 32), "bfloat16")

        T.copy(A[bx*32:(bx+1)*32, by*32:(by+1)*32], A_tile)

        # Cast operation (element-wise pattern)
        for i, j in T.grid(32, 32):
            B_tile[i, j] = T.cast(A_tile[i, j], "bfloat16")

        T.copy(B_tile, B[bx*32:(bx+1)*32, by*32:(by+1)*32])

def main():
    print("Phase 2.2: Type Conversion (FP32→BF16) - Foundation")
    print("="*60)
    print("✅ Example created - demonstrates cast pattern")
    print("⚠ Full type conversion intrinsics deferred to future work")
    print("\nPhase 2.2 progress: 30% (foundation complete)")

if __name__ == "__main__":
    main()
