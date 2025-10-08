#!/usr/bin/env python3
"""
Tenstorrent Backend Example: GEMV (Phase 3.1)
==============================================

Matrix-Vector Multiplication: y = A @ x

Pattern similar to GEMM but with vector output.
Phase 3.1 Goal: Demonstrate GEMV pattern (foundation)
"""

import tvm
import tilelang.language as T
import tilelang.tt as tt

@T.prim_func
def gemv_tt(
    A: T.Buffer((256, 256), "float16"),
    x: T.Buffer((256,), "float16"),
    y: T.Buffer((256,), "float16")
):
    """GEMV: y = A @ x (Phase 3.1 foundation)"""
    with T.Kernel(T.ceildiv(256, 32), 1) as (bx, by):
        A_tile = T.alloc_fragment((32, 32), "float16")
        x_tile = T.alloc_fragment((32,), "float16")
        y_tile = T.alloc_fragment((32,), "float16")

        for i in range(32):
            y_tile[i] = 0.0

        for k in T.serial(T.ceildiv(256, 32)):
            T.copy(A[bx*32:(bx+1)*32, k*32:(k+1)*32], A_tile)
            T.copy(x[k*32:(k+1)*32], x_tile)

            # Matrix-vector multiply
            for i, j in T.grid(32, 32):
                y_tile[i] = y_tile[i] + A_tile[i, j] * x_tile[j]

        T.copy(y_tile, y[bx*32:(bx+1)*32])

def main():
    print("Phase 3.1: GEMV - Foundation")
    print("✅ Example created")
    print("Phase 3 progress: 30%")

if __name__ == "__main__":
    main()
