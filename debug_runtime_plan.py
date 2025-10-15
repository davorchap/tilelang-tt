#!/usr/bin/env python3
"""Debug script to check runtime plan structure"""

import json
import sys

sys.path.insert(0, '.')

import tilelang
import tilelang.language as T
from tilelang.utils.target import TENSTORRENT_TARGET


@tilelang.jit(target=TENSTORRENT_TARGET, out_idx=[-1])
def kernel(M, N):

    @T.prim_func
    def func(A: T.Buffer((M, N), "float16"), B: T.Buffer((M, N), "float16")):
        with T.Kernel(T.ceildiv(N, 32), T.ceildiv(M, 32)) as (bx, by):
            for i in T.serial(32):
                for j in T.serial(32):
                    row = by * 32 + i
                    col = bx * 32 + j
                    if row < M and col < N:
                        B[row, col] = A[row, col]

    return func


k = kernel(64, 64)
source = k.get_kernel_source()
artifacts = json.loads(source)

# Parse runtime plan
plan = json.loads(artifacts["tt.plan.json"])

print("Runtime plan structure:")
print(json.dumps(plan, indent=2))
