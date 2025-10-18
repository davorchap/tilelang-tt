#!/usr/bin/env python3
"""
Diagnostic script to check pipeline output and identify codegen issues.
"""

import logging

logging.basicConfig(level=logging.INFO)

import tilelang
import tilelang.language as T
from tilelang.utils.target import TENSTORRENT_TARGET


@tilelang.jit(target=TENSTORRENT_TARGET, out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

    @T.prim_func
    def gemm(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def main():
    print("=" * 80)
    print("Pipeline Diagnostic Script")
    print("=" * 80)

    # Create kernel
    kernel = matmul(256, 256, 256, 32, 32, 32)

    # Get the internal IRModule from the kernel object
    # This requires accessing internal attributes
    if hasattr(kernel, '_func'):
        mod = kernel._func
        print(f"\n✓ Found IRModule with {len(list(mod.functions_items()))} functions:")

        for name, func in mod.functions_items():
            print(f"\n  Function: {name}")
            if hasattr(func, 'attrs') and func.attrs:
                print("    Attributes:")
                for key in func.attrs.keys():
                    val = func.attrs[key]
                    print(f"      {key}: {val}")

                # Check for kernel role
                role = func.attrs.get("tt.kernel_role")
                if role:
                    print(f"    ⭐ Has kernel role: {role}")
                else:
                    print("    ⚠️  No kernel role attribute!")
            else:
                print("    ⚠️  No attributes!")

    else:
        print("⚠️  Cannot access IRModule from kernel object")

    # Check generated code
    import json
    source = kernel.get_kernel_source()
    artifacts = json.loads(source)

    print(f"\n✓ Generated {len(artifacts)} artifacts:")
    for name in artifacts.keys():
        print(f"  - {name}")

    # Check for NOC operations in reader
    if "reader.cpp" in artifacts:
        reader_code = artifacts["reader.cpp"]
        print(f"\n🔍 Checking reader.cpp ({len(reader_code)} bytes):")

        # Check for expected operations
        has_cb_reserve = "cb_reserve_back" in reader_code
        has_noc_read = "noc_async_read" in reader_code
        has_buffer_access = ("A[" in reader_code or "B[" in reader_code)

        print(f"  ✓ Has cb_reserve_back: {has_cb_reserve}")
        print(f"  ✓ Has noc_async_read: {has_noc_read}")
        print(f"  ⚠️  Has raw buffer access (A[...] or B[...]): {has_buffer_access}")

        if has_buffer_access and not has_noc_read:
            print("\n❌ PROBLEM: Reader has raw buffer accesses instead of NOC operations!")
            print("   This means the lowering passes didn't transform the IR correctly.")

            # Show first few lines
            lines = reader_code.split('\n')
            print("\n   First 30 lines of reader.cpp:")
            for i, line in enumerate(lines[:30], 1):
                print(f"   {i:3d}: {line}")


if __name__ == "__main__":
    main()
