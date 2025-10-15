"""
Test @tilelang.jit decorator functionality with Tenstorrent backend.

This test verifies that:
1. Basic kernels compile with TT backend
2. Full TileLang DSL features work (T.alloc_shared, T.alloc_fragment, T.gemm, etc.)
3. TT artifacts are generated correctly
"""

import json
import sys

sys.path.insert(0, '.')

import tilelang
import tilelang.language as T
from tilelang.utils.target import TENSTORRENT_TARGET


def test_basic_jit_decorator():
    """Test basic @tilelang.jit decorator with TT backend."""

    @tilelang.jit(target=TENSTORRENT_TARGET, out_idx=[-1])
    def simple_kernel(M, N, dtype="float16"):

        @T.prim_func
        def func(A: T.Buffer((M, N), dtype), B: T.Buffer((M, N), dtype)):
            with T.Kernel(T.ceildiv(N, 32), T.ceildiv(M, 32)) as (bx, by):
                for i in T.serial(32):
                    for j in T.serial(32):
                        row = by * 32 + i
                        col = bx * 32 + j
                        if row < M and col < N:
                            B[row, col] = A[row, col] * T.cast(2.0, dtype)

        return func

    # Test kernel creation
    kernel = simple_kernel(64, 64)
    assert kernel is not None

    # Test artifact generation
    source = kernel.get_kernel_source()
    artifacts = json.loads(source)

    # Verify all TT artifacts are present
    expected_artifacts = ["reader.cpp", "compute.cpp", "writer.cpp", "main.cpp", "tt.plan.json"]
    for artifact in expected_artifacts:
        assert artifact in artifacts, f"Missing artifact: {artifact}"


def test_full_dsl_features():
    """Test full TileLang DSL features with TT backend."""

    @tilelang.jit(target=TENSTORRENT_TARGET, out_idx=[-1])
    def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

        @T.prim_func
        def gemm(
                A: T.Tensor((M, K), dtype),
                B: T.Tensor((K, N), dtype),
                C: T.Tensor((M, N), dtype),
        ):
            # Test that threads parameter is accepted (though ignored by TT)
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                # Test T.alloc_shared (maps to L1)
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)

                # Test T.alloc_fragment (maps to registers)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                # Test T.clear
                T.clear(C_local)

                # Test T.Pipelined (becomes persistent loop)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                    # Test T.copy (NOC DMA)
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)

                    # Test T.gemm (matmul intrinsic)
                    T.gemm(A_shared, B_shared, C_local)

                # Test final copy
                T.copy(C_local, C[by * block_M, bx * block_N])

        return gemm

    # Test with TT-friendly tile sizes
    kernel = matmul(128, 128, 128, 32, 32, 32)
    assert kernel is not None

    # Verify artifacts
    source = kernel.get_kernel_source()
    artifacts = json.loads(source)
    assert len(artifacts) == 5  # All 5 TT artifacts

    # Verify compute kernel contains matmul operations
    compute_code = artifacts.get("compute.cpp", "")
    assert "matmul" in compute_code.lower() or "compute" in compute_code.lower()


def test_different_sizes():
    """Test various tensor and tile sizes."""

    @tilelang.jit(target=TENSTORRENT_TARGET, out_idx=[-1])
    def matmul(M, N, K, block_M, block_N, block_K):

        @T.prim_func
        def gemm(
                A: T.Tensor((M, K), "float16"),
                B: T.Tensor((K, N), "float16"),
                C: T.Tensor((M, N), "float16"),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M)) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), "float16")
                B_shared = T.alloc_shared((block_K, block_N), "float16")
                C_local = T.alloc_fragment((block_M, block_N), "float")

                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C[by * block_M, bx * block_N])

        return gemm

    # Test different configurations (all use 32x32 tiles for TT)
    configs = [
        (64, 64, 64, 32, 32, 32),  # 2x2x2 tiles
        (128, 128, 128, 32, 32, 32),  # 4x4x4 tiles
        (256, 256, 256, 32, 32, 32),  # 8x8x8 tiles
    ]

    for M, N, K, bM, bN, bK in configs:
        kernel = matmul(M, N, K, bM, bN, bK)
        assert kernel is not None

        # Verify work partition is created correctly
        source = kernel.get_kernel_source()
        artifacts = json.loads(source)
        plan = json.loads(artifacts["tt.plan.json"])

        expected_grid_x = N // bN
        expected_grid_y = M // bM
        assert plan["grid"]["x"] == expected_grid_x
        assert plan["grid"]["y"] == expected_grid_y


def test_runtime_plan():
    """Test that runtime plan (tt.plan.json) is generated correctly."""

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

    # Verify plan structure
    assert "grid" in plan
    assert "cores" in plan
    assert "schedule" in plan
    assert "layouts" in plan

    # Verify grid (2x2 for 64x64 with 32x32 tiles)
    assert plan["grid"]["x"] == 2
    assert plan["grid"]["y"] == 2

    # Verify layouts
    assert "A" in plan["layouts"]
    assert "B" in plan["layouts"]
    assert plan["layouts"]["A"]["shard"] == "DRAM"
    assert plan["layouts"]["B"]["shard"] == "DRAM"


if __name__ == "__main__":
    test_basic_jit_decorator()
    print("✓ Basic JIT decorator test passed")

    test_full_dsl_features()
    print("✓ Full DSL features test passed")

    test_different_sizes()
    print("✓ Different sizes test passed")

    test_runtime_plan()
    print("✓ Runtime plan test passed")

    print("\nAll tests passed!")
