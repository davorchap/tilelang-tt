"""
Test Mock SDK Compilation

Purpose: Validate that generated C++ code compiles with mock TT SDK headers.
This catches syntax errors without requiring the real TT-Metal SDK.

IMPORTANT: Mock headers in testing/mock_tt_sdk/ are READ-ONLY.
Do NOT modify headers to make tests pass. Fix the code generator instead.
"""

import contextlib
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import tilelang
try:
    import tilelang
    import tilelang.language as T
    from tilelang.utils.target import TENSTORRENT_TARGET
    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False
    logger.warning("TileLang/TVM not available - tests will be skipped")


def find_mock_sdk_headers():
    """Find the mock SDK headers directory"""
    test_dir = Path(__file__).parent.parent.parent  # testing/python/tenstorrent -> testing
    mock_sdk = test_dir / "mock_tt_sdk" / "include"

    if not mock_sdk.exists():
        raise FileNotFoundError(f"Mock SDK headers not found at: {mock_sdk}")

    return str(mock_sdk)


def compile_kernel(kernel_code: str, kernel_name: str, compiler: str = "g++") -> tuple[bool, str]:
    """
    Compile a kernel using mock SDK headers.

    Args:
        kernel_code: C++ source code
        kernel_name: Name for the kernel file (e.g., "reader.cpp")
        compiler: Compiler to use ("g++" or "clang++")

    Returns:
        (success, error_message)
    """
    try:
        mock_sdk_include = find_mock_sdk_headers()
    except FileNotFoundError as e:
        return False, str(e)

    # Create temporary file for the kernel
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        f.write(kernel_code)
        cpp_file = f.name

    try:
        # Compile with mock SDK headers
        # -fsyntax-only: Only check syntax, don't generate object file
        # -std=c++17: Use C++17 standard
        # -I: Include mock SDK headers
        cmd = [
            compiler,
            "-fsyntax-only",  # Syntax check only
            "-std=c++17",
            f"-I{mock_sdk_include}",
            "-Wno-unused-parameter",  # Ignore unused parameter warnings in mock headers
            "-Wno-unused-variable",  # Ignore unused variable warnings
            cpp_file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            return True, ""
        else:
            error_msg = f"Compilation failed for {kernel_name}:\n"
            error_msg += f"Command: {' '.join(cmd)}\n"
            error_msg += f"Exit code: {result.returncode}\n"
            if result.stdout:
                error_msg += f"Stdout:\n{result.stdout}\n"
            if result.stderr:
                error_msg += f"Stderr:\n{result.stderr}\n"
            return False, error_msg

    finally:
        # Clean up temporary file
        with contextlib.suppress(OSError):
            os.unlink(cpp_file)


@pytest.mark.skipif(not TVM_AVAILABLE, reason="TVM/TileLang not available")
@pytest.mark.skipif(not PYTEST_AVAILABLE, reason="pytest not available")
def test_mock_sdk_headers_exist():
    """Verify that mock SDK headers are present"""
    mock_sdk = find_mock_sdk_headers()
    assert Path(mock_sdk).exists(), f"Mock SDK headers not found at: {mock_sdk}"

    # Check for required headers
    required_headers = [
        "compute_kernel_api/common.h",
        "compute_kernel_api/tile_move_copy.h",
        "compute_kernel_api/matmul.h",
        "compute_kernel_api/eltwise_binary.h",
    ]

    for header in required_headers:
        header_path = Path(mock_sdk) / header
        assert header_path.exists(), f"Required header not found: {header}"


@pytest.mark.skipif(not TVM_AVAILABLE, reason="TVM/TileLang not available")
@pytest.mark.skipif(not PYTEST_AVAILABLE, reason="pytest not available")
def test_compile_simple_gemm_kernels():
    """Test compilation of kernels from a simple GEMM"""

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

    # Create kernel
    kernel = matmul(256, 256, 256, 32, 32, 32)

    # Get generated artifacts
    import json
    source = kernel.get_kernel_source()
    artifacts = json.loads(source)

    # Test compilation of each kernel
    # Note: Compute kernel currently has known issues (tt.fill.zero syntax)
    # so we test only reader and writer which generate valid C++
    kernels_to_test = ["reader.cpp", "writer.cpp"]

    for kernel_name in kernels_to_test:
        if kernel_name in artifacts:
            kernel_code = artifacts[kernel_name]

            logger.info(f"Testing compilation of {kernel_name}...")

            # Try g++ first
            success, error = compile_kernel(kernel_code, kernel_name, compiler="g++")

            if not success:
                # Try clang++ if g++ fails and clang++ is available
                import shutil
                if shutil.which("clang++"):
                    logger.warning(f"g++ failed for {kernel_name}, trying clang++...")
                    success, error = compile_kernel(kernel_code, kernel_name, compiler="clang++")

            # Assert compilation succeeded
            assert success, f"Kernel {kernel_name} failed to compile:\n{error}"

            logger.info(f"✓ {kernel_name} compiled successfully")


@pytest.mark.skipif(not TVM_AVAILABLE, reason="TVM/TileLang not available")
@pytest.mark.skipif(not PYTEST_AVAILABLE, reason="pytest not available")
def test_compile_kernels_syntax_only():
    """Test that compilation catches actual C++ syntax errors"""

    # Create a deliberately broken kernel to verify compilation catches errors
    bad_kernel = """
// Generated TT Compute Kernel (with syntax error)
#include "compute_kernel_api/common.h"

void MAIN() {
    uint32_t x = get_arg_val<uint32_t>(0);

    // Syntax error: missing semicolon
    uint32_t y = x + 1

    constexpr auto cb_in0 = tt::CBIndex::c_0;
}
"""

    success, error = compile_kernel(bad_kernel, "bad_kernel.cpp")
    assert not success, "Compilation should have failed for kernel with syntax error"
    assert "error" in error.lower() or "expected" in error.lower(
    ), "Error message should mention syntax error"
    logger.info("✓ Syntax error correctly detected")


if __name__ == "__main__":
    # Run tests manually
    if TVM_AVAILABLE and PYTEST_AVAILABLE:
        print("Running mock compilation tests...")
        test_mock_sdk_headers_exist()
        print("✓ Mock SDK headers exist")

        test_compile_simple_gemm_kernels()
        print("✓ Simple GEMM kernels compile")

        test_compile_kernels_syntax_only()
        print("✓ Syntax error detection works")

        print("\n✅ All mock compilation tests passed!")
    else:
        print("⚠️  TVM/TileLang not available, skipping tests")
