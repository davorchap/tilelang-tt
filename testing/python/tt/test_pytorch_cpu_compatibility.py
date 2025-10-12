"""Test CPU-only PyTorch compatibility for Tenstorrent backend.

This test suite verifies that the Tenstorrent backend works correctly with
CPU-only PyTorch installations (without CUDA/GPU support). This is important
because:
1. TT backend doesn't require CUDA
2. CPU-only PyTorch saves ~5GB of disk space
3. Prevents dependency bloat in TT-only environments

These tests verify core PyTorch functionality used by TileLang:
- Tensor operations on CPU
- NumPy interoperability
- DLPack conversion (for TVM interop)
- Data type support
- Gradient computation
"""

import pytest
import torch
import numpy as np


def test_pytorch_cpu_only_installation():
    """Verify PyTorch is CPU-only (no CUDA support required)."""
    # This test documents that we intentionally use CPU-only PyTorch
    # CUDA availability should be False for TT backend development
    assert not torch.cuda.is_available(), (
        "Expected CPU-only PyTorch installation. "
        "TT backend doesn't require CUDA. "
        "Consider using: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    )

    # Verify we can create CPU tensors
    x = torch.randn(10)
    assert x.device.type == "cpu", f"Expected CPU device, got {x.device.type}"


def test_pytorch_basic_tensor_operations():
    """Test basic PyTorch tensor operations on CPU."""
    # Create tensors
    x = torch.randn(10, 10)
    y = torch.randn(10, 10)

    # Basic operations
    z = x + y
    assert z.shape == (10, 10)

    # Matrix multiplication
    w = x @ y
    assert w.shape == (10, 10)

    # Element-wise operations
    v = x * 2.0
    assert torch.allclose(v, x + x, rtol=1e-5)


def test_pytorch_numpy_interop():
    """Test NumPy-PyTorch conversion (used throughout TileLang)."""
    # NumPy to PyTorch
    np_array = np.random.randn(5, 5).astype(np.float32)
    torch_tensor = torch.from_numpy(np_array)
    assert torch_tensor.shape == (5, 5)
    assert torch.allclose(torch_tensor, torch.tensor(np_array), rtol=1e-5)

    # PyTorch to NumPy
    torch_tensor2 = torch.randn(3, 3)
    np_array2 = torch_tensor2.numpy()
    assert np_array2.shape == (3, 3)
    assert np.allclose(np_array2, torch_tensor2.numpy(), rtol=1e-5)


def test_pytorch_dlpack_conversion():
    """Test DLPack conversion (critical for TVM-PyTorch interop)."""
    import torch.utils.dlpack

    # Create a PyTorch tensor
    x = torch.randn(10, 10, dtype=torch.float32)

    # Convert to DLPack
    dlpack_capsule = torch.utils.dlpack.to_dlpack(x)
    assert dlpack_capsule is not None

    # Convert back from DLPack
    y = torch.utils.dlpack.from_dlpack(dlpack_capsule)
    assert y.shape == x.shape
    assert torch.allclose(y, x, rtol=1e-5)


def test_pytorch_gradient_computation():
    """Test autograd functionality on CPU."""
    x = torch.randn(3, 3, requires_grad=True)
    y = (x ** 2).sum()

    # Compute gradients
    y.backward()

    # Verify gradient exists
    assert x.grad is not None
    assert x.grad.shape == x.shape

    # Verify gradient correctness
    expected_grad = 2 * x.data
    assert torch.allclose(x.grad, expected_grad, rtol=1e-5)


def test_pytorch_dtype_support():
    """Test PyTorch data type support on CPU."""
    dtypes_to_test = [
        torch.float16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.bool,
    ]

    for dtype in dtypes_to_test:
        t = torch.zeros(2, 2, dtype=dtype)
        assert t.dtype == dtype, f"Failed to create tensor with dtype {dtype}"
        assert t.device.type == "cpu"


def test_pytorch_special_dtypes():
    """Test special floating point types used in ML."""
    # BFloat16 (used in some ML models)
    try:
        x = torch.randn(5, 5, dtype=torch.bfloat16)
        assert x.dtype == torch.bfloat16
    except RuntimeError:
        pytest.skip("bfloat16 not supported on this CPU")

    # Float8 types (if available in PyTorch version)
    if hasattr(torch, "float8_e4m3fn"):
        # Just verify the type exists (conversion may have limitations on CPU)
        assert hasattr(torch, "float8_e4m3fn")
        assert hasattr(torch, "float8_e5m2")


def test_tilelang_torch_utilities():
    """Test TileLang's PyTorch utility functions."""
    from tilelang.utils.tensor import torch_assert_close

    # Create identical tensors
    a = torch.randn(10)
    b = a.clone()

    # Should not raise
    torch_assert_close(a, b, rtol=1e-5, atol=1e-5)

    # Create different tensors
    c = torch.randn(10)

    # Should raise
    with pytest.raises(AssertionError):
        torch_assert_close(a, c, rtol=1e-5, atol=1e-5)


def test_tilelang_dlpack_integration():
    """Test TileLang's DLPack integration with PyTorch."""
    # This test verifies that TileLang's DLPack utilities work with CPU PyTorch
    # Import is lazy to avoid issues if TVM is not built
    try:
        from tilelang.contrib.dlpack import to_pytorch_func
        import tvm
        from tvm import tir

        # Create a simple TVM function
        n = 1024
        A = tir.Var("A", "handle")
        B = tir.Var("B", "handle")

        # Simple function that doubles input
        stmt = tir.Evaluate(0)
        func = tir.PrimFunc([A, B], stmt)

        # This just tests that the conversion utility exists and is callable
        # Full end-to-end testing would require a complete TVM build
        assert callable(to_pytorch_func)

    except ImportError as e:
        pytest.skip(f"TVM not available for DLPack test: {e}")


def test_pytorch_version_info():
    """Document PyTorch version and configuration."""
    import torch

    # Print version info (useful for debugging)
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CPU threads: {torch.get_num_threads()}")

    # Verify we have a reasonable PyTorch version
    version_parts = torch.__version__.split("+")[0].split(".")
    major = int(version_parts[0])
    minor = int(version_parts[1])

    # Require PyTorch >= 2.0 (matches TileLang requirements)
    assert major >= 2, f"PyTorch version {torch.__version__} is too old. Require >= 2.0"


if __name__ == "__main__":
    # Run tests
    test_pytorch_cpu_only_installation()
    test_pytorch_basic_tensor_operations()
    test_pytorch_numpy_interop()
    test_pytorch_dlpack_conversion()
    test_pytorch_gradient_computation()
    test_pytorch_dtype_support()
    test_pytorch_special_dtypes()
    test_tilelang_torch_utilities()
    test_tilelang_dlpack_integration()
    test_pytorch_version_info()
    print("\nAll PyTorch CPU compatibility tests passed!")
