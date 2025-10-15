"""
Test that all Tenstorrent examples run without errors.

This test module imports and executes all TT backend examples
to ensure they compile and run without raising exceptions.
"""

import sys
import os
import subprocess
import importlib.util

import pytest

# Add examples directory to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
EXAMPLES_DIR = os.path.join(REPO_ROOT, 'examples', 'tenstorrent')
sys.path.insert(0, REPO_ROOT)


def import_and_run_module(module_path, module_name):
    """Import and run a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    # Execute the module
    spec.loader.exec_module(module)
    return module


def test_example_gemm_tt():
    """Test that example_gemm_tt.py runs without errors."""
    example_path = os.path.join(EXAMPLES_DIR, 'example_gemm_tt.py')
    if not os.path.exists(example_path):
        pytest.skip(f"Example file not found: {example_path}")

    # Run as subprocess to isolate execution
    result = subprocess.run([sys.executable, example_path],
                            capture_output=True,
                            text=True,
                            cwd=REPO_ROOT,
                            env=os.environ.copy())

    # Check for success
    assert result.returncode == 0, f"example_gemm_tt.py failed:\n{result.stderr}"

    # Verify expected output
    assert "TT artifacts generated successfully" in result.stdout or \
           "artifacts" in result.stdout.lower(), \
           "Expected artifact generation message not found in output"


def test_example_gemm_tt_minimal():
    """Test that example_gemm_tt_minimal.py runs without errors."""
    example_path = os.path.join(EXAMPLES_DIR, 'example_gemm_tt_minimal.py')
    if not os.path.exists(example_path):
        pytest.skip(f"Example file not found: {example_path}")

    # Run as subprocess
    result = subprocess.run([sys.executable, example_path],
                            capture_output=True,
                            text=True,
                            cwd=REPO_ROOT,
                            env=os.environ.copy())

    # Check for success
    assert result.returncode == 0, f"example_gemm_tt_minimal.py failed:\n{result.stderr}"

    # Verify expected output markers - be flexible about output format
    # Check that TT artifacts are mentioned in the output
    assert any(
        artifact in result.stdout
        for artifact in ["reader.cpp", "compute.cpp", "writer.cpp", "main.cpp", "tt.plan.json"
                        ]), "Expected TT artifacts not mentioned in output"


def test_run_gemm_with_tt_backend():
    """Test that run_gemm_with_tt_backend.py runs without errors."""
    example_path = os.path.join(EXAMPLES_DIR, 'run_gemm_with_tt_backend.py')
    if not os.path.exists(example_path):
        pytest.skip(f"Example file not found: {example_path}")

    # Run the example (it demonstrates multiple approaches)
    result = subprocess.run(
        [sys.executable, example_path],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=30,  # 30 second timeout
        env=os.environ.copy())

    # Check for success
    assert result.returncode == 0, f"run_gemm_with_tt_backend.py failed:\n{result.stderr}"

    # Verify output contains key sections from the demonstration
    assert "Running Original GEMM Example with TT Backend" in result.stdout, "Original example section not found"
    assert "TT-Optimized GEMM" in result.stdout, "TT-optimized section not found"
    assert "Generated" in result.stdout and "TT artifacts" in result.stdout, "Artifacts mention not found"


def test_new_pipeline_example():
    """Test that new_pipeline_example.py runs without errors."""
    example_path = os.path.join(EXAMPLES_DIR, 'new_pipeline_example.py')
    if not os.path.exists(example_path):
        pytest.skip(f"Example file not found: {example_path}")

    # Run as subprocess
    result = subprocess.run([sys.executable, example_path],
                            capture_output=True,
                            text=True,
                            cwd=REPO_ROOT,
                            timeout=30,
                            env=os.environ.copy())

    # Check for success (allow non-zero exit if it's just missing SDK)
    if "TT_METAL_HOME" in result.stderr or "SDK" in result.stderr:
        pytest.skip("TT-Metalium SDK not available, skipping hardware example")

    assert result.returncode == 0, f"new_pipeline_example.py failed:\n{result.stderr}"


def test_example_gemm_basic():
    """Test the basic example_gemm.py if it exists."""
    example_path = os.path.join(EXAMPLES_DIR, 'example_gemm.py')
    if not os.path.exists(example_path):
        pytest.skip(f"Example file not found: {example_path}")

    # Run as subprocess
    result = subprocess.run([sys.executable, example_path],
                            capture_output=True,
                            text=True,
                            cwd=REPO_ROOT,
                            timeout=30,
                            env=os.environ.copy())

    # Check for success
    assert result.returncode == 0, f"example_gemm.py failed:\n{result.stderr}"


def test_all_examples_have_docstrings():
    """Verify all example files have proper documentation."""
    example_files = [
        f for f in os.listdir(EXAMPLES_DIR) if f.endswith('.py') and f.startswith('example_')
    ]

    for filename in example_files:
        filepath = os.path.join(EXAMPLES_DIR, filename)
        with open(filepath, 'r') as f:
            content = f.read()

        # Check for docstring or header comment
        assert '"""' in content or "'''" in content or content.startswith('#'), \
            f"{filename} lacks documentation (docstring or header comment)"


def test_examples_import_tilelang():
    """Verify all examples can import tilelang module."""
    example_files = [
        f for f in os.listdir(EXAMPLES_DIR) if f.endswith('.py') and not f.startswith('_')
    ]

    for filename in example_files:
        filepath = os.path.join(EXAMPLES_DIR, filename)
        with open(filepath, 'r') as f:
            content = f.read()

        # Check that tilelang is imported
        assert 'import tilelang' in content or 'from tilelang' in content, \
            f"{filename} doesn't import tilelang"


def test_examples_use_tenstorrent_target():
    """Verify examples use the Tenstorrent target."""
    example_files = [
        f for f in os.listdir(EXAMPLES_DIR) if f.endswith('.py') and 'gemm' in f.lower()
    ]

    for filename in example_files:
        filepath = os.path.join(EXAMPLES_DIR, filename)
        with open(filepath, 'r') as f:
            content = f.read()

        # Check for TT target usage
        assert 'TENSTORRENT_TARGET' in content or 'tenstorrent' in content.lower(), \
            f"{filename} doesn't appear to use Tenstorrent target"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
