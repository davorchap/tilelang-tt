"""Test memory space lowering (now integrated into GridToPersistentTT).

NOTE: This test uses the old pass API (GridToPersistentTT) which is being deprecated
in favor of the v5 pipeline. This test is skipped as it tests legacy behavior.
"""

import pytest

# Skip the entire module - tests old pass API being deprecated
pytestmark = pytest.mark.skip(
    reason="Tests legacy pass API (GridToPersistentTT) - use test_v5_passes.py instead")

# Import tilelang first to get proper TVM
from tilelang import tvm
from tvm import tir
from tilelang.tenstorrent.passes import GridToPersistentTT


def test_memory_space_lower_integrated():
    """Memory space lowering is now integrated into GridToPersistentTT."""
    # This functionality is now part of the final lowering pass
    # GridToPersistentTT handles circular buffer allocation

    A = tir.decl_buffer((256, 256), "float16", name="A")
    func = tir.PrimFunc([A], tir.Evaluate(0))
    func = func.with_attr("tt.core_grid", [8, 8])
    func = func.with_attr("tt.work_partition", {})

    mod = tvm.IRModule({"main": func})

    # Memory space lowering happens here
    mod = GridToPersistentTT()(mod)

    # The pass marks the function as transformed
    func = mod["main"]
    assert "tt.persistent_kernel" in func.attrs
    print("Memory space lowering integrated into GridToPersistentTT")


if __name__ == "__main__":
    test_memory_space_lower_integrated()
    print("Test passed!")
