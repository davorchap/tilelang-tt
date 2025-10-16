"""Test tile padding (now integrated into TTTilesToCoreMap).

NOTE: This test uses the old pass API (TTTilesToCoreMap) which is being deprecated
in favor of the v5 pipeline. This test is skipped as it tests legacy behavior.
"""

import pytest

# Skip the entire module - tests old pass API being deprecated
pytestmark = pytest.mark.skip(
    reason="Tests legacy pass API (TTTilesToCoreMap) - use test_v5_metadata_passes.py instead")

# Import tilelang first to get proper TVM
from tilelang import tvm
from tvm import tir
from tilelang.tenstorrent.passes import TTTilesToCoreMap


def test_tile_pad_integrated():
    """Tile padding is now integrated into TTTilesToCoreMap."""
    # Padding to tile boundaries happens automatically in TTTilesToCoreMap

    A = tir.decl_buffer((100, 100), "float16", name="A")  # Non-tile-aligned
    func = tir.PrimFunc([A], tir.Evaluate(0))
    func = func.with_attr("tt.core_grid", [8, 8])

    mod = tvm.IRModule({"main": func})

    # Padding happens here
    mod = TTTilesToCoreMap()(mod)

    func = mod["main"]
    # Should have work partition with padded dimensions
    assert "tt.work_partition" in func.attrs
    print("Tile padding integrated into TTTilesToCoreMap")


if __name__ == "__main__":
    test_tile_pad_integrated()
    print("Test passed!")
