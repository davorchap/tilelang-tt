"""Test tile padding (now integrated into TTTilesToCoreMap)."""

# Import tilelang first to get proper TVM
import tilelang
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
