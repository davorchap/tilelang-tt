"""Test TilePadTT pass (WS3 Phase 2).

This pass computes padding metadata for non-tile-aligned buffers.
"""

import pytest
import tvm
from tvm import tir


def create_func_with_padding_metadata(needs_padding=True):
    """Create a mock PrimFunc with WS2 padding metadata."""
    # Create buffers - 250×250 if padding needed, 256×256 if aligned
    if needs_padding:
        A = tir.decl_buffer((250, 250), "float16", name="A", scope="global")
    else:
        A = tir.decl_buffer((256, 256), "float16", name="A", scope="global")

    B = tir.decl_buffer((256, 256), "float16", name="B", scope="global")
    C = tir.decl_buffer((256, 256), "float16", name="C", scope="global")

    body = tir.Evaluate(0)
    func = tir.PrimFunc([A, B, C], body)

    # Add TT defaults
    func = func.with_attr("tt_schedule_policy", "contiguous")

    # Add WS2 padding metadata
    func = func.with_attr("tt_buffer_A_needs_padding",
                          tvm.tir.IntImm("int32", 1 if needs_padding else 0))
    func = func.with_attr(
        "tt_buffer_A_tile_shape",
        [tvm.tir.IntImm("int32", 32), tvm.tir.IntImm("int32", 32)])

    func = func.with_attr("tt_buffer_B_needs_padding", tvm.tir.IntImm("int32", 0))
    func = func.with_attr(
        "tt_buffer_B_tile_shape",
        [tvm.tir.IntImm("int32", 32), tvm.tir.IntImm("int32", 32)])

    func = func.with_attr("tt_buffer_C_needs_padding", tvm.tir.IntImm("int32", 0))
    func = func.with_attr(
        "tt_buffer_C_tile_shape",
        [tvm.tir.IntImm("int32", 32), tvm.tir.IntImm("int32", 32)])

    return func


def test_tile_pad_tt_basic():
    """Test TilePadTT generates padding metadata."""
    from tilelang.tt.passes import tile_pad_tt

    func = create_func_with_padding_metadata(needs_padding=True)
    mod = tvm.IRModule({"main": func})

    # Apply TilePadTT
    mod = tile_pad_tt(mod)
    func = mod["main"]

    # Verify padding metadata attached
    assert func.attrs is not None, "Function should have attributes"
    assert "tt_padding_info" in func.attrs, "Should have tt_padding_info attribute"


def test_tile_pad_tt_padded_dimensions():
    """Test TilePadTT computes correct padded dimensions."""
    from tilelang.tt.passes import tile_pad_tt

    func = create_func_with_padding_metadata(needs_padding=True)
    mod = tvm.IRModule({"main": func})

    mod = tile_pad_tt(mod)
    func = mod["main"]

    padding_info = func.attrs["tt_padding_info"]

    # Check buffer A (250×250 → 256×256)
    assert "A" in padding_info, "Should have padding info for buffer A"

    a_info = padding_info["A"]
    assert "needs_padding" in a_info, "Should have needs_padding flag"
    assert bool(a_info["needs_padding"]), "Buffer A should need padding"

    assert "padded_shape" in a_info, "Should have padded_shape"
    padded_shape = a_info["padded_shape"]
    assert len(padded_shape) == 2, "Padded shape should be 2D"
    assert int(padded_shape[0]) == 256, f"Expected padded height 256, got {int(padded_shape[0])}"
    assert int(padded_shape[1]) == 256, f"Expected padded width 256, got {int(padded_shape[1])}"


def test_tile_pad_tt_padding_amount():
    """Test TilePadTT calculates correct padding amounts."""
    from tilelang.tt.passes import tile_pad_tt

    func = create_func_with_padding_metadata(needs_padding=True)
    mod = tvm.IRModule({"main": func})

    mod = tile_pad_tt(mod)
    func = mod["main"]

    padding_info = func.attrs["tt_padding_info"]
    a_info = padding_info["A"]

    assert "padding_amount" in a_info, "Should have padding_amount"
    padding_amount = a_info["padding_amount"]

    # For 250×250 → 256×256, padding is 6 per dimension
    assert len(padding_amount) == 2, "Padding amount should be 2D"
    assert int(
        padding_amount[0]) == 6, f"Expected padding 6 for height, got {int(padding_amount[0])}"
    assert int(
        padding_amount[1]) == 6, f"Expected padding 6 for width, got {int(padding_amount[1])}"


def test_tile_pad_tt_original_shape():
    """Test TilePadTT preserves original shape."""
    from tilelang.tt.passes import tile_pad_tt

    func = create_func_with_padding_metadata(needs_padding=True)
    mod = tvm.IRModule({"main": func})

    mod = tile_pad_tt(mod)
    func = mod["main"]

    padding_info = func.attrs["tt_padding_info"]
    a_info = padding_info["A"]

    assert "original_shape" in a_info, "Should have original_shape"
    original_shape = a_info["original_shape"]

    assert len(original_shape) == 2, "Original shape should be 2D"
    assert int(
        original_shape[0]) == 250, f"Expected original height 250, got {int(original_shape[0])}"
    assert int(
        original_shape[1]) == 250, f"Expected original width 250, got {int(original_shape[1])}"


def test_tile_pad_tt_skip_aligned_buffers():
    """Test TilePadTT skips already aligned buffers."""
    from tilelang.tt.passes import tile_pad_tt

    func = create_func_with_padding_metadata(needs_padding=False)
    mod = tvm.IRModule({"main": func})

    mod = tile_pad_tt(mod)
    func = mod["main"]

    # Should not add padding_info if no buffers need padding
    if "tt_padding_info" in func.attrs:
        padding_info = func.attrs["tt_padding_info"]
        # Buffer A is aligned (256×256), should not be in padding_info
        assert "A" not in padding_info, "Aligned buffer A should not have padding info"


def test_tile_pad_tt_skip_non_tt_functions():
    """Test TilePadTT skips functions without TT attributes."""
    from tilelang.tt.passes import tile_pad_tt

    # Create function WITHOUT TT attributes
    A = tir.decl_buffer((250, 250), "float16", name="A")
    body = tir.Evaluate(0)
    func = tir.PrimFunc([A], body)

    mod = tvm.IRModule({"main": func})

    # Apply pass
    mod = tile_pad_tt(mod)
    func = mod["main"]

    # Should NOT add padding metadata
    assert func.attrs is None or "tt_padding_info" not in func.attrs, "Should not add padding without TT attributes"


def test_tile_pad_tt_multiple_buffers():
    """Test TilePadTT handles multiple buffers with different padding needs."""
    from tilelang.tt.passes import tile_pad_tt

    # Create buffers with different padding needs
    A = tir.decl_buffer((250, 250), "float16", name="A", scope="global")  # Needs padding
    B = tir.decl_buffer((100, 100), "float16", name="B", scope="global")  # Needs padding
    C = tir.decl_buffer((256, 256), "float16", name="C", scope="global")  # Aligned

    body = tir.Evaluate(0)
    func = tir.PrimFunc([A, B, C], body)

    # Add TT defaults
    func = func.with_attr("tt_schedule_policy", "contiguous")

    # Add WS2 metadata
    func = func.with_attr("tt_buffer_A_needs_padding", tvm.tir.IntImm("int32", 1))
    func = func.with_attr(
        "tt_buffer_A_tile_shape",
        [tvm.tir.IntImm("int32", 32), tvm.tir.IntImm("int32", 32)])

    func = func.with_attr("tt_buffer_B_needs_padding", tvm.tir.IntImm("int32", 1))
    func = func.with_attr(
        "tt_buffer_B_tile_shape",
        [tvm.tir.IntImm("int32", 32), tvm.tir.IntImm("int32", 32)])

    func = func.with_attr("tt_buffer_C_needs_padding", tvm.tir.IntImm("int32", 0))
    func = func.with_attr(
        "tt_buffer_C_tile_shape",
        [tvm.tir.IntImm("int32", 32), tvm.tir.IntImm("int32", 32)])

    mod = tvm.IRModule({"main": func})

    mod = tile_pad_tt(mod)
    func = mod["main"]

    padding_info = func.attrs["tt_padding_info"]

    # Should have info for A and B, not C
    assert "A" in padding_info, "Should have padding info for buffer A"
    assert "B" in padding_info, "Should have padding info for buffer B"
    assert "C" not in padding_info, "Aligned buffer C should not have padding info"

    # Check A: 250×250 → 256×256
    a_padded = padding_info["A"]["padded_shape"]
    assert int(a_padded[0]) == 256 and int(a_padded[1]) == 256

    # Check B: 100×100 → 128×128 (4 tiles per dimension)
    b_padded = padding_info["B"]["padded_shape"]
    assert int(b_padded[0]) == 128 and int(b_padded[1]) == 128


def test_tile_pad_tt_integration_with_ws1_ws2():
    """Test TilePadTT integrates with full WS1→WS2→TilePadTT pipeline."""
    from tilelang.tt.passes import apply_ws2_passes, tile_pad_tt
    from tilelang.tt.target import apply_tt_defaults

    # Create function with non-aligned buffer
    A = tir.decl_buffer((250, 250), "float16", name="A", scope="global")
    B = tir.decl_buffer((256, 256), "float16", name="B", scope="global")
    C = tir.decl_buffer((256, 256), "float16", name="C", scope="global")

    body = tir.Evaluate(0)
    func = tir.PrimFunc([A, B, C], body)

    # Add grid metadata (normally from TileLang frontend)
    func = func.with_attr("tl.grid_x", tvm.tir.IntImm("int32", 8))
    func = func.with_attr("tl.grid_y", tvm.tir.IntImm("int32", 8))

    mod = tvm.IRModule({"main": func})

    # Apply WS1 → WS2 → TilePadTT
    mod = apply_tt_defaults(mod)
    mod = apply_ws2_passes(mod)
    mod = tile_pad_tt(mod)

    func = mod["main"]

    # Verify all metadata exists
    assert "tt_schedule_policy" in func.attrs, "Should have WS1 defaults"
    assert "tt_buffer_A_needs_padding" in func.attrs, "Should have WS2 padding detection"
    assert "tt_padding_info" in func.attrs, "Should have TilePadTT output"


if __name__ == "__main__":
    # Run tests
    test_tile_pad_tt_basic()
    test_tile_pad_tt_padded_dimensions()
    test_tile_pad_tt_padding_amount()
    test_tile_pad_tt_original_shape()
    test_tile_pad_tt_skip_aligned_buffers()
    test_tile_pad_tt_skip_non_tt_functions()
    test_tile_pad_tt_multiple_buffers()
    test_tile_pad_tt_integration_with_ws1_ws2()
    print("All TilePadTT tests passed!")
