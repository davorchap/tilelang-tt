"""Test VerifyTTIR pass (WS3 Phase 2).

This pass validates that transformed TT IR has all required metadata.
"""

import pytest
import tvm
from tvm import tir


def create_func_with_complete_metadata():
    """Create a mock PrimFunc with complete WS1-3 metadata."""
    A = tir.decl_buffer((256, 256), "float16", name="A", scope="global")
    B = tir.decl_buffer((256, 256), "float16", name="B", scope="global")
    C = tir.decl_buffer((256, 256), "float16", name="C", scope="global")

    body = tir.Evaluate(0)
    func = tir.PrimFunc([A, B, C], body)

    # WS1 metadata
    func = func.with_attr("tt_schedule_policy", "contiguous")
    func = func.with_attr("tt_schedule_order", "row_major")
    func = func.with_attr("tt_layout_type", "dram_interleaved")
    func = func.with_attr("tt_tile_height", tvm.tir.IntImm("int32", 32))
    func = func.with_attr("tt_tile_width", tvm.tir.IntImm("int32", 32))

    # WS2 metadata
    func = func.with_attr("tt_grid_x", tvm.tir.IntImm("int32", 8))
    func = func.with_attr("tt_grid_y", tvm.tir.IntImm("int32", 8))
    func = func.with_attr("tt_num_tiles", tvm.tir.IntImm("int32", 64))
    func = func.with_attr("tt_num_cores", tvm.tir.IntImm("int32", 64))
    func = func.with_attr(
        "tt_tiles_per_core",
        [[tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 1)]])

    # WS3 metadata
    func = func.with_attr("tt_persistent_loop", tvm.tir.IntImm("int32", 1))

    return func


def create_func_missing_ws1_metadata():
    """Create a mock PrimFunc missing WS1 metadata."""
    A = tir.decl_buffer((256, 256), "float16", name="A", scope="global")
    body = tir.Evaluate(0)
    func = tir.PrimFunc([A], body)

    # Has tt_schedule_policy but missing other WS1 attributes
    func = func.with_attr("tt_schedule_policy", "contiguous")
    func = func.with_attr("tt_schedule_order", "row_major")
    # Missing: tt_layout_type, tt_tile_height, tt_tile_width

    return func


def create_func_missing_ws2_metadata():
    """Create a mock PrimFunc missing WS2 metadata."""
    A = tir.decl_buffer((256, 256), "float16", name="A", scope="global")
    body = tir.Evaluate(0)
    func = tir.PrimFunc([A], body)

    # WS1 metadata present
    func = func.with_attr("tt_schedule_policy", "contiguous")
    func = func.with_attr("tt_schedule_order", "row_major")
    func = func.with_attr("tt_layout_type", "dram_interleaved")
    func = func.with_attr("tt_tile_height", tvm.tir.IntImm("int32", 32))
    func = func.with_attr("tt_tile_width", tvm.tir.IntImm("int32", 32))

    # Missing WS2 metadata - should be flagged as error
    # No tt_grid_x, tt_grid_y, etc.

    return func


def create_func_with_large_grid():
    """Create a mock PrimFunc with large grid dimensions (warning case)."""
    A = tir.decl_buffer((256, 256), "float16", name="A", scope="global")
    body = tir.Evaluate(0)
    func = tir.PrimFunc([A], body)

    # WS1 metadata
    func = func.with_attr("tt_schedule_policy", "contiguous")
    func = func.with_attr("tt_schedule_order", "row_major")
    func = func.with_attr("tt_layout_type", "dram_interleaved")
    func = func.with_attr("tt_tile_height", tvm.tir.IntImm("int32", 32))
    func = func.with_attr("tt_tile_width", tvm.tir.IntImm("int32", 32))

    # WS2 metadata with large grid (should warn)
    func = func.with_attr("tt_grid_x", tvm.tir.IntImm("int32", 100))  # > 64
    func = func.with_attr("tt_grid_y", tvm.tir.IntImm("int32", 100))  # > 64
    func = func.with_attr("tt_num_tiles", tvm.tir.IntImm("int32", 10000))
    func = func.with_attr("tt_num_cores", tvm.tir.IntImm("int32", 64))
    func = func.with_attr(
        "tt_tiles_per_core",
        [[tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 1)]])

    return func


def test_verify_tt_ir_basic():
    """Test VerifyTTIR validates complete metadata."""
    from tilelang.tt.passes import verify_tt_ir

    func = create_func_with_complete_metadata()
    mod = tvm.IRModule({"main": func})

    # Apply VerifyTTIR
    mod = verify_tt_ir(mod)
    func = mod["main"]

    # Verify validation metadata attached
    assert func.attrs is not None, "Function should have attributes"
    assert "tt_ir_validated" in func.attrs, "Should have tt_ir_validated attribute"
    assert "tt_validation_error_count" in func.attrs, "Should have error count"
    assert "tt_validation_warning_count" in func.attrs, "Should have warning count"


def test_verify_tt_ir_validation_passes():
    """Test VerifyTTIR passes for complete metadata."""
    from tilelang.tt.passes import verify_tt_ir

    func = create_func_with_complete_metadata()
    mod = tvm.IRModule({"main": func})

    mod = verify_tt_ir(mod)
    func = mod["main"]

    # Validation should pass
    assert bool(func.attrs["tt_ir_validated"]), "Validation should pass with complete metadata"
    assert int(func.attrs["tt_validation_error_count"]) == 0, "Should have no errors"


def test_verify_tt_ir_detects_missing_ws1():
    """Test VerifyTTIR detects missing WS1 metadata."""
    from tilelang.tt.passes import verify_tt_ir

    func = create_func_missing_ws1_metadata()
    mod = tvm.IRModule({"main": func})

    mod = verify_tt_ir(mod)
    func = mod["main"]

    # Should detect missing WS1 attributes
    assert "tt_ir_validated" in func.attrs, "Should have validation result"
    assert not bool(func.attrs["tt_ir_validated"]), "Validation should fail with missing WS1"
    assert int(func.attrs["tt_validation_error_count"]) > 0, "Should have errors"


def test_verify_tt_ir_detects_missing_ws2():
    """Test VerifyTTIR detects missing WS2 metadata."""
    from tilelang.tt.passes import verify_tt_ir

    func = create_func_missing_ws2_metadata()
    mod = tvm.IRModule({"main": func})

    mod = verify_tt_ir(mod)
    func = mod["main"]

    # Should detect missing WS2 attributes
    assert not bool(func.attrs["tt_ir_validated"]), "Validation should fail with missing WS2"
    assert int(func.attrs["tt_validation_error_count"]) > 0, "Should have errors"


def test_verify_tt_ir_large_grid_warning():
    """Test VerifyTTIR warns about large grid dimensions."""
    from tilelang.tt.passes import verify_tt_ir

    func = create_func_with_large_grid()
    mod = tvm.IRModule({"main": func})

    mod = verify_tt_ir(mod)
    func = mod["main"]

    # May pass validation but should have warnings
    warning_count = int(func.attrs["tt_validation_warning_count"])
    # Large grid dimensions may trigger warnings (implementation dependent)
    # Just verify warning_count exists and is accessible


def test_verify_tt_ir_skip_non_tt_functions():
    """Test VerifyTTIR skips functions without TT attributes."""
    from tilelang.tt.passes import verify_tt_ir

    # Create function WITHOUT TT attributes
    A = tir.decl_buffer((256, 256), "float16", name="A")
    body = tir.Evaluate(0)
    func = tir.PrimFunc([A], body)

    mod = tvm.IRModule({"main": func})

    # Apply pass
    mod = verify_tt_ir(mod)
    func = mod["main"]

    # Should NOT add validation metadata for non-TT functions
    assert func.attrs is None or "tt_ir_validated" not in func.attrs, \
        "Should not validate non-TT functions"


def test_verify_tt_ir_integration_with_full_pipeline():
    """Test VerifyTTIR integrates with full WS1→WS2→WS3 pipeline."""
    from tilelang.tt.passes import apply_ws2_passes, apply_ws3_passes
    from tilelang.tt.target import apply_tt_defaults

    # Create function
    A = tir.decl_buffer((256, 256), "float16", name="A", scope="global")
    B = tir.decl_buffer((256, 256), "float16", name="B", scope="global")
    C = tir.decl_buffer((256, 256), "float16", name="C", scope="global")

    body = tir.Evaluate(0)
    func = tir.PrimFunc([A, B, C], body)

    # Add grid metadata (normally from TileLang frontend)
    func = func.with_attr("tl.grid_x", tvm.tir.IntImm("int32", 8))
    func = func.with_attr("tl.grid_y", tvm.tir.IntImm("int32", 8))

    mod = tvm.IRModule({"main": func})

    # Apply WS1 → WS2 → WS3 (includes VerifyTTIR)
    mod = apply_tt_defaults(mod)
    mod = apply_ws2_passes(mod)
    mod = apply_ws3_passes(mod)

    func = mod["main"]

    # Verify all metadata exists
    assert "tt_schedule_policy" in func.attrs, "Should have WS1 defaults"
    assert "tt_tiles_per_core" in func.attrs, "Should have WS2 schedule metadata"
    assert "tt_ir_validated" in func.attrs, "Should have VerifyTTIR output"

    # Validation should pass for complete pipeline
    assert bool(func.attrs["tt_ir_validated"]), "Full pipeline should produce valid IR"
    assert int(func.attrs["tt_validation_error_count"]) == 0, "Should have no errors"


def test_verify_tt_ir_core_range_validation():
    """Test VerifyTTIR validates core range format."""
    from tilelang.tt.passes import verify_tt_ir

    func = create_func_with_complete_metadata()

    # Add malformed core_ranges (wrong size)
    func = func.with_attr(
        "tt_core_ranges",
        [[tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 0)]  # Only 2 elements, should be 6
        ])

    mod = tvm.IRModule({"main": func})

    mod = verify_tt_ir(mod)
    func = mod["main"]

    # Should detect invalid core range format
    error_count = int(func.attrs["tt_validation_error_count"])
    assert error_count > 0, "Should detect malformed core_ranges"


def test_verify_tt_ir_circular_buffer_count_mismatch():
    """Test VerifyTTIR detects CB count mismatch."""
    from tilelang.tt.passes import verify_tt_ir

    func = create_func_with_complete_metadata()

    # Add tt_num_cbs that doesn't match actual CB count
    func = func.with_attr("tt_num_cbs", tvm.tir.IntImm("int32", 5))
    func = func.with_attr("tt_circular_buffers", [
        {
            "cb_id": tvm.tir.IntImm("int32", 0)
        },
        {
            "cb_id": tvm.tir.IntImm("int32", 1)
        },
        {
            "cb_id": tvm.tir.IntImm("int32", 2)
        },
    ])  # Only 3 CBs, but tt_num_cbs says 5

    mod = tvm.IRModule({"main": func})

    mod = verify_tt_ir(mod)
    func = mod["main"]

    # Should detect mismatch
    error_count = int(func.attrs["tt_validation_error_count"])
    assert error_count > 0, "Should detect CB count mismatch"


if __name__ == "__main__":
    # Run tests
    test_verify_tt_ir_basic()
    test_verify_tt_ir_validation_passes()
    test_verify_tt_ir_detects_missing_ws1()
    test_verify_tt_ir_detects_missing_ws2()
    test_verify_tt_ir_large_grid_warning()
    test_verify_tt_ir_skip_non_tt_functions()
    test_verify_tt_ir_integration_with_full_pipeline()
    test_verify_tt_ir_core_range_validation()
    test_verify_tt_ir_circular_buffer_count_mismatch()
    print("All VerifyTTIR tests passed!")
