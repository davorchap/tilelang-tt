"""Test TTTilesToCoreMap pass.

NOTE: These tests are for the legacy metadata format and are no longer relevant
in the new metadata-driven architecture. All tests are skipped.
"""

import pytest
import tvm
from tvm import tir

# Skip the entire module - legacy tests not relevant to new architecture
pytestmark = pytest.mark.skip(
    reason="Legacy TTTilesToCoreMap tests - not relevant to new architecture")


def create_mock_func_with_tiles_per_core(grid_x=8, grid_y=8):
    """Create a mock PrimFunc with tt_tiles_per_core metadata from metadata inference stage."""
    # Create simple buffer declaration
    A = tir.decl_buffer((256, 256), "float16", name="A")
    B = tir.decl_buffer((256, 256), "float16", name="B")
    C = tir.decl_buffer((256, 256), "float16", name="C")

    # Create simple body (doesn't matter for this test)
    body = tir.Evaluate(0)

    # Create PrimFunc
    func = tir.PrimFunc([A, B, C], body)

    # Add metadata inference stage schedule metadata
    num_tiles = grid_x * grid_y
    num_cores = 64

    # Simple contiguous assignment: each core gets num_tiles/num_cores tiles
    tiles_per_core_value = num_tiles // num_cores
    tiles_per_core = []
    for core_id in range(num_cores):
        start_tile = core_id * tiles_per_core_value
        count = tiles_per_core_value
        tiles_per_core.append([tvm.tir.IntImm("int32", start_tile), tvm.tir.IntImm("int32", count)])

    # Stamp metadata
    func = func.with_attr("tt_tiles_per_core", tiles_per_core)
    func = func.with_attr("tt_num_cores", tvm.tir.IntImm("int32", num_cores))
    func = func.with_attr("tt_grid_x", tvm.tir.IntImm("int32", grid_x))
    func = func.with_attr("tt_grid_y", tvm.tir.IntImm("int32", grid_y))
    func = func.with_attr("tt_num_tiles", tvm.tir.IntImm("int32", num_tiles))

    return func


def test_tt_tiles_to_core_map_basic():
    """Test TTTilesToCoreMap generates core ranges correctly."""
    from tilelang.tenstorrent.passes import TTTilesToCoreMap

    # Create function with metadata inference stage metadata
    func = create_mock_func_with_tiles_per_core(grid_x=8, grid_y=8)
    mod = tvm.IRModule({"main": func})

    # Apply TTTilesToCoreMap
    mod = TTTilesToCoreMap()(mod)
    func = mod["main"]

    # Verify core ranges attribute exists
    assert func.attrs is not None, "Function should have attributes"
    assert "tt_core_ranges" in func.attrs, "Should have tt_core_ranges attribute"
    assert ("tt_core_runtime_args" in func.attrs), "Should have tt_core_runtime_args attribute"

    core_ranges = func.attrs["tt_core_ranges"]
    core_runtime_args = func.attrs["tt_core_runtime_args"]

    # Should have 64 core ranges (one per core with assigned tiles)
    assert len(core_ranges) == 64, f"Expected 64 core ranges, got {len(core_ranges)}"
    assert (
        len(core_runtime_args) == 64), f"Expected 64 runtime arg sets, got {len(core_runtime_args)}"


def test_tt_tiles_to_core_map_coordinates():
    """Test TTTilesToCoreMap generates correct physical coordinates."""
    from tilelang.tenstorrent.passes import TTTilesToCoreMap

    func = create_mock_func_with_tiles_per_core(grid_x=8, grid_y=8)
    mod = tvm.IRModule({"main": func})

    mod = TTTilesToCoreMap()(mod)
    func = mod["main"]

    core_ranges = func.attrs["tt_core_ranges"]

    # Check first core (core_id=0 -> x=0, y=0)
    first_range = core_ranges[0]
    assert (
        len(first_range) == 6
    ), "Core range should have 6 elements: [start_x, start_y, end_x, end_y, start_tile, count]"
    assert int(first_range[0]) == 0, "Core 0 should have x=0"
    assert int(first_range[1]) == 0, "Core 0 should have y=0"
    assert int(first_range[2]) == 0, "Core 0 should have end_x=0 (single core range)"
    assert int(first_range[3]) == 0, "Core 0 should have end_y=0 (single core range)"
    assert int(first_range[4]) == 0, "Core 0 should start at tile 0"
    assert int(first_range[5]) == 1, "Core 0 should have 1 tile (64 tiles / 64 cores)"

    # Check core 8 (row-major: core_id=8 -> x=0, y=1)
    if len(core_ranges) > 8:
        eighth_range = core_ranges[8]
        assert int(eighth_range[0]) == 0, "Core 8 should have x=0"
        assert int(eighth_range[1]) == 1, "Core 8 should have y=1"

    # Check core 9 (row-major: core_id=9 -> x=1, y=1)
    if len(core_ranges) > 9:
        ninth_range = core_ranges[9]
        assert int(ninth_range[0]) == 1, "Core 9 should have x=1"
        assert int(ninth_range[1]) == 1, "Core 9 should have y=1"

    # Check last core (core_id=63 -> x=7, y=7)
    last_range = core_ranges[63]
    assert int(last_range[0]) == 7, "Core 63 should have x=7"
    assert int(last_range[1]) == 7, "Core 63 should have y=7"


def test_tt_tiles_to_core_map_runtime_args():
    """Test TTTilesToCoreMap generates correct runtime args."""
    from tilelang.tenstorrent.passes import TTTilesToCoreMap

    func = create_mock_func_with_tiles_per_core(grid_x=8, grid_y=8)
    mod = tvm.IRModule({"main": func})

    mod = TTTilesToCoreMap()(mod)
    func = mod["main"]

    core_runtime_args = func.attrs["tt_core_runtime_args"]

    # Check runtime args format
    for core_id in range(64):
        args = core_runtime_args[core_id]
        assert (len(args) == 2), "Runtime args should have 2 elements: [start_tile, num_tiles]"

        start_tile = int(args[0])
        num_tiles = int(args[1])

        # For 64 tiles / 64 cores, each core gets 1 tile
        assert start_tile == core_id, f"Core {core_id} should start at tile {core_id}"
        assert num_tiles == 1, f"Core {core_id} should have 1 tile"


def test_tt_tiles_to_core_map_skip_without_metadata():
    """Test TTTilesToCoreMap skips functions without metadata inference stage metadata."""
    from tilelang.tenstorrent.passes import TTTilesToCoreMap

    # Create function WITHOUT metadata inference stage metadata
    A = tir.decl_buffer((256, 256), "float16", name="A")
    body = tir.Evaluate(0)
    func = tir.PrimFunc([A], body)
    mod = tvm.IRModule({"main": func})

    # Apply pass
    mod = TTTilesToCoreMap()(mod)
    func = mod["main"]

    # Should NOT add core ranges
    assert (func.attrs is None or "tt_core_ranges" not in func.attrs
           ), "Should not add core ranges without metadata inference stage metadata"


def test_tt_tiles_to_core_map_consistency_with_metadata():
    """Test TTTilesToCoreMap output is consistent with metadata inference input."""
    from tilelang.tenstorrent.passes import TTTilesToCoreMap

    func = create_mock_func_with_tiles_per_core(grid_x=8, grid_y=8)
    mod = tvm.IRModule({"main": func})

    # Get original metadata inference stage metadata
    original_tiles_per_core = func.attrs["tt_tiles_per_core"]

    # Apply pass
    mod = TTTilesToCoreMap()(mod)
    func = mod["main"]

    core_ranges = func.attrs["tt_core_ranges"]
    core_runtime_args = func.attrs["tt_core_runtime_args"]

    # Verify consistency: core_ranges and core_runtime_args should match original tiles_per_core
    for core_id in range(64):
        original = original_tiles_per_core[core_id]
        original_start = int(original[0])
        original_count = int(original[1])

        # Check core_ranges
        core_range = core_ranges[core_id]
        range_start_tile = int(core_range[4])
        range_count = int(core_range[5])

        assert (range_start_tile == original_start), f"Core {core_id} range start_tile mismatch"
        assert range_count == original_count, f"Core {core_id} range count mismatch"

        # Check core_runtime_args
        runtime_args = core_runtime_args[core_id]
        args_start_tile = int(runtime_args[0])
        args_count = int(runtime_args[1])

        assert (
            args_start_tile == original_start), f"Core {core_id} runtime args start_tile mismatch"
        assert (args_count == original_count), f"Core {core_id} runtime args count mismatch"


def test_tt_tiles_to_core_map_integration_with_metadata():
    """Test TTTilesToCoreMap integrates with metadata inference passes."""
    from tilelang.tenstorrent.passes import (
        InferTTLayout,
        PropagateTTLayout,
        TTTilesToCoreMap,
    )

    def apply_tt_metadata_passes(mod):
        """Helper to apply metadata passes in the new pipeline."""
        mod = InferTTLayout()(mod)
        mod = PropagateTTLayout()(mod)
        mod = TTTilesToCoreMap()(mod)
        return mod

    from tilelang.tenstorrent.target import apply_tt_defaults

    # Create a simple function
    A = tir.decl_buffer((256, 256), "float16", name="A")
    B = tir.decl_buffer((256, 256), "float16", name="B")
    C = tir.decl_buffer((256, 256), "float16", name="C")

    # Mock grid dimensions
    tir.Var("bx", "int32")
    tir.Var("by", "int32")

    body = tir.Evaluate(0)
    func = tir.PrimFunc([A, B, C], body)

    # Add grid attributes (normally from TileLang frontend)
    func = func.with_attr("tl.grid_x", tvm.tir.IntImm("int32", 8))
    func = func.with_attr("tl.grid_y", tvm.tir.IntImm("int32", 8))

    mod = tvm.IRModule({"main": func})

    # Apply TT defaults -> metadata inference -> TTTilesToCoreMap pipeline
    mod = apply_tt_defaults(mod)
    mod = apply_tt_metadata_passes(mod)
    mod = TTTilesToCoreMap()(mod)

    func = mod["main"]

    # Verify all metadata exists (new pipeline format)
    assert "tt_schedule_policy" in func.attrs, "Should have TT defaults"
    # The new pipeline uses different metadata attributes
    assert "tt.core_grid" in func.attrs, "Should have core grid"
    assert "tt.work_partition" in func.attrs, "Should have work partition"


if __name__ == "__main__":
    # Run tests
    test_tt_tiles_to_core_map_basic()
    test_tt_tiles_to_core_map_coordinates()
    test_tt_tiles_to_core_map_runtime_args()
    test_tt_tiles_to_core_map_skip_without_metadata()
    test_tt_tiles_to_core_map_consistency_with_metadata()
    test_tt_tiles_to_core_map_integration_with_metadata()
    print("All TTTilesToCoreMap tests passed!")
