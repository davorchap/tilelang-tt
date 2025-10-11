"""Test MemorySpaceLowerTT pass (persistent transform stage Phase 2).

This pass lowers abstract buffer allocations to TT circular buffer configurations.
"""

import tvm
from tvm import tir


def create_func_with_tile_buffers():
    """Create a mock PrimFunc with tile-sized buffer allocations."""
    # Function parameters (DRAM buffers)
    A = tir.decl_buffer((256, 256), "float16", name="A", scope="global")
    B = tir.decl_buffer((256, 256), "float16", name="B", scope="global")
    C = tir.decl_buffer((256, 256), "float16", name="C", scope="global")

    # Tile buffers (to be converted to circular buffers)
    A_tile = tir.decl_buffer((32, 32), "float16", name="A_tile")
    B_tile = tir.decl_buffer((32, 32), "float16", name="B_tile")
    C_tile = tir.decl_buffer((32, 32), "float16", name="C_tile")

    # Simple body with DeclBuffer statements
    body = tir.DeclBuffer(A_tile, tir.DeclBuffer(B_tile, tir.DeclBuffer(C_tile, tir.Evaluate(0))))

    func = tir.PrimFunc([A, B, C], body)

    # Add TT defaults (required for MemorySpaceLowerTT to activate)
    func = func.with_attr("tt_schedule_policy", "contiguous")
    func = func.with_attr("tt_schedule_order", "row_major")

    return func


def test_memory_space_lower_tt_basic():
    """Test MemorySpaceLowerTT assigns circular buffer IDs."""
    from tilelang.tt.passes import memory_space_lower_tt

    func = create_func_with_tile_buffers()
    mod = tvm.IRModule({"main": func})

    # Apply MemorySpaceLowerTT
    mod = memory_space_lower_tt(mod)
    func = mod["main"]

    # Verify CB metadata attached
    assert func.attrs is not None, "Function should have attributes"
    assert "tt_circular_buffers" in func.attrs, "Should have tt_circular_buffers attribute"
    assert "tt_num_cbs" in func.attrs, "Should have tt_num_cbs attribute"

    cb_configs = func.attrs["tt_circular_buffers"]
    num_cbs = int(func.attrs["tt_num_cbs"])

    # Should have 3 circular buffers (A_tile, B_tile, C_tile)
    assert num_cbs == 3, f"Expected 3 CBs, got {num_cbs}"
    assert len(cb_configs) == 3, f"Expected 3 CB configs, got {len(cb_configs)}"


def test_memory_space_lower_tt_cb_ids():
    """Test MemorySpaceLowerTT assigns unique sequential CB IDs."""
    from tilelang.tt.passes import memory_space_lower_tt

    func = create_func_with_tile_buffers()
    mod = tvm.IRModule({"main": func})

    mod = memory_space_lower_tt(mod)
    func = mod["main"]

    cb_configs = func.attrs["tt_circular_buffers"]

    # Check CB IDs are unique and sequential (order may vary due to traversal)
    cb_ids = sorted([int(config["cb_id"]) for config in cb_configs])
    assert cb_ids == [0, 1, 2], f"Expected sorted CB IDs [0, 1, 2], got {cb_ids}"


def test_memory_space_lower_tt_num_pages():
    """Test MemorySpaceLowerTT assigns correct num_pages (double-buffer for inputs, single for accumulator)."""
    from tilelang.tt.passes import memory_space_lower_tt

    func = create_func_with_tile_buffers()
    mod = tvm.IRModule({"main": func})

    mod = memory_space_lower_tt(mod)
    func = mod["main"]

    cb_configs = func.attrs["tt_circular_buffers"]

    # Check num_pages
    for config in cb_configs:
        cb_id = int(config["cb_id"])
        num_pages = int(config["num_pages"])
        name = str(config["name"])

        # Heuristic: buffers with "C" in name should have 1 page (accumulator)
        # Others should have 2 pages (double-buffer)
        if "C" in name:
            assert num_pages == 1, f"CB{cb_id} ({name}) should have 1 page (accumulator), got {num_pages}"
        else:
            assert num_pages == 2, f"CB{cb_id} ({name}) should have 2 pages (double-buffer), got {num_pages}"


def test_memory_space_lower_tt_tile_size():
    """Test MemorySpaceLowerTT calculates tile size correctly."""
    from tilelang.tt.passes import memory_space_lower_tt

    func = create_func_with_tile_buffers()
    mod = tvm.IRModule({"main": func})

    mod = memory_space_lower_tt(mod)
    func = mod["main"]

    cb_configs = func.attrs["tt_circular_buffers"]

    # For 32×32 float16 tiles: 32 * 32 * 2 bytes = 2048 bytes
    expected_tile_size = 32 * 32 * 2

    for config in cb_configs:
        tile_size = int(config["tile_size"])
        assert tile_size == expected_tile_size, f"Expected tile_size={expected_tile_size}, got {tile_size}"


def test_memory_space_lower_tt_skip_non_tt_functions():
    """Test MemorySpaceLowerTT skips functions without TT attributes."""
    from tilelang.tt.passes import memory_space_lower_tt

    # Create function WITHOUT TT attributes
    A = tir.decl_buffer((256, 256), "float16", name="A")
    A_tile = tir.decl_buffer((32, 32), "float16", name="A_tile")

    body = tir.DeclBuffer(A_tile, tir.Evaluate(0))
    func = tir.PrimFunc([A], body)

    mod = tvm.IRModule({"main": func})

    # Apply pass
    mod = memory_space_lower_tt(mod)
    func = mod["main"]

    # Should NOT add CB metadata
    assert func.attrs is None or "tt_circular_buffers" not in func.attrs, "Should not add CBs without TT attributes"


def test_memory_space_lower_tt_skip_non_tile_buffers():
    """Test MemorySpaceLowerTT only processes tile-sized buffers."""
    from tilelang.tt.passes import memory_space_lower_tt

    # Function parameters
    A = tir.decl_buffer((256, 256), "float16", name="A", scope="global")

    # Tile buffer (should be converted)
    A_tile = tir.decl_buffer((32, 32), "float16", name="A_tile")

    # Non-tile buffer (large, should be skipped)
    large_buf = tir.decl_buffer((1024, 1024), "float16", name="large_buf")

    # Scalar buffer (should be skipped)
    scalar_buf = tir.decl_buffer((1,), "float16", name="scalar")

    body = tir.DeclBuffer(A_tile,
                          tir.DeclBuffer(large_buf, tir.DeclBuffer(scalar_buf, tir.Evaluate(0))))

    func = tir.PrimFunc([A], body)
    func = func.with_attr("tt_schedule_policy", "contiguous")

    mod = tvm.IRModule({"main": func})

    mod = memory_space_lower_tt(mod)
    func = mod["main"]

    # Should only have 1 CB (for A_tile)
    num_cbs = int(func.attrs.get("tt_num_cbs", 0))
    assert num_cbs == 1, f"Expected 1 CB (only A_tile), got {num_cbs}"


def test_memory_space_lower_tt_integration_with_defaults_and_metadata():
    """Test MemorySpaceLowerTT integrates with full apply_tt_defaults→metadata inference→transform pipeline."""
    from tilelang.tt.passes import apply_tt_metadata_passes, memory_space_lower_tt
    from tilelang.tt.target import apply_tt_defaults

    # Create function
    A = tir.decl_buffer((256, 256), "float16", name="A", scope="global")
    B = tir.decl_buffer((256, 256), "float16", name="B", scope="global")
    C = tir.decl_buffer((256, 256), "float16", name="C", scope="global")

    A_tile = tir.decl_buffer((32, 32), "float16", name="A_tile")
    B_tile = tir.decl_buffer((32, 32), "float16", name="B_tile")
    C_tile = tir.decl_buffer((32, 32), "float16", name="C_tile")

    body = tir.DeclBuffer(A_tile, tir.DeclBuffer(B_tile, tir.DeclBuffer(C_tile, tir.Evaluate(0))))

    func = tir.PrimFunc([A, B, C], body)

    # Add grid metadata (normally from TileLang frontend)
    func = func.with_attr("tl.grid_x", tvm.tir.IntImm("int32", 8))
    func = func.with_attr("tl.grid_y", tvm.tir.IntImm("int32", 8))

    mod = tvm.IRModule({"main": func})

    # Apply TT defaults → metadata inference → MemorySpaceLowerTT
    mod = apply_tt_defaults(mod)
    mod = apply_tt_metadata_passes(mod)
    mod = memory_space_lower_tt(mod)

    func = mod["main"]

    # Verify all metadata exists
    assert "tt_schedule_policy" in func.attrs, "Should have TT defaults"
    assert "tt_tiles_per_core" in func.attrs, "Should have metadata inference schedule metadata"
    assert "tt_circular_buffers" in func.attrs, "Should have MemorySpaceLowerTT output"
    assert "tt_num_cbs" in func.attrs, "Should have num CBs"


def test_memory_space_lower_tt_different_tile_sizes():
    """Test MemorySpaceLowerTT handles different tile sizes correctly."""
    from tilelang.tt.passes import memory_space_lower_tt

    # Function parameters
    A = tir.decl_buffer((256, 256), "float32", name="A", scope="global")

    # Different tile sizes
    tile_16x16 = tir.decl_buffer((16, 16), "float32", name="tile_16x16")
    tile_32x32 = tir.decl_buffer((32, 32), "float32", name="tile_32x32")
    tile_64x64 = tir.decl_buffer((64, 64), "float32", name="tile_64x64")

    body = tir.DeclBuffer(tile_16x16,
                          tir.DeclBuffer(tile_32x32, tir.DeclBuffer(tile_64x64, tir.Evaluate(0))))

    func = tir.PrimFunc([A], body)
    func = func.with_attr("tt_schedule_policy", "contiguous")

    mod = tvm.IRModule({"main": func})

    mod = memory_space_lower_tt(mod)
    func = mod["main"]

    cb_configs = func.attrs["tt_circular_buffers"]

    # All three should be converted (all are square and <= 64)
    assert len(cb_configs) == 3, f"Expected 3 CBs, got {len(cb_configs)}"

    # Check tile sizes (float32 = 4 bytes)
    expected_sizes = {
        "tile_16x16": 16 * 16 * 4,
        "tile_32x32": 32 * 32 * 4,
        "tile_64x64": 64 * 64 * 4,
    }

    for config in cb_configs:
        name = str(config["name"])
        tile_size = int(config["tile_size"])
        assert tile_size == expected_sizes[
            name], f"{name}: expected {expected_sizes[name]}, got {tile_size}"


if __name__ == "__main__":
    # Run tests
    test_memory_space_lower_tt_basic()
    test_memory_space_lower_tt_cb_ids()
    test_memory_space_lower_tt_num_pages()
    test_memory_space_lower_tt_tile_size()
    test_memory_space_lower_tt_skip_non_tt_functions()
    test_memory_space_lower_tt_skip_non_tile_buffers()
    test_memory_space_lower_tt_integration_with_defaults_and_metadata()
    test_memory_space_lower_tt_different_tile_sizes()
    print("All MemorySpaceLowerTT tests passed!")
