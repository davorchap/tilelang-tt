"""Integration tests for Workstream 2: Schedule and Sharding Inference Passes.

This module tests the schedule and sharding inference passes that inject
TT-specific metadata into IRModules. These tests verify that:

1. Schedule inference computes correct per-core tile ranges
2. Sharding inference generates correct layout metadata
3. Metadata format matches expectations for downstream WS3/WS4 passes
"""

import pytest
from tilelang import tvm
import tilelang.language as T
from tilelang.tt import apply_tt_defaults, infer_default_tt_schedule, infer_default_tt_shard, apply_ws2_passes


class TestScheduleInference:
    """Test schedule inference pass (WS2)."""

    def test_schedule_inference_8x8_grid(self):
        """Test schedule inference on 8x8 grid (64 tiles, perfect fit for 64 cores)."""

        @T.prim_func
        def gemm_8x8(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                     C: T.Buffer((256, 256), "float16")):
            with T.Kernel(T.ceildiv(256, 32), T.ceildiv(256, 32)) as (bx, by):
                pass  # Stub kernel

        # Convert to IRModule and apply WS1 defaults
        mod = tvm.IRModule.from_expr(gemm_8x8.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)

        # Apply schedule inference
        mod = infer_default_tt_schedule(mod)

        # Verify metadata
        func = mod["main"]
        assert "tt_num_tiles" in func.attrs, "Missing tt_num_tiles attribute"
        assert "tt_grid_x" in func.attrs, "Missing tt_grid_x attribute"
        assert "tt_grid_y" in func.attrs, "Missing tt_grid_y attribute"
        assert "tt_grid_z" in func.attrs, "Missing tt_grid_z attribute"
        assert "tt_num_cores" in func.attrs, "Missing tt_num_cores attribute"
        assert "tt_tiles_per_core" in func.attrs, "Missing tt_tiles_per_core attribute"

        # Verify values
        assert int(func.attrs["tt_grid_x"]) == 8, "Incorrect grid_x"
        assert int(func.attrs["tt_grid_y"]) == 8, "Incorrect grid_y"
        assert int(func.attrs["tt_grid_z"]) == 1, "Incorrect grid_z"
        assert int(func.attrs["tt_num_tiles"]) == 64, "Incorrect num_tiles (should be 8*8=64)"
        assert int(func.attrs["tt_num_cores"]) == 64, "Incorrect num_cores"

        # Verify per-core ranges (64 tiles / 64 cores = 1 tile per core)
        tiles_per_core = func.attrs["tt_tiles_per_core"]
        assert len(tiles_per_core) == 64, "Should have 64 core ranges"

        for i, (start, count) in enumerate(tiles_per_core):
            assert int(start) == i, f"Core {i} should start at tile {i}"
            assert int(count) == 1, f"Core {i} should get 1 tile"

    def test_schedule_inference_4x4_grid(self):
        """Test schedule inference on 4x4 grid (16 tiles, non-perfect fit for 64 cores)."""

        @T.prim_func
        def gemm_4x4(A: T.Buffer((128, 128), "float16"), B: T.Buffer((128, 128), "float16"),
                     C: T.Buffer((128, 128), "float16")):
            with T.Kernel(T.ceildiv(128, 32), T.ceildiv(128, 32)) as (bx, by):
                pass  # Stub kernel

        # Convert to IRModule and apply WS1 defaults
        mod = tvm.IRModule.from_expr(gemm_4x4.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)

        # Apply schedule inference
        mod = infer_default_tt_schedule(mod)

        # Verify metadata
        func = mod["main"]
        assert int(func.attrs["tt_grid_x"]) == 4
        assert int(func.attrs["tt_grid_y"]) == 4
        assert int(func.attrs["tt_num_tiles"]) == 16

        # Verify per-core ranges (16 tiles / 64 cores = 0 tiles per core for most, 1 tile for first 16)
        tiles_per_core = func.attrs["tt_tiles_per_core"]
        assert len(tiles_per_core) == 64

        # First 16 cores get 1 tile each
        for i in range(16):
            start, count = tiles_per_core[i]
            assert int(count) == 1, f"Core {i} should get 1 tile"

        # Remaining cores get 0 tiles
        for i in range(16, 64):
            start, count = tiles_per_core[i]
            assert int(count) == 0, f"Core {i} should get 0 tiles (inactive)"

    def test_schedule_inference_16x16_grid(self):
        """Test schedule inference on 16x16 grid (256 tiles > 64 cores)."""

        @T.prim_func
        def gemm_16x16(A: T.Buffer((512, 512), "float16"), B: T.Buffer((512, 512), "float16"),
                       C: T.Buffer((512, 512), "float16")):
            with T.Kernel(T.ceildiv(512, 32), T.ceildiv(512, 32)) as (bx, by):
                pass  # Stub kernel

        # Convert to IRModule and apply WS1 defaults
        mod = tvm.IRModule.from_expr(gemm_16x16.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)

        # Apply schedule inference
        mod = infer_default_tt_schedule(mod)

        # Verify metadata
        func = mod["main"]
        assert int(func.attrs["tt_num_tiles"]) == 256

        # Verify per-core ranges (256 tiles / 64 cores = 4 tiles per core)
        tiles_per_core = func.attrs["tt_tiles_per_core"]
        total_tiles = sum(int(count) for start, count in tiles_per_core)
        assert total_tiles == 256, f"Total tiles should be 256, got {total_tiles}"

        # Each core should get 4 tiles (256/64 = 4 with no remainder)
        for i in range(64):
            start, count = tiles_per_core[i]
            assert int(count) == 4, f"Core {i} should get 4 tiles"


class TestShardInference:
    """Test sharding inference pass (WS2)."""

    def test_shard_inference_tile_aligned(self):
        """Test sharding inference on tile-aligned buffers (256x256, multiples of 32)."""

        @T.prim_func
        def gemm_aligned(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                         C: T.Buffer((256, 256), "float16")):
            with T.Kernel(8, 8) as (bx, by):
                pass  # Stub kernel

        # Convert to IRModule and apply WS1 defaults
        mod = tvm.IRModule.from_expr(gemm_aligned.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)

        # Apply sharding inference
        mod = infer_default_tt_shard(mod)

        # Verify metadata exists for each buffer
        func = mod["main"]

        for buffer_name in ["A", "B", "C"]:
            # Check layout metadata
            layout_key = f"tt_buffer_{buffer_name}_layout"
            assert layout_key in func.attrs, f"Missing {layout_key}"
            assert str(func.attrs[layout_key]) == "dram_interleaved"

            # Check tile shape
            tile_shape_key = f"tt_buffer_{buffer_name}_tile_shape"
            assert tile_shape_key in func.attrs, f"Missing {tile_shape_key}"
            tile_shape = func.attrs[tile_shape_key]
            assert len(tile_shape) == 2
            assert int(tile_shape[0]) == 32
            assert int(tile_shape[1]) == 32

            # Check tile counts
            tiles_height_key = f"tt_buffer_{buffer_name}_num_tiles_height"
            tiles_width_key = f"tt_buffer_{buffer_name}_num_tiles_width"
            assert tiles_height_key in func.attrs, f"Missing {tiles_height_key}"
            assert tiles_width_key in func.attrs, f"Missing {tiles_width_key}"
            assert int(func.attrs[tiles_height_key]) == 8  # 256/32 = 8
            assert int(func.attrs[tiles_width_key]) == 8  # 256/32 = 8

            # Check padding (should be False for tile-aligned)
            needs_padding_key = f"tt_buffer_{buffer_name}_needs_padding"
            assert needs_padding_key in func.attrs, f"Missing {needs_padding_key}"
            assert int(func.attrs[needs_padding_key]) == 0, f"{buffer_name} should not need padding"

    def test_shard_inference_non_tile_aligned(self):
        """Test sharding inference on non-tile-aligned buffers (100x100, not multiples of 32)."""

        @T.prim_func
        def gemm_unaligned(A: T.Buffer((100, 100), "float16"), B: T.Buffer((100, 100), "float16"),
                           C: T.Buffer((100, 100), "float16")):
            with T.Kernel(T.ceildiv(100, 32), T.ceildiv(100, 32)) as (bx, by):
                pass  # Stub kernel

        # Convert to IRModule and apply WS1 defaults
        mod = tvm.IRModule.from_expr(gemm_unaligned.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)

        # Apply sharding inference
        mod = infer_default_tt_shard(mod)

        # Verify metadata
        func = mod["main"]

        for buffer_name in ["A", "B", "C"]:
            # Check tile counts (should be ceil(100/32) = 4)
            tiles_height_key = f"tt_buffer_{buffer_name}_num_tiles_height"
            tiles_width_key = f"tt_buffer_{buffer_name}_num_tiles_width"
            assert int(func.attrs[tiles_height_key]) == 4  # ceil(100/32) = 4
            assert int(func.attrs[tiles_width_key]) == 4  # ceil(100/32) = 4

            # Check padding (should be True)
            needs_padding_key = f"tt_buffer_{buffer_name}_needs_padding"
            assert needs_padding_key in func.attrs, f"Missing {needs_padding_key}"
            assert int(func.attrs[needs_padding_key]) == 1, f"{buffer_name} should need padding"

            # Check padded shape (should be 128x128)
            padded_shape_key = f"tt_buffer_{buffer_name}_padded_shape"
            assert padded_shape_key in func.attrs, f"Missing {padded_shape_key}"
            padded_shape = func.attrs[padded_shape_key]
            assert len(padded_shape) == 2
            assert int(padded_shape[0]) == 128  # 4 tiles * 32 = 128
            assert int(padded_shape[1]) == 128  # 4 tiles * 32 = 128


class TestWS2Integration:
    """Test integration of WS1 + WS2 passes."""

    def test_full_ws2_pipeline(self):
        """Test full WS1 -> WS2 pipeline on realistic GEMM."""

        @T.prim_func
        def gemm(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                 C: T.Buffer((256, 256), "float16")):
            with T.Kernel(T.ceildiv(256, 32), T.ceildiv(256, 32)) as (bx, by):
                pass  # Stub kernel

        # Convert to IRModule
        mod = tvm.IRModule.from_expr(gemm.with_attr("global_symbol", "main"))

        # Apply WS1 defaults
        mod = apply_tt_defaults(mod)

        # Apply both WS2 passes
        mod = apply_ws2_passes(mod)

        # Verify all metadata is present
        func = mod["main"]

        # WS1 metadata
        assert "tt_schedule_policy" in func.attrs
        assert "tt_schedule_order" in func.attrs
        assert "tt_layout_type" in func.attrs

        # WS2 schedule metadata
        assert "tt_num_tiles" in func.attrs
        assert "tt_grid_x" in func.attrs
        assert "tt_tiles_per_core" in func.attrs

        # WS2 shard metadata (check one buffer)
        assert "tt_buffer_A_layout" in func.attrs
        assert "tt_buffer_A_tile_shape" in func.attrs
        assert "tt_buffer_A_num_tiles_height" in func.attrs

    def test_ws2_convenience_function(self):
        """Test the apply_ws2_passes convenience function."""

        @T.prim_func
        def kernel(A: T.Buffer((128, 128), "float16")):
            with T.Kernel(4, 4) as (bx, by):
                pass

        mod = tvm.IRModule.from_expr(kernel.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)

        # Use convenience function
        mod = apply_ws2_passes(mod)

        func = mod["main"]
        # Should have both schedule and shard metadata
        assert "tt_num_tiles" in func.attrs  # From schedule inference
        assert "tt_buffer_A_layout" in func.attrs  # From shard inference


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
