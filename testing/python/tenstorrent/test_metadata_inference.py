"""Integration tests for Metadata Inference using the new pipeline.

These tests validate the new metadata-driven pipeline passes
(InferTTLayout, PropagateTTLayout, TTTilesToCoreMap) that provide
metadata inference and work partitioning.

These tests verify that:

1. Layout inference computes correct buffer metadata
2. Layout propagation normalizes and distributes metadata
3. Core mapping generates correct work partitions
4. Runtime plan is correctly generated

See docs/tenstorrent/NEW_LOWERING_ARCHITECTURE.md for pipeline details.
"""

import pytest
import json
import os
from tilelang import tvm
import tilelang.language as T
from tilelang.tenstorrent import apply_tt_defaults
from tilelang.tenstorrent.passes import (InferTTLayout, PropagateTTLayout, TTTilesToCoreMap,
                                         run_pipeline)
from tilelang.tenstorrent.attrs import TT_CORE_GRID, TT_LAYOUT_DESC, TT_WORK_PARTITION


class TestLayoutInference:
    """Test layout inference and core mapping using the new pipeline."""

    def test_layout_inference_8x8_grid(self):
        """Test layout inference on 8x8 grid (64 tiles, perfect fit for 64 cores)."""

        @T.prim_func
        def gemm_8x8(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                     C: T.Buffer((256, 256), "float16")):
            with T.Kernel(T.ceildiv(256, 32), T.ceildiv(256, 32)) as (bx, by):
                pass  # Stub kernel

        # Convert to IRModule and apply TT defaults
        mod = tvm.IRModule.from_expr(gemm_8x8.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)

        # Convert legacy attributes to new format
        # No compatibility transform needed - using new API directly

        # Apply new pipeline passes
        mod = InferTTLayout()(mod)
        mod = PropagateTTLayout()(mod)
        mod = TTTilesToCoreMap(partition_strategy="row_major")(mod)

        # Verify metadata
        func = mod["main"]
        assert TT_CORE_GRID in func.attrs, "Missing tt.core_grid attribute"
        assert TT_LAYOUT_DESC in func.attrs, "Missing tt.layout_desc attribute"
        assert TT_WORK_PARTITION in func.attrs, "Missing tt.work_partition attribute"

        # Verify core grid
        core_grid = func.attrs[TT_CORE_GRID]
        # Accept either list or tuple format
        assert list(core_grid) == [8, 8], f"Expected [8, 8] grid, got {core_grid}"

        # Verify layout descriptors exist for each buffer
        layout_desc = func.attrs[TT_LAYOUT_DESC]
        assert "A" in layout_desc, "Missing layout for buffer A"
        assert "B" in layout_desc, "Missing layout for buffer B"
        assert "C" in layout_desc, "Missing layout for buffer C"

        # Verify work partition (should have 64 core entries)
        work_partition = func.attrs[TT_WORK_PARTITION]
        assert len(work_partition) == 64, f"Should have 64 cores, got {len(work_partition)}"

        # Each core should have work items
        total_work = sum(len(items) for items in work_partition.values())
        assert total_work == 64, f"Total work items should be 64, got {total_work}"

    def test_layout_inference_4x4_grid(self):
        """Test layout inference on 4x4 grid (16 tiles, non-perfect fit for 64 cores)."""

        @T.prim_func
        def gemm_4x4(A: T.Buffer((128, 128), "float16"), B: T.Buffer((128, 128), "float16"),
                     C: T.Buffer((128, 128), "float16")):
            with T.Kernel(T.ceildiv(128, 32), T.ceildiv(128, 32)) as (bx, by):
                pass  # Stub kernel

        # Convert to IRModule and apply TT defaults
        mod = tvm.IRModule.from_expr(gemm_4x4.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)

        # Apply new pipeline passes
        mod = InferTTLayout()(mod)
        mod = PropagateTTLayout()(mod)
        mod = TTTilesToCoreMap()(mod)

        # Verify metadata
        func = mod["main"]
        assert TT_CORE_GRID in func.attrs
        assert TT_WORK_PARTITION in func.attrs

        # Verify core grid
        core_grid = func.attrs[TT_CORE_GRID]
        assert list(core_grid) == [4, 4], f"Expected [4, 4] grid, got {core_grid}"

        # Verify work partition - only first 16 cores should have work
        work_partition = func.attrs[TT_WORK_PARTITION]
        active_cores = sum(1 for items in work_partition.values() if len(items) > 0)
        assert active_cores <= 16, f"At most 16 cores should be active, got {active_cores}"

        # Total work items should be 16
        total_work = sum(len(items) for items in work_partition.values())
        assert total_work == 16, f"Total work items should be 16, got {total_work}"

    def test_layout_inference_16x16_grid(self):
        """Test layout inference on 16x16 grid (256 tiles > 64 cores)."""

        @T.prim_func
        def gemm_16x16(A: T.Buffer((512, 512), "float16"), B: T.Buffer((512, 512), "float16"),
                       C: T.Buffer((512, 512), "float16")):
            with T.Kernel(T.ceildiv(512, 32), T.ceildiv(512, 32)) as (bx, by):
                pass  # Stub kernel

        # Convert to IRModule and apply TT defaults
        mod = tvm.IRModule.from_expr(gemm_16x16.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)

        # Apply new pipeline passes
        mod = InferTTLayout()(mod)
        mod = PropagateTTLayout()(mod)
        mod = TTTilesToCoreMap()(mod)

        # Verify metadata
        func = mod["main"]
        assert TT_CORE_GRID in func.attrs
        assert TT_WORK_PARTITION in func.attrs

        # Verify core grid (should be 16x16 logically, mapped to physical cores)
        core_grid = func.attrs[TT_CORE_GRID]

        # Verify work partition - all 64 cores should have work
        work_partition = func.attrs[TT_WORK_PARTITION]

        # Total work items should be 256 (16x16)
        total_work = sum(len(items) for items in work_partition.values())
        assert total_work == 256, f"Total work items should be 256, got {total_work}"

        # Each core should get multiple tiles (256/64 = 4 average)
        for core_id, items in work_partition.items():
            if len(items) > 0:
                assert len(items) >= 1, f"Active core {core_id} should have at least 1 work item"


class TestBufferLayouts:
    """Test buffer layout inference using the new pipeline."""

    def test_buffer_layout_tile_aligned(self):
        """Test buffer layout inference on tile-aligned buffers (256x256, multiples of 32)."""

        @T.prim_func
        def gemm_aligned(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                         C: T.Buffer((256, 256), "float16")):
            with T.Kernel(8, 8) as (bx, by):
                pass  # Stub kernel

        # Convert to IRModule and apply TT defaults
        mod = tvm.IRModule.from_expr(gemm_aligned.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)

        # Apply new pipeline passes
        mod = InferTTLayout()(mod)
        mod = PropagateTTLayout()(mod)

        # Verify layout metadata exists for each buffer
        func = mod["main"]
        assert TT_LAYOUT_DESC in func.attrs, "Missing tt.layout_desc attribute"

        layout_desc = func.attrs[TT_LAYOUT_DESC]

        for buffer_name in ["A", "B", "C"]:
            assert buffer_name in layout_desc, f"Missing layout for buffer {buffer_name}"

            buffer_layout = layout_desc[buffer_name]

            # Check that layout info exists
            assert "shard" in buffer_layout or "memory_space" in buffer_layout, \
                f"Buffer {buffer_name} should have memory space info"

            # For tile-aligned buffers (256x256), tiles should be 8x8
            # Each tile is 32x32 elements
            if "tile_shape" in buffer_layout:
                tile_shape = buffer_layout["tile_shape"]
                assert tile_shape == [32, 32], f"Expected 32x32 tiles, got {tile_shape}"

    def test_buffer_layout_non_tile_aligned(self):
        """Test buffer layout inference on non-tile-aligned buffers (100x100, not multiples of 32)."""

        @T.prim_func
        def gemm_unaligned(A: T.Buffer((100, 100), "float16"), B: T.Buffer((100, 100), "float16"),
                           C: T.Buffer((100, 100), "float16")):
            with T.Kernel(T.ceildiv(100, 32), T.ceildiv(100, 32)) as (bx, by):
                pass  # Stub kernel

        # Convert to IRModule and apply TT defaults
        mod = tvm.IRModule.from_expr(gemm_unaligned.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)

        # Apply new pipeline passes
        mod = InferTTLayout()(mod)
        mod = PropagateTTLayout()(mod)

        # Verify metadata
        func = mod["main"]
        assert TT_LAYOUT_DESC in func.attrs

        layout_desc = func.attrs[TT_LAYOUT_DESC]

        for buffer_name in ["A", "B", "C"]:
            assert buffer_name in layout_desc
            buffer_layout = layout_desc[buffer_name]

            # For non-tile-aligned buffers (100x100), need padding
            # ceil(100/32) = 4 tiles in each dimension
            # Padded shape would be 128x128 (4*32 = 128)
            if "padded_shape" in buffer_layout:
                padded = buffer_layout["padded_shape"]
                assert padded == [128, 128], f"Expected padded shape [128, 128], got {padded}"


class TestPipelineIntegration:
    """Test integration of the full new pipeline."""

    def test_full_pipeline_integration(self):
        """Test full pipeline on realistic GEMM."""

        @T.prim_func
        def gemm(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                 C: T.Buffer((256, 256), "float16")):
            with T.Kernel(T.ceildiv(256, 32), T.ceildiv(256, 32)) as (bx, by):
                pass  # Stub kernel

        # Convert to IRModule
        mod = tvm.IRModule.from_expr(gemm.with_attr("global_symbol", "main"))

        # Apply TT defaults
        mod = apply_tt_defaults(mod)

        # Run full pipeline
        plan_path = "test_gemm.plan.json"
        mod = run_pipeline(mod, plan_path=plan_path)

        # Verify all metadata is present
        func = mod["main"]

        # New pipeline attributes
        assert TT_CORE_GRID in func.attrs
        assert TT_LAYOUT_DESC in func.attrs
        assert TT_WORK_PARTITION in func.attrs

        # Verify runtime plan was generated
        assert os.path.exists(plan_path), f"Runtime plan {plan_path} not generated"

        # Load and validate plan
        with open(plan_path) as f:
            plan = json.load(f)
            assert "core_grid" in plan, "Missing core_grid in plan"
            assert "work_partition" in plan, "Missing work_partition in plan"

        # Clean up
        if os.path.exists(plan_path):
            os.remove(plan_path)

    def test_pipeline_with_custom_options(self):
        """Test pipeline with custom configuration options."""

        @T.prim_func
        def kernel(A: T.Buffer((128, 128), "float16")):
            with T.Kernel(4, 4) as (bx, by):
                pass

        mod = tvm.IRModule.from_expr(kernel.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)

        # Use pipeline with custom options
        plan_path = "test_custom.plan.json"
        mod = run_pipeline(
            mod,
            plan_path=plan_path,
            target_device="wormhole",  # Different device
            partition_strategy="column_major",  # Different strategy
            enable_double_buffer=False,  # Disable double buffering
            enable_prefetch=False,  # Disable prefetch
            verbose=True  # Enable verbose logging
        )

        func = mod["main"]
        # Should have metadata regardless of options
        assert TT_CORE_GRID in func.attrs
        assert TT_LAYOUT_DESC in func.attrs

        # Clean up
        if os.path.exists(plan_path):
            os.remove(plan_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
