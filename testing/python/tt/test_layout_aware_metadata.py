"""Tests for layout-aware metadata passes."""

import pytest

from tilelang import tvm
import tilelang.language as T

from tilelang.tt import (
    annotate_tt_layout,
    apply_tt_defaults,
    apply_tt_metadata_passes,
    apply_layout_aware_metadata_passes,
    infer_tt_layout,
    propagate_tt_layout,
    layout_aware_work_partition_tt,
)


def _build_simple_module():

    @T.prim_func
    def matmul(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"), C: T.Buffer(
        (256, 256), "float16")):
        with T.Kernel(8, 8) as (bx, by):
            T.evaluate(0)

    prim = matmul.with_attr("global_symbol", "main")
    return tvm.IRModule.from_expr(prim)


class TestLayoutAwareMetadata:

    def test_pipeline_generates_buffer_and_cb_metadata(self):
        mod = _build_simple_module()
        mod = apply_tt_defaults(mod)
        mod = apply_tt_metadata_passes(mod)
        mod = apply_layout_aware_metadata_passes(mod)

        func = mod["main"]

        # Buffer metadata
        assert "tt.buffer.A" in func.attrs, "Expected tt.buffer.A metadata"
        buf_attr = func.attrs["tt.buffer.A"]
        tile_shape = buf_attr["tile_shape"]
        assert int(tile_shape[0]) == 32
        assert int(tile_shape[1]) == 32
        assert str(buf_attr["layout"]) == "dram_interleaved"

        # Circular buffer metadata
        assert "tt.cb.A" in func.attrs, "Expected tt.cb.A metadata"
        cb_attr = func.attrs["tt.cb.A"]
        assert int(cb_attr["depth"]) == 2
        assert int(cb_attr["page_size"]) > 0

        # Partition metadata
        assert str(func.attrs["tt.partition_mode"]) == "global"
        grid_tiles = func.attrs["tt.grid_tiles"]
        assert int(grid_tiles[0]) == 8
        assert int(grid_tiles[1]) == 8

        runtime_args = [str(x) for x in func.attrs["tt.runtime_arg_names"]]
        assert runtime_args == ["tt_start_tile", "tt_tile_count", "Mt", "Kt", "Nt"]
        constants = func.attrs["tt.runtime_constants"]
        assert int(constants["Mt"]) == 8
        assert int(constants["Nt"]) == 8

    def test_user_annotation_overrides_defaults(self):

        @T.prim_func
        def annotated(A: T.Buffer((128, 128), "bfloat16"), B: T.Buffer((128, 128), "bfloat16")):
            with T.Kernel(4, 4) as (bx, by):
                T.evaluate(0)

        annotated = annotate_tt_layout(
            annotated,
            {
                "A": {
                    "memory": "L1",
                    "layout": "sharded",
                    "tile_shape": [16, 16],
                }
            },
        ).with_attr("global_symbol", "main")

        mod = tvm.IRModule.from_expr(annotated)
        mod = apply_tt_defaults(mod)
        mod = apply_tt_metadata_passes(mod)

        # Run passes independently to exercise direct APIs
        mod = infer_tt_layout(mod)
        mod = propagate_tt_layout(mod)
        mod = layout_aware_work_partition_tt(mod)

        func = mod["main"]
        buf_attr = func.attrs["tt.buffer.A"]
        assert str(buf_attr["memory"]) == "L1"
        shape = buf_attr["tile_shape"]
        assert int(shape[0]) == 16
        assert int(shape[1]) == 16

        cb_attr = func.attrs["tt.cb.A"]
        assert int(cb_attr["page_size"]) == 16 * 16 * 2  # bf16 → 2 bytes
        assert [str(x) for x in func.attrs["tt.runtime_arg_names"]][:2] == [
            "tt_start_tile",
            "tt_tile_count",
        ]

    def test_sharded_layout_projects_axes(self):

        @T.prim_func
        def sharded(A: T.Buffer((128, 256), "float16")):
            with T.Kernel(4, 8) as (bx, by):
                T.evaluate(0)

        sharded = annotate_tt_layout(
            sharded,
            {
                "A": {
                    "memory": "L1",
                    "layout": "sharded",
                    "nd_shard": {
                        "axes": ["M", "N"],
                        "grid": [2, 4],
                        "shard_shape_elems": [64, 128],
                    },
                }
            },
        ).with_attr("global_symbol", "main")

        mod = tvm.IRModule.from_expr(sharded)
        mod = apply_tt_defaults(mod)
        mod = apply_tt_metadata_passes(mod)
        mod = apply_layout_aware_metadata_passes(mod)

        func = mod["main"]
        meta = func.attrs["tt.buffer.A"]
        nd_shard = meta["nd_shard"]
        assert [int(x) for x in nd_shard["projected_grid"]] == [2, 4]
        assert [int(x) for x in nd_shard["projected_shard_tiles"]] == [2, 4]

    def test_l1_shard_requires_tile_alignment(self):

        @T.prim_func
        def kernel(A: T.Buffer((64, 64), "float16")):
            with T.Kernel(2, 2) as (bx, by):
                T.evaluate(0)

        misaligned = annotate_tt_layout(
            kernel,
            {
                "A": {
                    "memory": "L1",
                    "layout": "sharded",
                    "nd_shard": {
                        "axes": ["M", "N"],
                        "grid": [2, 2],
                        "shard_shape_elems": [48, 80],
                    },
                }
            },
        ).with_attr("global_symbol", "main")

        mod = tvm.IRModule.from_expr(misaligned)
        mod = apply_tt_defaults(mod)
        mod = apply_tt_metadata_passes(mod)

        with pytest.raises(ValueError, match="tile-aligned"):
            apply_layout_aware_metadata_passes(mod)
