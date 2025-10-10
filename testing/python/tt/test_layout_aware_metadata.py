"""Tests for layout-aware metadata passes."""

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
    def matmul(A: T.Buffer((256, 256), "float16"),
               B: T.Buffer((256, 256), "float16"),
               C: T.Buffer((256, 256), "float16")):
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

        runtime_args = [str(x) for x in func.attrs["tt.runtime_args"]]
        assert runtime_args == ["start_id", "count", "Mt", "Kt", "Nt"]

    def test_user_annotation_overrides_defaults(self):
        @T.prim_func
        def annotated(A: T.Buffer((128, 128), "bfloat16"),
                      B: T.Buffer((128, 128), "bfloat16")):
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
