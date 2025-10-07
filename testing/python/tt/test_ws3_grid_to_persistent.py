"""Basic tests for Workstream 3: GridToPersistentTT pass.

This module tests the foundation of WS3 - the GridToPersistentTT transform
that converts grid-style kernels to persistent per-core loops.
"""

import pytest
import tilelang
from tilelang import tvm
import tilelang.language as T
from tilelang.tt import apply_tt_defaults, apply_ws2_passes, grid_to_persistent_tt, apply_ws3_passes


class TestGridToPersistentTT:
    """Test GridToPersistentTT transformation."""

    def test_grid_to_persistent_basic(self):
        """Test basic grid-to-persistent transformation on simple kernel."""

        @T.prim_func
        def simple_kernel(
            A: T.Buffer((256, 256), "float16"),
            C: T.Buffer((256, 256), "float16")
        ):
            with T.Kernel(T.ceildiv(256, 32), T.ceildiv(256, 32)) as (bx, by):
                # Stub kernel body
                pass

        # Convert to IRModule and apply WS1+WS2
        mod = tvm.IRModule.from_expr(simple_kernel.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)
        mod = apply_ws2_passes(mod)

        # Apply GridToPersistentTT
        mod = grid_to_persistent_tt(mod)

        # Verify metadata was added
        func = mod["main"]
        assert "tt_runtime_args" in func.attrs, "Missing tt_runtime_args attribute"
        assert "tt_persistent_loop" in func.attrs, "Missing tt_persistent_loop attribute"

        # Verify runtime args schema
        runtime_args = func.attrs["tt_runtime_args"]
        assert "start_id" in runtime_args
        assert "count" in runtime_args
        assert "grid_x" in runtime_args
        assert "grid_y" in runtime_args

    def test_apply_ws3_passes(self):
        """Test full WS3 pipeline on simple kernel."""

        @T.prim_func
        def kernel(A: T.Buffer((128, 128), "float16")):
            with T.Kernel(4, 4) as (bx, by):
                pass

        mod = tvm.IRModule.from_expr(kernel.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)
        mod = apply_ws2_passes(mod)

        # Apply WS3 pipeline
        mod = apply_ws3_passes(mod)

        func = mod["main"]
        # Should have both WS2 and WS3 metadata
        assert "tt_num_tiles" in func.attrs  # From WS2
        assert "tt_runtime_args" in func.attrs  # From WS3
        assert "tt_persistent_loop" in func.attrs  # From WS3


class TestWS1WS2WS3Integration:
    """Test integration of all three workstreams."""

    def test_full_pipeline_integration(self):
        """Test WS1 → WS2 → WS3 pipeline on realistic GEMM."""

        @T.prim_func
        def gemm(
            A: T.Buffer((256, 256), "float16"),
            B: T.Buffer((256, 256), "float16"),
            C: T.Buffer((256, 256), "float16")
        ):
            with T.Kernel(T.ceildiv(256, 32), T.ceildiv(256, 32)) as (bx, by):
                # Stub kernel body
                pass

        # Full pipeline
        mod = tvm.IRModule.from_expr(gemm.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)  # WS1
        mod = apply_ws2_passes(mod)   # WS2
        mod = apply_ws3_passes(mod)   # WS3

        func = mod["main"]

        # Verify WS1 metadata
        assert "tt_schedule_policy" in func.attrs
        assert "tt_layout_type" in func.attrs

        # Verify WS2 metadata
        assert "tt_num_tiles" in func.attrs
        assert "tt_tiles_per_core" in func.attrs

        # Verify WS3 metadata
        assert "tt_runtime_args" in func.attrs
        assert "tt_persistent_loop" in func.attrs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
