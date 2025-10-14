"""Basic tests for Persistent transform stage: GridToPersistentTT pass.

This module tests the foundation of persistent transform stage – the GridToPersistentTT transform
that converts grid-style kernels to persistent per-core loops.
"""

import pytest
from tilelang import tvm
import tilelang.language as T
from tilelang.tenstorrent import (
    apply_tt_defaults,
    apply_tt_metadata_passes,
    apply_layout_aware_metadata_passes,
    grid_to_persistent_tt,
    apply_tt_transform_passes,
    annotate_tt_schedule,
    annotate_tt_layout,
)


class TestGridToPersistentTT:
    """Test GridToPersistentTT transformation."""

    def test_grid_to_persistent_basic(self):
        """Test basic grid-to-persistent transformation on simple kernel."""

        @T.prim_func
        def simple_kernel(A: T.Buffer((256, 256), "float16"), C: T.Buffer((256, 256), "float16")):
            with T.Kernel(T.ceildiv(256, 32), T.ceildiv(256, 32)) as (bx, by):
                # Stub kernel body
                pass

        # Convert to IRModule and apply TT defaults stage + metadata inference stage
        mod = tvm.IRModule.from_expr(simple_kernel.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)
        mod = apply_tt_metadata_passes(mod)

        # Apply GridToPersistentTT
        mod = grid_to_persistent_tt(mod)

        # Verify metadata was added
        func = mod["main"]
        assert "tt_runtime_args" in func.attrs, "Missing tt_runtime_args attribute"
        assert "tt_persistent_loop" in func.attrs, "Missing tt_persistent_loop attribute"

        # Verify runtime args schema
        runtime_args = func.attrs["tt_runtime_args"]
        assert "start_tile" in runtime_args
        assert "tile_count" in runtime_args
        assert "grid_shape" in runtime_args
        assert "param_order" in runtime_args

        start_info = runtime_args["start_tile"]
        count_info = runtime_args["tile_count"]

        assert start_info["name"] == "tt_start_tile"
        assert count_info["name"] == "tt_tile_count"
        assert start_info["dtype"] == "int32"
        assert count_info["dtype"] == "int32"

        param_order = [str(p) for p in runtime_args["param_order"]]
        assert param_order[:2] == ["tt_start_tile", "tt_tile_count"]
        assert param_order[2:] == ["Mt", "Kt", "Nt"]

        assert [int(x) for x in runtime_args["grid_tiles"]] == [8, 8]
        assert [int(x) for x in runtime_args["local_shape_tiles"]] == [8, 8]
        assert [int(x) for x in runtime_args["shard_grid"]] == [1, 1]
        assert str(runtime_args["partition_mode"]) == "global"

        grid_shape = runtime_args["grid_shape"]
        assert len(grid_shape) == 3, "Grid shape should describe (x, y, z)"
        assert int(runtime_args["iteration_ndims"]) == 2
        assert list(map(str, runtime_args["iteration_symbols"])) == ["bx", "by"]

        # New runtime parameters should be appended to PrimFunc params
        param_names = [p.name for p in func.params]
        assert param_names[-2:] == ["tt_start_tile", "tt_tile_count"]

        # Persistent loop should wrap the body
        assert isinstance(func.body, tvm.tir.For)
        assert int(func.attrs["tt_persistent_iteration_ndims"]) == 2

    def test_grid_to_persistent_one_dim(self):
        """Ensure 1D kernels produce correct metadata and params."""

        @T.prim_func
        def kernel_1d(A: T.Buffer((1024,), "float16")):
            with T.Kernel(32) as bx:
                T.evaluate(A[bx])

        mod = tvm.IRModule.from_expr(kernel_1d.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)
        mod = apply_tt_metadata_passes(mod)
        mod = grid_to_persistent_tt(mod)

        func = mod["main"]
        runtime_args = func.attrs["tt_runtime_args"]
        assert int(runtime_args["iteration_ndims"]) == 1
        assert list(map(str, runtime_args["iteration_symbols"])) == ["bx"]
        assert int(func.attrs["tt_persistent_iteration_ndims"]) == 1
        assert [int(x) for x in runtime_args["grid_tiles"]] == [1, 32]
        assert runtime_args["partition_mode"] == "global"

    def test_grid_to_persistent_local_shard(self):

        @T.prim_func
        def shard_kernel(A: T.Buffer((128, 256), "float16")):
            with T.Kernel(2, 4) as (bx, by):
                T.evaluate(A[bx, by])

        # Annotate with sharded layout required for local_shard mode
        prim = annotate_tt_layout(
            shard_kernel,
            {
                "A": {
                    "memory": "L1",
                    "layout": "sharded",
                    "nd_shard": {
                        "axes": ["M", "N"],
                        "grid": [2, 4],
                        "shard_shape_elems": [64, 64],
                    },
                }
            },
        )

        prim = annotate_tt_schedule(
            prim,
            {
                "partition_mode": "local_shard",
            },
        ).with_attr("global_symbol", "main")

        mod = tvm.IRModule.from_expr(prim)
        mod = apply_tt_defaults(mod)
        mod = apply_tt_metadata_passes(mod)
        mod = apply_layout_aware_metadata_passes(mod)
        mod = grid_to_persistent_tt(mod)

        func = mod["main"]
        runtime_args = func.attrs["tt_runtime_args"]
        assert str(runtime_args["partition_mode"]) == "local_shard"
        param_order = [str(x) for x in runtime_args["param_order"]]
        assert param_order[:4] == [
            "tt_start_tile",
            "tt_tile_count",
            "Mt",
            "Kt",
        ]
        assert param_order[-2:] == ["tt_shard_coord_y", "tt_shard_coord_x"]

        param_names = [p.name for p in func.params]
        assert param_names[-4:] == [
            "tt_start_tile",
            "tt_tile_count",
            "tt_shard_coord_y",
            "tt_shard_coord_x",
        ]

    def test_grid_to_persistent_three_dim(self):
        """Ensure 3D kernels expose full iteration metadata."""

        @T.prim_func
        def kernel_3d(A: T.Buffer((64,), "float16")):
            with T.Kernel(2, 2, 2) as (bx, by, bz):
                T.evaluate(A[bx + by + bz])

        mod = tvm.IRModule.from_expr(kernel_3d.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)
        mod = apply_tt_metadata_passes(mod)
        mod = grid_to_persistent_tt(mod)

        func = mod["main"]
        runtime_args = func.attrs["tt_runtime_args"]
        assert int(runtime_args["iteration_ndims"]) == 3
        assert list(map(str, runtime_args["iteration_symbols"])) == ["bx", "by", "bz"]
        assert int(func.attrs["tt_persistent_iteration_ndims"]) == 3

    def test_apply_tt_transform_passes(self):
        """Test full persistent transform stage pipeline on simple kernel."""

        @T.prim_func
        def kernel(A: T.Buffer((128, 128), "float16")):
            with T.Kernel(4, 4) as (bx, by):
                pass

        mod = tvm.IRModule.from_expr(kernel.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)
        mod = apply_tt_metadata_passes(mod)

        # Apply persistent transform stage pipeline
        mod = apply_tt_transform_passes(mod)

        func = mod["main"]
        # Should have both metadata inference stage and persistent transform stage metadata
        assert "tt_num_tiles" in func.attrs  # From metadata inference stage
        assert "tt_runtime_args" in func.attrs  # From persistent transform stage
        assert "tt_persistent_loop" in func.attrs  # From persistent transform stage


class TestPipelineIntegration:
    """Test integration of all three workstreams."""

    def test_full_pipeline_integration(self):
        """Test TT defaults stage → metadata inference stage → persistent transform stage pipeline on realistic GEMM."""

        @T.prim_func
        def gemm(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                 C: T.Buffer((256, 256), "float16")):
            with T.Kernel(T.ceildiv(256, 32), T.ceildiv(256, 32)) as (bx, by):
                # Stub kernel body
                pass

        # Full pipeline
        mod = tvm.IRModule.from_expr(gemm.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)  # TT defaults stage
        mod = apply_tt_metadata_passes(mod)  # metadata inference stage
        mod = apply_tt_transform_passes(mod)  # persistent transform stage

        func = mod["main"]

        # Verify TT defaults stage metadata
        assert "tt_schedule_policy" in func.attrs
        assert "tt_layout_type" in func.attrs

        # Verify metadata inference stage metadata
        assert "tt_num_tiles" in func.attrs
        assert "tt_tiles_per_core" in func.attrs

        # Verify persistent transform stage metadata
        assert "tt_runtime_args" in func.attrs
        assert "tt_persistent_loop" in func.attrs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
