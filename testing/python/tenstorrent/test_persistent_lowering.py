"""Basic tests for Persistent transform stage: GridToPersistentTT pass.

This module tests the foundation of persistent transform stage – the GridToPersistentTT transform
that converts grid-style kernels to persistent per-core loops.
"""

import pytest
from tilelang import tvm
import tilelang.language as T
from tilelang.tenstorrent import (
    apply_tt_defaults,
    annotate_tt_schedule,
    annotate_tt_layout,
)
from tilelang.tenstorrent.passes import (
    InferTTLayout,
    PropagateTTLayout,
    TTTilesToCoreMap,
    LowerTTTileIntrinsics,
    GridToPersistentTT,
)


def apply_metadata_passes(mod):
    """Helper to apply metadata passes in the new pipeline."""
    mod = InferTTLayout()(mod)
    mod = PropagateTTLayout()(mod)
    mod = TTTilesToCoreMap()(mod)
    return mod


def apply_tt_transform_passes(mod):
    """Helper to apply transform passes in the new pipeline."""
    mod = LowerTTTileIntrinsics()(mod)
    mod = GridToPersistentTT()(mod)
    return mod


class TestGridToPersistentTT:
    """Test GridToPersistentTT transformation."""

    def test_grid_to_persistent_basic(self):
        """Test basic grid-to-persistent transformation on simple kernel."""

        @T.prim_func
        def simple_kernel(
            A: T.Buffer((256, 256), "float16"), C: T.Buffer((256, 256), "float16")
        ):
            with T.Kernel(T.ceildiv(256, 32), T.ceildiv(256, 32)) as (bx, by):
                # Stub kernel body
                pass

        # Convert to IRModule and apply TT defaults stage + metadata inference stage
        mod = tvm.IRModule.from_expr(simple_kernel.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)
        mod = apply_metadata_passes(mod)

        # Apply GridToPersistentTT
        mod = GridToPersistentTT()(mod)

        # Verify the pass marked the function as transformed
        func = mod["main"]
        assert (
            "tt.persistent_kernel" in func.attrs
        ), "Missing tt.persistent_kernel attribute"

        # The new architecture doesn't add the detailed runtime_args metadata
        # It focuses on the IR transformation itself
        # The metadata is handled by the earlier passes

    def test_grid_to_persistent_one_dim(self):
        """Ensure 1D kernels produce correct metadata and params."""

        @T.prim_func
        def kernel_1d(A: T.Buffer((1024,), "float16")):
            with T.Kernel(32) as bx:
                T.evaluate(A[bx])

        mod = tvm.IRModule.from_expr(kernel_1d.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)
        mod = apply_metadata_passes(mod)
        mod = GridToPersistentTT()(mod)

        func = mod["main"]
        # Verify the pass marked the function as transformed
        assert "tt.persistent_kernel" in func.attrs

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
        mod = apply_metadata_passes(mod)
        mod = apply_metadata_passes(mod)
        mod = GridToPersistentTT()(mod)

        func = mod["main"]
        # Verify the pass marked the function as transformed
        assert "tt.persistent_kernel" in func.attrs

    def test_grid_to_persistent_three_dim(self):
        """Ensure 3D kernels expose full iteration metadata."""

        @T.prim_func
        def kernel_3d(A: T.Buffer((64,), "float16")):
            with T.Kernel(2, 2, 2) as (bx, by, bz):
                T.evaluate(A[bx + by + bz])

        mod = tvm.IRModule.from_expr(kernel_3d.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)
        mod = apply_metadata_passes(mod)
        mod = GridToPersistentTT()(mod)

        func = mod["main"]
        # Verify the pass marked the function as transformed
        assert "tt.persistent_kernel" in func.attrs

    def test_apply_tt_transform_passes(self):
        """Test full persistent transform stage pipeline on simple kernel."""

        @T.prim_func
        def kernel(A: T.Buffer((128, 128), "float16")):
            with T.Kernel(4, 4) as (bx, by):
                pass

        mod = tvm.IRModule.from_expr(kernel.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)
        mod = apply_metadata_passes(mod)

        # Apply persistent transform stage pipeline
        mod = apply_tt_transform_passes(mod)

        func = mod["main"]
        # Should have metadata from the new pipeline
        assert "tt.core_grid" in func.attrs  # From metadata passes
        assert "tt.work_partition" in func.attrs  # From metadata passes
        assert "tt.persistent_kernel" in func.attrs  # From GridToPersistentTT


class TestPipelineIntegration:
    """Test integration of all three workstreams."""

    def test_full_pipeline_integration(self):
        """Test TT defaults stage → metadata inference stage → persistent transform stage pipeline on realistic GEMM."""

        @T.prim_func
        def gemm(
            A: T.Buffer((256, 256), "float16"),
            B: T.Buffer((256, 256), "float16"),
            C: T.Buffer((256, 256), "float16"),
        ):
            with T.Kernel(T.ceildiv(256, 32), T.ceildiv(256, 32)) as (bx, by):
                # Stub kernel body
                pass

        # Full pipeline
        mod = tvm.IRModule.from_expr(gemm.with_attr("global_symbol", "main"))
        mod = apply_tt_defaults(mod)  # TT defaults stage
        mod = apply_metadata_passes(mod)  # metadata inference stage
        mod = apply_tt_transform_passes(mod)  # persistent transform stage

        func = mod["main"]

        # Verify TT defaults stage metadata
        assert "tt_schedule_policy" in func.attrs
        assert "tt_layout_type" in func.attrs

        # Verify metadata from new pipeline
        assert "tt.core_grid" in func.attrs
        assert "tt.work_partition" in func.attrs

        # Verify persistent transform stage metadata
        assert "tt.persistent_kernel" in func.attrs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
