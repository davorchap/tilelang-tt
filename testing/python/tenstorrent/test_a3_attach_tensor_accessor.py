"""
Test Suite for A3: AttachTensorAccessorTT Pass
Tests the tensor accessor attachment functionality following v5 specification.
"""

import pytest
import tvm
from tvm.script import tir as T
import tvm.script
import sys
import os

# Add passes directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../tilelang/tenstorrent/passes"))

# Import the A3 pass
from attach_tensor_accessor_tt import AttachTensorAccessorTT


class TestAttachTensorAccessorTT:
    """Test A3: AttachTensorAccessorTT pass"""

    def test_basic_accessor_creation(self):
        """Test that accessors are created for all buffers"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def gemm(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                     C: T.Buffer((256, 256), "float16")):
                for i, j in T.grid(256, 256):
                    C[i, j] = A[i, j] + B[i, j]

        func = TestModule["gemm"]

        # Simulate A1 output (buffer layouts)
        func = func.with_attr("tt.buffer.A", {
            "memory": "DRAM",
            "layout": "interleaved",
            "tile_shape": [32, 32],
            "dtype": "bf16"
        })
        func = func.with_attr("tt.buffer.B", {
            "memory": "DRAM",
            "layout": "interleaved",
            "tile_shape": [32, 32],
            "dtype": "bf16"
        })
        func = func.with_attr("tt.buffer.C", {
            "memory": "DRAM",
            "layout": "interleaved",
            "tile_shape": [32, 32],
            "dtype": "bf16"
        })

        TestModule["gemm"] = func

        # Apply A3 pass
        pass_a3 = AttachTensorAccessorTT()
        result = pass_a3(TestModule)

        func = result["gemm"]

        # Check that accessors were created for all buffers
        assert "tt.tensor_accessor.A" in func.attrs
        assert "tt.tensor_accessor.B" in func.attrs
        assert "tt.tensor_accessor.C" in func.attrs

        # Check accessor summary
        assert "tt.accessor_summary" in func.attrs
        summary = func.attrs["tt.accessor_summary"]
        assert summary["total_accessors"] == 3

    def test_accessor_structure(self):
        """Test that accessors have correct structure"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16")):
                T.evaluate(0)

        func = TestModule["func"]
        func = func.with_attr("tt.buffer.A", {
            "memory": "DRAM",
            "layout": "interleaved",
            "tile_shape": [32, 32],
            "dtype": "bf16"
        })
        TestModule["func"] = func

        # Apply A3 pass
        pass_a3 = AttachTensorAccessorTT()
        result = pass_a3(TestModule)

        func = result["func"]
        accessor_a = func.attrs["tt.tensor_accessor.A"]

        # Check required fields
        assert accessor_a["type"] == "abstract"
        assert accessor_a["buffer_name"] == "A"
        assert accessor_a["layout_ref"] == "tt.buffer.A"
        assert "stride_mode" in accessor_a
        assert "access_pattern" in accessor_a
        assert "tile_dims" in accessor_a
        assert "tiles_per_dim" in accessor_a
        assert "memory" in accessor_a
        assert "layout_type" in accessor_a

        # Check runtime binding fields are null (filled by D2)
        assert accessor_a["base_offset"] is None
        assert accessor_a["runtime_arg_idx"] is None

    def test_stride_mode_determination(self):
        """Test stride mode is correctly determined based on layout"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                     C: T.Buffer((256, 256), "float16")):
                T.evaluate(0)

        func = TestModule["func"]

        # DRAM interleaved -> tiled
        func = func.with_attr("tt.buffer.A", {
            "memory": "DRAM",
            "layout": "interleaved",
            "tile_shape": [32, 32],
            "dtype": "bf16"
        })

        # L1 always tiled
        func = func.with_attr("tt.buffer.B", {
            "memory": "L1",
            "layout": "interleaved",
            "tile_shape": [32, 32],
            "dtype": "bf16"
        })

        # L1 sharded -> sharded
        func = func.with_attr(
            "tt.buffer.C", {
                "memory": "L1",
                "layout": "sharded",
                "tile_shape": [32, 32],
                "dtype": "bf16",
                "nd_shard": {
                    "axes": ["M", "N"],
                    "grid": [2, 4],
                    "projected_grid": [2, 4],
                    "projected_shard_tiles": [4, 2]
                }
            })

        TestModule["func"] = func

        pass_a3 = AttachTensorAccessorTT()
        result = pass_a3(TestModule)
        func = result["func"]

        # Check stride modes
        assert func.attrs["tt.tensor_accessor.A"]["stride_mode"] == "tiled"
        assert func.attrs["tt.tensor_accessor.B"]["stride_mode"] == "tiled"
        assert func.attrs["tt.tensor_accessor.C"]["stride_mode"] == "sharded"

    def test_access_pattern_detection(self):
        """Test access pattern detection based on buffer names"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(input_data: T.Buffer((256, 256), "float16"), weights: T.Buffer(
                (256, 256), "float16"), output: T.Buffer((256, 256), "float16")):
                T.evaluate(0)

        func = TestModule["func"]

        # Add layouts for each buffer
        for buffer_name in ["input_data", "weights", "output"]:
            func = func.with_attr(f"tt.buffer.{buffer_name}", {
                "memory": "DRAM",
                "layout": "interleaved",
                "tile_shape": [32, 32],
                "dtype": "bf16"
            })

        TestModule["func"] = func

        pass_a3 = AttachTensorAccessorTT()
        result = pass_a3(TestModule)
        func = result["func"]

        # Check access patterns
        assert func.attrs["tt.tensor_accessor.input_data"]["access_pattern"] == "input"
        assert func.attrs["tt.tensor_accessor.weights"]["access_pattern"] == "weight"
        assert func.attrs["tt.tensor_accessor.output"]["access_pattern"] == "output"

    def test_tile_parameter_calculation(self):
        """Test tile parameter calculation"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(A: T.Buffer((512, 384), "float16")  # Not evenly divisible
                    ):
                T.evaluate(0)

        func = TestModule["func"]
        func = func.with_attr("tt.buffer.A", {
            "memory": "DRAM",
            "layout": "interleaved",
            "tile_shape": [32, 32],
            "dtype": "bf16"
        })
        TestModule["func"] = func

        pass_a3 = AttachTensorAccessorTT()
        result = pass_a3(TestModule)
        func = result["func"]

        accessor_a = func.attrs["tt.tensor_accessor.A"]

        # Check tile dimensions (convert TVM Array to list)
        assert list(accessor_a["tile_dims"]) == [32, 32]

        # Check tiles per dimension (512/32=16, 384/32=12) (convert TVM Array to list)
        assert list(accessor_a["tiles_per_dim"]) == [16, 12]

        # Check tile size in bytes (32*32*2 for bf16)
        assert accessor_a["tile_size_bytes"] == 2048

    def test_sharding_info_extraction(self):
        """Test that sharding info is correctly extracted"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16")):
                T.evaluate(0)

        func = TestModule["func"]

        # Add sharded buffer layout
        func = func.with_attr(
            "tt.buffer.A", {
                "memory": "L1",
                "layout": "sharded",
                "tile_shape": [32, 32],
                "dtype": "bf16",
                "nd_shard": {
                    "axes": ["M", "N"],
                    "grid": [2, 4],
                    "projected_grid": [2, 4],
                    "projected_shard_tiles": [4, 2],
                    "order": "row_major"
                }
            })
        TestModule["func"] = func

        pass_a3 = AttachTensorAccessorTT()
        result = pass_a3(TestModule)
        func = result["func"]

        accessor_a = func.attrs["tt.tensor_accessor.A"]
        sharding = accessor_a["sharding"]

        # Check sharding info (convert TVM Arrays to lists where needed)
        assert sharding["enabled"] is True

        # Convert axes if it's a TVM Array
        axes = sharding["axes"]
        if hasattr(axes, '__iter__') and not isinstance(axes, str):
            axes = list(axes)
        assert axes == ["M", "N"]

        # Convert grid if it's a TVM Array
        grid = sharding["grid"]
        if hasattr(grid, '__iter__') and not isinstance(grid, str):
            grid = list(grid)
        assert grid == [2, 4]

        # Convert shard_tiles if it's a TVM Array
        shard_tiles = sharding["shard_tiles"]
        if hasattr(shard_tiles, '__iter__') and not isinstance(shard_tiles, str):
            shard_tiles = list(shard_tiles)
        assert shard_tiles == [4, 2]

        assert sharding["order"] == "row_major"

    def test_no_layouts_skip(self):
        """Test that pass gracefully skips when no layouts present"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16")):
                T.evaluate(0)

        # No buffer layouts added

        pass_a3 = AttachTensorAccessorTT()
        result = pass_a3(TestModule)
        func = result["func"]

        # Should not have any accessor attributes
        assert "tt.tensor_accessor.A" not in func.attrs
        assert "tt.accessor_summary" not in func.attrs

    def test_accessor_summary(self):
        """Test accessor summary generation"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                     C: T.Buffer((256, 256), "float16")):
                T.evaluate(0)

        func = TestModule["func"]

        # Mix of different memory and patterns
        func = func.with_attr("tt.buffer.A", {
            "memory": "DRAM",
            "layout": "interleaved",
            "tile_shape": [32, 32],
            "dtype": "bf16"
        })
        func = func.with_attr(
            "tt.buffer.B", {
                "memory": "L1",
                "layout": "sharded",
                "tile_shape": [32, 32],
                "dtype": "bf16",
                "nd_shard": {
                    "axes": ["M", "N"],
                    "grid": [2, 2],
                    "projected_grid": [2, 2],
                    "projected_shard_tiles": [4, 4]
                }
            })
        func = func.with_attr("tt.buffer.C", {
            "memory": "L1",
            "layout": "interleaved",
            "tile_shape": [32, 32],
            "dtype": "bf16"
        })

        TestModule["func"] = func

        pass_a3 = AttachTensorAccessorTT()
        result = pass_a3(TestModule)
        func = result["func"]

        summary = func.attrs["tt.accessor_summary"]

        # Check summary contents
        assert summary["total_accessors"] == 3

        # Check access patterns count
        assert "access_patterns" in summary
        assert summary["access_patterns"]["input"] == 1  # A
        assert summary["access_patterns"]["weight"] == 1  # B (matches "b" in weight patterns)
        assert summary["access_patterns"]["output"] == 1  # C

        # Check stride modes count
        assert "stride_modes" in summary
        assert summary["stride_modes"]["tiled"] == 2  # A and C
        assert summary["stride_modes"]["sharded"] == 1  # B

        # Check memory types count
        assert "memory_types" in summary
        assert summary["memory_types"]["DRAM"] == 1  # A
        assert summary["memory_types"]["L1"] == 2  # B and C


class TestA3Integration:
    """Test A3 integration with other passes"""

    def test_a1_a2_a3_pipeline(self):
        """Test A1->A2->A3 pipeline integration"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def gemm(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                     C: T.Buffer((256, 256), "float16")):
                for i, j in T.grid(256, 256):
                    C[i, j] = A[i, j] + B[i, j]

        # Import A1 and A2 passes
        from infer_tt_layout_v5 import InferTTLayout_v5
        from propagate_tt_layout_v5 import PropagateTTLayout_v5

        # Apply passes in sequence
        # A1: InferTTLayout
        pass_a1 = InferTTLayout_v5()
        result = pass_a1(TestModule)

        # A2: PropagateTTLayout
        pass_a2 = PropagateTTLayout_v5()
        result = pass_a2(result)

        # A3: AttachTensorAccessorTT
        pass_a3 = AttachTensorAccessorTT()
        result = pass_a3(result)

        func = result["gemm"]

        # Verify complete metadata chain
        # From A1
        assert "tt.buffer.A" in func.attrs
        assert "tt.buffer.B" in func.attrs
        assert "tt.buffer.C" in func.attrs

        # From A2
        assert "tt.cb_descriptors" in func.attrs
        assert "tt.cb_summary" in func.attrs

        # From A3
        assert "tt.tensor_accessor.A" in func.attrs
        assert "tt.tensor_accessor.B" in func.attrs
        assert "tt.tensor_accessor.C" in func.attrs
        assert "tt.accessor_summary" in func.attrs

        # Verify accessor links to layout
        accessor_a = func.attrs["tt.tensor_accessor.A"]
        assert accessor_a["layout_ref"] == "tt.buffer.A"
        assert accessor_a["memory"] == "DRAM"  # Should match layout

    def test_a3_with_b1_integration(self):
        """Test A3 working with B1 partition info"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(C: T.Buffer((256, 256), "float16")):
                T.evaluate(0)

        # Import required passes
        from infer_tt_layout_v5 import InferTTLayout_v5
        from propagate_tt_layout_v5 import PropagateTTLayout_v5
        from layout_aware_work_partition_tt_v5 import LayoutAwareWorkPartitionTT_v5

        # Create L1 sharded layout
        user_annot = {
            "C": {
                "memory": "L1",
                "layout": "sharded",
                "nd_shard": {
                    "axes": ["M", "N"],
                    "grid": [2, 2],
                    "projected_grid": [2, 2],
                    "projected_shard_tiles": [4, 4]
                }
            }
        }

        # Apply pass pipeline
        pass_a1 = InferTTLayout_v5(user_annot)
        result = pass_a1(TestModule)

        pass_a2 = PropagateTTLayout_v5()
        result = pass_a2(result)

        pass_a3 = AttachTensorAccessorTT()
        result = pass_a3(result)

        pass_b1 = LayoutAwareWorkPartitionTT_v5()
        result = pass_b1(result)

        func = result["func"]

        # Check that A3 accessor correctly reflects sharding
        accessor_c = func.attrs["tt.tensor_accessor.C"]
        assert accessor_c["stride_mode"] == "sharded"
        assert accessor_c["sharding"]["enabled"] is True

        # Convert grid if it's a TVM Array
        grid = accessor_c["sharding"]["grid"]
        if hasattr(grid, '__iter__') and not isinstance(grid, str):
            grid = list(grid)
        assert grid == [2, 2]

        # Check that B1 partition mode aligns
        assert func.attrs["tt.partition_mode"] == "local_shard"

        # Convert shard_grid if it's a TVM Array
        shard_grid = func.attrs["tt.shard_grid"]
        if hasattr(shard_grid, '__iter__') and not isinstance(shard_grid, str):
            shard_grid = list(shard_grid)
        assert shard_grid == [2, 2]


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
