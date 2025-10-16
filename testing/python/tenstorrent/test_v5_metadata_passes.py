"""
Test Suite for v5 Metadata Passes (A1, A2, B1)
Tests the updated passes:
- A1: InferTTLayout (v5 spec)
- A2: PropagateTTLayout (v5 spec)
- B1: LayoutAwareWorkPartitionTT (v5 spec)
"""

import pytest
# Import tilelang first to get proper TVM
import tilelang
from tilelang import tvm
from tvm.script import tir as T
import tvm.script
import sys
import os

# Add passes directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../tilelang/tenstorrent/passes"))

# Import v5 passes
from infer_tt_layout_v5 import InferTTLayout_v5
from propagate_tt_layout_v5 import PropagateTTLayout_v5
from layout_aware_work_partition_tt_v5 import LayoutAwareWorkPartitionTT_v5


class TestInferTTLayout_v5:
    """Test A1: InferTTLayout v5 pass"""

    def test_default_layout_inference(self):
        """Test that default layouts are properly inferred"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                     C: T.Buffer((256, 256), "float16")):
                T.evaluate(0)

        # Apply A1 pass
        pass_a1 = InferTTLayout_v5()
        result = pass_a1(TestModule)
        func = result["func"]

        # Check that layouts were created with v5 schema
        assert "tt.buffer.A" in func.attrs
        assert "tt.buffer.B" in func.attrs
        assert "tt.buffer.C" in func.attrs

        # Verify layout structure
        layout_a = func.attrs["tt.buffer.A"]
        assert "memory" in layout_a  # Not "shard"
        assert "layout" in layout_a  # Not "interleave"
        assert "tile_shape" in layout_a
        assert "dtype" in layout_a

        # Check defaults
        assert layout_a["memory"] == "DRAM"
        assert layout_a["layout"] == "interleaved"
        assert list(layout_a["tile_shape"]) == [32, 32]

    def test_user_annotations(self):
        """Test that user annotations are properly applied"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16")):
                T.evaluate(0)

        # Apply with user annotations
        user_annot = {
            "A": {
                "memory": "L1",
                "layout": "sharded",
                "nd_shard": {
                    "axes": ["M", "N"],
                    "grid": [2, 4]
                }
            }
        }

        pass_a1 = InferTTLayout_v5(user_annot)
        result = pass_a1(TestModule)
        func = result["func"]

        layout_a = func.attrs["tt.buffer.A"]
        assert layout_a["memory"] == "L1"
        assert layout_a["layout"] == "sharded"
        assert "nd_shard" in layout_a
        assert list(layout_a["nd_shard"]["grid"]) == [2, 4]

    def test_l1_validation(self):
        """Test L1 buffer validation"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16")):
                T.evaluate(0)

        # Try L1 sharded without nd_shard (should fail)
        user_annot = {
            "A": {
                "memory": "L1",
                "layout": "sharded"
                # Missing nd_shard
            }
        }

        pass_a1 = InferTTLayout_v5(user_annot)

        with pytest.raises(ValueError, match="L1 sharded.*requires nd_shard"):
            pass_a1(TestModule)

    def test_halo_rejection(self):
        """Test that halo metadata is rejected"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16")):
                T.evaluate(0)

        user_annot = {
            "A": {
                "memory": "DRAM",
                "layout": "interleaved",
                "halo": [1, 1]  # Not supported
            }
        }

        pass_a1 = InferTTLayout_v5(user_annot)

        with pytest.raises(ValueError, match="Halo metadata not supported"):
            pass_a1(TestModule)


class TestPropagateTTLayout_v5:
    """Test A2: PropagateTTLayout v5 pass"""

    def test_cb_descriptor_generation(self):
        """Test CB descriptor generation from layouts"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                     C: T.Buffer((256, 256), "float16")):
                T.evaluate(0)

        func = TestModule["func"]

        # Simulate A1 output
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
            "dtype": "fp16"
        })
        func = func.with_attr("tt.buffer.C", {
            "memory": "DRAM",
            "layout": "interleaved",
            "tile_shape": [32, 32],
            "dtype": "bf16"
        })

        TestModule["func"] = func

        # Apply A2 pass
        pass_a2 = PropagateTTLayout_v5()
        result = pass_a2(TestModule)
        func = result["func"]

        # Check CB descriptors
        assert "tt.cb_descriptors" in func.attrs
        cb_descs = func.attrs["tt.cb_descriptors"]

        # Should have CBs for each buffer
        assert len(cb_descs) == 3

        # Check CB properties
        for _cb_name, cb_desc in cb_descs.items():
            assert "page_size" in cb_desc
            assert "depth" in cb_desc
            assert "data_format" in cb_desc

            # Page size should be 2KB for bf16/fp16 32x32 tiles
            assert cb_desc["page_size"] == 2048  # 32*32*2

            # Default depth is 2 (double buffering)
            if cb_desc["source_memory"] == "DRAM":
                assert cb_desc["depth"] == 2

    def test_data_format_conversion(self):
        """Test dtype to Metalium format conversion"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(A: T.Buffer((256, 256), "bfloat16"), B: T.Buffer((256, 256), "float16"),
                     C: T.Buffer((256, 256), "float32")):
                T.evaluate(0)

        func = TestModule["func"]

        # Add layouts with different dtypes
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
            "dtype": "fp16"
        })
        func = func.with_attr("tt.buffer.C", {
            "memory": "DRAM",
            "layout": "interleaved",
            "tile_shape": [32, 32],
            "dtype": "fp32"
        })

        TestModule["func"] = func

        pass_a2 = PropagateTTLayout_v5()
        result = pass_a2(TestModule)
        func = result["func"]

        cb_descs = func.attrs["tt.cb_descriptors"]

        # Check data format conversions
        found_formats = set()
        for cb_desc in cb_descs.values():
            found_formats.add(cb_desc["data_format"])

        assert "Float16_b" in found_formats  # bf16
        assert "Float16" in found_formats  # fp16
        assert "Float32" in found_formats  # fp32

    def test_cb_summary_generation(self):
        """Test CB summary metadata"""

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

        pass_a2 = PropagateTTLayout_v5()
        result = pass_a2(TestModule)
        func = result["func"]

        # Check summary
        assert "tt.cb_summary" in func.attrs
        summary = func.attrs["tt.cb_summary"]

        assert "total_cbs" in summary
        assert "total_l1_bytes" in summary
        assert "fits_in_l1" in summary


class TestLayoutAwareWorkPartitionTT_v5:
    """Test B1: LayoutAwareWorkPartitionTT v5 pass"""

    def test_global_partition_mode(self):
        """Test global partition mode selection"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16"), C: T.Buffer((256, 256), "float16")):
                T.evaluate(0)

        func = TestModule["func"]

        # All DRAM interleaved -> global mode
        func = func.with_attr("tt.buffer.A", {
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
        TestModule["func"] = func

        pass_b1 = LayoutAwareWorkPartitionTT_v5()
        result = pass_b1(TestModule)
        func = result["func"]

        # Check partition mode
        assert func.attrs["tt.partition_mode"] == "global"

        # Check runtime args for global mode
        runtime_args = func.attrs["tt.runtime_args"]
        assert "start_id" in runtime_args
        assert "count" in runtime_args
        assert "Mt" in runtime_args
        assert "Kt" in runtime_args
        assert "Nt" in runtime_args
        # Should NOT have shard args
        assert "Sm" not in runtime_args
        assert "sy" not in runtime_args

    def test_local_shard_partition_mode(self):
        """Test local shard partition mode selection"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(C: T.Buffer((256, 256), "float16")):
                T.evaluate(0)

        func = TestModule["func"]

        # L1 sharded -> local_shard mode
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

        pass_b1 = LayoutAwareWorkPartitionTT_v5()
        result = pass_b1(TestModule)
        func = result["func"]

        # Check partition mode
        assert func.attrs["tt.partition_mode"] == "local_shard"

        # Check shard-specific attributes
        assert "tt.shard_grid" in func.attrs
        assert list(func.attrs["tt.shard_grid"]) == [2, 4]
        assert "tt.local_shape_tiles" in func.attrs
        assert list(func.attrs["tt.local_shape_tiles"]) == [4, 2]

        # Check runtime args for local_shard mode
        runtime_args = func.attrs["tt.runtime_args"]
        assert "Sm" in runtime_args  # Shard M tiles
        assert "Sn" in runtime_args  # Shard N tiles
        assert "Gy" in runtime_args  # Grid Y
        assert "Gx" in runtime_args  # Grid X
        assert "sy" in runtime_args  # Shard coord Y
        assert "sx" in runtime_args  # Shard coord X

    def test_work_partition_generation(self):
        """Test work partition assignment generation"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(C: T.Buffer((256, 256), "float16")):
                T.evaluate(0)

        func = TestModule["func"]
        func = func.with_attr("tt.buffer.C", {
            "memory": "DRAM",
            "layout": "interleaved",
            "tile_shape": [32, 32],
            "dtype": "bf16"
        })
        TestModule["func"] = func

        pass_b1 = LayoutAwareWorkPartitionTT_v5()
        result = pass_b1(TestModule)
        func = result["func"]

        # Check work partition
        assert "tt.work_partition" in func.attrs
        work_partition = func.attrs["tt.work_partition"]

        # Should have assignments for each core
        assert len(work_partition) > 0

        # Check core ranges
        assert "tt.core_ranges" in func.attrs
        core_ranges = func.attrs["tt.core_ranges"]
        assert len(core_ranges) > 0


class TestPassIntegration_v5:
    """Test integration of all v5 metadata passes"""

    def test_full_pipeline_global_mode(self):
        """Test A1->A2->B1 pipeline for global mode"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def gemm(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                     C: T.Buffer((256, 256), "float16")):
                for i, j in T.grid(256, 256):
                    C[i, j] = A[i, j] + B[i, j]

        # Apply passes in sequence
        # A1: InferTTLayout
        pass_a1 = InferTTLayout_v5()
        result = pass_a1(TestModule)

        # A2: PropagateTTLayout
        pass_a2 = PropagateTTLayout_v5()
        result = pass_a2(result)

        # B1: LayoutAwareWorkPartitionTT
        pass_b1 = LayoutAwareWorkPartitionTT_v5()
        result = pass_b1(result)

        func = result["gemm"]

        # Verify complete metadata
        # From A1
        assert "tt.buffer.A" in func.attrs
        assert "tt.buffer.B" in func.attrs
        assert "tt.buffer.C" in func.attrs

        # From A2
        assert "tt.cb_descriptors" in func.attrs
        assert "tt.cb_summary" in func.attrs

        # From B1
        assert "tt.partition_mode" in func.attrs
        assert "tt.core_grid" in func.attrs
        assert "tt.grid_tiles" in func.attrs
        assert "tt.work_partition" in func.attrs
        assert "tt.runtime_args" in func.attrs

        # Should be global mode
        assert func.attrs["tt.partition_mode"] == "global"

    def test_full_pipeline_local_shard_mode(self):
        """Test A1->A2->B1 pipeline for local shard mode"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(C: T.Buffer((256, 256), "float16")):
                T.evaluate(0)

        # Apply with L1 sharded annotation
        user_annot = {
            "C": {
                "memory": "L1",
                "layout": "sharded",
                "nd_shard": {
                    "axes": ["M", "N"],
                    "grid": [4, 4],
                    "projected_grid": [4, 4],
                    "projected_shard_tiles": [2, 2]
                }
            }
        }

        # A1: InferTTLayout
        pass_a1 = InferTTLayout_v5(user_annot)
        result = pass_a1(TestModule)

        # A2: PropagateTTLayout
        pass_a2 = PropagateTTLayout_v5()
        result = pass_a2(result)

        # B1: LayoutAwareWorkPartitionTT
        pass_b1 = LayoutAwareWorkPartitionTT_v5()
        result = pass_b1(result)

        func = result["func"]

        # Should be local_shard mode
        assert func.attrs["tt.partition_mode"] == "local_shard"
        assert list(func.attrs["tt.shard_grid"]) == [4, 4]
        assert list(func.attrs["tt.local_shape_tiles"]) == [2, 2]

        # Runtime args should include shard parameters
        runtime_args = func.attrs["tt.runtime_args"]
        assert "Sm" in runtime_args
        assert "sy" in runtime_args


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
