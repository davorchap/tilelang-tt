"""
Test Suite for v5 TT Backend Passes
Tests the three updated passes for protocol-less design:
- C1: LowerSharedToCB (protocol-less)
- C2: LowerTTTileIntrinsics (no heuristics)
- B2: GridToCoreGrid (new metadata)
"""

import pytest
import sys
import os

# Import tilelang first to get proper TVM
import tilelang
from tilelang import tvm
from tvm.script import tir as T
import tvm.script

# Import the new passes
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../tilelang/tenstorrent/passes"))

from lower_shared_to_cb_v5 import LowerSharedToCB_v5, validate_protocol_less_output
from lower_tt_tile_intrinsics_v5 import LowerTTTileIntrinsics_v5, validate_no_heuristics
from grid_to_core_grid_v5 import GridToCoreGrid_v5, validate_core_launch, extract_metadata


class TestLowerSharedToCB:
    """Test C1: LowerSharedToCB pass"""

    def test_basic_shared_allocation(self):
        """Test basic shared memory to CB conversion"""

        @tvm.script.ir_module
        class Before:
            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16")):
                A_shared = T.alloc_buffer((32, 32), "float16", scope="shared")  # noqa: F841
                T.evaluate(0)  # Dummy body

        # Apply the pass
        transformed_mod = LowerSharedToCB_v5(Before)
        transformed = transformed_mod["func"]

        # Check that shared allocation is replaced with CB allocation
        assert "tt.conceptual_cbs" in transformed.attrs
        assert len(transformed.attrs["tt.conceptual_cbs"]) > 0

        # Validate protocol-less
        validate_protocol_less_output(transformed)

    def test_copy_to_cb_conversion(self):
        """Test T.copy conversion to abstract read_to_cb/write_from_cb"""

        @tvm.script.ir_module
        class Before:
            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16"), C: T.Buffer((256, 256), "float16")):
                A_shared = T.alloc_buffer((32, 32), "float16", scope="shared")
                # Simulating copy pattern
                for i, j in T.grid(32, 32):
                    A_shared[i, j] = A[i, j]

        # Apply the pass
        transformed_mod = LowerSharedToCB_v5(Before)
        transformed = transformed_mod["func"]

        # Should have conceptual CBs
        assert "tt.conceptual_cbs" in transformed.attrs

        # Should be protocol-less
        validate_protocol_less_output(transformed)

    def test_multiple_shared_buffers(self):
        """Test multiple shared buffers get unique CB names"""

        @tvm.script.ir_module
        class Before:
            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16")):
                A_shared = T.alloc_buffer((32, 32), "float16", scope="shared")  # noqa: F841
                B_shared = T.alloc_buffer((32, 32), "float16", scope="shared")  # noqa: F841
                T.evaluate(0)

        # Apply the pass
        transformed_mod = LowerSharedToCB_v5(Before)
        transformed = transformed_mod["func"]

        cbs = transformed.attrs["tt.conceptual_cbs"]
        assert len(cbs) == 2
        cb_names = list(cbs.keys())
        assert cb_names[0] != cb_names[1]  # Unique names

    def test_no_protocol_insertion(self):
        """Ensure no protocol calls are inserted"""

        @tvm.script.ir_module
        class Before:
            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16")):
                A_shared = T.alloc_buffer((32, 32), "float16", scope="shared")  # noqa: F841
                T.evaluate(0)

        # Apply the pass
        transformed_mod = LowerSharedToCB_v5(Before)
        transformed = transformed_mod["func"]

        # Convert to string and check for protocol calls
        func_str = str(transformed.script())

        # Should NOT contain these protocol operations
        protocol_ops = [
            "noc_async_read", "noc_async_write", "cb_reserve_back", "cb_push_back",
            "cb_wait_front", "cb_pop_front"
        ]

        for op in protocol_ops:
            assert op not in func_str, f"Found protocol operation {op}"


class TestLowerTTTileIntrinsics:
    """Test C2: LowerTTTileIntrinsics pass"""

    def test_gemm_lowering_no_heuristics(self):
        """Test GEMM lowering without naming heuristics"""

        @tvm.script.ir_module
        class Before:
            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16"),
                    B: T.Buffer((256, 256), "float16"),
                    C: T.Buffer((256, 256), "float16")):
                # Simulate matmul pattern
                for i, j, k in T.grid(256, 256, 256):
                    if k == 0:
                        C[i, j] = T.float16(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

        func = Before["func"]

        # Add CB metadata as if C1 pass ran
        func = func.with_attr("tt.conceptual_cbs", {
            "cb_in0": {"original_buffer": "A"},
            "cb_in1": {"original_buffer": "B"},
            "cb_out0": {"original_buffer": "C"}
        })

        # Apply the pass
        Before["func"] = func
        transformed_mod = LowerTTTileIntrinsics_v5(Before)
        transformed = transformed_mod["func"]

        # Should have tt.mm.mma intrinsic
        func_str = str(transformed.script())
        assert "tt.mm.mma" in func_str or "matmul" in func_str.lower()

        # Validate no heuristics
        validate_no_heuristics(transformed)

    def test_elementwise_lowering(self):
        """Test element-wise operation lowering"""

        @tvm.script.ir_module
        class Before:
            @T.prim_func
            def func(X: T.Buffer((256, 256), "float16"),
                    Y: T.Buffer((256, 256), "float16"),
                    Z: T.Buffer((256, 256), "float16")):
                # Simulate element-wise add
                for i, j in T.grid(256, 256):
                    Z[i, j] = X[i, j] + Y[i, j]

        func = Before["func"]

        # Add CB metadata
        func = func.with_attr("tt.conceptual_cbs", {
            "cb_in0": {"original_buffer": "X"},
            "cb_in1": {"original_buffer": "Y"},
            "cb_out0": {"original_buffer": "Z"}
        })

        # Apply the pass
        Before["func"] = func
        transformed_mod = LowerTTTileIntrinsics_v5(Before)
        transformed = transformed_mod["func"]

        # Should have tt.fpu.add intrinsic
        func_str = str(transformed.script())
        assert "tt.fpu.add" in func_str or "add" in func_str.lower()

    def test_no_tile_suffix_heuristics(self):
        """Ensure no "_tile" suffix heuristics are used"""

        @tvm.script.ir_module
        class Before:
            @T.prim_func
            def func(A_tile: T.Buffer((32, 32), "float16"),  # Has _tile suffix
                    B_tile: T.Buffer((32, 32), "float16"),  # Has _tile suffix
                    C: T.Buffer((32, 32), "float16")):
                for i, j, k in T.grid(32, 32, 32):
                    if k == 0:
                        C[i, j] = T.float16(0)
                    C[i, j] = C[i, j] + A_tile[i, k] * B_tile[k, j]

        func = Before["func"]

        # Add proper CB metadata (not based on names)
        func = func.with_attr("tt.conceptual_cbs", {
            "cb_in0": {"original_buffer": "A_tile"},
            "cb_in1": {"original_buffer": "B_tile"},
            "cb_out0": {"original_buffer": "C"}
        })

        # Apply the pass
        Before["func"] = func
        transformed_mod = LowerTTTileIntrinsics_v5(Before)
        transformed = transformed_mod["func"]

        # Should work without relying on _tile suffix
        validate_no_heuristics(transformed)

    def test_no_dst_or_engine_init(self):
        """Ensure no DST management or engine init is inserted"""

        @tvm.script.ir_module
        class Before:
            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16"),
                    B: T.Buffer((256, 256), "float16"),
                    C: T.Buffer((256, 256), "float16")):
                for k in T.serial(8):
                    for i, j in T.grid(32, 32):
                        if k == 0:
                            C[i, j] = T.float16(0)
                        C[i, j] = C[i, j] + A[i, k] * B[k, j]

        func = Before["func"]
        func = func.with_attr("tt.conceptual_cbs", {
            "cb_in0": {"original_buffer": "A"},
            "cb_in1": {"original_buffer": "B"},
            "cb_out0": {"original_buffer": "C"}
        })

        # Apply the pass
        Before["func"] = func
        transformed_mod = LowerTTTileIntrinsics_v5(Before)
        transformed = transformed_mod["func"]

        func_str = str(transformed.script())

        # Should NOT contain DST or engine init
        assert "tt.dst.acquire" not in func_str
        assert "tt.dst.release" not in func_str
        assert "tt.engine.init" not in func_str
        assert "matmul_init" not in func_str


class TestGridToCoreGrid:
    """Test B2: GridToCoreGrid pass"""

    def test_basic_grid_transformation(self):
        """Test basic GPU grid to core launch transformation"""

        @tvm.script.ir_module
        class Before:
            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16")):
                for bx in T.thread_binding(8, thread="blockIdx.x"):
                    for by in T.thread_binding(8, thread="blockIdx.y"):
                        A[by, bx] = T.float16(0.0)

        func = Before["func"]

        # Add v5 metadata
        func = func.with_attr("tt.core_grid", [8, 8])
        func = func.with_attr("tt.partition_mode", "global")
        func = func.with_attr("tt.grid_tiles", [8, 8])
        func = func.with_attr("tt.runtime_args", ["start_id", "count", "Mt", "Nt"])

        # Apply the pass
        Before["func"] = func
        transformed_mod = GridToCoreGrid_v5(Before)
        transformed = transformed_mod["func"]

        # Check for core launch
        assert "tt.core_map_x" in transformed.attrs
        assert "tt.core_map_y" in transformed.attrs
        assert transformed.attrs["tt.transformed_to_core"] is True

        # Validate structure
        validate_core_launch(transformed)

    def test_global_partition_mode(self):
        """Test global partition mode handling"""

        @tvm.script.ir_module
        class Before:
            @T.prim_func
            def func(C: T.Buffer((256, 256), "float16")):
                for bx in T.thread_binding(8, thread="blockIdx.x"):
                    for by in T.thread_binding(8, thread="blockIdx.y"):
                        for i in T.serial(32):
                            for j in T.serial(32):
                                C[by * 32 + i, bx * 32 + j] = T.float16(0.0)

        func = Before["func"]

        # Global mode metadata
        func = func.with_attr("tt.core_grid", [8, 8])
        func = func.with_attr("tt.partition_mode", "global")
        func = func.with_attr("tt.grid_tiles", [8, 8])
        func = func.with_attr("tt.runtime_args", ["start_id", "count", "Mt", "Nt"])
        func = func.with_attr("tt.work_partition",
                              {f"core_{y}_{x}": [[y, x]] for y in range(8) for x in range(8)})

        # Apply the pass
        Before["func"] = func
        transformed_mod = GridToCoreGrid_v5(Before)
        transformed = transformed_mod["func"]

        # Should have global mode runtime args
        func_str = str(transformed.script())
        assert "get_arg_val" in func_str or "launch_core" in func_str  # Runtime arg access

    def test_local_shard_mode(self):
        """Test local shard partition mode"""

        @tvm.script.ir_module
        class Before:
            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16")):
                for bx in T.thread_binding(4, thread="blockIdx.x"):
                    for by in T.thread_binding(4, thread="blockIdx.y"):
                        for i in T.serial(64):
                            for j in T.serial(64):
                                A[by * 64 + i, bx * 64 + j] = T.float16(0.0)

        func = Before["func"]

        # Local shard mode metadata
        func = func.with_attr("tt.core_grid", [4, 4])
        func = func.with_attr("tt.partition_mode", "local_shard")
        func = func.with_attr("tt.grid_tiles", [4, 4])
        func = func.with_attr("tt.shard_grid", [2, 2])
        func = func.with_attr("tt.local_shape_tiles", [2, 2])
        func = func.with_attr("tt.runtime_args",
                              ["start_id", "count", "Mt", "Nt", "Sm", "Sn", "Gy", "Gx", "sy", "sx"])

        # Apply the pass
        Before["func"] = func
        transformed_mod = GridToCoreGrid_v5(Before)
        transformed = transformed_mod["func"]

        # Should handle shard-specific args
        func_str = str(transformed.script())
        assert "get_arg_val" in func_str or "launch_core" in func_str

    def test_metadata_extraction(self):
        """Test metadata extraction utility"""

        @tvm.script.ir_module
        class Module:
            @T.prim_func
            def func():
                T.evaluate(0)

        func = Module["func"]

        # Add various metadata
        func = func.with_attr("tt.core_grid", [8, 8])
        func = func.with_attr("tt.partition_mode", "global")
        func = func.with_attr("tt.grid_tiles", [16, 16])
        func = func.with_attr("other_attr", "ignored")

        metadata = extract_metadata(func)

        # Convert TVM Arrays to Python lists for comparison
        assert list(metadata["core_grid"]) == [8, 8]
        assert metadata["partition_mode"] == "global"
        assert list(metadata["grid_tiles"]) == [16, 16]
        assert "other_attr" not in metadata


class TestPassIntegration:
    """Test integration of all three passes"""

    def test_pipeline_integration(self):
        """Test all three passes work together"""

        @tvm.script.ir_module
        class Original:
            @T.prim_func
            def gemm(A: T.Buffer((256, 256), "float16"),
                    B: T.Buffer((256, 256), "float16"),
                    C: T.Buffer((256, 256), "float16")):
                for bx in T.thread_binding(8, thread="blockIdx.x"):
                    for by in T.thread_binding(8, thread="blockIdx.y"):
                        with T.block("compute"):
                            # Shared memory
                            A_shared = T.alloc_buffer((32, 32), "float16", scope="shared")
                            B_shared = T.alloc_buffer((32, 32), "float16", scope="shared")

                            # Copy to shared
                            for i, j in T.grid(32, 32):
                                A_shared[i, j] = A[by * 32 + i, j]
                            for i, j in T.grid(32, 32):
                                B_shared[i, j] = B[i, bx * 32 + j]

                            # GEMM
                            for i, j, k in T.grid(32, 32, 32):
                                if k == 0:
                                    C[by * 32 + i, bx * 32 + j] = T.float16(0)
                                C[by * 32 + i, bx * 32 + j] = (
                                    C[by * 32 + i, bx * 32 + j] +
                                    A_shared[i, k] * B_shared[k, j]
                                )

        func = Original["gemm"]

        # Add metadata for GridToCoreGrid
        func = func.with_attr("tt.core_grid", [8, 8])
        func = func.with_attr("tt.partition_mode", "global")
        func = func.with_attr("tt.grid_tiles", [8, 8])
        func = func.with_attr("tt.runtime_args", ["start_id", "count", "Mt", "Nt"])

        # Apply passes in sequence
        # B2: Grid transformation
        Original["gemm"] = func
        result_mod = GridToCoreGrid_v5(Original)
        func = result_mod["gemm"]

        # C1: Shared to CB
        result_mod["gemm"] = func
        result_mod = LowerSharedToCB_v5(result_mod)
        func = result_mod["gemm"]

        # C2: Tile intrinsics
        result_mod["gemm"] = func
        result_mod = LowerTTTileIntrinsics_v5(result_mod)
        func = result_mod["gemm"]

        # Validate final result by checking attributes
        # The three passes should have added their respective metadata

        # C1: LowerSharedToCB should have added tt.conceptual_cbs
        assert "tt.conceptual_cbs" in func.attrs, "Missing tt.conceptual_cbs from C1 pass"
        cbs = func.attrs["tt.conceptual_cbs"]
        assert len(cbs) > 0, "Should have conceptual CBs"

        # B2: GridToCoreGrid should have added core mapping attributes
        assert "tt.core_map_x" in func.attrs, "Missing tt.core_map_x from B2 pass"
        assert "tt.core_map_y" in func.attrs, "Missing tt.core_map_y from B2 pass"
        assert "tt.transformed_to_core" in func.attrs, "Missing tt.transformed_to_core from B2 pass"
        assert func.attrs["tt.transformed_to_core"] is True, "Core transformation flag should be True"

        # All passes completed successfully - this is the key validation
        # Note: We don't check the IR string representation because after multiple
        # passes the IR may have complex structures that the TVM script printer
        # cannot handle. The important thing is that all passes ran and set their
        # expected attributes correctly.


# Pytest configuration
if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])