#!/usr/bin/env python3
"""
Simple test script for A3: AttachTensorAccessorTT Pass
Run without pytest dependency
"""

import sys
import os

# Add passes directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../tilelang/tenstorrent/passes"))

# Import tilelang first to get proper TVM
import tilelang
from tilelang import tvm
from tvm.script import tir as T
import tvm.script

# Import the A3 pass
from attach_tensor_accessor_tt import AttachTensorAccessorTT


def test_basic_accessor_creation():
    """Test that accessors are created for all buffers"""
    print("Testing basic accessor creation...")

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
    assert "tt.tensor_accessor.A" in func.attrs, "Missing accessor for buffer A"
    assert "tt.tensor_accessor.B" in func.attrs, "Missing accessor for buffer B"
    assert "tt.tensor_accessor.C" in func.attrs, "Missing accessor for buffer C"

    # Check accessor summary
    assert "tt.accessor_summary" in func.attrs, "Missing accessor summary"
    summary = func.attrs["tt.accessor_summary"]
    assert summary[
        "total_accessors"] == 3, f"Expected 3 accessors, got {summary['total_accessors']}"

    print("✓ Basic accessor creation test passed")


def test_accessor_structure():
    """Test that accessors have correct structure"""
    print("Testing accessor structure...")

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
    assert accessor_a["type"] == "abstract", f"Expected type='abstract', got {accessor_a['type']}"
    assert accessor_a[
        "buffer_name"] == "A", f"Expected buffer_name='A', got {accessor_a['buffer_name']}"
    assert accessor_a[
        "layout_ref"] == "tt.buffer.A", f"Expected layout_ref='tt.buffer.A', got {accessor_a['layout_ref']}"
    assert "stride_mode" in accessor_a, "Missing stride_mode"
    assert "access_pattern" in accessor_a, "Missing access_pattern"
    assert "tile_dims" in accessor_a, "Missing tile_dims"
    assert "tiles_per_dim" in accessor_a, "Missing tiles_per_dim"
    assert "memory" in accessor_a, "Missing memory"
    assert "layout_type" in accessor_a, "Missing layout_type"

    # Check runtime binding fields are null (filled by D2)
    assert accessor_a[
        "base_offset"] is None, f"base_offset should be None, got {accessor_a['base_offset']}"
    assert accessor_a[
        "runtime_arg_idx"] is None, f"runtime_arg_idx should be None, got {accessor_a['runtime_arg_idx']}"

    print("✓ Accessor structure test passed")


def test_stride_mode_determination():
    """Test stride mode is correctly determined based on layout"""
    print("Testing stride mode determination...")

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
    assert func.attrs["tt.tensor_accessor.A"]["stride_mode"] == "tiled", \
        f"Expected A stride_mode='tiled', got {func.attrs['tt.tensor_accessor.A']['stride_mode']}"
    assert func.attrs["tt.tensor_accessor.B"]["stride_mode"] == "tiled", \
        f"Expected B stride_mode='tiled', got {func.attrs['tt.tensor_accessor.B']['stride_mode']}"
    assert func.attrs["tt.tensor_accessor.C"]["stride_mode"] == "sharded", \
        f"Expected C stride_mode='sharded', got {func.attrs['tt.tensor_accessor.C']['stride_mode']}"

    print("✓ Stride mode determination test passed")


def test_sharding_info_extraction():
    """Test that sharding info is correctly extracted"""
    print("Testing sharding info extraction...")

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

    # Check sharding info
    assert sharding["enabled"] is True, f"Expected sharding enabled, got {sharding['enabled']}"
    assert list(sharding["axes"]) == ["M", "N"], f"Expected axes=['M', 'N'], got {sharding['axes']}"
    assert list(sharding["grid"]) == [2, 4], f"Expected grid=[2, 4], got {sharding['grid']}"
    assert list(sharding["shard_tiles"]) == [
        4, 2
    ], f"Expected shard_tiles=[4, 2], got {sharding['shard_tiles']}"
    assert sharding["order"] == "row_major", f"Expected order='row_major', got {sharding['order']}"

    print("✓ Sharding info extraction test passed")


def test_full_pipeline():
    """Test A1->A2->A3 pipeline integration"""
    print("Testing full A1->A2->A3 pipeline...")

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
    assert "tt.buffer.A" in func.attrs, "Missing buffer layout from A1"
    assert "tt.buffer.B" in func.attrs, "Missing buffer layout from A1"
    assert "tt.buffer.C" in func.attrs, "Missing buffer layout from A1"

    # From A2
    assert "tt.cb_descriptors" in func.attrs, "Missing CB descriptors from A2"
    assert "tt.cb_summary" in func.attrs, "Missing CB summary from A2"

    # From A3
    assert "tt.tensor_accessor.A" in func.attrs, "Missing accessor from A3"
    assert "tt.tensor_accessor.B" in func.attrs, "Missing accessor from A3"
    assert "tt.tensor_accessor.C" in func.attrs, "Missing accessor from A3"
    assert "tt.accessor_summary" in func.attrs, "Missing accessor summary from A3"

    # Verify accessor links to layout
    accessor_a = func.attrs["tt.tensor_accessor.A"]
    assert accessor_a["layout_ref"] == "tt.buffer.A", "Accessor doesn't link to layout"
    assert accessor_a["memory"] == "DRAM", "Accessor memory doesn't match layout"

    print("✓ Full pipeline test passed")


def print_accessor_details(func):
    """Helper to print accessor details"""
    print("\n=== Tensor Accessors ===")

    for buffer in ["A", "B", "C"]:
        key = f"tt.tensor_accessor.{buffer}"
        if key in func.attrs:
            accessor = func.attrs[key]
            print(f"\n{buffer}:")
            print(f"  Type: {accessor['type']}")
            print(f"  Stride mode: {accessor['stride_mode']}")
            print(f"  Access pattern: {accessor['access_pattern']}")
            print(f"  Memory: {accessor['memory']}")
            print(f"  Tile dims: {accessor['tile_dims']}")
            print(f"  Runtime arg idx: {accessor['runtime_arg_idx']} (will be set in D2)")
            if accessor['sharding']['enabled']:
                print(f"  Sharding: {accessor['sharding']['grid']}")

    if "tt.accessor_summary" in func.attrs:
        print("\n=== Accessor Summary ===")
        summary = func.attrs["tt.accessor_summary"]
        for key, value in summary.items():
            print(f"  {key}: {value}")


def main():
    print("=" * 60)
    print("Testing A3: AttachTensorAccessorTT Pass")
    print("=" * 60)

    try:
        test_basic_accessor_creation()
        test_accessor_structure()
        test_stride_mode_determination()
        test_sharding_info_extraction()
        test_full_pipeline()

        print("\n" + "=" * 60)
        print("All tests passed successfully! ✅")
        print("=" * 60)

        # Example with output
        print("\nExample output:")

        @tvm.script.ir_module
        class ExampleModule:

            @T.prim_func
            def gemm(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                     C: T.Buffer((256, 256), "float16")):
                T.evaluate(0)

        func = ExampleModule["gemm"]
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
        ExampleModule["gemm"] = func

        pass_a3 = AttachTensorAccessorTT()
        result = pass_a3(ExampleModule)
        print_accessor_details(result["gemm"])

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
