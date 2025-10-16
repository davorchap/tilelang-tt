"""
Test the BlockTransformer utilities for v5 passes.
This demonstrates how to properly handle TVM Block structures.
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

from block_transformer import (BlockTransformer, is_shared_buffer, extract_buffer_info,
                               create_cb_intrinsic)


class TestBlockTransformer:
    """Test BlockTransformer base class"""

    def test_basic_block_traversal(self):
        """Test that BlockTransformer can traverse Block structures"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16")):
                A_shared = T.alloc_buffer((32, 32), "float16", scope="shared")
                T.evaluate(0)

        # Create a simple transformer that counts blocks
        class BlockCounter(BlockTransformer):

            def __init__(self):
                super().__init__()
                self.block_count = 0
                self.shared_buffers = []

            def visit_block(self, block):
                self.block_count += 1

                # Check for shared buffers
                for buffer in block.alloc_buffers:
                    if is_shared_buffer(buffer):
                        self.shared_buffers.append(buffer.name)

                return super().visit_block(block)

        func = TestModule["func"]
        counter = BlockCounter()
        counter.visit(func.body)

        assert counter.block_count > 0, "Should find at least one Block"
        assert "A_shared" in counter.shared_buffers, "Should find A_shared buffer"

    def test_shared_buffer_detection(self):
        """Test detection of shared memory buffers"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16")):
                A_shared = T.alloc_buffer((32, 32), "float16", scope="shared")
                B_local = T.alloc_buffer((16, 16), "float16", scope="local")
                T.evaluate(0)

        class SharedBufferCollector(BlockTransformer):

            def __init__(self):
                super().__init__()
                self.shared_buffers = []
                self.local_buffers = []

            def process_alloc_buffers(self, alloc_buffers):
                for buffer in alloc_buffers:
                    if is_shared_buffer(buffer):
                        self.shared_buffers.append(extract_buffer_info(buffer))
                    elif buffer.scope() == "local":
                        self.local_buffers.append(extract_buffer_info(buffer))

                return alloc_buffers, []

        func = TestModule["func"]
        collector = SharedBufferCollector()
        collector.visit(func.body)

        assert len(collector.shared_buffers) == 1
        assert collector.shared_buffers[0]['name'] == "A_shared"
        assert collector.shared_buffers[0]['shape'] == [32, 32]

        assert len(collector.local_buffers) == 1
        assert collector.local_buffers[0]['name'] == "B_local"

    def test_cb_allocation_insertion(self):
        """Test inserting CB allocations for shared buffers"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16")):
                A_shared = T.alloc_buffer((32, 32), "float16", scope="shared")
                T.evaluate(0)

        class SharedToCBTransformer(BlockTransformer):

            def __init__(self):
                super().__init__()
                self.cb_counter = 0
                self.cb_map = {}

            def process_alloc_buffers(self, alloc_buffers):
                cb_metadata = []

                for buffer in alloc_buffers:
                    if is_shared_buffer(buffer):
                        cb_name = f"cb_in{self.cb_counter}"
                        self.cb_counter += 1
                        self.cb_map[buffer.name] = cb_name

                        cb_metadata.append({
                            'cb_name': cb_name,
                            'shape': [int(d) for d in buffer.shape],
                            'dtype': str(buffer.dtype),
                            'original': buffer.name
                        })

                return alloc_buffers, cb_metadata

            def create_cb_allocation(self, cb_info):
                return tvm.tir.Evaluate(
                    create_cb_intrinsic(cb_info['cb_name'], cb_info['shape'], cb_info['dtype']))

        func = TestModule["func"]
        transformer = SharedToCBTransformer()
        new_body = transformer.visit(func.body)

        # Check that CB was mapped
        assert "A_shared" in transformer.cb_map
        assert transformer.cb_map["A_shared"] == "cb_in0"

        # The transformed body should be different
        assert new_body != func.body

    def test_block_reconstruction(self):
        """Test that Blocks are properly reconstructed after transformation"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16")):
                with T.block("compute"):
                    A_shared = T.alloc_buffer((32, 32), "float16", scope="shared")
                    for i, j in T.grid(32, 32):
                        A_shared[i, j] = A[i, j]

        class BodyModifier(BlockTransformer):

            def visit_block(self, block):
                # Process the block normally
                result = super().visit_block(block)

                # Verify block structure is preserved
                assert hasattr(result, 'name_hint')
                assert hasattr(result, 'alloc_buffers')
                assert hasattr(result, 'body')

                return result

        func = TestModule["func"]
        modifier = BodyModifier()
        new_body = modifier.visit(func.body)

        # Should still be a valid IR structure
        assert new_body is not None

    def test_nested_blocks(self):
        """Test handling of nested Block structures"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func(A: T.Buffer((256, 256), "float16")):
                with T.block("outer"):
                    A_shared_outer = T.alloc_buffer((64, 64), "float16", scope="shared")
                    with T.block("inner"):
                        A_shared_inner = T.alloc_buffer((32, 32), "float16", scope="shared")
                        T.evaluate(0)

        class NestedBlockCounter(BlockTransformer):

            def __init__(self):
                super().__init__()
                self.blocks = []

            def visit_block(self, block):
                self.blocks.append(block.name_hint if hasattr(block, 'name_hint') else "unnamed")
                return super().visit_block(block)

        func = TestModule["func"]
        counter = NestedBlockCounter()
        counter.visit(func.body)

        # Should find both outer and inner blocks (and possibly root)
        assert len(counter.blocks) >= 2


class TestCBIntrinsicGeneration:
    """Test CB intrinsic generation utilities"""

    def test_create_cb_intrinsic(self):
        """Test CB intrinsic creation"""
        intrinsic = create_cb_intrinsic("cb_test", [32, 32], "float16")

        assert isinstance(intrinsic, tvm.tir.Call)
        # The call_extern op is 'tir.call_extern', function name is first arg
        assert str(intrinsic.op) == "Op(tir.call_extern)"

        # Check arguments: [function_name, cb_name, dim1, dim2, dtype]
        args = intrinsic.args
        assert len(args) == 5  # function_name, cb_name, dim1, dim2, dtype
        assert args[0] == "tt.alloc_cb"  # Function name
        assert args[1].value == "cb_test"  # CB name
        assert args[2].value == 32  # dim1
        assert args[3].value == 32  # dim2
        assert args[4].value == "float16"  # dtype

    def test_buffer_info_extraction(self):
        """Test extracting buffer information"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def func():
                A = T.alloc_buffer((128, 64), "float32", scope="shared")
                T.evaluate(0)

        # Get the buffer from the function
        func = TestModule["func"]

        # Find the buffer in the IR
        found_buffer = None

        def find_buffer(node):
            nonlocal found_buffer
            if isinstance(node, tvm.tir.Block):
                for buf in node.alloc_buffers:
                    if buf.name == "A":
                        found_buffer = buf

        tvm.tir.stmt_functor.post_order_visit(func.body, find_buffer)

        assert found_buffer is not None

        info = extract_buffer_info(found_buffer)
        assert info['name'] == "A"
        assert info['shape'] == [128, 64]
        assert info['dtype'] == "float32"
        assert info['scope'] == "shared"


# Run tests
if __name__ == "__main__":
    # Run with pytest if available, otherwise run directly
    try:
        pytest.main([__file__, "-v", "--tb=short"])
    except ImportError:
        # Manual test running
        print("Running tests manually (install pytest for better output)")

        test_block = TestBlockTransformer()
        test_block.test_basic_block_traversal()
        print("✓ Basic block traversal test passed")

        test_block.test_shared_buffer_detection()
        print("✓ Shared buffer detection test passed")

        test_block.test_cb_allocation_insertion()
        print("✓ CB allocation insertion test passed")

        test_block.test_block_reconstruction()
        print("✓ Block reconstruction test passed")

        test_block.test_nested_blocks()
        print("✓ Nested blocks test passed")

        test_cb = TestCBIntrinsicGeneration()
        test_cb.test_create_cb_intrinsic()
        print("✓ CB intrinsic generation test passed")

        test_cb.test_buffer_info_extraction()
        print("✓ Buffer info extraction test passed")

        print("\nAll tests passed!")
