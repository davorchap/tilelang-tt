"""
Pass C1: LowerSharedToCB (Protocol-less Version) - Fixed with BlockTransformer
Version: 5.1
Date: 2025-10-15

Purpose: Convert shared memory allocations and copies to conceptual circular buffers
         WITHOUT inserting protocol calls. This maintains analyzable mid-level TIR.

This version properly handles TVM Block structures using BlockTransformer.

Input: TIR with T.alloc_shared and T.copy operations in Block structures
Output: TIR with abstract tt.alloc_cb and tt.read_to_cb/write_from_cb operations
"""

import tvm
from tvm import tir
from tvm.tir import stmt_functor
import sys
import os

# Import our BlockTransformer base class
sys.path.append(os.path.dirname(__file__))
from block_transformer import BlockTransformer, is_shared_buffer, extract_buffer_info, create_cb_intrinsic


@tvm.tir.transform.prim_func_pass(opt_level=0)
def LowerSharedToCB_v5(func, mod, ctx):
    """
    Lower shared memory to conceptual circular buffers (protocol-less).

    This pass:
    1. Detects shared memory in Block.alloc_buffers
    2. Replaces with conceptual CB allocations
    3. Transforms T.copy to abstract CB operations
    4. Does NOT insert protocol calls
    """

    class SharedToCBTransformer(BlockTransformer):

        def __init__(self):
            super().__init__()
            self.shared_to_cb_map = {}  # Map shared buffer names to CB names
            self.cb_counter = {"input": 0, "output": 0, "intermediate": 0}
            self.buffer_info = {}  # Store buffer metadata for function attributes

        def process_alloc_buffers(self, alloc_buffers):
            """Process Block.alloc_buffers to detect and convert shared memory"""
            cb_metadata = []

            for buffer in alloc_buffers:
                if is_shared_buffer(buffer):
                    # Generate conceptual CB name
                    cb_name = self._generate_cb_name(buffer.name)
                    self.shared_to_cb_map[buffer.name] = cb_name

                    # Extract buffer information
                    buffer_info = extract_buffer_info(buffer)

                    # Store for later metadata attachment
                    self.buffer_info[cb_name] = {
                        "shape": buffer_info['shape'],
                        "dtype": buffer_info['dtype'],
                        "original_buffer": buffer.name
                    }

                    # Prepare CB allocation metadata
                    cb_metadata.append({
                        'cb_name': cb_name,
                        'shape': buffer_info['shape'],
                        'dtype': buffer_info['dtype'],
                        'original': buffer.name
                    })

            # Return original buffers and CB metadata
            # We keep alloc_buffers unchanged but insert CB allocations
            return alloc_buffers, cb_metadata

        def create_cb_allocation(self, cb_info):
            """Create a CB allocation intrinsic"""
            return tir.Evaluate(
                create_cb_intrinsic(cb_info['cb_name'], cb_info['shape'], cb_info['dtype']))

        def visit_evaluate(self, op):
            """Transform T.copy operations to abstract CB operations"""

            # Check if this is a copy operation
            if self._is_copy_intrinsic(op):
                src, dst = self._extract_copy_args(op)

                # Determine if this involves shared buffers
                if src and dst:
                    # Check if dst is a shared buffer (DRAM -> CB)
                    dst_cb = self._get_cb_for_buffer(dst)
                    if dst_cb:
                        return self._create_read_to_cb(src, dst_cb)

                    # Check if src is a shared buffer (CB -> DRAM)
                    src_cb = self._get_cb_for_buffer(src)
                    if src_cb:
                        return self._create_write_from_cb(src_cb, dst)

            return op

        def visit_buffer_store(self, op):
            """Handle stores to shared buffers"""
            buffer_name = self._get_buffer_name(op.buffer)

            # If storing to a shared buffer, could transform to CB write
            if buffer_name in self.shared_to_cb_map:
                # For now, keep as-is (protocol-less design)
                # Later passes will handle actual CB protocol
                pass

            return op

        def visit_buffer_load(self, op):
            """Handle loads from shared buffers"""
            buffer_name = self._get_buffer_name(op.buffer)

            # If loading from a shared buffer, could transform to CB read
            if buffer_name in self.shared_to_cb_map:
                # For now, keep as-is (protocol-less design)
                pass

            return op

        # Helper methods

        def _generate_cb_name(self, buffer_name):
            """Generate conceptual CB name based on buffer usage"""
            name_lower = buffer_name.lower()

            # Determine CB type based on name patterns
            if any(x in name_lower for x in ["input", "a_", "b_", "x_", "in"]):
                idx = self.cb_counter["input"]
                self.cb_counter["input"] += 1
                return f"cb_in{idx}"
            elif any(x in name_lower for x in ["output", "c_", "y_", "z_", "out", "result"]):
                idx = self.cb_counter["output"]
                self.cb_counter["output"] += 1
                return f"cb_out{idx}"
            else:
                idx = self.cb_counter["intermediate"]
                self.cb_counter["intermediate"] += 1
                return f"cb_intermed{idx}"

        def _is_copy_intrinsic(self, evaluate_node):
            """Check if this is a copy intrinsic"""
            if isinstance(evaluate_node.value, tir.Call):
                call = evaluate_node.value
                if hasattr(call, 'op'):
                    op_name = str(call.op)
                    return any(
                        pattern in op_name for pattern in ["tir.copy", "T.copy", "builtin.copy"])
            return False

        def _extract_copy_args(self, evaluate_node):
            """Extract source and destination from copy call"""
            call = evaluate_node.value
            if len(call.args) >= 2:
                return call.args[0], call.args[1]
            return None, None

        def _get_buffer_name(self, buffer):
            """Get buffer name from buffer object"""
            if hasattr(buffer, 'name'):
                return buffer.name
            return None

        def _get_cb_for_buffer(self, buffer_ref):
            """Get CB name if buffer is mapped to a CB"""
            # Handle different buffer reference types
            if hasattr(buffer_ref, 'buffer'):
                buffer_name = self._get_buffer_name(buffer_ref.buffer)
                return self.shared_to_cb_map.get(buffer_name)
            elif hasattr(buffer_ref, 'name'):
                return self.shared_to_cb_map.get(buffer_ref.name)
            return None

        def _create_read_to_cb(self, src, cb_name):
            """Create abstract read_to_cb operation"""
            return tir.Evaluate(
                tir.call_extern(
                    "void",
                    "tt.read_to_cb",
                    src,  # Source buffer/slice
                    tir.StringImm(cb_name)  # Destination CB
                ))

        def _create_write_from_cb(self, cb_name, dst):
            """Create abstract write_from_cb operation"""
            return tir.Evaluate(
                tir.call_extern(
                    "void",
                    "tt.write_from_cb",
                    tir.StringImm(cb_name),  # Source CB
                    dst  # Destination buffer/slice
                ))

    # Apply transformation
    transformer = SharedToCBTransformer()
    new_body = transformer.visit(func.body)

    # Update function with new body
    func = func.with_body(new_body)

    # Attach CB metadata to function attributes
    if transformer.buffer_info:
        func = func.with_attr("tt.conceptual_cbs", transformer.buffer_info)

    # Add summary metadata
    if transformer.shared_to_cb_map:
        func = func.with_attr("tt.shared_to_cb_map", transformer.shared_to_cb_map)

    return func


def validate_protocol_less_output(func):
    """
    Validate that output is protocol-less.
    Should NOT contain concrete CB IDs or protocol operations.
    """

    class ProtocolChecker:

        def __init__(self):
            self.has_protocol = False
            self.protocol_calls = []

        def check(self, stmt):
            """Check for protocol operations"""

            def visitor(node):
                if isinstance(node, tir.Call):
                    call_name = str(node.op)

                    # Check for protocol operations that shouldn't be here
                    protocol_ops = [
                        "noc_async_read", "noc_async_write", "cb_reserve_back", "cb_push_back",
                        "cb_wait_front", "cb_pop_front", "get_write_ptr", "get_read_ptr"
                    ]

                    if any(proto in call_name for proto in protocol_ops):
                        self.has_protocol = True
                        self.protocol_calls.append(call_name)

            stmt_functor.post_order_visit(stmt, visitor)

    checker = ProtocolChecker()
    checker.check(func.body)

    if checker.has_protocol:
        raise ValueError(f"Output contains protocol operations: {checker.protocol_calls}")

    return True


# Example usage and testing
if __name__ == "__main__":
    import tvm.script
    from tvm.script import tir as T

    # Create test module with shared memory in Blocks
    @tvm.script.ir_module
    class TestModule:

        @T.prim_func
        def gemm_with_shared(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                             C: T.Buffer((256, 256), "float16")):
            # Shared memory allocations in Block
            A_shared = T.alloc_buffer((32, 32), "float16", scope="shared")
            B_shared = T.alloc_buffer((32, 32), "float16", scope="shared")

            # Simulated copy operations
            for i in range(32):
                for j in range(32):
                    A_shared[i, j] = A[i, j]
                    B_shared[i, j] = B[i, j]

            # Computation
            for i, j in T.grid(32, 32):
                C[i, j] = A_shared[i, j] + B_shared[i, j]

    # Get original function
    func = TestModule["gemm_with_shared"]

    # Apply the pass
    transformed_mod = LowerSharedToCB_v5(TestModule)
    transformed = transformed_mod["gemm_with_shared"]

    # Print results
    print("=== Original Function ===")
    print(func.script())
    print("\n=== Transformed Function (Protocol-less) ===")
    print(transformed.script())

    # Check for CB metadata
    if "tt.conceptual_cbs" in transformed.attrs:
        print("\n=== CB Metadata ===")
        for cb_name, info in transformed.attrs["tt.conceptual_cbs"].items():
            print(f"{cb_name}: {info}")

    # Validate
    try:
        validate_protocol_less_output(transformed)
        print("\n✅ Pass validation successful: Output is protocol-less")
    except ValueError as e:
        print(f"\n❌ Validation failed: {e}")
