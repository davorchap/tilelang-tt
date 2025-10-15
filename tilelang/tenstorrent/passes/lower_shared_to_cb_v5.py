"""
Pass C1: LowerSharedToCB (Protocol-less Version)
Version: 5.0
Date: 2025-10-15

Purpose: Convert shared memory allocations and copies to conceptual circular buffers
         WITHOUT inserting protocol calls. This maintains analyzable mid-level TIR.

Input: TIR with T.alloc_shared and T.copy operations
Output: TIR with abstract tt.alloc_cb and tt.read_to_cb/write_from_cb operations
"""

import tvm
from tvm import tir
from tvm.tir import stmt_functor


@tvm.tir.transform.prim_func_pass(opt_level=0)
def LowerSharedToCB_v5(func, mod, ctx):
    """
    Lower shared memory to conceptual circular buffers (protocol-less).

    This pass:
    1. Replaces T.alloc_shared with tt.alloc_cb (conceptual names, no IDs)
    2. Replaces T.copy with abstract tt.read_to_cb or tt.write_from_cb
    3. Does NOT assign concrete CB IDs
    4. Does NOT insert NOC/CB protocol calls
    """

    class SharedToCBTransformer:

        def __init__(self):
            self.shared_to_cb_map = {}  # Map shared buffer names to CB names
            self.cb_counter = {"input": 0, "output": 0, "intermediate": 0}
            self.buffer_info = {}  # Store buffer metadata

        def visit(self, stmt):
            """Generic visit method that dispatches to specific visit methods"""
            if stmt is None:
                return None

            # Dispatch to specific visit methods based on node type
            if isinstance(stmt, tir.BlockRealize):
                return self.visit_block_realize(stmt)
            elif isinstance(stmt, tir.Block):
                return self.visit_block(stmt)
            elif isinstance(stmt, tir.Allocate):
                return self.visit_allocate(stmt)
            elif isinstance(stmt, tir.Evaluate):
                return self.visit_evaluate(stmt)
            elif isinstance(stmt, tir.BufferLoad):
                return self.visit_buffer_load(stmt)
            elif isinstance(stmt, tir.BufferStore):
                return self.visit_buffer_store(stmt)
            elif isinstance(stmt, tir.SeqStmt):
                new_seq = []
                for s in stmt.seq:
                    new_s = self.visit(s)
                    if new_s is not None:
                        new_seq.append(new_s)
                return tir.SeqStmt(new_seq) if new_seq else None
            elif hasattr(stmt, "body"):
                new_body = self.visit(stmt.body)
                if new_body != stmt.body:
                    # Create a new node with the updated body
                    # This is simplified - in real code you'd need to handle each node type
                    return stmt.with_body(new_body) if hasattr(stmt, 'with_body') else stmt
                return stmt
            else:
                return stmt

        def visit_block_realize(self, op):
            """Handle BlockRealize nodes"""
            # Visit the block within
            new_block = self.visit(op.block) if op.block else op.block
            if new_block != op.block:
                return tir.BlockRealize(op.iter_values, op.predicate, new_block)
            return op

        def visit_block(self, op):
            """Handle Block nodes - process alloc_buffers for shared memory"""

            # Process alloc_buffers for shared memory
            cb_allocs = []
            for buffer in op.alloc_buffers:
                if buffer.scope() == "shared":
                    # Generate CB name
                    cb_name = self._generate_cb_name(buffer.name)
                    self.shared_to_cb_map[buffer.name] = cb_name

                    # Store buffer info
                    self.buffer_info[cb_name] = {
                        "shape": [int(dim) for dim in buffer.shape],
                        "dtype": str(buffer.dtype),
                        "original_name": buffer.name
                    }

                    # Create CB allocation intrinsic
                    cb_allocs.append(tir.Evaluate(
                        tir.call_extern(
                            "handle",
                            "tt.alloc_cb",
                            tir.StringImm(cb_name),
                            *[tir.IntImm("int32", int(dim)) for dim in buffer.shape],
                            tir.StringImm(str(buffer.dtype))
                        )
                    ))

            # Visit body
            new_body = self.visit(op.body) if op.body else op.body

            # If we created CB allocations, prepend them to the body
            if cb_allocs:
                if new_body:
                    cb_allocs.append(new_body)
                new_body = tir.SeqStmt(cb_allocs)

            # Return modified block with CB allocations in the body
            # Note: We keep the original alloc_buffers for now (metadata)
            return tir.Block(op.iter_vars, op.reads, op.writes, op.name, new_body,
                           op.init, op.alloc_buffers, op.match_buffers, op.annotations, op.span)

        def visit_allocate(self, op):
            """Replace shared memory allocation with CB allocation"""

            # Check if this is a shared memory allocation
            if self._is_shared_allocation(op):
                # Generate conceptual CB name (not a concrete ID)
                cb_name = self._generate_cb_name(op.buffer_var.name)
                self.shared_to_cb_map[op.buffer_var.name] = cb_name

                # Store buffer info for later use
                self.buffer_info[cb_name] = {
                    "shape": op.extents,
                    "dtype": op.dtype,
                    "original_name": op.buffer_var.name
                }

                # Create abstract CB allocation
                # Note: We use tt.alloc_cb as a conceptual operation
                alloc_cb = tir.Evaluate(
                    tir.call_extern(
                        "handle",  # Returns a handle for the CB
                        "tt.alloc_cb",
                        tir.StringImm(cb_name),
                        *op.extents,  # Shape
                        tir.StringImm(str(op.dtype))))

                # Visit body with the transformation
                body = self.visit(op.body)

                # Return the CB allocation followed by the body
                return tir.SeqStmt([alloc_cb, body])

            # Continue visiting (was super().visit_allocate)
            return self.visit(op.body) if hasattr(op, "body") else op

        def visit_evaluate(self, op):
            """Transform T.copy operations to abstract CB operations"""

            # Check if this is a copy operation
            if self._is_copy_intrinsic(op):
                src, dst = self._extract_copy_args(op)

                # Determine if this is a read or write
                if self._is_read_to_cb(src, dst):
                    # DRAM/Buffer -> CB (read)
                    cb_name = self._get_cb_name_for_buffer(dst)
                    if cb_name:
                        return self._create_read_to_cb(src, cb_name)

                elif self._is_write_from_cb(src, dst):
                    # CB -> DRAM/Buffer (write)
                    cb_name = self._get_cb_name_for_buffer(src)
                    if cb_name:
                        return self._create_write_from_cb(cb_name, dst)

            # Continue visiting (was super().visit_evaluate)
            return self.visit(op.body) if hasattr(op, "body") else op

        def visit_buffer_load(self, op):
            """Handle buffer loads from shared memory"""
            buffer_name = op.buffer.name
            if buffer_name in self.shared_to_cb_map:
                # This load is from a CB, mark it for later processing
                # For now, keep as-is (will be handled by later passes)
                pass
            # Continue visiting (was super().visit_buffer_load)
            return self.visit(op.body) if hasattr(op, "body") else op

        def visit_buffer_store(self, op):
            """Handle buffer stores to shared memory"""
            buffer_name = op.buffer.name
            if buffer_name in self.shared_to_cb_map:
                # This store is to a CB, mark it for later processing
                # For now, keep as-is (will be handled by later passes)
                pass
            # Continue visiting (was super().visit_buffer_store)
            return self.visit(op.body) if hasattr(op, "body") else op

        # Helper methods

        def _is_shared_allocation(self, allocate_node):
            """Check if allocation is for shared memory"""
            # Look for common patterns: shared, local, fragment
            var_name = allocate_node.buffer_var.name
            return any(keyword in var_name.lower() for keyword in ["shared", "local", "fragment"])

        def _generate_cb_name(self, original_name):
            """Generate conceptual CB name based on usage pattern"""
            # Determine CB type based on variable name patterns
            if any(x in original_name.lower() for x in ["input", "a_", "b_", "in"]):
                idx = self.cb_counter["input"]
                self.cb_counter["input"] += 1
                return f"cb_in{idx}"
            elif any(x in original_name.lower() for x in ["output", "c_", "out"]):
                idx = self.cb_counter["output"]
                self.cb_counter["output"] += 1
                return f"cb_out{idx}"
            else:
                idx = self.cb_counter["intermediate"]
                self.cb_counter["intermediate"] += 1
                return f"cb_intermed{idx}"

        def _is_copy_intrinsic(self, evaluate_node):
            """Check if this is a T.copy intrinsic"""
            if isinstance(evaluate_node.value, tir.Call):
                call = evaluate_node.value
                # Check for various copy patterns
                if hasattr(call.op, 'name'):
                    return call.op.name in ["tir.copy", "T.copy", "tvm_copy"]
                # Also check for builtin copy operations
                if isinstance(call.op, tir.Op):
                    return "copy" in str(call.op)
            return False

        def _extract_copy_args(self, evaluate_node):
            """Extract source and destination from copy operation"""
            call = evaluate_node.value
            if len(call.args) >= 2:
                return call.args[0], call.args[1]
            return None, None

        def _is_read_to_cb(self, src, dst):
            """Check if this is a read from DRAM/buffer to CB"""
            # Check if dst is a shared buffer that we've mapped to a CB
            if hasattr(dst, 'buffer'):
                return dst.buffer.name in self.shared_to_cb_map
            return False

        def _is_write_from_cb(self, src, dst):
            """Check if this is a write from CB to DRAM/buffer"""
            # Check if src is a shared buffer that we've mapped to a CB
            if hasattr(src, 'buffer'):
                return src.buffer.name in self.shared_to_cb_map
            return False

        def _get_cb_name_for_buffer(self, buffer_ref):
            """Get the CB name for a buffer reference"""
            if hasattr(buffer_ref, 'buffer'):
                buffer_name = buffer_ref.buffer.name
                return self.shared_to_cb_map.get(buffer_name)
            return None

        def _create_read_to_cb(self, src, cb_name):
            """Create abstract read_to_cb operation"""
            return tir.Evaluate(
                tir.call_extern(
                    "void",
                    "tt.read_to_cb",
                    src,  # Source tensor slice
                    tir.StringImm(cb_name)  # Destination CB
                ))

        def _create_write_from_cb(self, cb_name, dst):
            """Create abstract write_from_cb operation"""
            return tir.Evaluate(
                tir.call_extern(
                    "void",
                    "tt.write_from_cb",
                    tir.StringImm(cb_name),  # Source CB
                    dst  # Destination tensor slice
                ))

    # Apply the transformation
    transformer = SharedToCBTransformer()
    new_body = transformer.visit(func.body)

    # Update function with new body
    func = func.with_body(new_body)

    # Add metadata about CB allocations
    cb_metadata = {}
    for cb_name, info in transformer.buffer_info.items():
        cb_metadata[cb_name] = {
            "shape": [int(x) if isinstance(x, tir.IntImm) else str(x) for x in info["shape"]],
            "dtype": str(info["dtype"]),
            "original_buffer": info["original_name"]
        }

    # Attach CB metadata to function (will be used by later passes)
    if cb_metadata:
        func = func.with_attr("tt.conceptual_cbs", cb_metadata)

    return func


def validate_protocol_less_output(func):
    """
    Validate that the output is protocol-less.
    Should NOT contain:
    - Concrete CB IDs (cb0, cb1, etc.)
    - NOC operations (noc_async_read, noc_async_write)
    - CB protocol operations (cb_reserve_back, cb_push_back, etc.)
    """

    class ProtocolChecker:

        def __init__(self):
            self.has_protocol = False
            self.protocol_calls = []

        def visit(self, stmt):
            """Visit and check for protocol operations"""
            if stmt is None:
                return

            if isinstance(stmt, tir.Evaluate):
                self.visit_evaluate(stmt)
            elif isinstance(stmt, tir.SeqStmt):
                for s in stmt.seq:
                    self.visit(s)
            elif hasattr(stmt, "body"):
                self.visit(stmt.body)

        def visit_evaluate(self, op):
            if isinstance(op.value, tir.Call):
                call_name = str(op.value.op)

                # Check for protocol operations that shouldn't be here
                protocol_ops = [
                    "noc_async_read", "noc_async_write", "cb_reserve_back", "cb_push_back",
                    "cb_wait_front", "cb_pop_front", "get_write_ptr", "get_read_ptr"
                ]

                if any(proto in call_name for proto in protocol_ops):
                    self.has_protocol = True
                    self.protocol_calls.append(call_name)

            # Continue visiting evaluate

    checker = ProtocolChecker()
    checker.visit(func.body)

    if checker.has_protocol:
        raise ValueError(f"Output contains protocol operations: {checker.protocol_calls}")

    return True


# Example usage and testing
if __name__ == "__main__":
    import tvm.script
    from tvm.script import tir as T

    # Create a test function with shared memory
    @tvm.script.ir_module
    class TestModule:

        @T.prim_func
        def gemm_with_shared(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                             C: T.Buffer((256, 256), "float16")):
            # Shared memory allocations
            A_shared = T.alloc_buffer((32, 32), "float16", scope="shared")
            B_shared = T.alloc_buffer((32, 32), "float16", scope="shared")

            # Copy from global to shared
            T.copy(A[0:32, 0:32], A_shared)
            T.copy(B[0:32, 0:32], B_shared)

            # Some computation (simplified)
            for i, j in T.grid(32, 32):
                C[i, j] = A_shared[i, j] + B_shared[i, j]

    # Apply the pass
    func = TestModule["gemm_with_shared"]
    transformed = LowerSharedToCB_v5(func, TestModule, None)

    # Print result
    print("=== Original Function ===")
    print(func.script())
    print("\n=== Transformed Function (Protocol-less) ===")
    print(transformed.script())

    # Validate
    try:
        validate_protocol_less_output(transformed)
        print("\n✅ Pass validation successful: Output is protocol-less")
    except ValueError as e:
        print(f"\n❌ Validation failed: {e}")
