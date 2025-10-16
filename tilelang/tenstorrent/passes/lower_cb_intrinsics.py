"""
Pass D3: LowerCBIntrinsics (v5 Specification)
Version: 5.0
Date: 2025-10-15

Purpose: Lower abstract data movements to NOC/CB protocol calls.
         Inserts protocol only in reader/writer kernels, not compute.

Input: Abstract tt.read_to_cb and tt.write_from_cb operations
Output: Concrete NOC/CB protocol sequences
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import logging
from enum import Enum

try:
    import tvm
    from tvm import tir, IRModule
except ImportError:
    tvm = None
    tir = None
    IRModule = object

logger = logging.getLogger(__name__)


class ProtocolType(Enum):
    """Types of protocol sequences"""
    READ_TILE = "read_tile"  # DRAM -> CB via NOC
    WRITE_TILE = "write_tile"  # CB -> DRAM via NOC
    CB_SYNC = "cb_sync"  # CB synchronization


class CBProtocolInserter:
    """Mutator to replace abstract ops with protocol sequences"""

    def __init__(self, kernel_role: str, tensor_accessors: Dict[str, Dict[str, Any]],
                 cb_indices: Dict[str, int]):
        self.kernel_role = kernel_role
        self.tensor_accessors = tensor_accessors
        self.cb_indices = cb_indices
        self.in_pipeline_loop = False
        self.pipeline_depth = 3  # Default pipeline depth

    def visit(self, stmt):
        """Recursive IR transformation"""
        if tir is None or stmt is None:
            return stmt

        # Handle different statement types
        if isinstance(stmt, tir.Evaluate):
            return self.visit_evaluate(stmt)
        elif isinstance(stmt, tir.For):
            return self.visit_for(stmt)
        elif isinstance(stmt, tir.SeqStmt):
            # Transform each statement in the sequence
            new_stmts = [self.visit(s) for s in stmt.seq]
            return tir.SeqStmt(new_stmts)
        elif isinstance(stmt, tir.LetStmt):
            # Transform the body
            new_body = self.visit(stmt.body)
            return tir.LetStmt(stmt.var, stmt.value, new_body)
        elif isinstance(stmt, tir.IfThenElse):
            new_then = self.visit(stmt.then_case)
            new_else = self.visit(stmt.else_case) if stmt.else_case else None
            return tir.IfThenElse(stmt.condition, new_then, new_else)
        elif isinstance(stmt, tir.Allocate):
            new_body = self.visit(stmt.body)
            return tir.Allocate(stmt.buffer_var, stmt.dtype, stmt.extents, stmt.condition, new_body)
        elif isinstance(stmt, tir.AttrStmt):
            new_body = self.visit(stmt.body)
            return tir.AttrStmt(stmt.node, stmt.attr_key, stmt.value, new_body)
        elif isinstance(stmt, tir.AssertStmt):
            new_body = self.visit(stmt.body)
            return tir.AssertStmt(stmt.condition, stmt.message, new_body)
        elif isinstance(stmt, (tir.BufferStore, tir.BufferRealize, tir.ProducerStore)):
            # Leaf nodes - return as-is
            return stmt
        else:
            # Unknown node type - return as-is
            return stmt

    def visit_evaluate(self, op):
        """Visit T.evaluate nodes to replace abstract operations"""

        if hasattr(op, 'value') and hasattr(op.value, 'op'):
            call = op.value
            op_name = str(call.op) if hasattr(call.op, 'name') else str(call.op)

            # Handle different abstract operations
            if "read_to_cb" in op_name and self.kernel_role == "reader":
                return self._lower_read_to_cb(call)
            elif "write_from_cb" in op_name and self.kernel_role == "writer":
                return self._lower_write_from_cb(call)
            else:
                # Keep other operations as-is
                return tir.Evaluate(call)

        return tir.Evaluate(op.value) if hasattr(op, 'value') else op

    def visit_for(self, op):
        """Track pipeline loops for protocol generation"""

        # Check if this is a pipeline loop
        is_pipeline = self._is_pipeline_loop(op)

        if is_pipeline:
            self.in_pipeline_loop = True
            # Process the loop body with the flag set
            new_body = self.visit(op.body) if hasattr(op, 'body') else op.body
            self.in_pipeline_loop = False
            # Return modified loop
            return tir.For(op.loop_var, op.min, op.extent, op.kind, new_body) if hasattr(
                op, 'loop_var') else op
        else:
            # Process normally
            new_body = self.visit(op.body) if hasattr(op, 'body') else op.body
            return tir.For(op.loop_var, op.min, op.extent, op.kind, new_body) if hasattr(
                op, 'loop_var') else op

    def _lower_read_to_cb(self, call) -> List[tir.Stmt]:
        """Lower tt.read_to_cb to NOC read protocol"""

        if len(call.args) < 2:
            return tir.Evaluate(call)

        buffer_slice = call.args[0]
        cb_name = self._extract_cb_name(call.args[1])

        if not cb_name:
            return tir.Evaluate(call)

        # Get CB index
        cb_index = self.cb_indices.get(cb_name, 0)

        # Get buffer accessor
        buffer_name = self._extract_buffer_name(buffer_slice)
        accessor = self.tensor_accessors.get(buffer_name)

        # Generate protocol sequence
        protocol_stmts = []

        # 1. Reserve space in CB
        protocol_stmts.append(tir.Evaluate(tir.call_extern("void", "cb_reserve_back", cb_index, 1)))

        # 2. Get write pointer
        write_ptr_var = tir.Var("write_ptr", "handle")
        protocol_stmts.append(
            tir.LetStmt(
                write_ptr_var,
                tir.call_extern("handle", "get_write_ptr", cb_index),
                tir.Evaluate(0)  # Placeholder body, will be replaced
            ))

        # 3. Issue NOC read
        if accessor:
            # Use accessor for address calculation
            tile_id_expr = self._create_tile_id_expr(buffer_slice)
            protocol_stmts.append(
                tir.Evaluate(
                    tir.call_extern("void", "noc_async_read_tile", tile_id_expr,
                                    accessor.get("runtime_arg_idx", 0), write_ptr_var)))
        else:
            # Fallback without accessor
            protocol_stmts.append(
                tir.Evaluate(
                    tir.call_extern(
                        "void",
                        "noc_async_read",
                        buffer_slice,  # Use slice directly
                        write_ptr_var,
                        2048  # Default tile size
                    )))

        # 4. Wait for NOC transfer
        protocol_stmts.append(tir.Evaluate(tir.call_extern("void", "noc_async_read_barrier")))

        # 5. Push to CB
        protocol_stmts.append(tir.Evaluate(tir.call_extern("void", "cb_push_back", cb_index, 1)))

        # Combine statements
        return self._combine_statements(protocol_stmts)

    def _lower_write_from_cb(self, call) -> List[tir.Stmt]:
        """Lower tt.write_from_cb to NOC write protocol"""

        if len(call.args) < 2:
            return tir.Evaluate(call)

        cb_name = self._extract_cb_name(call.args[0])
        buffer_slice = call.args[1]

        if not cb_name:
            return tir.Evaluate(call)

        # Get CB index
        cb_index = self.cb_indices.get(cb_name, 16)  # Default output CB

        # Get buffer accessor
        buffer_name = self._extract_buffer_name(buffer_slice)
        accessor = self.tensor_accessors.get(buffer_name)

        # Generate protocol sequence
        protocol_stmts = []

        # 1. Wait for data in CB
        protocol_stmts.append(tir.Evaluate(tir.call_extern("void", "cb_wait_front", cb_index, 1)))

        # 2. Get read pointer
        read_ptr_var = tir.Var("read_ptr", "handle")
        protocol_stmts.append(
            tir.LetStmt(
                read_ptr_var,
                tir.call_extern("handle", "get_read_ptr", cb_index),
                tir.Evaluate(0)  # Placeholder
            ))

        # 3. Issue NOC write
        if accessor:
            tile_id_expr = self._create_tile_id_expr(buffer_slice)
            protocol_stmts.append(
                tir.Evaluate(
                    tir.call_extern("void", "noc_async_write_tile", tile_id_expr,
                                    accessor.get("runtime_arg_idx", 0), read_ptr_var)))
        else:
            protocol_stmts.append(
                tir.Evaluate(
                    tir.call_extern(
                        "void",
                        "noc_async_write",
                        read_ptr_var,
                        buffer_slice,
                        2048  # Default tile size
                    )))

        # 4. Wait for NOC transfer
        protocol_stmts.append(tir.Evaluate(tir.call_extern("void", "noc_async_write_barrier")))

        # 5. Pop from CB
        protocol_stmts.append(tir.Evaluate(tir.call_extern("void", "cb_pop_front", cb_index, 1)))

        return self._combine_statements(protocol_stmts)

    def _is_pipeline_loop(self, loop) -> bool:
        """Check if this is a pipeline loop"""

        # Look for T.Pipelined annotation or specific patterns
        if hasattr(loop, 'annotations'):
            for annot in loop.annotations:
                if "pipeline" in str(annot).lower():
                    return True

        # Check loop variable name
        if hasattr(loop, 'loop_var') and hasattr(loop.loop_var, 'name'):
            if any(x in loop.loop_var.name.lower() for x in ["pipe", "stage"]):
                return True

        return False

    def _extract_cb_name(self, arg) -> Optional[str]:
        """Extract CB name from argument"""

        if isinstance(arg, str):
            return arg
        elif hasattr(arg, 'value'):
            return str(arg.value) if arg.value else None
        else:
            cb_str = str(arg)
            if "cb" in cb_str.lower():
                # Try to extract CB name from string
                for cb_name in self.cb_indices.keys():
                    if cb_name in cb_str:
                        return cb_name
            return None

    def _extract_buffer_name(self, buffer_slice) -> Optional[str]:
        """Extract buffer name from slice expression"""

        if hasattr(buffer_slice, 'buffer'):
            if hasattr(buffer_slice.buffer, 'name'):
                return buffer_slice.buffer.name
            elif hasattr(buffer_slice.buffer, 'data'):
                if hasattr(buffer_slice.buffer.data, 'name'):
                    return buffer_slice.buffer.data.name
        elif hasattr(buffer_slice, 'name'):
            return buffer_slice.name

        # Try string parsing
        str_repr = str(buffer_slice)
        if '[' in str_repr:
            return str_repr.split('[')[0].strip()

        return None

    def _create_tile_id_expr(self, buffer_slice) -> tir.PrimExpr:
        """Create tile ID expression from buffer slice"""

        # This would be more sophisticated in practice
        # For now, return a simple expression
        return tir.IntImm("int32", 0)  # Placeholder

    def _combine_statements(self, stmts: List[tir.Stmt]) -> tir.Stmt:
        """Combine multiple statements into a sequence"""

        if not stmts:
            return tir.Evaluate(0)
        elif len(stmts) == 1:
            return stmts[0]
        else:
            # Create sequence of statements
            result = stmts[-1]
            for stmt in reversed(stmts[:-1]):
                if isinstance(stmt, tir.LetStmt):
                    # Special handling for LetStmt - create new LetStmt with updated body
                    result = tir.LetStmt(stmt.var, stmt.value, result)
                else:
                    result = tir.SeqStmt([stmt, result])
            return result


class LowerCBIntrinsics:
    """
    Pass to lower abstract CB operations to NOC/CB protocol.

    This pass:
    1. Replaces tt.read_to_cb with NOC read protocol (reader kernel)
    2. Replaces tt.write_from_cb with NOC write protocol (writer kernel)
    3. Adds CB synchronization (reserve/push/wait/pop)
    4. Leaves compute kernel unchanged (no protocol there)
    """

    def __init__(self) -> None:
        """Initialize the pass."""
        pass

    def __call__(self, mod: IRModule) -> IRModule:
        """Apply the pass to an IRModule."""
        if tvm is None:
            return mod

        new_funcs = {}
        for gvar, func in mod.functions_items():
            if not isinstance(func, tir.PrimFunc):
                new_funcs[gvar] = func
                continue

            # Process this function
            func = self._process_function(func)
            new_funcs[gvar] = func

        return tvm.IRModule(new_funcs)

    def _process_function(self, func: "tir.PrimFunc") -> "tir.PrimFunc":
        """Process a single function to insert CB protocol."""

        # Get kernel role
        kernel_role = None
        if func.attrs and "tt.kernel_role" in func.attrs:
            kernel_role = func.attrs["tt.kernel_role"]

        # Only process reader and writer kernels
        if kernel_role not in ["reader", "writer"]:
            logger.debug(f"Skipping {kernel_role} kernel for CB protocol insertion")
            return func

        # Collect tensor accessors
        tensor_accessors = self._collect_tensor_accessors(func)

        # Get CB indices
        cb_indices = {}
        if func.attrs and "tt.cb_indices" in func.attrs:
            cb_indices = self._convert_to_dict(func.attrs["tt.cb_indices"])

        # Apply protocol insertion
        inserter = CBProtocolInserter(kernel_role, tensor_accessors, cb_indices)
        new_body = inserter.visit(func.body)

        # Create new function with protocol
        new_func = tir.PrimFunc(
            params=func.params,
            body=new_body,
            ret_type=func.ret_type,
            buffer_map=func.buffer_map,
            attrs=func.attrs)

        # Mark that protocol has been inserted
        new_func = new_func.with_attr("tt.cb_protocol_inserted", True)

        logger.info(f"Inserted CB protocol for {kernel_role} kernel")

        return new_func

    def _collect_tensor_accessors(self, func: "tir.PrimFunc") -> Dict[str, Dict[str, Any]]:
        """Collect all tensor accessors from function attributes."""

        accessors = {}

        if func.attrs:
            for key in func.attrs.keys():
                if key.startswith("tt.tensor_accessor."):
                    buffer_name = key.replace("tt.tensor_accessor.", "")
                    accessor = self._convert_to_dict(func.attrs[key])
                    accessors[buffer_name] = accessor

        return accessors

    def _convert_to_dict(self, attr_value: Any) -> Dict[str, Any]:
        """Convert TVM attribute value to Python dict."""

        if isinstance(attr_value, dict):
            return attr_value

        if hasattr(attr_value, "items"):
            result = {}
            for k, v in attr_value.items():
                if hasattr(v, "items"):
                    result[str(k)] = self._convert_to_dict(v)
                elif isinstance(v, (list, tuple)):
                    result[str(k)] = list(v)
                else:
                    result[str(k)] = self._convert_value(v)
            return result

        return {"value": self._convert_value(attr_value)}

    def _convert_value(self, value: Any) -> Any:
        """Convert TVM value to Python type."""

        if hasattr(value, "value"):
            return value.value
        elif isinstance(value, (int, float, str, bool)):
            return value
        else:
            return str(value)


# Module-level pass function for compatibility
def lower_cb_intrinsics(mod: IRModule) -> IRModule:
    """Apply LowerCBIntrinsics pass to a module."""
    pass_instance = LowerCBIntrinsics()
    return pass_instance(mod)


# Example usage and testing
if __name__ == "__main__":
    import tvm.script
    from tvm.script import tir as T

    # Create test module with reader kernel
    @tvm.script.ir_module
    class TestModule:

        @T.prim_func
        def gemm_reader(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16")):
            # Simulate abstract operations from split kernel
            T.evaluate(T.call_extern("void", "tt.read_to_cb", A[0:128, 0:32], "cb_in0"))
            T.evaluate(T.call_extern("void", "tt.read_to_cb", B[0:32, 0:128], "cb_in1"))

        @T.prim_func
        def gemm_writer(C: T.Buffer((256, 256), "float16")):
            # Simulate abstract write operation
            T.evaluate(T.call_extern("void", "tt.write_from_cb", "cb_out", C[0:128, 0:128]))

    # Add metadata
    reader_func = TestModule["gemm_reader"]
    reader_func = reader_func.with_attr("tt.kernel_role", "reader")
    reader_func = reader_func.with_attr("tt.cb_indices", {"cb_in0": 0, "cb_in1": 1})
    reader_func = reader_func.with_attr("tt.tensor_accessor.A", {
        "type": "bound",
        "runtime_arg_idx": 0,
        "tile_size_bytes": 2048
    })
    reader_func = reader_func.with_attr("tt.tensor_accessor.B", {
        "type": "bound",
        "runtime_arg_idx": 1,
        "tile_size_bytes": 2048
    })
    TestModule["gemm_reader"] = reader_func

    writer_func = TestModule["gemm_writer"]
    writer_func = writer_func.with_attr("tt.kernel_role", "writer")
    writer_func = writer_func.with_attr("tt.cb_indices", {"cb_out": 16})
    writer_func = writer_func.with_attr("tt.tensor_accessor.C", {
        "type": "bound",
        "runtime_arg_idx": 0,
        "tile_size_bytes": 2048
    })
    TestModule["gemm_writer"] = writer_func

    # Apply D3 pass
    pass_d3 = LowerCBIntrinsics()
    result = pass_d3(TestModule)

    # Check results
    print("=== CB Protocol Insertion Results ===\n")
    for name, func in result.functions_items():
        if func.attrs and "tt.kernel_role" in func.attrs:
            print(f"{name} ({func.attrs['tt.kernel_role']}):")
            if "tt.cb_protocol_inserted" in func.attrs:
                print(f"  Protocol inserted: {func.attrs['tt.cb_protocol_inserted']}")
            print(f"  Body preview: {str(func.body)[:200]}...")
            print()
