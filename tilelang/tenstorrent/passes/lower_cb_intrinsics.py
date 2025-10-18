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
    """Helper class to transform abstract ops to protocol sequences"""

    def __init__(self,
                 kernel_role: str,
                 tensor_accessors: Dict[str, Dict[str, Any]],
                 cb_indices: Dict[str, int],
                 loop_var: Optional[tir.Var] = None):
        self.kernel_role = kernel_role
        self.tensor_accessors = tensor_accessors
        self.cb_indices = cb_indices
        self.in_pipeline_loop = False
        self.pipeline_depth = 3  # Default pipeline depth
        self.loop_var = loop_var  # The 'k' loop variable for tile_id calculation

    def transform_evaluate(self, op):
        """Visit T.evaluate nodes to replace abstract operations"""

        if hasattr(op, 'value') and hasattr(op.value, 'op'):
            call = op.value

            # Extract operation name - handle both direct name and call_extern
            op_name = None
            if hasattr(call.op, 'name'):
                op_name = call.op.name
                # For call_extern, look through all args to find the function name StringImm
                if op_name == "tir.call_extern" and hasattr(call, 'args'):
                    for arg in call.args:
                        if (isinstance(arg, tir.StringImm) and arg.value not in ["void"] and
                                not arg.value.startswith("uint") and
                                not arg.value.startswith("int")):
                            op_name = arg.value
                            break

            if not op_name:
                return tir.Evaluate(call)

            # Handle different abstract operations
            if "read_to_cb" in op_name and self.kernel_role == "reader":
                return self._lower_read_to_cb(call)
            elif "write_from_cb" in op_name and self.kernel_role == "writer":
                return self._lower_write_from_cb(call)
            elif "tt.copy.protocol_less" in op_name:
                # Handle tt.copy.protocol_less operations
                logger.debug(f"Transforming tt.copy.protocol_less in {self.kernel_role} kernel")
                return self._lower_protocol_less_copy(call)
            else:
                # Keep other operations as-is
                return tir.Evaluate(call)

        return tir.Evaluate(op.value) if hasattr(op, 'value') else op

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
        return hasattr(loop, 'loop_var') and hasattr(loop.loop_var, 'name') and any(
            x in loop.loop_var.name.lower() for x in ["pipe", "stage"])

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
            elif hasattr(buffer_slice.buffer, 'data') and hasattr(buffer_slice.buffer.data, 'name'):
                return buffer_slice.buffer.data.name
        elif hasattr(buffer_slice, 'name'):
            return buffer_slice.name

        # Try string parsing
        str_repr = str(buffer_slice)
        if '[' in str_repr:
            return str_repr.split('[')[0].strip()

        return None

    def _create_tile_id_expr(self, buffer_name: Optional[str] = None) -> tir.PrimExpr:
        """Create tile ID expression from loop variable and runtime args.

        For global partition mode:
            tile_id = start_id + k

        Args:
            buffer_name: Name of buffer being accessed (for future shard-aware logic)

        Returns:
            TIR expression computing the tile ID
        """
        # If we have a loop variable, use start_id + loop_var
        if self.loop_var is not None:
            # Create variable reference to start_id runtime arg
            start_id_var = tir.Var("start_id", "int32")

            # Compute: tile_id = start_id + k
            tile_id_expr = tir.Add(start_id_var, self.loop_var)

            return tile_id_expr
        else:
            # No loop context - use constant 0
            return tir.IntImm("int32", 0)

    def _lower_protocol_less_copy(self, call):
        """Lower tt.copy.protocol_less to appropriate protocol based on kernel role.

        Args:
            call: The tt.copy.protocol_less call with args:
                  [0]: Source region (T.region)
                  [1]: Destination region (T.region)

        For reader kernels: DRAM->shared copies become NOC read + CB protocol
        For writer kernels: shared/local->DRAM copies become CB wait + NOC write
        For compute kernels: Should not be called (compute is skipped)
        """

        if len(call.args) < 3:  # Need at least: "void", "tt.copy.protocol_less", src, dst
            return tir.Evaluate(call)

        # Args are: func_name_str, src_region, dst_region, ...
        # (TVM call_extern doesn't include return type in args, only in the op)
        src_region = call.args[1] if len(call.args) > 1 else None
        dst_region = call.args[2] if len(call.args) > 2 else None

        if src_region is None or dst_region is None:
            return tir.Evaluate(call)

        # Determine the scope/memory of source and destination
        src_scope = self._get_region_scope(src_region)
        dst_scope = self._get_region_scope(dst_region)

        # Determine if this is a read (DRAM->L1) or write (L1->DRAM)
        is_dram_read = (
            src_scope in ["global", "dram", ""] and dst_scope in ["shared", "shared.dyn", "local"])
        is_dram_write = (
            src_scope in ["shared", "shared.dyn", "local", "local.fragment"] and
            dst_scope in ["global", "dram", ""])

        # Handle based on kernel role and operation type
        if self.kernel_role == "reader" and is_dram_read:
            # This is a DRAM read operation in reader kernel
            # Transform to: cb_reserve_back -> noc_async_read -> noc_async_read_barrier -> cb_push_back
            return self._generate_dram_read_protocol(src_region, dst_region)

        elif self.kernel_role == "writer" and is_dram_write:
            # This is a DRAM write operation in writer kernel
            # Transform to: cb_wait_front -> noc_async_write -> noc_async_write_barrier -> cb_pop_front
            return self._generate_dram_write_protocol(src_region, dst_region)

        else:
            # Not a DRAM operation or wrong kernel role - keep as-is
            # This handles local copies, compute kernel ops, etc.
            return tir.Evaluate(call)

    def _get_region_scope(self, region):
        """Extract the memory scope from a T.region expression."""

        # T.region has the buffer as first argument
        if not hasattr(region, 'args') or len(region.args) < 1:
            return ""

        buffer_arg = region.args[0]

        # Check if it's a buffer load/store
        if hasattr(buffer_arg, 'buffer'):
            buffer = buffer_arg.buffer
            if hasattr(buffer, 'scope'):
                # scope might be a property or callable - try both
                scope_val = buffer.scope() if callable(buffer.scope) else buffer.scope
                if scope_val and isinstance(scope_val, str):
                    return scope_val
            elif hasattr(buffer, 'data') and hasattr(buffer.data, 'type_annotation'):
                scope_str = str(buffer.data.type_annotation)
                if "shared" in scope_str:
                    return "shared"
                elif "local" in scope_str:
                    return "local"

        # Try to infer from buffer name
        buffer_str = str(buffer_arg)
        if "shared" in buffer_str.lower():
            return "shared.dyn"  # Use shared.dyn to match allocation
        elif "local" in buffer_str.lower():
            if "fragment" in buffer_str.lower():
                return "local.fragment"
            return "local"

        # Default: assume global/DRAM
        return "global"

    def _generate_dram_read_protocol(self, src_region, dst_region):
        """Generate NOC read protocol for DRAM->L1 copy.

        Generated structure:
        1. cb_reserve_back(cb_index, 1);
        2. uint32_t l1_write_addr = get_write_ptr(cb_index);
        3. noc_async_read_tile(tile_id, accessor, l1_write_addr);
        4. noc_async_read_barrier();
        5. cb_push_back(cb_index, 1);
        """

        # Extract source buffer name for CB mapping and tile_id calculation
        src_buffer_name = self._extract_buffer_name_from_region(src_region)

        # Map buffer to CB (A->cb_in0, B->cb_in1, etc.)
        cb_name = self._infer_cb_from_buffer(src_buffer_name)
        cb_index = self.cb_indices.get(cb_name, 0)

        # Get tensor accessor for address calculation
        accessor = self.tensor_accessors.get(src_buffer_name, {})

        # Step 1: Reserve space in CB (happens first, outside LetStmt)
        cb_reserve = tir.Evaluate(
            tir.call_extern("void", "cb_reserve_back", tir.IntImm("int32", cb_index),
                            tir.IntImm("int32", 1)))

        # Step 2-5: Build the statements that use l1_write_addr
        # These will be the body of the LetStmt

        # Create variable for write pointer
        write_ptr_var = tir.Var("l1_write_addr", "uint32")

        # Step 3: Issue NOC read using write_ptr_var
        # Compute tile_id from loop variable (start_id + k)
        tile_id_expr = self._create_tile_id_expr(src_buffer_name)

        noc_read = tir.Evaluate(
            tir.call_extern(
                "void",
                "noc_async_read_tile",
                tile_id_expr,  # Computed tile_id
                tir.IntImm("int32", accessor.get("runtime_arg_idx", 0)),
                write_ptr_var))

        # Step 4: Wait for NOC transfer
        noc_barrier = tir.Evaluate(tir.call_extern("void", "noc_async_read_barrier"))

        # Step 5: Push to CB
        cb_push = tir.Evaluate(
            tir.call_extern("void", "cb_push_back", tir.IntImm("int32", cb_index),
                            tir.IntImm("int32", 1)))

        # Combine steps 3-5 into body of LetStmt
        let_body = tir.SeqStmt([noc_read, noc_barrier, cb_push])

        # Step 2: Create LetStmt that defines l1_write_addr and contains steps 3-5
        let_stmt = tir.LetStmt(
            write_ptr_var, tir.call_extern("uint32", "get_write_ptr",
                                           tir.IntImm("int32", cb_index)), let_body)

        # Final sequence: cb_reserve, then LetStmt
        return tir.SeqStmt([cb_reserve, let_stmt])

    def _generate_dram_write_protocol(self, src_region, dst_region):
        """Generate NOC write protocol for L1->DRAM copy.

        Generated structure:
        1. cb_wait_front(cb_index, 1);
        2. uint32_t l1_read_addr = get_read_ptr(cb_index);
        3. noc_async_write_tile(tile_id, accessor, l1_read_addr);
        4. noc_async_write_barrier();
        5. cb_pop_front(cb_index, 1);
        """

        # Extract destination buffer name for CB mapping and tile_id calculation
        dst_buffer_name = self._extract_buffer_name_from_region(dst_region)

        # Map buffer to CB (C_local->cb_out0, etc.)
        cb_name = self._infer_cb_from_buffer(dst_buffer_name)
        cb_index = self.cb_indices.get(cb_name, 16)  # Output CBs start at 16

        # Get tensor accessor
        accessor = self.tensor_accessors.get(dst_buffer_name, {})

        # Step 1: Wait for data in CB (happens first, outside LetStmt)
        cb_wait = tir.Evaluate(
            tir.call_extern("void", "cb_wait_front", tir.IntImm("int32", cb_index),
                            tir.IntImm("int32", 1)))

        # Step 2-5: Build statements that use l1_read_addr
        # These will be the body of the LetStmt

        # Create variable for read pointer
        read_ptr_var = tir.Var("l1_read_addr", "uint32")

        # Step 3: Issue NOC write using read_ptr_var
        # Compute tile_id from loop variable (start_id + k)
        tile_id_expr = self._create_tile_id_expr(dst_buffer_name)

        noc_write = tir.Evaluate(
            tir.call_extern(
                "void",
                "noc_async_write_tile",
                tile_id_expr,  # Computed tile_id
                tir.IntImm("int32", accessor.get("runtime_arg_idx", 0)),
                read_ptr_var))

        # Step 4: Wait for NOC transfer
        noc_barrier = tir.Evaluate(tir.call_extern("void", "noc_async_write_barrier"))

        # Step 5: Pop from CB
        cb_pop = tir.Evaluate(
            tir.call_extern("void", "cb_pop_front", tir.IntImm("int32", cb_index),
                            tir.IntImm("int32", 1)))

        # Combine steps 3-5 into body of LetStmt
        let_body = tir.SeqStmt([noc_write, noc_barrier, cb_pop])

        # Step 2: Create LetStmt that defines l1_read_addr and contains steps 3-5
        let_stmt = tir.LetStmt(
            read_ptr_var, tir.call_extern("uint32", "get_read_ptr", tir.IntImm("int32", cb_index)),
            let_body)

        # Final sequence: cb_wait, then LetStmt
        return tir.SeqStmt([cb_wait, let_stmt])

    def _extract_buffer_name_from_region(self, region):
        """Extract buffer name from T.region expression."""

        if not hasattr(region, 'args') or len(region.args) < 1:
            return None

        buffer_arg = region.args[0]

        # Try various ways to get the buffer name
        if hasattr(buffer_arg, 'buffer'):
            if hasattr(buffer_arg.buffer, 'name'):
                return buffer_arg.buffer.name
            elif hasattr(buffer_arg.buffer, 'data') and hasattr(buffer_arg.buffer.data, 'name'):
                return buffer_arg.buffer.data.name

        # Parse from string
        buffer_str = str(buffer_arg)
        if '[' in buffer_str:
            return buffer_str.split('[')[0].strip()

        return None

    def _infer_cb_from_buffer(self, buffer_name):
        """Infer CB name from buffer name."""

        if not buffer_name:
            return "cb_in0"

        # Common mappings
        buffer_lower = buffer_name.lower()
        if 'a' in buffer_lower and 'shared' in buffer_lower:
            return "cb_in0"
        elif 'b' in buffer_lower and 'shared' in buffer_lower:
            return "cb_in1"
        elif 'c' in buffer_lower or 'out' in buffer_lower:
            return "cb_out0"

        # Default
        return "cb_in0"

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

        # Extract the 'k' loop variable for tile_id calculation
        k_loop_var = self._find_tile_iteration_loop_var(func.body)
        if k_loop_var is not None:
            logger.info(f"[{kernel_role}] Found tile iteration loop variable: {k_loop_var.name}")

        # Create helper for transformation
        inserter = CBProtocolInserter(kernel_role, tensor_accessors, cb_indices, k_loop_var)

        # Define post-visit function for ir_transform
        def post_visit(stmt):
            """Transform Evaluate nodes with tt.copy.protocol_less"""
            if isinstance(stmt, tir.Evaluate):
                return inserter.transform_evaluate(stmt)
            return stmt

        # Apply transformation using ir_transform
        from tvm.tir import stmt_functor
        new_body = stmt_functor.ir_transform(func.body, None, post_visit)

        # Create new function with protocol
        new_func = func.with_body(new_body)

        # Mark that protocol has been inserted
        from tilelang.tenstorrent.attrs import TT_CB_PROTOCOL_INSERTED
        new_func = new_func.with_attr(TT_CB_PROTOCOL_INSERTED, True)

        logger.info(f"Inserted CB protocol for {kernel_role} kernel")

        return new_func

    def _find_tile_iteration_loop_var(self, body) -> Optional[tir.Var]:
        """Find the 'k' loop variable used for tile iteration.

        Returns:
            The loop variable (typically named 'k') or None if not found
        """
        found_var = None

        def visitor(node):
            nonlocal found_var
            if (isinstance(node, tir.For) and hasattr(node, 'loop_var') and
                    hasattr(node.loop_var, 'name') and node.loop_var.name == 'k'):
                found_var = node.loop_var

        tir.stmt_functor.post_order_visit(body, visitor)
        return found_var

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
