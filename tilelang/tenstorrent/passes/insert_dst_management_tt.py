"""
Pass D5: InsertDSTManagementTT (v5 Specification)
Version: 5.0
Date: 2025-10-15

Purpose: Insert DST (destination register) lifecycle management in compute kernels.
         Wraps compute with acquire/commit/wait/release and packing.

Input: Compute kernel with engine init but no DST protocol
Output: Compute kernel with full DST lifecycle management
"""

from __future__ import annotations
from typing import Dict, Any, Optional
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

from tilelang.tenstorrent.attrs import TT_DST_MANAGEMENT_INSERTED, TT_DST_PATTERN


class DSTPattern(Enum):
    """DST management patterns"""
    ACCUMULATION = "accumulation"  # K-loop pattern (GEMM)
    SINGLE_TILE = "single_tile"  # Per-tile pattern (eltwise)
    REDUCTION = "reduction"  # Other reduction patterns


class ComputePatternAnalyzer:
    """Analyze compute patterns to determine DST management strategy"""

    def __init__(self):
        self.has_k_loop = False
        self.has_accumulation = False
        self.compute_ops = []
        self.loop_info = []
        self.cb_usage = {"inputs": set(), "outputs": set()}
        self.current_loop_var = None

    def visit(self, stmt):
        """Recursive IR traversal (analysis only)"""
        if tir is None or stmt is None:
            return

        # Handle different statement types
        if isinstance(stmt, tir.Evaluate):
            self.visit_evaluate(stmt)
        elif isinstance(stmt, tir.For):
            self.visit_for(stmt)
        elif isinstance(stmt, tir.SeqStmt):
            for s in stmt.seq:
                self.visit(s)
        elif isinstance(stmt, tir.LetStmt):
            self.visit(stmt.body)
        elif isinstance(stmt, tir.IfThenElse):
            self.visit(stmt.then_case)
            if stmt.else_case:
                self.visit(stmt.else_case)
        elif isinstance(stmt, (tir.Allocate, tir.AttrStmt, tir.AssertStmt)):
            self.visit(stmt.body)

    def visit_for(self, op):
        """Track loops and their patterns"""

        prev_loop_var = self.current_loop_var
        self.current_loop_var = op.loop_var

        # Check if this is a K-loop or reduction loop
        if hasattr(op.loop_var, 'name'):
            var_name = op.loop_var.name.lower()
            if any(x in var_name for x in ['k', 'reduction', 'accumulate']):
                self.has_k_loop = True
                self.loop_info.append({
                    "var": op.loop_var,
                    "name": var_name,
                    "extent": op.extent,
                    "type": "reduction"
                })

        # Visit loop body
        self.visit(op.body)

        self.current_loop_var = prev_loop_var

    def visit_evaluate(self, op):
        """Track compute operations and their patterns"""

        if hasattr(op, 'value') and hasattr(op.value, 'op'):
            call = op.value
            op_name = str(call.op) if hasattr(call.op, 'name') else str(call.op)

            # Check for compute operations
            if any(x in op_name for x in ["mm.mma", "fpu.", "sfpu.", "matmul", "add", "mul"]):
                # Check for accumulation parameter
                accumulate = False
                if len(call.args) > 3:
                    # Check if there's an accumulate flag
                    for arg in call.args[3:]:
                        if self._is_accumulate_arg(arg):
                            accumulate = True
                            self.has_accumulation = True
                            break

                self.compute_ops.append({
                    "op_name": op_name,
                    "args": call.args,
                    "accumulate": accumulate,
                    "in_loop": self.current_loop_var is not None
                })

                # Track CB usage
                for i, arg in enumerate(call.args):
                    cb_name = self._extract_cb_name(arg)
                    if cb_name:
                        if i < 2:  # First two args are usually inputs
                            self.cb_usage["inputs"].add(cb_name)
                        elif "out" in cb_name:
                            self.cb_usage["outputs"].add(cb_name)

        # super().visit_evaluate(op) - no parent class

    def _is_accumulate_arg(self, arg) -> bool:
        """Check if argument indicates accumulation"""

        # Check for boolean true or comparison > 0
        if hasattr(arg, 'value'):
            if isinstance(arg.value, bool):
                return arg.value
            elif isinstance(arg.value, int):
                return arg.value != 0

        # Check for comparisons like kt > 0
        return hasattr(arg, 'op') and hasattr(arg.op, 'name') and arg.op.name in ["GT", "GE", "NE"]

    def _extract_cb_name(self, arg) -> Optional[str]:
        """Extract CB name from argument"""

        if isinstance(arg, str) and "cb" in arg.lower():
            return arg
        elif hasattr(arg, 'value'):
            val_str = str(arg.value)
            if "cb" in val_str.lower():
                return val_str
        return None


class DSTProtocolInserter:
    """Insert DST lifecycle management around compute operations"""

    def __init__(self, dst_pattern: DSTPattern, cb_info: Dict[str, Any]):
        self.dst_pattern = dst_pattern
        self.cb_info = cb_info
        self.dst_wrapped = False
        self.in_target_loop = False

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
            return self.visit_seq_stmt(stmt)
        elif isinstance(stmt, tir.LetStmt):
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
        elif isinstance(stmt, (tir.BufferStore, tir.BufferRealize)) or (hasattr(
                tir, 'ProducerStore') and isinstance(stmt, tir.ProducerStore)):
            return stmt
        else:
            return stmt

    def visit_for(self, op):
        """Wrap loops with DST management for accumulation pattern"""

        # Check if this is the K-loop to wrap
        if self.dst_pattern == DSTPattern.ACCUMULATION and not self.dst_wrapped and self._is_reduction_loop(
                op):
            # Wrap entire K-loop with DST management
            wrapped_body = self._wrap_accumulation_loop(op)
            self.dst_wrapped = True
            return wrapped_body

        return op

    def visit_seq_stmt(self, op):
        """Handle sequence of statements for single-tile pattern"""

        if self.dst_pattern == DSTPattern.SINGLE_TILE and not self.dst_wrapped:
            # Find compute operations and wrap them
            new_seq = []
            for stmt in op.seq:
                if self._contains_compute(stmt):
                    # Wrap this statement with DST protocol
                    wrapped = self._wrap_single_tile_compute(stmt)
                    new_seq.append(wrapped)
                    self.dst_wrapped = True
                else:
                    new_seq.append(self.visit(stmt))

            if new_seq:
                return tir.SeqStmt(new_seq)

        return self.visit(op)

    def visit_evaluate(self, op):
        """Wrap single compute operations if not in loop"""

        if self.dst_pattern == DSTPattern.SINGLE_TILE and not self.dst_wrapped and self._is_compute_op(
                op):
            # Wrap with DST protocol
            wrapped = self._wrap_single_tile_compute(op)
            self.dst_wrapped = True
            return wrapped

        return op

    def _wrap_accumulation_loop(self, loop_stmt) -> tir.Stmt:
        """Wrap K-loop with DST accumulation protocol"""

        stmts = []

        # 1. Acquire DST before loop
        stmts.append(tir.Evaluate(tir.call_extern("void", "tt.dst.acquire")))

        # 2. Modified loop body with CB wait/pop
        modified_body = self._insert_cb_sync_in_loop(loop_stmt.body, is_accumulation=True)
        modified_loop = tir.For(loop_stmt.loop_var, loop_stmt.min, loop_stmt.extent, loop_stmt.kind,
                                modified_body, loop_stmt.thread_binding, loop_stmt.annotations)
        stmts.append(modified_loop)

        # 3. After loop: commit, wait, pack, release
        cb_out = self.cb_info.get("cb_out", 16)

        # Reserve space in output CB
        stmts.append(tir.Evaluate(tir.call_extern("void", "cb_reserve_back", cb_out, 1)))

        # Commit DST
        stmts.append(tir.Evaluate(tir.call_extern("void", "tt.dst.commit")))

        # Wait for DST
        stmts.append(tir.Evaluate(tir.call_extern("void", "tt.dst.wait")))

        # Pack to output CB
        stmts.append(tir.Evaluate(tir.call_extern("void", "pack_tile", 0, cb_out, 0)))

        # Release DST
        stmts.append(tir.Evaluate(tir.call_extern("void", "tt.dst.release")))

        # Push to CB
        stmts.append(tir.Evaluate(tir.call_extern("void", "cb_push_back", cb_out, 1)))

        return tir.SeqStmt(stmts)

    def _wrap_single_tile_compute(self, compute_stmt) -> tir.Stmt:
        """Wrap single-tile compute with DST protocol"""

        stmts = []

        cb_in0 = self.cb_info.get("cb_in0", 0)
        cb_in1 = self.cb_info.get("cb_in1", 1)
        cb_out = self.cb_info.get("cb_out", 16)

        # 1. Wait for input CBs
        stmts.append(tir.Evaluate(tir.call_extern("void", "cb_wait_front", cb_in0, 1)))
        if cb_in1 != cb_in0:  # Binary operation
            stmts.append(tir.Evaluate(tir.call_extern("void", "cb_wait_front", cb_in1, 1)))

        # 2. Acquire DST
        stmts.append(tir.Evaluate(tir.call_extern("void", "tt.dst.acquire")))

        # 3. Compute operation (original statement)
        stmts.append(compute_stmt)

        # 4. Pop input CBs
        stmts.append(tir.Evaluate(tir.call_extern("void", "cb_pop_front", cb_in0, 1)))
        if cb_in1 != cb_in0:
            stmts.append(tir.Evaluate(tir.call_extern("void", "cb_pop_front", cb_in1, 1)))

        # 5. Reserve output CB
        stmts.append(tir.Evaluate(tir.call_extern("void", "cb_reserve_back", cb_out, 1)))

        # 6. Commit DST
        stmts.append(tir.Evaluate(tir.call_extern("void", "tt.dst.commit")))

        # 7. Wait for DST
        stmts.append(tir.Evaluate(tir.call_extern("void", "tt.dst.wait")))

        # 8. Pack to output CB
        stmts.append(tir.Evaluate(tir.call_extern("void", "pack_tile", 0, cb_out, 0)))

        # 9. Release DST
        stmts.append(tir.Evaluate(tir.call_extern("void", "tt.dst.release")))

        # 10. Push to CB
        stmts.append(tir.Evaluate(tir.call_extern("void", "cb_push_back", cb_out, 1)))

        return tir.SeqStmt(stmts)

    def _insert_cb_sync_in_loop(self, body, is_accumulation: bool) -> tir.Stmt:
        """Insert CB wait/pop inside the loop for accumulation"""

        cb_in0 = self.cb_info.get("cb_in0", 0)
        cb_in1 = self.cb_info.get("cb_in1", 1)

        # Create CB synchronization statements
        pre_stmts = [
            tir.Evaluate(tir.call_extern("void", "cb_wait_front", cb_in0, 1)),
            tir.Evaluate(tir.call_extern("void", "cb_wait_front", cb_in1, 1))
        ]

        post_stmts = [
            tir.Evaluate(tir.call_extern("void", "cb_pop_front", cb_in0, 1)),
            tir.Evaluate(tir.call_extern("void", "cb_pop_front", cb_in1, 1))
        ]

        # Combine with original body
        return tir.SeqStmt(pre_stmts + [body] + post_stmts)

    def _is_reduction_loop(self, loop) -> bool:
        """Check if this is a reduction/K-loop"""

        if hasattr(loop.loop_var, 'name'):
            var_name = loop.loop_var.name.lower()
            return any(x in var_name for x in ['k', 'reduction', 'accumulate'])
        return False

    def _contains_compute(self, stmt) -> bool:
        """Check if statement contains compute operations"""

        class ComputeChecker:

            def __init__(self):
                self.has_compute = False

            def visit_evaluate(self, op):
                if hasattr(op, 'value') and hasattr(op.value, 'op'):
                    op_name = str(op.value.op)
                    if any(x in op_name for x in ["mm.mma", "fpu.", "sfpu."]):
                        self.has_compute = True

            def visit(self, stmt):
                """Simple visitor to check for compute ops"""
                if isinstance(stmt, tir.Evaluate):
                    self.visit_evaluate(stmt)
                elif hasattr(stmt, 'body'):
                    self.visit(stmt.body)
                elif isinstance(stmt, tir.SeqStmt):
                    for s in stmt.seq:
                        self.visit(s)
                        if self.has_compute:
                            break

        checker = ComputeChecker()
        checker.visit(stmt)
        return checker.has_compute

    def _is_compute_op(self, evaluate_stmt) -> bool:
        """Check if evaluate statement is a compute operation"""

        if hasattr(evaluate_stmt, 'value') and hasattr(evaluate_stmt.value, 'op'):
            op_name = str(evaluate_stmt.value.op)
            return any(x in op_name for x in ["mm.mma", "fpu.", "sfpu.", "matmul", "add", "mul"])
        return False


class InsertDSTManagementTT:
    """
    Pass to insert DST lifecycle management in compute kernels.

    This pass:
    1. Analyzes compute patterns (accumulation vs single-tile)
    2. Wraps compute with DST acquire/release
    3. Adds CB synchronization (wait/pop)
    4. Inserts packing from DST to output CB
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
        """Process a single function to insert DST management."""

        # Get kernel role
        kernel_role = None
        if func.attrs and "tt.kernel_role" in func.attrs:
            kernel_role = func.attrs["tt.kernel_role"]

        # Only process compute kernels
        if kernel_role != "compute":
            logger.debug(f"Skipping {kernel_role} kernel for DST management")
            return func

        # Analyze compute pattern
        analyzer = ComputePatternAnalyzer()
        analyzer.visit(func.body)

        # Determine DST pattern
        dst_pattern = self._determine_dst_pattern(analyzer)

        # Get CB information
        cb_info = self._extract_cb_info(func, analyzer)

        # Insert DST management
        inserter = DSTProtocolInserter(dst_pattern, cb_info)
        new_body = inserter.visit(func.body)

        # Create new function with DST management
        new_func = tir.PrimFunc(
            params=func.params,
            body=new_body,
            ret_type=func.ret_type,
            buffer_map=func.buffer_map,
            attrs=func.attrs)

        # Mark that DST management has been inserted
        new_func = new_func.with_attr(TT_DST_MANAGEMENT_INSERTED, True)
        new_func = new_func.with_attr(TT_DST_PATTERN, dst_pattern.value)

        logger.info(f"Inserted DST management with {dst_pattern.value} pattern")

        return new_func

    def _determine_dst_pattern(self, analyzer: ComputePatternAnalyzer) -> DSTPattern:
        """Determine DST management pattern based on compute analysis."""

        if analyzer.has_k_loop and analyzer.has_accumulation:
            # K-loop with accumulation -> GEMM pattern
            return DSTPattern.ACCUMULATION
        elif analyzer.has_k_loop:
            # K-loop without explicit accumulation -> still use accumulation pattern
            return DSTPattern.ACCUMULATION
        else:
            # No K-loop -> single-tile pattern (element-wise)
            return DSTPattern.SINGLE_TILE

    def _extract_cb_info(self, func: "tir.PrimFunc",
                         analyzer: ComputePatternAnalyzer) -> Dict[str, Any]:
        """Extract CB information for DST management."""

        cb_info = {"cb_in0": 0, "cb_in1": 1, "cb_out": 16}

        # Get CB indices from function attributes
        if func.attrs and "tt.cb_indices" in func.attrs:
            cb_indices = self._convert_to_dict(func.attrs["tt.cb_indices"])
            for cb_name, index in cb_indices.items():
                if "in0" in cb_name:
                    cb_info["cb_in0"] = index
                elif "in1" in cb_name:
                    cb_info["cb_in1"] = index
                elif "out" in cb_name:
                    cb_info["cb_out"] = index

        # Override with analyzer findings if more specific
        for cb_name in analyzer.cb_usage["inputs"]:
            if "in0" in cb_name:
                cb_info["cb_in0"] = self._get_cb_index_from_name(cb_name)
            elif "in1" in cb_name:
                cb_info["cb_in1"] = self._get_cb_index_from_name(cb_name)

        for cb_name in analyzer.cb_usage["outputs"]:
            if "out" in cb_name:
                cb_info["cb_out"] = self._get_cb_index_from_name(cb_name)

        return cb_info

    def _get_cb_index_from_name(self, cb_name: str) -> int:
        """Get CB index from name (heuristic)."""

        if "in0" in cb_name:
            return 0
        elif "in1" in cb_name:
            return 1
        elif "in2" in cb_name:
            return 2
        elif "out" in cb_name:
            return 16
        else:
            return 0

    def _convert_to_dict(self, attr_value: Any) -> Dict[str, Any]:
        """Convert TVM attribute value to Python dict."""

        if isinstance(attr_value, dict):
            return attr_value

        if hasattr(attr_value, "items"):
            result = {}
            for k, v in attr_value.items():
                result[str(k)] = self._convert_value(v)
            return result

        return {}

    def _convert_value(self, value: Any) -> Any:
        """Convert TVM value to Python type."""

        if hasattr(value, "value"):
            return value.value
        elif isinstance(value, (int, float, str, bool)):
            return value
        else:
            return str(value)


# Module-level pass function for compatibility
def insert_dst_management_tt(mod: IRModule) -> IRModule:
    """Apply InsertDSTManagementTT pass to a module."""
    pass_instance = InsertDSTManagementTT()
    return pass_instance(mod)


# Example usage and testing
if __name__ == "__main__":
    import tvm.script
    from tvm.script import tir as T

    # Create test module with compute kernels
    @tvm.script.ir_module
    class TestModule:

        @T.prim_func
        def gemm_compute():
            # GEMM with K-loop (accumulation pattern)
            for kt in T.serial(8):
                T.evaluate(T.call_extern("void", "tt.mm.mma", "cb_in0", "cb_in1", 0, kt > 0))

        @T.prim_func
        def add_compute():
            # Element-wise add (single-tile pattern)
            T.evaluate(T.call_extern("void", "tt.fpu.add", "cb_in0", "cb_in1", 0))

    # Add metadata for GEMM
    gemm_func = TestModule["gemm_compute"]
    gemm_func = gemm_func.with_attr("tt.kernel_role", "compute")
    gemm_func = gemm_func.with_attr("tt.cb_indices", {"cb_in0": 0, "cb_in1": 1, "cb_out": 16})
    TestModule["gemm_compute"] = gemm_func

    # Add metadata for element-wise
    add_func = TestModule["add_compute"]
    add_func = add_func.with_attr("tt.kernel_role", "compute")
    add_func = add_func.with_attr("tt.cb_indices", {"cb_in0": 0, "cb_in1": 1, "cb_out": 16})
    TestModule["add_compute"] = add_func

    # Apply D5 pass
    pass_d5 = InsertDSTManagementTT()
    result = pass_d5(TestModule)

    # Check results
    print("=== DST Management Results ===\n")
    for name, func in result.functions_items():
        if func.attrs and "tt.kernel_role" in func.attrs:
            print(f"{name} ({func.attrs['tt.kernel_role']}):")
            if "tt.dst_management_inserted" in func.attrs:
                print(f"  DST management inserted: {func.attrs['tt.dst_management_inserted']}")
                print(f"  DST pattern: {func.attrs.get(TT_DST_PATTERN, 'unknown')}")
            print(f"  Body preview: {str(func.body)[:300]}...")
            print()
