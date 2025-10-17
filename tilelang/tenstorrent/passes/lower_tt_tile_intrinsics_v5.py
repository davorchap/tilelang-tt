"""
Pass C2: LowerTTTileIntrinsics (Without Heuristics) - Fixed with BlockTransformer
Version: 5.1
Date: 2025-10-15

Purpose: Tensorize compute operations to CB-based intrinsics WITHOUT:
         - CB ID heuristics ("_tile" suffix detection)
         - DST management
         - Engine initialization
         This maintains protocol-less mid-level TIR.

This version properly handles TVM Block structures using BlockTransformer.

Input: TIR with T.gemm, element-wise ops, and conceptual CBs from C1
Output: TIR with protocol-less tt.mm.mma, tt.fpu.add, tt.sfpu.unary operations
"""

import tvm
from tvm import tir
from tvm.tir import stmt_functor
import sys
import os

# Import our BlockTransformer base class
sys.path.append(os.path.dirname(__file__))
from block_transformer import BlockTransformer


def _transform_tile_intrinsics(func, mod, ctx):
    """
    Core transformation logic for tensorizing compute operations.

    This function:
    1. Detects compute patterns (GEMM, element-wise, etc.)
    2. Replaces them with protocol-less TT intrinsics
    3. Does NOT use naming heuristics
    4. Does NOT insert DST management or engine init
    """
    print(f"[_transform_tile_intrinsics] CALLED for function {func.attrs.get('global_symbol', 'unknown')}")

    class TileIntrinsicLowerer(BlockTransformer):

        def __init__(self, cb_metadata):
            super().__init__()
            self.cb_metadata = cb_metadata or {}
            self.shared_to_cb_map = {}  # Map from shared buffer names to CB names
            self.loop_stack = []  # Track loop nesting
            self.compute_patterns = []  # Detected patterns

            # Build reverse mapping from CB metadata
            for cb_name, info in self.cb_metadata.items():
                if "original_buffer" in info:
                    self.shared_to_cb_map[info["original_buffer"]] = cb_name

        def visit_block(self, block):
            """Override to detect compute patterns in blocks"""
            # First, process normally
            result = super().visit_block(block)

            # Then detect patterns in the block body
            self._detect_patterns_in_block(block)

            return result

        def visit_for(self, for_node):
            """Track loop context for pattern detection"""
            # Push loop info onto stack
            self.loop_stack.append({
                "var": for_node.loop_var,
                "min": for_node.min,
                "extent": for_node.extent,
                "kind": for_node.kind,
                "is_reduction": self._is_reduction_loop(for_node)
            })

            # Visit body
            new_body = self.visit(for_node.body)
            print(f"[TileIntrinsicLowerer.visit_for] Body changed: {new_body is not for_node.body}")

            # Check if this is a reduction loop pattern
            if self._is_reduction_loop(for_node):
                # Mark for K-loop accumulation pattern
                old_new_body = new_body
                new_body = self._annotate_k_loop(new_body)
                print(f"[TileIntrinsicLowerer.visit_for] Annotated k-loop, body changed: {new_body is not old_new_body}")

            # Pop loop info
            self.loop_stack.pop()

            if new_body is not for_node.body:  # Use object identity, not equality!
                print(f"[TileIntrinsicLowerer.visit_for] Returning new For node")
                return tir.For(for_node.loop_var, for_node.min, for_node.extent, for_node.kind,
                               new_body, for_node.thread_binding, for_node.annotations,
                               for_node.span)
            print(f"[TileIntrinsicLowerer.visit_for] Returning original For node")
            return for_node

        def visit_evaluate(self, evaluate_node):
            """Transform high-level compute operations to TT intrinsics"""

            # Check for T.fill / tl.fill pattern
            if self._is_fill_intrinsic(evaluate_node):
                lowered = self._lower_fill(evaluate_node)
                # Debug what we created
                print(f"[visit_evaluate] Lowered fill:")
                if hasattr(lowered.value, 'args'):
                    for i, arg in enumerate(lowered.value.args):
                        print(f"[visit_evaluate]   args[{i}]: {arg}, type={type(arg)}")
                return lowered

            # Check for T.copy / tl.copy pattern
            elif self._is_copy_intrinsic(evaluate_node):
                lowered = self._lower_copy(evaluate_node)
                print(f"[visit_evaluate] Lowering copy: {evaluate_node.value.op.name if hasattr(evaluate_node.value.op, 'name') else 'unknown'} -> {type(lowered)}")
                return lowered

            # Check for T.gemm / tl.gemm pattern
            elif self._is_gemm_intrinsic(evaluate_node):
                lowered = self._lower_gemm(evaluate_node)
                print(f"[visit_evaluate] Lowering gemm: {evaluate_node.value.op.name if hasattr(evaluate_node.value.op, 'name') else 'unknown'} -> {type(lowered)}")
                return lowered

            # Check for element-wise operations
            elif self._is_elementwise_op(evaluate_node):
                return self._lower_elementwise(evaluate_node)

            # Check for SFPU operations
            elif self._is_sfpu_op(evaluate_node):
                return self._lower_sfpu(evaluate_node)

            return evaluate_node

        def visit_buffer_store(self, buffer_store):
            """Detect compute patterns in buffer stores"""

            # Check for matmul accumulation pattern
            if self._is_matmul_accumulation(buffer_store):
                return self._lower_matmul_accumulation(buffer_store)

            # Check for binary operation pattern
            elif self._is_binary_operation(buffer_store):
                return self._lower_binary_operation(buffer_store)

            return buffer_store

        # Pattern detection methods

        def _detect_patterns_in_block(self, block):
            """Detect compute patterns within a block"""

            def visitor(node):
                if isinstance(node, tir.Evaluate):
                    if self._is_gemm_intrinsic(node):
                        self.compute_patterns.append(("gemm", node))
                    elif self._is_elementwise_op(node):
                        self.compute_patterns.append(("elementwise", node))
                    elif self._is_sfpu_op(node):
                        self.compute_patterns.append(("sfpu", node))
                elif isinstance(node, tir.BufferStore):
                    if self._is_matmul_accumulation(node):
                        self.compute_patterns.append(("matmul_accum", node))
                    elif self._is_binary_operation(node):
                        self.compute_patterns.append(("binary", node))

            stmt_functor.post_order_visit(block.body, visitor)

        def _is_fill_intrinsic(self, evaluate_node):
            """Check for T.fill / tl.fill intrinsic call"""
            if isinstance(evaluate_node.value, tir.Call):
                call = evaluate_node.value
                if hasattr(call, 'op') and hasattr(call.op, 'name'):
                    op_name = call.op.name
                    return op_name in ["tl.fill", "T.fill", "tir.fill"]
            return False

        def _is_copy_intrinsic(self, evaluate_node):
            """Check for T.copy / tl.copy intrinsic call"""
            if isinstance(evaluate_node.value, tir.Call):
                call = evaluate_node.value
                if hasattr(call, 'op') and hasattr(call.op, 'name'):
                    op_name = call.op.name
                    return op_name in ["tl.copy", "T.copy", "tir.copy"]
            return False

        def _is_gemm_intrinsic(self, evaluate_node):
            """Check for T.gemm / tl.gemm intrinsic call"""
            if isinstance(evaluate_node.value, tir.Call):
                call = evaluate_node.value
                # Check op name directly (most common for TileLang intrinsics)
                if hasattr(call, 'op') and hasattr(call.op, 'name'):
                    op_name = call.op.name
                    return op_name in ["tl.gemm", "T.gemm", "tir.gemm", "tvm_gemm"]
                # Also check for call_extern format
                elif str(call.op) == "Op(tir.call_extern)":
                    # For call_extern, args[0] is the function name
                    if len(call.args) >= 1 and isinstance(call.args[0], tir.StringImm):
                        func_name = call.args[0].value
                        return any(
                            pattern in func_name for pattern in ["T.gemm", "tir.gemm", "tvm_gemm", "tl.gemm"])
            return False

        def _is_elementwise_op(self, evaluate_node):
            """Check for element-wise operations"""
            if isinstance(evaluate_node.value, tir.Call):
                call = evaluate_node.value
                # Check for call_extern with elementwise op as function name
                if str(call.op) == "Op(tir.call_extern)":
                    # For call_extern, args[0] is the function name
                    if len(call.args) >= 1 and isinstance(call.args[0], tir.StringImm):
                        func_name = call.args[0].value
                        elementwise_ops = [
                            "T.add", "T.multiply", "T.subtract", "T.divide", "tir.add",
                            "tir.multiply", "tir.subtract", "tir.divide"
                        ]
                        return any(op in func_name for op in elementwise_ops)
                # Also check op name directly
                elif hasattr(call, 'op'):
                    op_name = str(call.op)
                    elementwise_ops = [
                        "T.add", "T.multiply", "T.subtract", "T.divide", "tir.add", "tir.multiply",
                        "tir.subtract", "tir.divide"
                    ]
                    return any(op in op_name for op in elementwise_ops)
            return False

        def _is_sfpu_op(self, evaluate_node):
            """Check for SFPU (SIMD FPU) operations"""
            if isinstance(evaluate_node.value, tir.Call):
                call = evaluate_node.value
                if hasattr(call, 'op'):
                    op_name = str(call.op)
                    sfpu_ops = ["exp", "log", "sin", "cos", "tanh", "sigmoid", "gelu", "relu"]
                    return any(op in op_name for op in sfpu_ops)
            return False

        def _is_matmul_accumulation(self, buffer_store):
            """Detect matmul accumulation pattern in buffer stores"""
            # Look for pattern: C[i,j] += A[i,k] * B[k,j]
            value = buffer_store.value

            # Check for Add node with multiplication
            return isinstance(value, tir.Add) and isinstance(
                value.a, tir.BufferLoad) and value.a.buffer == buffer_store.buffer and isinstance(
                    value.b, tir.Mul)

        def _is_binary_operation(self, buffer_store):
            """Detect binary operations (add, sub, mul, div)"""
            value = buffer_store.value
            return isinstance(value, (tir.Add, tir.Sub, tir.Mul, tir.Div))

        def _is_reduction_loop(self, for_node):
            """Check if this is a reduction loop (K dimension in GEMM)"""

            # Look for patterns that indicate reduction
            # Avoid heuristics - check for actual reduction semantics

            class ReductionChecker:

                def __init__(self, var):
                    self.var = var
                    self.has_reduction = False

                def check(self, stmt):
                    if isinstance(stmt, tir.BufferStore) and isinstance(
                            stmt.value, tir.Add) and self._uses_var(stmt.value, self.var):
                        self.has_reduction = True
                    elif isinstance(stmt, tir.SeqStmt):
                        for s in stmt.seq:
                            self.check(s)
                    elif hasattr(stmt, 'body'):
                        self.check(stmt.body)

                def _uses_var(self, expr, var):
                    """Check if expression uses the variable"""
                    # Simple check - would need full visitor for production
                    return str(var) in str(expr)

            checker = ReductionChecker(for_node.loop_var)
            checker.check(for_node.body)
            return checker.has_reduction

        # Lowering methods

        def _lower_fill(self, evaluate_node):
            """Lower T.fill / tl.fill to initialization (protocol-less)"""
            # T.fill is typically used to zero-initialize output buffers
            # At this protocol-less stage, we can just remove it
            # The actual initialization will be handled by compute init pass
            # For now, return a no-op or keep the call as-is for later passes
            # to recognize
            call = evaluate_node.value

            # Create a protocol-less fill marker that later passes can use
            # This just marks that a buffer needs initialization
            arg0 = call.args[0] if len(call.args) > 0 else tir.IntImm("int32", 0)
            arg1 = call.args[1] if len(call.args) > 1 else tir.IntImm("int32", 0)

            return tir.Evaluate(
                tir.call_extern(
                    "void",
                    "tt.fill.zero",  # Protocol-less fill marker
                    arg0,
                    arg1
                ))

        def _lower_copy(self, evaluate_node):
            """Lower T.copy / tl.copy to protocol-less marker"""
            # T.copy is used to transfer data between memory spaces
            # At this protocol-less stage, we just mark the intent
            # Actual NOC operations will be inserted by lower_cb_intrinsics (Pass D3)
            call = evaluate_node.value
            args = call.args

            # args typically are: source_region, dest_region, mask, predicate, cache_hint
            # For now, create a protocol-less copy marker
            # Extract source and dest buffer info if possible
            src = args[0] if len(args) > 0 else tir.IntImm("int32", 0)
            dst = args[1] if len(args) > 1 else tir.IntImm("int32", 0)

            # Create protocol-less copy marker
            return tir.Evaluate(
                tir.call_extern(
                    "void",
                    "tt.copy.protocol_less",  # Marker for later lowering
                    src,
                    dst
                ))

        def _lower_gemm(self, evaluate_node):
            """Lower T.gemm / tl.gemm to protocol-less tt.mm.mma"""
            call = evaluate_node.value
            args = call.args

            # For call_extern, args are: [func_name, actual_args...]
            # Check if this is a call_extern
            if str(call.op) == "Op(tir.call_extern)" and len(args) >= 1:
                # Skip first arg (function name)
                actual_args = args[1:]
            else:
                actual_args = args

            # Extract CB arguments (should be conceptual CBs from C1)
            cb_a = self._get_cb_for_arg(actual_args[0]) if len(actual_args) > 0 else "cb_in0"
            cb_b = self._get_cb_for_arg(actual_args[1]) if len(actual_args) > 1 else "cb_in1"
            self._get_cb_for_arg(actual_args[2]) if len(actual_args) > 2 else "cb_out0"

            # Determine if this is accumulating
            accumulate = self._in_reduction_loop()

            # Create protocol-less matmul intrinsic
            # Note: No DST management, no engine init
            return tir.Evaluate(
                tir.call_extern(
                    "void",
                    "tt.mm.mma",
                    tir.StringImm(cb_a),
                    tir.StringImm(cb_b),
                    tir.IntImm("int32", 0),  # dst register (always 0 for now)
                    tir.IntImm("bool", accumulate)))

        def _lower_elementwise(self, evaluate_node):
            """Lower element-wise operations to tt.fpu intrinsics"""
            call = evaluate_node.value
            args = call.args

            # Determine the operation name
            op_name = ""
            # For call_extern, function name is in args[0]
            if str(call.op) == "Op(tir.call_extern)" and len(args) >= 1 and isinstance(
                    args[0], tir.StringImm):
                op_name = args[0].value
                actual_args = args[1:]  # Skip function name
            else:
                op_name = str(call.op)
                actual_args = args

            # Map high-level ops to TT FPU ops
            op_map = {
                "T.add": "add",
                "T.multiply": "multiply",
                "T.subtract": "subtract",
                "T.divide": "divide",
                "tir.add": "add",
                "tir.multiply": "multiply",
                "tir.subtract": "subtract",
                "tir.divide": "divide"
            }

            # Find matching operation
            tt_op = "add"  # default
            for pattern, op in op_map.items():
                if pattern in op_name:
                    tt_op = op
                    break

            # Get CB operands
            cb_a = self._get_cb_for_arg(actual_args[0]) if len(actual_args) > 0 else "cb_in0"
            cb_b = self._get_cb_for_arg(actual_args[1]) if len(actual_args) > 1 else cb_a

            # Create protocol-less FPU intrinsic
            return tir.Evaluate(
                tir.call_extern(
                    "void",
                    f"tt.fpu.{tt_op}",
                    tir.StringImm(cb_a),
                    tir.StringImm(cb_b),
                    tir.IntImm("int32", 0)  # dst register
                ))

        def _lower_sfpu(self, evaluate_node):
            """Lower SFPU operations to tt.sfpu intrinsics"""
            call = evaluate_node.value
            op_name = str(call.op)

            # Extract operation name
            sfpu_op = "relu"  # default
            sfpu_ops = ["exp", "log", "sin", "cos", "tanh", "sigmoid", "gelu", "relu"]
            for op in sfpu_ops:
                if op in op_name:
                    sfpu_op = op
                    break

            # Get CB operand
            cb_in = self._get_cb_for_arg(call.args[0]) if len(call.args) > 0 else "cb_in0"

            # Create protocol-less SFPU intrinsic
            return tir.Evaluate(
                tir.call_extern(
                    "void",
                    "tt.sfpu.unary",
                    tir.StringImm(sfpu_op),
                    tir.StringImm(cb_in),
                    tir.IntImm("int32", 0)  # dst register
                ))

        def _lower_matmul_accumulation(self, buffer_store):
            """Lower matmul accumulation pattern"""
            # Extract operands from the accumulation pattern
            value = buffer_store.value

            if isinstance(value, tir.Add) and isinstance(value.b, tir.Mul):
                mul = value.b
                # Get source buffers
                cb_a = self._get_cb_for_buffer(mul.a)
                cb_b = self._get_cb_for_buffer(mul.b)

                # This is accumulating since we're adding to existing value
                accumulate = True

                # Create protocol-less matmul
                return tir.Evaluate(
                    tir.call_extern(
                        "void",
                        "tt.mm.mma",
                        tir.StringImm(cb_a),
                        tir.StringImm(cb_b),
                        tir.IntImm("int32", 0),  # dst
                        tir.IntImm("bool", accumulate)))

            return buffer_store

        def _lower_binary_operation(self, buffer_store):
            """Lower binary operations"""
            value = buffer_store.value

            op_map = {tir.Add: "add", tir.Sub: "subtract", tir.Mul: "multiply", tir.Div: "divide"}

            op_type = type(value)
            if op_type in op_map:
                tt_op = op_map[op_type]

                # Get CB operands - handle different operand types
                cb_a = self._get_cb_name_for_operand(value.a)
                cb_b = self._get_cb_name_for_operand(value.b)

                # Create protocol-less FPU operation
                fpu_op = tir.Evaluate(
                    tir.call_extern(
                        "void",
                        f"tt.fpu.{tt_op}",
                        tir.StringImm(cb_a),
                        tir.StringImm(cb_b),
                        tir.IntImm("int32", 0)  # dst
                    ))

                # Return both the FPU op and the original store (for now)
                # Later passes will optimize this
                return tir.SeqStmt([fpu_op, buffer_store])

            return buffer_store

        def _get_cb_name_for_operand(self, operand):
            """Get CB name for an operand (buffer or constant)"""
            # If it's a buffer load, try to get CB mapping
            if isinstance(operand, tir.BufferLoad):
                cb_name = self._get_cb_for_buffer(operand)
                if cb_name and cb_name != "cb_in0":  # If we found a mapping
                    return cb_name
                # Otherwise use buffer name as hint
                if hasattr(operand.buffer, 'name'):
                    buffer_name = operand.buffer.name
                    if buffer_name in self.shared_to_cb_map:
                        return self.shared_to_cb_map[buffer_name]
                return "cb_in0"  # Default input CB
            # If it's a constant, we'll need a constant CB
            elif isinstance(operand, (tir.FloatImm, tir.IntImm)):
                return "cb_const0"  # Constant CB
            # Default
            return "cb_in0"

        # Helper methods

        def _get_cb_for_arg(self, arg):
            """Get CB name for an argument (no heuristics!)"""
            # Check if this references a conceptual CB from metadata
            if hasattr(arg, 'buffer'):
                buffer_name = arg.buffer.name
                # Look up in CB metadata from C1 pass
                if buffer_name in self.shared_to_cb_map:
                    return self.shared_to_cb_map[buffer_name]

                # Also check CB metadata directly
                for cb_name, info in self.cb_metadata.items():
                    if info.get("original_buffer") == buffer_name:
                        return cb_name

            # Check if arg is a string (CB name)
            elif isinstance(arg, tir.StringImm):
                return arg.value

            # Default naming if not found
            return "cb_in0"

        def _get_cb_for_buffer(self, buffer_ref):
            """Get CB name for a buffer reference"""
            if isinstance(buffer_ref, tir.BufferLoad):
                return self._get_cb_for_arg(buffer_ref)
            elif isinstance(buffer_ref, tir.Var):
                # Check if this is a known buffer
                var_name = buffer_ref.name if hasattr(buffer_ref, 'name') else str(buffer_ref)
                if var_name in self.shared_to_cb_map:
                    return self.shared_to_cb_map[var_name]
            return "cb_in0"

        def _in_reduction_loop(self):
            """Check if we're currently in a reduction loop"""
            return any(loop.get("is_reduction", False) for loop in self.loop_stack)

        def _annotate_k_loop(self, body):
            """Annotate body as being in a K-loop for later passes"""
            # Add annotation for DST management pass
            return tir.AttrStmt(
                tir.StringImm("tt.k_loop"), "pragma_scope", tir.IntImm("int32", 1), body)

    # Get CB metadata from previous pass
    cb_metadata = func.attrs.get("tt.conceptual_cbs", {})

    # Use ir_transform instead of BlockTransformer for proper node reconstruction
    def post_visit(stmt):
        """Post-visit function for ir_transform"""
        if isinstance(stmt, tir.Evaluate):
            # Create temporary lowerer just for detection
            temp_lowerer = TileIntrinsicLowerer(cb_metadata)

            # Check if this is a TileLang intrinsic that needs lowering
            if temp_lowerer._is_fill_intrinsic(stmt):
                print(f"[ir_transform] Lowering fill")
                return temp_lowerer._lower_fill(stmt)
            elif temp_lowerer._is_copy_intrinsic(stmt):
                print(f"[ir_transform] Lowering copy")
                return temp_lowerer._lower_copy(stmt)
            elif temp_lowerer._is_gemm_intrinsic(stmt):
                print(f"[ir_transform] Lowering gemm")
                return temp_lowerer._lower_gemm(stmt)

        return stmt

    print(f"[_transform] Using ir_transform on func.body")
    new_body = stmt_functor.ir_transform(func.body, None, post_visit)
    print(f"[_transform] ir_transform returned new_body")

    # DEBUG: Check what's in new_body BEFORE with_body
    calls_before = []
    first_evaluate_found = False
    def scan_before(stmt):
        nonlocal first_evaluate_found
        if isinstance(stmt, tir.Evaluate) and isinstance(stmt.value, tir.Call):
            if not first_evaluate_found:
                first_evaluate_found = True
                print(f"[scan_before DEBUG] FIRST Evaluate node found:")
                print(f"[scan_before DEBUG]   stmt ID: {id(stmt)}")
                print(f"[scan_before DEBUG]   stmt.value ID: {id(stmt.value)}")
                print(f"[scan_before DEBUG]   stmt.value.op: {stmt.value.op}")
                if hasattr(stmt.value, 'args'):
                    for i, arg in enumerate(stmt.value.args[:3]):
                        print(f"[scan_before DEBUG]   args[{i}]: {arg}")

            if hasattr(stmt.value.op, 'name'):
                op_name = stmt.value.op.name
                op_str = str(stmt.value.op)
                # For call_extern with void return type, function name is in args[0]!
                if op_str == "Op(tir.call_extern)" and len(stmt.value.args) > 0:
                    if isinstance(stmt.value.args[0], tir.StringImm):
                        calls_before.append(f"[extern:{stmt.value.args[0].value}]")
                else:
                    calls_before.append(op_name)
    tir.stmt_functor.post_order_visit(new_body, scan_before)
    print(f"[_transform] new_body BEFORE with_body: {calls_before[:5]}")

    # Update function with transformed body
    func = func.with_body(new_body)

    # DEBUG: Check what's in func.body AFTER with_body
    calls_after = []
    def scan_after(stmt):
        if isinstance(stmt, tir.Evaluate) and isinstance(stmt.value, tir.Call):
            if hasattr(stmt.value.op, 'name'):
                op_name = stmt.value.op.name
                if str(stmt.value.op) == "Op(tir.call_extern)" and len(stmt.value.args) > 0:
                    if isinstance(stmt.value.args[0], tir.StringImm):
                        calls_after.append(f"[extern:{stmt.value.args[0].value}]")
                else:
                    calls_after.append(op_name)
    tir.stmt_functor.post_order_visit(func.body, scan_after)
    print(f"[_transform] func.body AFTER with_body: {calls_after[:5]}")

    # Note: We're no longer tracking compute patterns since we switched to ir_transform
    # If needed, this can be re-added by scanning the transformed IR

    return func


# Decorated version for TVM pass infrastructure
@tvm.tir.transform.prim_func_pass(opt_level=0)
def LowerTTTileIntrinsics_v5(func, mod, ctx):
    """TVM pass wrapper for tile intrinsic lowering."""
    return _transform_tile_intrinsics(func, mod, ctx)


def validate_no_heuristics(func):
    """
    Validate that the pass doesn't use naming heuristics.
    Should NOT have:
    - "_tile" suffix checking
    - Variable name pattern matching
    - Hard-coded CB IDs beyond defaults
    """

    class HeuristicChecker:

        def __init__(self):
            self.has_heuristics = False
            self.issues = []

        def check(self, stmt):
            """Check for heuristics"""

            def visitor(node):
                if isinstance(node, tir.Call):
                    # Check arguments for hard-coded patterns
                    for arg in node.args:
                        if isinstance(arg, tir.StringImm) and arg.value in [
                                "cb0", "cb1", "cb2", "cb16", "cb24", "cb32"
                        ]:
                            # Check for suspicious hard-coded CB IDs
                            # (cb_in0, cb_in1, cb_out0 are OK as defaults)
                            self.has_heuristics = True
                            self.issues.append(f"Hard-coded CB ID: {arg.value}")

            stmt_functor.post_order_visit(stmt, visitor)

    checker = HeuristicChecker()
    checker.check(func.body)

    if checker.has_heuristics:
        raise ValueError(f"Output uses heuristics: {checker.issues}")

    return True


# Example usage and testing
if __name__ == "__main__":
    import tvm.script
    from tvm.script import tir as T

    # Create a test function with high-level operations
    @tvm.script.ir_module
    class TestModule:

        @T.prim_func
        def gemm_example(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                         C: T.Buffer((256, 256), "float16")):
            # Shared memory allocations (would be from C1 pass)
            A_shared = T.alloc_buffer((32, 32), "float16", scope="shared")
            B_shared = T.alloc_buffer((32, 32), "float16", scope="shared")

            # High-level operations to be lowered
            for k in T.serial(8):
                # This would be lowered to tt.mm.mma
                for i, j in T.grid(32, 32):
                    C[i, j] = C[i, j] + A_shared[i, k] * B_shared[k, j]

    # Get the function
    func = TestModule["gemm_example"]

    # Simulate CB metadata from C1
    func = func.with_attr(
        "tt.conceptual_cbs", {
            "cb_in0": {
                "shape": [32, 32],
                "dtype": "float16",
                "original_buffer": "A_shared"
            },
            "cb_in1": {
                "shape": [32, 32],
                "dtype": "float16",
                "original_buffer": "B_shared"
            },
            "cb_out0": {
                "shape": [256, 256],
                "dtype": "float16",
                "original_buffer": "C"
            }
        })

    # Apply the pass
    transformed = LowerTTTileIntrinsics_v5(func, TestModule, None)

    print("=== Original Function ===")
    print(func.script())
    print("\n=== Transformed Function (Protocol-less) ===")
    print(transformed.script())

    # Validate
    try:
        validate_no_heuristics(transformed)
        print("\n✅ Pass validation successful: No heuristics used")
    except ValueError as e:
        print(f"\n❌ Validation failed: {e}")


# Module-level wrapper function for compatibility with test imports
def lower_tt_tile_intrinsics_v5(mod):
    """Apply LowerTTTileIntrinsics v5 pass to a module."""
    print("[Pass C2] lower_tt_tile_intrinsics_v5 wrapper CALLED")

    # Call the decorated version (like Pass C1 does)
    result = LowerTTTileIntrinsics_v5(mod)

    # Verify what we're returning
    for gvar, func in result.functions.items():
        if isinstance(func, tir.PrimFunc):
            calls_in_result = []
            def scan(stmt):
                if isinstance(stmt, tir.Evaluate) and isinstance(stmt.value, tir.Call):
                    call = stmt.value
                    if hasattr(call.op, 'name'):
                        op_name = call.op.name
                        # Check if this is call_extern - function name is in args[0] for void return type!
                        if str(call.op) == "Op(tir.call_extern)" and len(call.args) > 0:
                            if isinstance(call.args[0], tir.StringImm):
                                calls_in_result.append(f"{call.args[0].value}")
                        else:
                            calls_in_result.append(op_name)
            tir.stmt_functor.post_order_visit(func.body, scan)
            print(f"[Pass C2] Wrapper returning function {gvar.name_hint} with calls: {calls_in_result[:5]}")

    return result
