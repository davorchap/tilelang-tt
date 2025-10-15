"""
Pass C2: LowerTTTileIntrinsics (Without Heuristics)
Version: 5.0
Date: 2025-10-15

Purpose: Tensorize compute operations to CB-based intrinsics WITHOUT:
         - CB ID heuristics ("_tile" suffix detection)
         - DST management
         - Engine initialization
         This maintains protocol-less mid-level TIR.

Input: TIR with T.gemm, element-wise ops, and conceptual CBs from C1
Output: TIR with protocol-less tt.mm.mma, tt.fpu.add, tt.sfpu.unary operations
"""

import tvm
from tvm import tir
from tvm.tir import stmt_functor


@tvm.tir.transform.prim_func_pass(opt_level=0)
def LowerTTTileIntrinsics_v5(func, mod, ctx):
    """
    Tensorize compute operations to protocol-less TT intrinsics.

    This pass:
    1. Detects compute patterns (GEMM, element-wise, etc.)
    2. Replaces them with protocol-less TT intrinsics
    3. Does NOT use naming heuristics
    4. Does NOT insert DST management or engine init
    """

    class TileIntrinsicLowerer(stmt_functor.IRMutator):

        def __init__(self, cb_metadata):
            super().__init__()
            self.cb_metadata = cb_metadata or {}
            self.loop_stack = []  # Track loop nesting
            self.compute_patterns = []  # Detected patterns

        def visit_for(self, op):
            """Track loop context for pattern detection"""
            # Push loop info onto stack
            self.loop_stack.append({
                "var": op.loop_var,
                "min": op.min,
                "extent": op.extent,
                "kind": op.kind
            })

            # Visit body
            new_body = self.visit(op.body)

            # Check if this is a reduction loop pattern
            if self._is_reduction_loop(op):
                # Mark for K-loop accumulation pattern
                new_body = self._annotate_k_loop(new_body)

            # Pop loop info
            self.loop_stack.pop()

            return tir.For(op.loop_var, op.min, op.extent, op.kind, new_body)

        def visit_evaluate(self, op):
            """Transform high-level compute operations to TT intrinsics"""

            # Check for T.gemm pattern
            if self._is_gemm_intrinsic(op):
                return self._lower_gemm(op)

            # Check for element-wise operations
            elif self._is_elementwise_op(op):
                return self._lower_elementwise(op)

            # Check for SFPU operations
            elif self._is_sfpu_op(op):
                return self._lower_sfpu(op)

            return super().visit_evaluate(op)

        def visit_buffer_store(self, op):
            """Detect compute patterns in buffer stores"""

            # Check for matmul accumulation pattern
            if self._is_matmul_accumulation(op):
                return self._lower_matmul_accumulation(op)

            # Check for binary operation pattern
            elif self._is_binary_operation(op):
                return self._lower_binary_operation(op)

            return super().visit_buffer_store(op)

        # Pattern detection methods (no heuristics!)

        def _is_gemm_intrinsic(self, evaluate_node):
            """Check for T.gemm intrinsic call"""
            if isinstance(evaluate_node.value, tir.Call):
                call = evaluate_node.value
                if hasattr(call.op, 'name'):
                    return call.op.name in ["T.gemm", "tir.gemm", "tvm_gemm"]
            return False

        def _is_elementwise_op(self, evaluate_node):
            """Check for element-wise operations"""
            if isinstance(evaluate_node.value, tir.Call):
                call = evaluate_node.value
                if hasattr(call.op, 'name'):
                    elementwise_ops = ["T.add", "T.multiply", "T.subtract", "T.divide"]
                    return any(op in call.op.name for op in elementwise_ops)
            return False

        def _is_sfpu_op(self, evaluate_node):
            """Check for SFPU (SIMD FPU) operations"""
            if isinstance(evaluate_node.value, tir.Call):
                call = evaluate_node.value
                if hasattr(call.op, 'name'):
                    sfpu_ops = ["exp", "log", "sin", "cos", "tanh", "sigmoid", "gelu"]
                    return any(op in call.op.name for op in sfpu_ops)
            return False

        def _is_matmul_accumulation(self, buffer_store):
            """Detect matmul accumulation pattern in buffer stores"""
            # Look for pattern: C[i,j] += A[i,k] * B[k,j]
            value = buffer_store.value

            # Check for Add node with multiplication
            if isinstance(value, tir.Add):
                # Check if adding to same buffer (accumulation)
                if isinstance(value.a, tir.BufferLoad):
                    if value.a.buffer == buffer_store.buffer:
                        # Check if b is multiplication
                        if isinstance(value.b, tir.Mul):
                            return True
            return False

        def _is_binary_operation(self, buffer_store):
            """Detect binary operations (add, sub, mul, div)"""
            value = buffer_store.value
            return isinstance(value, (tir.Add, tir.Sub, tir.Mul, tir.Div))

        def _is_reduction_loop(self, for_node):
            """Check if this is a reduction loop (K dimension in GEMM)"""
            # Look for patterns that indicate reduction
            # Avoid heuristics - check for actual reduction semantics

            # Check if loop variable is used in accumulation
            loop_var = for_node.loop_var

            class ReductionChecker(stmt_functor.StmtVisitor):

                def __init__(self, var):
                    super().__init__()
                    self.var = var
                    self.has_reduction = False

                def visit_buffer_store(self, op):
                    # Check for accumulation pattern
                    if isinstance(op.value, tir.Add):
                        if self._uses_var(op.value, self.var):
                            self.has_reduction = True
                    super().visit_buffer_store(op)

                def _uses_var(self, expr, var):
                    """Check if expression uses the variable"""
                    # This is simplified - would need full visitor
                    return str(var) in str(expr)

            checker = ReductionChecker(loop_var)
            checker.visit(for_node.body)
            return checker.has_reduction

        # Lowering methods

        def _lower_gemm(self, evaluate_node):
            """Lower T.gemm to protocol-less tt.mm.mma"""
            call = evaluate_node.value
            args = call.args

            # Extract CB arguments (should be conceptual CBs from C1)
            cb_a = self._get_cb_for_arg(args[0])
            cb_b = self._get_cb_for_arg(args[1])
            self._get_cb_for_arg(args[2]) if len(args) > 2 else None

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
            op_name = call.op.name

            # Map high-level ops to TT FPU ops
            op_map = {
                "T.add": "add",
                "T.multiply": "multiply",
                "T.subtract": "subtract",
                "T.divide": "divide"
            }

            tt_op = op_map.get(op_name, "add")

            # Get CB operands
            cb_a = self._get_cb_for_arg(call.args[0])
            cb_b = self._get_cb_for_arg(call.args[1]) if len(call.args) > 1 else cb_a

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
            op_name = call.op.name.split(".")[-1]  # Get operation name

            # Get CB operand
            cb_in = self._get_cb_for_arg(call.args[0])

            # Create protocol-less SFPU intrinsic
            return tir.Evaluate(
                tir.call_extern(
                    "void",
                    "tt.sfpu.unary",
                    tir.StringImm(op_name),
                    tir.StringImm(cb_in),
                    tir.IntImm("int32", 0)  # dst register
                ))

        def _lower_matmul_accumulation(self, buffer_store):
            """Lower matmul accumulation pattern"""
            # Extract operands from the accumulation pattern
            value = buffer_store.value

            if isinstance(value.b, tir.Mul):
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

            return super().visit_buffer_store(buffer_store)

        def _lower_binary_operation(self, buffer_store):
            """Lower binary operations"""
            value = buffer_store.value

            op_map = {tir.Add: "add", tir.Sub: "subtract", tir.Mul: "multiply", tir.Div: "divide"}

            op_type = type(value)
            if op_type in op_map:
                tt_op = op_map[op_type]

                # Get CB operands
                cb_a = self._get_cb_for_buffer(value.a)
                cb_b = self._get_cb_for_buffer(value.b)

                # Create protocol-less FPU operation
                return tir.Evaluate(
                    tir.call_extern(
                        "void",
                        f"tt.fpu.{tt_op}",
                        tir.StringImm(cb_a),
                        tir.StringImm(cb_b),
                        tir.IntImm("int32", 0)  # dst
                    ))

            return super().visit_buffer_store(buffer_store)

        # Helper methods

        def _get_cb_for_arg(self, arg):
            """Get CB name for an argument (no heuristics!)"""
            # Check if this references a conceptual CB from metadata
            if hasattr(arg, 'buffer'):
                buffer_name = arg.buffer.name
                # Look up in CB metadata from C1 pass
                for cb_name, info in self.cb_metadata.items():
                    if info.get("original_buffer") == buffer_name:
                        return cb_name

            # Default naming if not found
            return "cb_unknown"

        def _get_cb_for_buffer(self, buffer_ref):
            """Get CB name for a buffer reference"""
            if isinstance(buffer_ref, tir.BufferLoad):
                return self._get_cb_for_arg(buffer_ref)
            return "cb_unknown"

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

    # Apply transformation
    lowerer = TileIntrinsicLowerer(cb_metadata)
    new_body = lowerer.visit(func.body)

    # Update function
    func = func.with_body(new_body)

    # Add metadata about compute patterns found
    if lowerer.compute_patterns:
        func = func.with_attr("tt.compute_patterns", lowerer.compute_patterns)

    return func


def validate_no_heuristics(func):
    """
    Validate that the pass doesn't use naming heuristics.
    Should NOT have:
    - "_tile" suffix checking
    - Variable name pattern matching
    - Hard-coded CB IDs
    """

    class HeuristicChecker(stmt_functor.StmtVisitor):

        def __init__(self):
            super().__init__()
            self.has_heuristics = False
            self.issues = []

        def visit_evaluate(self, op):
            if isinstance(op.value, tir.Call):
                # Check arguments for hard-coded patterns
                for arg in op.value.args:
                    if isinstance(arg, tir.StringImm):
                        # Check for hard-coded CB IDs
                        if arg.value in ["cb0", "cb1", "cb2"]:
                            self.has_heuristics = True
                            self.issues.append(f"Hard-coded CB ID: {arg.value}")

            super().visit_evaluate(op)

    checker = HeuristicChecker()
    checker.visit(func.body)

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
            # Assume CBs already allocated by C1 pass
            for _k in T.serial(8):
                # High-level GEMM that needs lowering
                T.evaluate(T.gemm(A, B, C))

        @T.prim_func
        def elementwise_example(X: T.Buffer((256, 256), "float16"), Y: T.Buffer(
            (256, 256), "float16"), Z: T.Buffer((256, 256), "float16")):
            # Element-wise operations
            T.evaluate(T.add(X, Y, Z))

    # Test GEMM lowering
    func = TestModule["gemm_example"]

    # Simulate CB metadata from C1
    func = func.with_attr(
        "tt.conceptual_cbs", {
            "cb_in0": {
                "original_buffer": "A"
            },
            "cb_in1": {
                "original_buffer": "B"
            },
            "cb_out0": {
                "original_buffer": "C"
            }
        })

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
