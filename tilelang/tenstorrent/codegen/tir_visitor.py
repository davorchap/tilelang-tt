"""
TIR Visitor for Tenstorrent Code Generation
Version: 5.0
Date: 2025-10-15

Purpose: Traverse TIR and generate C++ code based on actual operations.
"""

from __future__ import annotations
import logging

try:
    import tvm
    from tvm import tir
except ImportError:
    tvm = None
    tir = None

logger = logging.getLogger(__name__)


class TIRToMetaliumVisitor:
    """Visitor that traverses TIR and generates Metalium C++ code"""

    def __init__(self, code_buffer):
        """Initialize the visitor with a code buffer"""
        self.code = code_buffer
        self.in_loop = False
        self.loop_vars = {}
        self.cb_indices = {}
        self.tensor_accessors = {}

    def visit(self, stmt):
        """Main visit dispatcher"""
        if isinstance(stmt, tir.For):
            self.visit_for(stmt)
        elif isinstance(stmt, tir.SeqStmt):
            self.visit_seq(stmt)
        elif isinstance(stmt, tir.Evaluate):
            self.visit_evaluate(stmt)
        elif isinstance(stmt, tir.IfThenElse):
            self.visit_if(stmt)
        elif isinstance(stmt, tir.LetStmt):
            self.visit_let(stmt)
        elif isinstance(stmt, tir.BufferStore):
            self.visit_buffer_store(stmt)
        elif isinstance(stmt, tir.BufferRealize):
            self.visit_buffer_realize(stmt)
        else:
            logger.debug(f"Unknown stmt type: {type(stmt)}")

    def visit_for(self, op):
        """Visit For loop"""
        loop_var = op.loop_var.name if hasattr(op.loop_var, 'name') else "i"
        min_val = self.expr_to_string(op.min)
        extent = self.expr_to_string(op.extent)

        # Generate for loop
        self.code.writeln(
            f"for (uint32_t {loop_var} = {min_val}; {loop_var} < {min_val} + {extent}; {loop_var}++) {{"
        )
        self.code.indent()

        # Track loop variable
        self.loop_vars[loop_var] = op
        self.in_loop = True

        # Visit body
        self.visit(op.body)

        self.in_loop = False
        del self.loop_vars[loop_var]

        self.code.dedent()
        self.code.writeln("}")

    def visit_seq(self, op):
        """Visit sequence of statements"""
        for stmt in op.seq:
            self.visit(stmt)

    def visit_evaluate(self, op):
        """Visit Evaluate node (function calls)"""
        if hasattr(op.value, 'op') and hasattr(op.value.op, 'name'):
            self._handle_intrinsic_call(op.value)
        else:
            # Regular expression evaluation
            expr_str = self.expr_to_string(op.value)
            if expr_str:
                self.code.writeln(f"{expr_str};")

    def visit_if(self, op):
        """Visit if-then-else statement"""
        cond = self.expr_to_string(op.condition)
        self.code.writeln(f"if ({cond}) {{")
        self.code.indent()
        self.visit(op.then_body)
        self.code.dedent()

        if op.else_body:
            self.code.writeln("} else {")
            self.code.indent()
            self.visit(op.else_body)
            self.code.dedent()

        self.code.writeln("}")

    def visit_let(self, op):
        """Visit let statement"""
        var_name = op.var.name if hasattr(op.var, 'name') else "tmp"
        value = self.expr_to_string(op.value)
        dtype = self._get_c_dtype(op.var.dtype if hasattr(op.var, 'dtype') else "int32")

        self.code.writeln(f"{dtype} {var_name} = {value};")
        self.visit(op.body)

    def visit_buffer_store(self, op):
        """Visit buffer store"""
        # This would handle buffer stores if needed
        pass

    def visit_buffer_realize(self, op):
        """Visit buffer realize"""
        # Visit the body
        self.visit(op.body)

    def _handle_intrinsic_call(self, call):
        """Handle TIR intrinsic calls and map to Metalium"""
        func_name = call.op.name
        args = [self.expr_to_string(arg) for arg in call.args]

        # Map intrinsics to Metalium APIs
        if func_name == "cb_reserve_back":
            cb_idx = args[0]
            count = args[1] if len(args) > 1 else "1"
            self.code.writeln(f"cb_reserve_back({cb_idx}, {count});")

        elif func_name == "cb_push_back":
            cb_idx = args[0]
            count = args[1] if len(args) > 1 else "1"
            self.code.writeln(f"cb_push_back({cb_idx}, {count});")

        elif func_name == "cb_wait_front":
            cb_idx = args[0]
            count = args[1] if len(args) > 1 else "1"
            self.code.writeln(f"cb_wait_front({cb_idx}, {count});")

        elif func_name == "cb_pop_front":
            cb_idx = args[0]
            count = args[1] if len(args) > 1 else "1"
            self.code.writeln(f"cb_pop_front({cb_idx}, {count});")

        elif func_name == "get_write_ptr":
            cb_idx = args[0]
            # This is usually assigned to a variable
            return f"get_write_ptr({cb_idx})"

        elif func_name == "get_read_ptr":
            cb_idx = args[0]
            return f"get_read_ptr({cb_idx})"

        elif func_name == "noc_async_read_tile":
            tile_id = args[0]
            accessor = args[1] if len(args) > 1 else "src_accessor"
            addr = args[2] if len(args) > 2 else "l1_write_addr"
            self.code.writeln(f"noc_async_read_tile({tile_id}, {accessor}, {addr});")

        elif func_name == "noc_async_write_tile":
            tile_id = args[0]
            accessor = args[1] if len(args) > 1 else "dst_accessor"
            addr = args[2] if len(args) > 2 else "l1_read_addr"
            self.code.writeln(f"noc_async_write_tile({tile_id}, {accessor}, {addr});")

        elif func_name == "noc_async_read_barrier":
            self.code.writeln("noc_async_read_barrier();")

        elif func_name == "noc_async_write_barrier":
            self.code.writeln("noc_async_write_barrier();")

        elif func_name == "tt.engine.init_common":
            cb_in0 = args[0]
            cb_in1 = args[1] if len(args) > 1 else cb_in0
            cb_out = args[2] if len(args) > 2 else "16"
            self.code.writeln(f"binary_op_init_common({cb_in0}, {cb_in1}, {cb_out});")

        elif func_name == "tt.fpu.matmul_init":
            cb_in0 = args[0]
            cb_in1 = args[1] if len(args) > 1 else cb_in0
            cb_out = args[2] if len(args) > 2 else "16"
            self.code.writeln(f"mm_init({cb_in0}, {cb_in1}, {cb_out});")

        elif func_name == "tt.dst.acquire":
            self.code.writeln("acquire_dst(tt::DstMode::Half);")

        elif func_name == "tt.dst.release":
            self.code.writeln("release_dst(tt::DstMode::Half);")

        elif func_name == "tt.dst.commit":
            # This is often combined with pack_tile
            pass

        elif func_name == "pack_tile":
            dst_idx = args[0] if args else "0"
            cb_idx = args[1] if len(args) > 1 else "cb_out"
            self.code.writeln(f"pack_tile({dst_idx}, {cb_idx});")

        elif func_name == "tt.mm.mma":
            cb_in0 = args[0]
            cb_in1 = args[1]
            dst_idx = args[2] if len(args) > 2 else "0"
            accumulate = args[3] if len(args) > 3 else "false"
            self.code.writeln(f"matmul_tiles({cb_in0}, {cb_in1}, 0, 0, {dst_idx}, {accumulate});")

        elif func_name == "tt.fpu.add":
            cb_in0 = args[0]
            cb_in1 = args[1]
            dst_idx = args[2] if len(args) > 2 else "0"
            self.code.writeln(f"add_tiles({cb_in0}, {cb_in1}, 0, 0, {dst_idx});")

        elif func_name == "tt.fpu.mul":
            cb_in0 = args[0]
            cb_in1 = args[1]
            dst_idx = args[2] if len(args) > 2 else "0"
            self.code.writeln(f"mul_tiles({cb_in0}, {cb_in1}, 0, 0, {dst_idx});")

        elif func_name == "tt.sfpu.exp":
            cb_in = args[0]
            dst_idx = args[1] if len(args) > 1 else "0"
            self.code.writeln(f"exp_tile({cb_in}, 0, {dst_idx});")

        elif func_name == "tt.sfpu.relu":
            cb_in = args[0]
            dst_idx = args[1] if len(args) > 1 else "0"
            self.code.writeln(f"relu_tile({cb_in}, 0, {dst_idx});")

        else:
            # Unknown intrinsic - generate comment
            self.code.writeln(f"// TODO: {func_name}({', '.join(args)});")

    def expr_to_string(self, expr) -> str:
        """Convert TIR expression to C++ string"""
        if expr is None:
            return ""

        if isinstance(expr, (int, float)):
            return str(expr)

        if isinstance(expr, tir.IntImm):
            return str(expr.value)

        if isinstance(expr, tir.FloatImm):
            return str(expr.value)

        if isinstance(expr, tir.StringImm):
            return f'"{expr.value}"'

        if isinstance(expr, tir.Var):
            return expr.name

        if isinstance(expr, tir.Add):
            left = self.expr_to_string(expr.a)
            right = self.expr_to_string(expr.b)
            return f"({left} + {right})"

        if isinstance(expr, tir.Sub):
            left = self.expr_to_string(expr.a)
            right = self.expr_to_string(expr.b)
            return f"({left} - {right})"

        if isinstance(expr, tir.Mul):
            left = self.expr_to_string(expr.a)
            right = self.expr_to_string(expr.b)
            return f"({left} * {right})"

        if isinstance(expr, tir.Div):
            left = self.expr_to_string(expr.a)
            right = self.expr_to_string(expr.b)
            return f"({left} / {right})"

        if isinstance(expr, tir.Mod):
            left = self.expr_to_string(expr.a)
            right = self.expr_to_string(expr.b)
            return f"({left} % {right})"

        if isinstance(expr, tir.LT):
            left = self.expr_to_string(expr.a)
            right = self.expr_to_string(expr.b)
            return f"({left} < {right})"

        if isinstance(expr, tir.GT):
            left = self.expr_to_string(expr.a)
            right = self.expr_to_string(expr.b)
            return f"({left} > {right})"

        if isinstance(expr, tir.EQ):
            left = self.expr_to_string(expr.a)
            right = self.expr_to_string(expr.b)
            return f"({left} == {right})"

        if isinstance(expr, tir.And):
            left = self.expr_to_string(expr.a)
            right = self.expr_to_string(expr.b)
            return f"({left} && {right})"

        if isinstance(expr, tir.Or):
            left = self.expr_to_string(expr.a)
            right = self.expr_to_string(expr.b)
            return f"({left} || {right})"

        if isinstance(expr, tir.Not):
            val = self.expr_to_string(expr.a)
            return f"(!{val})"

        if isinstance(expr, tir.Cast):
            val = self.expr_to_string(expr.value)
            dtype = self._get_c_dtype(expr.dtype)
            return f"({dtype})({val})"

        if isinstance(expr, tir.Call):
            return self._handle_call_expr(expr)

        if isinstance(expr, tir.BufferLoad):
            return self._handle_buffer_load(expr)

        # Default: return placeholder
        return f"/* expr: {type(expr).__name__} */"

    def _handle_call_expr(self, call):
        """Handle call expressions that return values"""
        func_name = call.op.name if hasattr(call.op, 'name') else str(call.op)
        args = [self.expr_to_string(arg) for arg in call.args]

        # Handle specific intrinsics that return values
        if func_name == "get_write_ptr":
            return f"get_write_ptr({args[0]})"
        elif func_name == "get_read_ptr":
            return f"get_read_ptr({args[0]})"
        elif func_name == "get_arg_val":
            # Runtime argument access
            idx = args[0]
            dtype = "uint32_t"  # Default
            return f"get_arg_val<{dtype}>({idx})"
        else:
            return f"{func_name}({', '.join(args)})"

    def _handle_buffer_load(self, load):
        """Handle buffer load expressions"""
        buffer_name = load.buffer.name if hasattr(load.buffer, 'name') else "buffer"
        indices = [self.expr_to_string(idx) for idx in load.indices]
        if indices:
            return f"{buffer_name}[{']['.join(indices)}]"
        return buffer_name

    def _get_c_dtype(self, tir_dtype) -> str:
        """Convert TIR dtype to C++ type"""
        dtype_map = {
            "int8": "int8_t",
            "int16": "int16_t",
            "int32": "int32_t",
            "int64": "int64_t",
            "uint8": "uint8_t",
            "uint16": "uint16_t",
            "uint32": "uint32_t",
            "uint64": "uint64_t",
            "float16": "uint16_t",  # Half stored as uint16
            "bfloat16": "uint16_t",  # BFloat16 stored as uint16
            "float32": "float",
            "float64": "double",
            "bool": "bool"
        }

        dtype_str = str(tir_dtype)
        return dtype_map.get(dtype_str, "uint32_t")
