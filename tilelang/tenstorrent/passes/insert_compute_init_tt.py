"""
Pass D4: InsertComputeInitTT (v5 Specification)
Version: 5.0
Date: 2025-10-15

Purpose: Insert compute engine initialization protocol in compute kernels.
         Adds Unpack/Math/Pack init calls before compute operations.

Input: Compute kernel with protocol-less compute ops
Output: Compute kernel with engine initialization
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Set, Tuple
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


class ComputeOpType(Enum):
    """Types of compute operations"""
    MATMUL = "matmul"
    BINARY_ADD = "binary_add"
    BINARY_MUL = "binary_mul"
    BINARY_SUB = "binary_sub"
    UNARY = "unary"
    SFPU = "sfpu"
    UNKNOWN = "unknown"


class ComputeOpAnalyzer(tir.stmt_functor.StmtVisitor):
    """Visitor to analyze compute operations in the kernel"""

    def __init__(self):
        super().__init__()
        self.compute_ops = []
        self.cb_names = set()
        self.has_k_loop = False
        self.loop_depth = 0

    def visit_evaluate(self, op):
        """Visit T.evaluate nodes to find compute operations"""

        if hasattr(op, 'value') and hasattr(op.value, 'op'):
            call = op.value
            op_name = str(call.op) if hasattr(call.op, 'name') else str(call.op)

            # Identify compute operations
            op_type = self._classify_compute_op(op_name)
            if op_type != ComputeOpType.UNKNOWN:
                self.compute_ops.append({
                    "type": op_type,
                    "op_name": op_name,
                    "args": call.args,
                    "loop_depth": self.loop_depth
                })

                # Extract CB names from args
                for arg in call.args:
                    cb_name = self._extract_cb_name(arg)
                    if cb_name:
                        self.cb_names.add(cb_name)

        super().visit_evaluate(op)

    def visit_for(self, op):
        """Track loop nesting and K-loops"""

        self.loop_depth += 1

        # Check if this is a K-loop (for GEMM)
        if hasattr(op, 'loop_var') and hasattr(op.loop_var, 'name'):
            var_name = op.loop_var.name.lower()
            if 'k' in var_name or 'reduction' in var_name:
                self.has_k_loop = True

        super().visit_for(op)
        self.loop_depth -= 1

    def _classify_compute_op(self, op_name: str) -> ComputeOpType:
        """Classify compute operation type"""

        op_lower = op_name.lower()

        if any(x in op_lower for x in ["mm.mma", "matmul", "gemm"]):
            return ComputeOpType.MATMUL
        elif "fpu.add" in op_lower or "add_tiles" in op_lower:
            return ComputeOpType.BINARY_ADD
        elif "fpu.mul" in op_lower or "mul_tiles" in op_lower:
            return ComputeOpType.BINARY_MUL
        elif "fpu.sub" in op_lower or "sub_tiles" in op_lower:
            return ComputeOpType.BINARY_SUB
        elif "sfpu.unary" in op_lower:
            return ComputeOpType.UNARY
        elif "sfpu" in op_lower:
            return ComputeOpType.SFPU
        elif any(x in op_lower for x in ["compute", "tile", "fpu", "math"]):
            return ComputeOpType.UNKNOWN  # Might be compute but unclear

        return ComputeOpType.UNKNOWN

    def _extract_cb_name(self, arg) -> Optional[str]:
        """Extract CB name from argument"""

        if isinstance(arg, str) and "cb" in arg.lower():
            return arg
        elif hasattr(arg, 'value'):
            val_str = str(arg.value)
            if "cb" in val_str.lower():
                return val_str
        return None


class ComputeInitInserter(tir.stmt_functor.StmtMutator):
    """Mutator to insert compute engine initialization"""

    def __init__(self, init_info: Dict[str, Any]):
        super().__init__()
        self.init_info = init_info
        self.init_inserted = False
        self.at_function_start = True

    def visit_seq_stmt(self, op):
        """Visit sequence statements to insert init at the beginning"""

        if self.at_function_start and not self.init_inserted:
            # Insert initialization at the start
            init_stmts = self._create_init_statements()

            # Continue with rest of the body
            self.at_function_start = False
            rest = super().visit_seq_stmt(op)

            # Combine init with rest
            if init_stmts:
                self.init_inserted = True
                return tir.SeqStmt(init_stmts + [rest])

        return super().visit_seq_stmt(op)

    def visit_evaluate(self, op):
        """Visit first evaluate to potentially insert init before it"""

        if self.at_function_start and not self.init_inserted:
            # Insert init before first statement
            init_stmts = self._create_init_statements()

            if init_stmts:
                self.init_inserted = True
                self.at_function_start = False
                # Return sequence of init followed by original statement
                return tir.SeqStmt(init_stmts + [op])

        self.at_function_start = False
        return super().visit_evaluate(op)

    def visit_for(self, op):
        """Visit for loop - insert init before if at start"""

        if self.at_function_start and not self.init_inserted:
            # Insert init before the loop
            init_stmts = self._create_init_statements()

            if init_stmts:
                self.init_inserted = True
                self.at_function_start = False
                # Return sequence of init followed by loop
                return tir.SeqStmt(init_stmts + [super().visit_for(op)])

        self.at_function_start = False
        return super().visit_for(op)

    def _create_init_statements(self) -> List[tir.Stmt]:
        """Create initialization statements based on compute type"""

        init_stmts = []

        # 1. Common init (unpack/math/pack)
        init_stmts.append(self._create_common_init())

        # 2. Operation-specific init
        op_type = self.init_info.get("primary_op_type")

        if op_type == ComputeOpType.MATMUL:
            init_stmts.append(self._create_matmul_init())
        elif op_type in [ComputeOpType.BINARY_ADD, ComputeOpType.BINARY_MUL, ComputeOpType.BINARY_SUB]:
            init_stmts.append(self._create_binary_init(op_type))
        elif op_type in [ComputeOpType.UNARY, ComputeOpType.SFPU]:
            init_stmts.append(self._create_sfpu_init(op_type))

        # 3. Additional configuration if needed
        if self.init_info.get("needs_transpose_init"):
            init_stmts.append(self._create_transpose_init())

        return init_stmts

    def _create_common_init(self) -> tir.Stmt:
        """Create common engine initialization"""

        cb_in0 = self.init_info.get("cb_in0", 0)
        cb_in1 = self.init_info.get("cb_in1", 1)
        cb_out = self.init_info.get("cb_out", 16)

        return tir.Evaluate(
            tir.call_extern("void", "tt.engine.init_common", cb_in0, cb_in1, cb_out)
        )

    def _create_matmul_init(self) -> tir.Stmt:
        """Create matmul-specific initialization"""

        cb_in0 = self.init_info.get("cb_in0", 0)
        cb_in1 = self.init_info.get("cb_in1", 1)
        cb_out = self.init_info.get("cb_out", 16)

        return tir.Evaluate(
            tir.call_extern("void", "tt.fpu.matmul_init", cb_in0, cb_in1, cb_out)
        )

    def _create_binary_init(self, op_type: ComputeOpType) -> tir.Stmt:
        """Create binary operation initialization"""

        cb_in0 = self.init_info.get("cb_in0", 0)
        cb_in1 = self.init_info.get("cb_in1", 1)
        cb_out = self.init_info.get("cb_out", 16)

        # Map operation type to string
        op_str_map = {
            ComputeOpType.BINARY_ADD: "add",
            ComputeOpType.BINARY_MUL: "mul",
            ComputeOpType.BINARY_SUB: "sub"
        }
        op_str = op_str_map.get(op_type, "add")

        return tir.Evaluate(
            tir.call_extern("void", "tt.fpu.binary_init", cb_in0, cb_in1, cb_out, op_str)
        )

    def _create_sfpu_init(self, op_type: ComputeOpType) -> tir.Stmt:
        """Create SFPU initialization"""

        cb_in = self.init_info.get("cb_in0", 0)
        cb_out = self.init_info.get("cb_out", 16)

        # Determine SFPU operation
        sfpu_op = self.init_info.get("sfpu_op", "identity")

        return tir.Evaluate(
            tir.call_extern("void", "tt.sfpu.init", sfpu_op, cb_in, cb_out)
        )

    def _create_transpose_init(self) -> tir.Stmt:
        """Create transpose initialization if needed"""

        return tir.Evaluate(
            tir.call_extern("void", "tt.pack.transpose_init")
        )


class InsertComputeInitTT:
    """
    Pass to insert compute engine initialization in compute kernels.

    This pass:
    1. Analyzes compute operations to determine initialization needs
    2. Inserts common init (unpack/math/pack)
    3. Adds operation-specific init (matmul_init, binary_init, etc.)
    4. Places initialization before compute loops
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
        """Process a single function to insert compute initialization."""

        # Get kernel role
        kernel_role = None
        if func.attrs and "tt.kernel_role" in func.attrs:
            kernel_role = func.attrs["tt.kernel_role"]

        # Only process compute kernels
        if kernel_role != "compute":
            logger.debug(f"Skipping {kernel_role} kernel for compute init")
            return func

        # Analyze compute operations
        analyzer = ComputeOpAnalyzer()
        analyzer.visit(func.body)

        if not analyzer.compute_ops:
            logger.warning("No compute operations found in compute kernel")
            return func

        # Determine initialization requirements
        init_info = self._determine_init_requirements(analyzer)

        # Insert initialization
        inserter = ComputeInitInserter(init_info)
        new_body = inserter.visit(func.body)

        # Create new function with initialization
        new_func = tir.PrimFunc(
            params=func.params,
            body=new_body,
            ret_type=func.ret_type,
            buffer_map=func.buffer_map,
            attrs=func.attrs
        )

        # Mark that initialization has been inserted
        new_func = new_func.with_attr("tt.compute_init_inserted", True)
        new_func = new_func.with_attr("tt.compute_init_info", tvm.runtime.convert(init_info))

        logger.info(f"Inserted compute initialization for {init_info['primary_op_type'].value} operation")

        return new_func

    def _determine_init_requirements(self, analyzer: ComputeOpAnalyzer) -> Dict[str, Any]:
        """Determine what initialization is required based on compute ops."""

        init_info = {
            "primary_op_type": ComputeOpType.UNKNOWN,
            "cb_in0": 0,
            "cb_in1": 1,
            "cb_out": 16,
            "has_k_loop": analyzer.has_k_loop,
            "needs_transpose_init": False,
            "sfpu_op": None
        }

        # Find primary operation type
        if analyzer.compute_ops:
            # Use first compute op as primary (could be more sophisticated)
            primary_op = analyzer.compute_ops[0]
            init_info["primary_op_type"] = primary_op["type"]

            # Extract CB indices from operation
            if primary_op["args"]:
                for i, arg in enumerate(primary_op["args"]):
                    cb_name = self._extract_cb_name(arg)
                    if cb_name:
                        if i == 0 and "in" in cb_name:
                            init_info["cb_in0"] = self._get_cb_index(cb_name)
                        elif i == 1 and "in" in cb_name:
                            init_info["cb_in1"] = self._get_cb_index(cb_name)
                        elif "out" in cb_name:
                            init_info["cb_out"] = self._get_cb_index(cb_name)

        # Check for special requirements
        for op in analyzer.compute_ops:
            if "transpose" in op["op_name"].lower():
                init_info["needs_transpose_init"] = True

            # Extract SFPU operation if present
            if op["type"] in [ComputeOpType.UNARY, ComputeOpType.SFPU]:
                # Try to extract operation name
                if "exp" in op["op_name"].lower():
                    init_info["sfpu_op"] = "exp"
                elif "log" in op["op_name"].lower():
                    init_info["sfpu_op"] = "log"
                elif "sqrt" in op["op_name"].lower():
                    init_info["sfpu_op"] = "sqrt"
                elif "recip" in op["op_name"].lower():
                    init_info["sfpu_op"] = "recip"
                else:
                    init_info["sfpu_op"] = "identity"

        return init_info

    def _extract_cb_name(self, arg) -> Optional[str]:
        """Extract CB name from argument"""

        if isinstance(arg, str):
            return arg
        elif hasattr(arg, 'value'):
            return str(arg.value) if arg.value else None
        else:
            arg_str = str(arg)
            if "cb" in arg_str.lower():
                # Try to extract CB name
                if "cb_in0" in arg_str:
                    return "cb_in0"
                elif "cb_in1" in arg_str:
                    return "cb_in1"
                elif "cb_out" in arg_str:
                    return "cb_out"
            return None

    def _get_cb_index(self, cb_name: str) -> int:
        """Get CB index from name (simple heuristic)"""

        if "in0" in cb_name:
            return 0
        elif "in1" in cb_name:
            return 1
        elif "in2" in cb_name:
            return 2
        elif "out" in cb_name:
            return 16
        else:
            # Default indices
            if "in" in cb_name:
                return 0
            else:
                return 16


# Module-level pass function for compatibility
def insert_compute_init_tt(mod: IRModule) -> IRModule:
    """Apply InsertComputeInitTT pass to a module."""
    pass_instance = InsertComputeInitTT()
    return pass_instance(mod)


# Example usage and testing
if __name__ == "__main__":
    import tvm.script
    from tvm.script import tir as T

    # Create test module with compute kernel
    @tvm.script.ir_module
    class TestModule:
        @T.prim_func
        def gemm_compute():
            # Simulate compute kernel with matmul
            for kt in T.serial(8):  # K-loop
                T.evaluate(T.call_extern("void", "tt.mm.mma", "cb_in0", "cb_in1", 0, kt > 0))

        @T.prim_func
        def add_compute():
            # Simulate element-wise add
            T.evaluate(T.call_extern("void", "tt.fpu.add", "cb_in0", "cb_in1", 0))

    # Add metadata for GEMM compute
    gemm_func = TestModule["gemm_compute"]
    gemm_func = gemm_func.with_attr("tt.kernel_role", "compute")
    gemm_func = gemm_func.with_attr("tt.cb_indices", {"cb_in0": 0, "cb_in1": 1, "cb_out": 16})
    TestModule["gemm_compute"] = gemm_func

    # Add metadata for element-wise compute
    add_func = TestModule["add_compute"]
    add_func = add_func.with_attr("tt.kernel_role", "compute")
    add_func = add_func.with_attr("tt.cb_indices", {"cb_in0": 0, "cb_in1": 1, "cb_out": 16})
    TestModule["add_compute"] = add_func

    # Apply D4 pass
    pass_d4 = InsertComputeInitTT()
    result = pass_d4(TestModule)

    # Check results
    print("=== Compute Engine Init Results ===\n")
    for name, func in result.functions_items():
        if func.attrs and "tt.kernel_role" in func.attrs:
            print(f"{name} ({func.attrs['tt.kernel_role']}):")
            if "tt.compute_init_inserted" in func.attrs:
                print(f"  Init inserted: {func.attrs['tt.compute_init_inserted']}")
                if "tt.compute_init_info" in func.attrs:
                    info = func.attrs["tt.compute_init_info"]
                    print(f"  Primary op: {info.get('primary_op_type', 'unknown')}")
                    print(f"  Has K-loop: {info.get('has_k_loop', False)}")
            print(f"  Body preview: {str(func.body)[:200]}...")
            print()