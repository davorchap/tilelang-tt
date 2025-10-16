"""
Enhanced Kernel Generators using TIR Visitor
Version: 5.0
Date: 2025-10-15

Purpose: Generate kernel code by traversing actual TIR instead of templates.
"""

from __future__ import annotations
from typing import Dict, Any
import logging

from .tir_visitor import TIRToMetaliumVisitor
from .codegen_tt import CodeBuffer

try:
    import tvm
    from tvm import tir
except ImportError:
    tvm = None
    tir = None

logger = logging.getLogger(__name__)


class EnhancedKernelGenerator:
    """Enhanced kernel generator that uses TIR visitor"""

    def __init__(self, func: "tir.PrimFunc", role: str):
        self.func = func
        self.role = role
        self.code = CodeBuffer()
        self.metadata = self._extract_metadata()
        self.visitor = TIRToMetaliumVisitor(self.code)

    def _extract_metadata(self) -> Dict[str, Any]:
        """Extract metadata from function attributes"""
        metadata = {}
        if self.func.attrs:
            for key in self.func.attrs.keys():
                if key.startswith("tt."):
                    metadata[key] = self.func.attrs[key]
        return metadata

    def generate(self) -> str:
        """Generate kernel code by visiting TIR"""
        # Generate includes
        self._generate_includes()

        # Generate MAIN function
        self.code.writeln("void MAIN {")
        self.code.indent()

        # Generate runtime args extraction
        self._generate_runtime_args()

        # Generate CB constants
        self._generate_cb_constants()

        # Visit the TIR body to generate code
        self._visit_body()

        self.code.dedent()
        self.code.writeln("}")

        return self.code.get_code()

    def _generate_includes(self):
        """Generate include directives based on role"""
        raise NotImplementedError("Subclass must implement")

    def _generate_runtime_args(self):
        """Generate runtime argument extraction"""
        runtime_args = self.metadata.get("tt.runtime_args", [])
        arg_info = self.metadata.get("tt.runtime_args_info", {})

        if not runtime_args:
            return

        self.code.writeln("// Runtime arguments")
        types = arg_info.get("types", {}) if isinstance(arg_info, dict) else {}

        for i, arg_name in enumerate(runtime_args):
            # Determine C++ type from metadata or name
            if arg_name in types:
                tir_type = types[arg_name]
                if "int64" in tir_type or "uint64" in tir_type or "addr" in arg_name:
                    cpp_type = "uint64_t"
                else:
                    cpp_type = "uint32_t"
            elif arg_name.endswith("_addr"):
                cpp_type = "uint64_t"
            else:
                cpp_type = "uint32_t"

            self.code.writeln(f"const {cpp_type} {arg_name} = get_arg_val<{cpp_type}>({i});")

        self.code.writeln()

    def _generate_cb_constants(self):
        """Generate CB index constants"""
        cb_indices = self.metadata.get("tt.cb_indices", {})

        if not cb_indices:
            return

        self.code.writeln("// Circular buffer indices")
        for cb_name, cb_index in cb_indices.items():
            # Filter based on role
            if self.role == "reader" and "in" not in cb_name:
                continue
            if self.role == "writer" and "out" not in cb_name:
                continue
            # Compute kernel needs all CBs

            self.code.writeln(f"constexpr uint32_t {cb_name} = {cb_index};")

        self.code.writeln()

        # Pass CB indices to visitor
        self.visitor.cb_indices = cb_indices

    def _visit_body(self):
        """Visit the function body to generate code"""
        # Pass metadata to visitor
        self.visitor.tensor_accessors = self.metadata.get("tt.tensor_accessors", {})

        # Check for persistent loop configuration
        persistent_config = self.metadata.get("tt.persistent_config", {})
        if persistent_config and self.role in ["reader", "writer"]:
            # Generate persistent loop wrapper
            loop_pattern = persistent_config.get("pattern", "")
            if loop_pattern:
                self.code.writeln("// Persistent loop")
                # The loop should be in the TIR body already after D passes
                # Just visit the body
                self.visitor.visit(self.func.body)
            else:
                # No persistent pattern, visit directly
                self.visitor.visit(self.func.body)
        else:
            # Compute kernel or no persistent config
            self.visitor.visit(self.func.body)


class EnhancedReaderKernelGenerator(EnhancedKernelGenerator):
    """Enhanced generator for reader kernels"""

    def _generate_includes(self):
        """Generate includes for reader kernel"""
        self.code.writeln('#include "compute_kernel_api/common.h"')
        self.code.writeln('#include "compute_kernel_api/tile_move_copy.h"')
        self.code.writeln()


class EnhancedComputeKernelGenerator(EnhancedKernelGenerator):
    """Enhanced generator for compute kernels"""

    def _generate_includes(self):
        """Generate includes for compute kernel"""
        self.code.writeln('#include "compute_kernel_api/common.h"')

        # Analyze function to determine needed includes
        includes_needed = self._analyze_compute_ops()

        if "matmul" in includes_needed:
            self.code.writeln('#include "compute_kernel_api/matmul.h"')
        if "binary" in includes_needed:
            self.code.writeln('#include "compute_kernel_api/eltwise_binary.h"')
        if "unary" in includes_needed:
            self.code.writeln('#include "compute_kernel_api/eltwise_unary.h"')

        self.code.writeln()

    def _analyze_compute_ops(self) -> set:
        """Analyze compute operations to determine includes"""

        class OpAnalyzer:

            def __init__(self):
                self.ops_found = set()

            def visit(self, stmt):
                """Recursive IR traversal to find compute ops"""
                if stmt is None:
                    return

                if isinstance(stmt, tir.Evaluate):
                    self.visit_evaluate(stmt)
                elif isinstance(stmt, tir.For):
                    self.visit(stmt.body)
                elif isinstance(stmt, tir.SeqStmt):
                    for s in stmt.seq:
                        self.visit(s)
                elif isinstance(stmt, tir.LetStmt):
                    self.visit(stmt.body)
                elif isinstance(stmt, tir.IfThenElse):
                    self.visit(stmt.then_case)
                    if stmt.else_case:
                        self.visit(stmt.else_case)
                elif isinstance(stmt, tir.Allocate):
                    self.visit(stmt.body)
                elif isinstance(stmt, tir.AttrStmt):
                    self.visit(stmt.body)
                elif isinstance(stmt, tir.AssertStmt):
                    self.visit(stmt.body)

            def visit_evaluate(self, op):
                if hasattr(op.value, 'op') and hasattr(op.value.op, 'name'):
                    op_name = op.value.op.name
                    if "mm" in op_name or "matmul" in op_name:
                        self.ops_found.add("matmul")
                    elif "fpu" in op_name and any(
                            x in op_name for x in ["add", "mul", "sub", "div"]):
                        self.ops_found.add("binary")
                    elif "sfpu" in op_name:
                        self.ops_found.add("unary")

        analyzer = OpAnalyzer()
        analyzer.visit(self.func.body)
        return analyzer.ops_found


class EnhancedWriterKernelGenerator(EnhancedKernelGenerator):
    """Enhanced generator for writer kernels"""

    def _generate_includes(self):
        """Generate includes for writer kernel"""
        self.code.writeln('#include "compute_kernel_api/common.h"')
        self.code.writeln('#include "compute_kernel_api/tile_move_copy.h"')
        self.code.writeln()


def create_kernel_generator(func: "tir.PrimFunc", role: str) -> EnhancedKernelGenerator:
    """Factory function to create appropriate kernel generator"""
    if role == "reader":
        return EnhancedReaderKernelGenerator(func, role)
    elif role == "compute":
        return EnhancedComputeKernelGenerator(func, role)
    elif role == "writer":
        return EnhancedWriterKernelGenerator(func, role)
    else:
        # Default to base generator
        return EnhancedKernelGenerator(func, role)
