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
from .codegen_tt import CodeBuffer, use_real_metalium

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
                # Get all tt.* attributes and legacy tt_* attributes
                if key.startswith("tt.") or key.startswith("tt_"):
                    metadata[key] = self.func.attrs[key]
                # Also specifically check for runtime arg names
                if key == "tt.runtime_arg_names":
                    metadata["tt.runtime_arg_names"] = self.func.attrs[key]
        return metadata

    def generate(self) -> str:
        """Generate kernel code by visiting TIR"""
        # Generate includes
        self._generate_includes()

        # Generate MAIN function
        self.code.writeln("void MAIN() {")
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
        # Try to get runtime args from different sources
        runtime_args = self.metadata.get("tt.runtime_args", [])
        if not runtime_args and "tt.runtime_arg_names" in self.metadata:
            runtime_args = self.metadata["tt.runtime_arg_names"]

        arg_info = self.metadata.get("tt.runtime_args_info", {})

        if not runtime_args:
            # Fallback to default runtime args
            runtime_args = ["tt_start_tile", "tt_tile_count", "A_addr", "B_addr", "C_addr"]

        self.code.writeln("// Runtime arguments")
        types = arg_info.get("types", {}) if isinstance(arg_info, dict) else {}

        for i, arg_name in enumerate(runtime_args):
            # Convert to string if needed
            arg_name_str = str(arg_name) if not isinstance(arg_name, str) else arg_name

            # Determine C++ type from metadata or name
            if arg_name_str in types:
                tir_type = types[arg_name_str]
                if "int64" in tir_type or "uint64" in tir_type or "addr" in arg_name_str:
                    cpp_type = "uint64_t"
                else:
                    cpp_type = "uint32_t"
            elif arg_name_str.endswith("_addr"):
                cpp_type = "uint64_t"
            else:
                cpp_type = "uint32_t"

            # For reader/writer kernels with shard coordinates, mark them as unused
            if self.role in ["reader", "writer"] and "tt_shard_coord" in arg_name_str:
                self.code.writeln(
                    f"const {cpp_type} {arg_name_str} = get_arg_val<{cpp_type}>({i});")
                self.code.writeln(f"(void){arg_name_str}; // Mark unused in {self.role} kernel")
            else:
                self.code.writeln(
                    f"const {cpp_type} {arg_name_str} = get_arg_val<{cpp_type}>({i});")

        self.code.writeln()

    def _generate_cb_constants(self):
        """Generate CB index constants"""
        cb_indices = self.metadata.get("tt.cb_indices", {})

        # If no CB indices in metadata, generate defaults based on role
        if not cb_indices:
            if self.role == "compute":
                self.code.writeln("// Circular buffer indices")
                self.code.writeln("constexpr auto cb_in0 = tt::CBIndex::c_0;")
                self.code.writeln("constexpr auto cb_in1 = tt::CBIndex::c_1;")
                self.code.writeln("constexpr auto cb_out0 = tt::CBIndex::c_16;")
                self.code.writeln()
                # Set default CB indices for visitor
                cb_indices = {"cb_in0": 0, "cb_in1": 1, "cb_out0": 16}
                # Pass CB indices to visitor and return early
                self.visitor.cb_indices = cb_indices
                return
            else:
                return

        self.code.writeln("// Circular buffer indices")
        for cb_name, cb_index in cb_indices.items():
            # Filter based on role
            if self.role == "reader" and "in" not in cb_name:
                continue
            if self.role == "writer" and "out" not in cb_name:
                continue
            # Compute kernel needs all CBs

            # Generate with proper format
            self.code.writeln(f"constexpr auto {cb_name} = tt::CBIndex::c_{cb_index};")

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
        self.code.writeln("// Generated TT Reader Kernel (IR-Driven)")
        if use_real_metalium():
            # Real SDK uses dataflow_api.h
            self.code.writeln('#include "dataflow_api.h"')
        else:
            # Mock mode uses compute_kernel_api headers
            self.code.writeln('#include "compute_kernel_api/common.h"')
            self.code.writeln('#include "compute_kernel_api/tile_move_copy.h"')
        self.code.writeln()

    def generate(self) -> str:
        """Generate reader kernel with proper structure"""
        # Generate includes
        self._generate_includes()

        # Use kernel_main for reader/writer kernels (Metalium convention)
        self.code.writeln("void kernel_main() {")
        self.code.indent()

        # Generate runtime args extraction
        self._generate_runtime_args()

        # Generate CB constants
        self._generate_cb_constants_reader()

        # Generate reader body with CB/NOC operations
        self._generate_reader_body()

        self.code.dedent()
        self.code.writeln("}")

        return self.code.get_code()

    def _generate_cb_constants_reader(self):
        """Generate CB constants for reader"""
        self.code.writeln("// Circular buffer indices")
        self.code.writeln("constexpr auto cb_in0 = tt::CBIndex::c_0;")
        self.code.writeln("constexpr auto cb_in1 = tt::CBIndex::c_1;")
        self.code.writeln()

    def _generate_reader_body(self):
        """Generate reader body with CB/NOC operations"""
        # Check if we have actual TIR body with operations
        if self.func.body and not self._is_empty_body(self.func.body):
            # Visit the actual TIR body
            self.visitor.visit(self.func.body)
        else:
            # Fail loudly - no template fallback allowed
            raise ValueError(
                "Reader kernel has empty or incomplete IR body. "
                "The IR must contain proper NOC/CB operations from the lowering passes. "
                "Check that passes C1-C2 and D1-D5 have run correctly.")

    def _is_empty_body(self, body):
        """Check if body is empty or just Evaluate(0)"""
        if isinstance(body, tir.Evaluate):
            return str(body.value) == "0"
        return False


class EnhancedComputeKernelGenerator(EnhancedKernelGenerator):
    """Enhanced generator for compute kernels"""

    def _generate_includes(self):
        """Generate includes for compute kernel"""
        self.code.writeln("// Generated TT Compute Kernel (IR-Driven)")

        # Add grid metadata comments if available
        if self.func.attrs:
            if "tt_grid_x" in self.func.attrs and "tt_grid_y" in self.func.attrs:
                grid_x = self.func.attrs["tt_grid_x"]
                grid_y = self.func.attrs["tt_grid_y"]
                grid_x_val = int(grid_x) if hasattr(grid_x, "__int__") else grid_x
                grid_y_val = int(grid_y) if hasattr(grid_y, "__int__") else grid_y
                self.code.writeln(f"// Grid: {grid_x_val}x{grid_y_val}")
            if "tt_num_cores" in self.func.attrs:
                num_cores = self.func.attrs["tt_num_cores"]
                num_cores_val = int(num_cores) if hasattr(num_cores, "__int__") else num_cores
                self.code.writeln(f"// Cores: {num_cores_val}")

        if use_real_metalium():
            # Real SDK requires ckernel_include.h first
            self.code.writeln('#include "ckernel_include.h"')
            self.code.writeln('#include "compute_kernel_api/common.h"')

            # Analyze function to determine needed includes
            includes_needed = self._analyze_compute_ops()

            if "matmul" in includes_needed:
                self.code.writeln('#include "compute_kernel_api/matmul.h"')
            if "binary" in includes_needed:
                self.code.writeln('#include "compute_kernel_api/eltwise_binary.h"')
            if "unary" in includes_needed:
                self.code.writeln('#include "compute_kernel_api/eltwise_unary.h"')
        else:
            # Mock mode
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
                elif isinstance(stmt, (tir.Allocate, tir.AttrStmt, tir.AssertStmt)):
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
        self.code.writeln("// Generated TT Writer Kernel (IR-Driven)")
        if use_real_metalium():
            # Real SDK uses dataflow_api.h
            self.code.writeln('#include "dataflow_api.h"')
        else:
            # Mock mode uses compute_kernel_api headers
            self.code.writeln('#include "compute_kernel_api/common.h"')
            self.code.writeln('#include "compute_kernel_api/tile_move_copy.h"')
        self.code.writeln()

    def generate(self) -> str:
        """Generate writer kernel with proper structure"""
        # Generate includes
        self._generate_includes()

        # Use kernel_main for reader/writer kernels (Metalium convention)
        self.code.writeln("void kernel_main() {")
        self.code.indent()

        # Generate runtime args extraction
        self._generate_runtime_args()

        # Generate CB constants
        self._generate_cb_constants_writer()

        # Generate writer body with CB/NOC operations
        self._generate_writer_body()

        self.code.dedent()
        self.code.writeln("}")

        return self.code.get_code()

    def _generate_cb_constants_writer(self):
        """Generate CB constants for writer"""
        self.code.writeln("// Circular buffer indices")
        self.code.writeln("constexpr auto cb_out0 = tt::CBIndex::c_16;")
        self.code.writeln()

    def _generate_writer_body(self):
        """Generate writer body with CB/NOC operations"""
        # Check if we have actual TIR body with operations
        if self.func.body and not self._is_empty_body(self.func.body):
            # Visit the actual TIR body
            self.visitor.visit(self.func.body)
        else:
            # Fail loudly - no template fallback allowed
            raise ValueError(
                "Writer kernel has empty or incomplete IR body. "
                "The IR must contain proper NOC/CB operations from the lowering passes. "
                "Check that passes C1-C2 and D1-D5 have run correctly.")

    def _is_empty_body(self, body):
        """Check if body is empty or just Evaluate(0)"""
        if isinstance(body, tir.Evaluate):
            return str(body.value) == "0"
        return False


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
