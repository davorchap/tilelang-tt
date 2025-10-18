"""
Pass E1: FinalizePersistentSignatureTT (v5 Specification)
Version: 5.0
Date: 2025-10-15

Purpose: Finalize runtime arguments and prepare signature for codegen.
         Adds persistent loop parameters and completes kernel signatures.

Input: Split kernels with partial runtime args
Output: Kernels with complete tt.runtime_args for persistent execution
"""

from __future__ import annotations
from typing import Dict, Any, List
import logging

try:
    import tvm
    from tvm import tir, IRModule
except ImportError:
    tvm = None
    tir = None
    IRModule = object

logger = logging.getLogger(__name__)


class RuntimeArgBuilder:
    """Helper to build complete runtime argument lists"""

    def __init__(self, kernel_role: str, partition_mode: str):
        self.kernel_role = kernel_role
        self.partition_mode = partition_mode
        self.args = []
        self.arg_types = {}
        self.arg_descriptions = {}

    def add_persistent_args(self):
        """Add persistent loop arguments"""
        self.add_arg("start_id", "int32", "Starting tile ID for this core")
        self.add_arg("count", "int32", "Number of tiles to process")

    def add_dimension_args(self, has_gemm: bool = False):
        """Add tile dimension arguments"""
        self.add_arg("Mt", "int32", "M dimension in tiles")
        self.add_arg("Nt", "int32", "N dimension in tiles")
        if has_gemm:
            self.add_arg("Kt", "int32", "K dimension in tiles (reduction)")

    def add_shard_args(self):
        """Add sharding arguments for local_shard mode"""
        if self.partition_mode == "local_shard":
            self.add_arg("Sm", "int32", "Shard M tiles")
            self.add_arg("Sn", "int32", "Shard N tiles")
            self.add_arg("Gy", "int32", "Grid Y dimension")
            self.add_arg("Gx", "int32", "Grid X dimension")
            self.add_arg("sy", "int32", "Shard Y coordinate")
            self.add_arg("sx", "int32", "Shard X coordinate")

    def add_buffer_args(self, buffers: List[str]):
        """Add buffer address arguments"""
        for buffer_name in buffers:
            self.add_arg(f"{buffer_name}_addr", "uint64", f"Address of {buffer_name} buffer")

    def add_kernel_specific_args(self):
        """Add kernel role-specific arguments"""
        if self.kernel_role == "reader":
            # Reader might need stride info
            self.add_arg("stride_m", "int32", "M-dimension stride in bytes")
            self.add_arg("stride_n", "int32", "N-dimension stride in bytes")
        elif self.kernel_role == "compute":
            # Compute might need accumulation flags
            pass  # Already handled by dimension args
        elif self.kernel_role == "writer":
            # Writer might need output stride
            self.add_arg("out_stride", "int32", "Output stride in bytes")

    def add_arg(self, name: str, dtype: str, description: str):
        """Add an argument if not already present"""
        if name not in self.args:
            self.args.append(name)
            self.arg_types[name] = dtype
            self.arg_descriptions[name] = description

    def get_final_args(self) -> List[str]:
        """Get the final ordered argument list"""
        # Define canonical ordering
        order = [
            # Buffer addresses first
            "A_addr",
            "B_addr",
            "C_addr",
            "input_addr",
            "weight_addr",
            "output_addr",
            # Persistent loop params
            "start_id",
            "count",
            # Dimensions
            "Mt",
            "Nt",
            "Kt",
            # Strides
            "stride_m",
            "stride_n",
            "out_stride",
            # Shard params
            "Sm",
            "Sn",
            "Gy",
            "Gx",
            "sy",
            "sx"
        ]

        # Order existing args according to canonical order
        ordered_args = []
        for arg in order:
            if arg in self.args:
                ordered_args.append(arg)

        # Add any remaining args not in canonical order
        for arg in self.args:
            if arg not in ordered_args:
                ordered_args.append(arg)

        return ordered_args


class FinalizePersistentSignatureTT:
    """
    Pass to finalize runtime arguments for persistent kernel execution.

    This pass:
    1. Completes runtime argument lists for each kernel
    2. Adds persistent loop parameters (start_id, count)
    3. Adds dimension parameters (Mt, Nt, Kt)
    4. Adds shard parameters for local_shard mode
    5. Orders arguments consistently
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
        """Process a single function to finalize runtime arguments."""

        # Get kernel role
        kernel_role = None
        if func.attrs and "tt.kernel_role" in func.attrs:
            kernel_role = func.attrs["tt.kernel_role"]

        if not kernel_role or kernel_role == "monolithic":
            logger.debug("Skipping non-split kernel")
            return func

        # Get partition mode
        partition_mode = "global"  # default
        if func.attrs and "tt.partition_mode" in func.attrs:
            partition_mode = func.attrs["tt.partition_mode"]

        # Get existing runtime args
        existing_args = []
        if func.attrs and "tt.runtime_args" in func.attrs:
            existing_args = self._convert_to_list(func.attrs["tt.runtime_args"])

        # Analyze function to determine requirements
        analysis = self._analyze_function(func)

        # Build complete runtime arguments
        builder = RuntimeArgBuilder(kernel_role, partition_mode)

        # Add arguments based on role and analysis
        if kernel_role == "reader":
            # Add buffer arguments for inputs
            input_buffers = analysis["input_buffers"]
            builder.add_buffer_args(input_buffers)

            # Add persistent and dimension args
            builder.add_persistent_args()
            builder.add_dimension_args(has_gemm=analysis["has_gemm"])

            # Add kernel-specific args
            builder.add_kernel_specific_args()

        elif kernel_role == "compute":
            # Compute doesn't usually need buffer addresses
            # Add dimension args for iteration
            if analysis["has_gemm"]:
                builder.add_arg("Kt", "int32", "K tiles for reduction")

            # Add persistent args for potential direct invocation
            builder.add_persistent_args()
            builder.add_dimension_args(has_gemm=analysis["has_gemm"])

        elif kernel_role == "writer":
            # Add buffer arguments for outputs
            output_buffers = analysis["output_buffers"]
            builder.add_buffer_args(output_buffers)

            # Add persistent and dimension args
            builder.add_persistent_args()
            builder.add_dimension_args(has_gemm=False)  # Writer doesn't need Kt

            # Add kernel-specific args
            builder.add_kernel_specific_args()

        # Add shard arguments if in local_shard mode
        builder.add_shard_args()

        # Merge with existing args (preserve any custom args)
        for arg in existing_args:
            if arg not in builder.args:
                builder.add_arg(arg, "int32", "Custom argument")

        # Get final ordered arguments
        final_args = builder.get_final_args()

        # Update function with finalized arguments
        from tilelang.tenstorrent.attrs import TT_RUNTIME_ARGS, TT_RUNTIME_ARGS_FINALIZED
        new_func = func.with_attr(TT_RUNTIME_ARGS, final_args)
        new_func = new_func.with_attr(TT_RUNTIME_ARGS_FINALIZED, True)

        # Add argument type information
        arg_info = {
            "types": builder.arg_types,
            "descriptions": builder.arg_descriptions,
            "count": len(final_args)
        }
        from tilelang.tenstorrent.attrs import TT_RUNTIME_ARGS_INFO, TT_PERSISTENT_CONFIG
        new_func = new_func.with_attr(TT_RUNTIME_ARGS_INFO, tvm.runtime.convert(arg_info))

        # Add persistent loop configuration
        persistent_config = {
            "loop_var": "tile_id",
            "start_expr": "start_id",
            "count_expr": "count",
            "pattern": "for (int tile_id = start_id; tile_id < start_id + count; tile_id++)"
        }
        new_func = new_func.with_attr(TT_PERSISTENT_CONFIG, tvm.runtime.convert(persistent_config))

        logger.info(f"Finalized {len(final_args)} runtime args for {kernel_role} kernel")

        return new_func

    def _analyze_function(self, func: "tir.PrimFunc") -> Dict[str, Any]:
        """Analyze function to determine runtime arg requirements."""

        analysis = {
            "has_gemm": False,
            "has_reduction": False,
            "input_buffers": [],
            "output_buffers": [],
            "cb_names": set(),
            "compute_ops": []
        }

        # Check for GEMM/reduction patterns
        class Analyzer:

            def __init__(self, analysis):
                self.analysis = analysis

            def visit_for(self, op):
                if hasattr(op.loop_var, 'name'):
                    var_name = op.loop_var.name.lower()
                    if 'k' in var_name or 'reduction' in var_name:
                        self.analysis["has_reduction"] = True

            def visit_evaluate(self, op):
                if hasattr(op, 'value') and hasattr(op.value, 'op'):
                    op_name = str(op.value.op)
                    if any(x in op_name for x in ["mm.mma", "matmul", "gemm"]):
                        self.analysis["has_gemm"] = True
                    if any(x in op_name for x in ["compute", "fpu", "sfpu"]):
                        self.analysis["compute_ops"].append(op_name)

            def visit(self, stmt):
                """Recursive IR traversal (analysis only)"""
                if tir is None or stmt is None:
                    return

                # Handle different statement types
                if isinstance(stmt, tir.Evaluate):
                    self.visit_evaluate(stmt)
                elif isinstance(stmt, tir.For):
                    self.visit_for(stmt)
                    self.visit(stmt.body)  # Visit loop body
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

        analyzer = Analyzer(analysis)
        analyzer.visit(func.body)

        # Get buffer names from function parameters
        for param in func.params:
            if param in func.buffer_map:
                buffer = func.buffer_map[param]
                buffer_name = buffer.name

                # Classify as input or output
                if any(x in buffer_name.lower() for x in ["a", "b", "input", "weight"]):
                    analysis["input_buffers"].append(buffer_name)
                elif any(x in buffer_name.lower() for x in ["c", "output", "result"]):
                    analysis["output_buffers"].append(buffer_name)

        # Check for CB usage
        if func.attrs and "tt.cb_indices" in func.attrs:
            cb_indices = self._convert_to_dict(func.attrs["tt.cb_indices"])
            analysis["cb_names"] = set(cb_indices.keys())

        return analysis

    def _convert_to_list(self, attr_value: Any) -> List[Any]:
        """Convert TVM attribute value to Python list."""

        if isinstance(attr_value, (list, tuple)):
            return list(attr_value)

        if hasattr(attr_value, "__iter__"):
            return [str(x) for x in attr_value]

        return [attr_value]

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
def finalize_persistent_signature_tt(mod: IRModule) -> IRModule:
    """Apply FinalizePersistentSignatureTT pass to a module."""
    pass_instance = FinalizePersistentSignatureTT()
    return pass_instance(mod)


# Example usage and testing
if __name__ == "__main__":
    import tvm.script
    from tvm.script import tir as T

    # Create test module with split kernels
    @tvm.script.ir_module
    class TestModule:

        @T.prim_func
        def gemm_reader(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16")):
            T.evaluate(0)  # Placeholder

        @T.prim_func
        def gemm_compute():
            for kt in T.serial(8):
                T.evaluate(T.call_extern("void", "tt.mm.mma", "cb_in0", "cb_in1", 0, kt > 0))

        @T.prim_func
        def gemm_writer(C: T.Buffer((256, 256), "float16")):
            T.evaluate(0)  # Placeholder

    # Add kernel metadata
    reader_func = TestModule["gemm_reader"]
    reader_func = reader_func.with_attr("tt.kernel_role", "reader")
    reader_func = reader_func.with_attr("tt.partition_mode", "global")
    reader_func = reader_func.with_attr("tt.runtime_args", ["A_addr", "B_addr"])
    TestModule["gemm_reader"] = reader_func

    compute_func = TestModule["gemm_compute"]
    compute_func = compute_func.with_attr("tt.kernel_role", "compute")
    compute_func = compute_func.with_attr("tt.partition_mode", "global")
    TestModule["gemm_compute"] = compute_func

    writer_func = TestModule["gemm_writer"]
    writer_func = writer_func.with_attr("tt.kernel_role", "writer")
    writer_func = writer_func.with_attr("tt.partition_mode", "global")
    writer_func = writer_func.with_attr("tt.runtime_args", ["C_addr"])
    TestModule["gemm_writer"] = writer_func

    # Apply E1 pass
    pass_e1 = FinalizePersistentSignatureTT()
    result = pass_e1(TestModule)

    # Check results
    print("=== Finalized Runtime Arguments ===\n")
    for name, func in result.functions_items():
        if func.attrs and "tt.kernel_role" in func.attrs:
            print(f"{name} ({func.attrs['tt.kernel_role']}):")
            if "tt.runtime_args" in func.attrs:
                args = func.attrs["tt.runtime_args"]
                print(f"  Arguments ({len(args)}): {args}")
            if "tt.runtime_args_finalized" in func.attrs:
                print(f"  Finalized: {func.attrs['tt.runtime_args_finalized']}")
            if "tt.persistent_config" in func.attrs:
                config = func.attrs["tt.persistent_config"]
                print(f"  Loop pattern: {config.get('pattern', 'N/A')}")
            print()
