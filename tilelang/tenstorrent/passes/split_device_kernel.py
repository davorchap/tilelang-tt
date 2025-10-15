"""
Pass D1: SplitDeviceKernel (v5 Specification)
Version: 5.0
Date: 2025-10-15

Purpose: Split monolithic kernel into reader/compute/writer kernels.
         This is the critical transformation to the 3-kernel architecture.

Input: Protocol-less IR with tt.tile_dfg from C3
Output: Three separate PrimFuncs with assigned roles
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Set
import logging
from dataclasses import dataclass
from enum import Enum

try:
    import tvm
    from tvm import tir, IRModule
except ImportError:
    tvm = None
    tir = None
    IRModule = object

logger = logging.getLogger(__name__)


class KernelRole(Enum):
    """Kernel roles in the 3-kernel architecture"""
    READER = "reader"
    COMPUTE = "compute"
    WRITER = "writer"
    MONOLITHIC = "monolithic"  # Before split


@dataclass
class KernelSlice:
    """Represents a slice of the original kernel for a specific role"""
    role: KernelRole
    statements: List[Any]  # TIR statements
    cb_names: Set[str]     # CBs used by this kernel
    buffer_names: Set[str] # External buffers accessed
    runtime_args: List[str] # Runtime arguments needed


class KernelSplitter(tir.stmt_functor.StmtMutator):
    """Mutator to extract statements for a specific kernel role"""

    def __init__(self, role: KernelRole, dfg_metadata: Dict[str, Any]):
        super().__init__()
        self.role = role
        self.dfg = dfg_metadata
        self.statements = []
        self.cb_names = set()
        self.buffer_names = set()
        self.skip_current = False

    def visit_evaluate(self, op):
        """Visit T.evaluate nodes to filter by role"""

        if hasattr(op, 'value') and hasattr(op.value, 'op'):
            call = op.value
            op_name = str(call.op) if hasattr(call.op, 'name') else str(call.op)

            # Determine if this statement belongs to current role
            should_include = self._should_include_statement(op_name, call)

            if should_include:
                # Track CBs and buffers used
                self._track_resources(call)
                return op
            else:
                return None  # Filter out

        return super().visit_evaluate(op)

    def visit_allocate(self, op):
        """Visit allocate nodes (CB allocations)"""

        # CB allocations go to the kernel that uses them
        # This is determined by the DFG analysis
        alloc_name = op.buffer_var.name if hasattr(op.buffer_var, 'name') else str(op.buffer_var)

        if self._cb_belongs_to_role(alloc_name):
            self.cb_names.add(alloc_name)
            return super().visit_allocate(op)
        else:
            # Process body without the allocation
            return self.visit(op.body)

    def visit_for(self, op):
        """Visit for loops - preserve structure for the appropriate role"""

        # Check if this loop is relevant to current role
        if self._loop_belongs_to_role(op):
            return super().visit_for(op)
        else:
            # Skip the loop but process its body
            return self.visit(op.body)

    def _should_include_statement(self, op_name: str, call) -> bool:
        """Determine if a statement belongs to the current role"""

        if self.role == KernelRole.READER:
            # Reader handles: alloc_cb for inputs, read_to_cb
            return "read_to_cb" in op_name or self._is_input_cb_alloc(op_name, call)

        elif self.role == KernelRole.COMPUTE:
            # Compute handles: compute ops, intermediate CB allocs
            return any(x in op_name for x in ["mm.mma", "fpu.", "sfpu.", "gemm", "add", "mul"])

        elif self.role == KernelRole.WRITER:
            # Writer handles: write_from_cb, output CB allocs
            return "write_from_cb" in op_name or self._is_output_cb_alloc(op_name, call)

        return False

    def _is_input_cb_alloc(self, op_name: str, call) -> bool:
        """Check if this is an input CB allocation"""

        if "alloc_cb" not in op_name:
            return False

        # Extract CB name
        if len(call.args) > 0:
            cb_name = self._extract_string(call.args[0])
            if cb_name and ("in" in cb_name or cb_name in self._get_input_cbs()):
                return True

        return False

    def _is_output_cb_alloc(self, op_name: str, call) -> bool:
        """Check if this is an output CB allocation"""

        if "alloc_cb" not in op_name:
            return False

        # Extract CB name
        if len(call.args) > 0:
            cb_name = self._extract_string(call.args[0])
            if cb_name and ("out" in cb_name or cb_name in self._get_output_cbs()):
                return True

        return False

    def _cb_belongs_to_role(self, cb_name: str) -> bool:
        """Check if a CB belongs to the current kernel role"""

        if not self.dfg or "kernel_roles" not in self.dfg:
            # Fallback to name-based heuristics
            if self.role == KernelRole.READER:
                return "in" in cb_name
            elif self.role == KernelRole.COMPUTE:
                return True  # Compute can access all CBs
            elif self.role == KernelRole.WRITER:
                return "out" in cb_name
            return False

        roles = self.dfg["kernel_roles"]
        role_str = self.role.value

        return cb_name in roles.get(role_str, [])

    def _loop_belongs_to_role(self, loop) -> bool:
        """Check if a loop is relevant to the current role"""

        # For now, keep all loops and let the body filtering handle it
        # More sophisticated analysis could check if loop contains relevant ops
        return True

    def _track_resources(self, call):
        """Track CBs and buffers used by this statement"""

        for arg in call.args:
            # Track CB names
            cb_name = self._extract_string(arg)
            if cb_name and "cb" in cb_name.lower():
                self.cb_names.add(cb_name)

            # Track buffer accesses
            buffer_name = self._extract_buffer_name(arg)
            if buffer_name:
                self.buffer_names.add(buffer_name)

    def _get_input_cbs(self) -> Set[str]:
        """Get input CB names from DFG"""

        if not self.dfg or "cb_reuse" not in self.dfg:
            return {"cb_in0", "cb_in1", "cb_in2"}

        input_cbs = set()
        for cb_name, info in self.dfg["cb_reuse"].items():
            if info.get("read_count", 0) > 0:
                input_cbs.add(cb_name)

        return input_cbs

    def _get_output_cbs(self) -> Set[str]:
        """Get output CB names from DFG"""

        if not self.dfg or "cb_reuse" not in self.dfg:
            return {"cb_out"}

        output_cbs = set()
        for cb_name, info in self.dfg["cb_reuse"].items():
            if info.get("write_count", 0) > 0:
                output_cbs.add(cb_name)

        return output_cbs

    def _extract_string(self, arg) -> Optional[str]:
        """Extract string from TIR argument"""

        if isinstance(arg, str):
            return arg
        elif hasattr(arg, 'value'):
            return str(arg.value) if arg.value else None
        else:
            return None

    def _extract_buffer_name(self, arg) -> Optional[str]:
        """Extract buffer name from argument"""

        if hasattr(arg, 'buffer'):
            if hasattr(arg.buffer, 'name'):
                return arg.buffer.name
        elif hasattr(arg, 'name'):
            return arg.name

        return None


class SplitDeviceKernel:
    """
    Pass to split monolithic kernel into reader/compute/writer.

    This pass:
    1. Analyzes the dataflow graph from C3
    2. Separates statements by role
    3. Creates three new PrimFuncs
    4. Assigns CB indices across kernels
    5. Sets up runtime arguments for each kernel
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

            # Check if already split
            if func.attrs and func.attrs.get("tt.kernel_role") in ["reader", "compute", "writer"]:
                logger.info(f"Function {gvar} already split, skipping")
                new_funcs[gvar] = func
                continue

            # Split this function
            reader_func, compute_func, writer_func = self._split_function(func, str(gvar))

            # Add all three functions to the module
            # Use naming convention: original_reader, original_compute, original_writer
            base_name = str(gvar)
            new_funcs[f"{base_name}_reader"] = reader_func
            new_funcs[f"{base_name}_compute"] = compute_func
            new_funcs[f"{base_name}_writer"] = writer_func

            logger.info(f"Split {gvar} into 3 kernels: reader, compute, writer")

        return tvm.IRModule(new_funcs)

    def _split_function(self, func: "tir.PrimFunc", func_name: str) -> Tuple["tir.PrimFunc", "tir.PrimFunc", "tir.PrimFunc"]:
        """Split a single function into three kernels."""

        # Get dataflow graph metadata
        dfg_metadata = {}
        if func.attrs and "tt.tile_dfg" in func.attrs:
            dfg_metadata = self._convert_to_dict(func.attrs["tt.tile_dfg"])

        # Get CB assignment
        cb_assignment = {}
        if func.attrs and "tt.cb_assignment" in func.attrs:
            cb_assignment = self._convert_to_dict(func.attrs["tt.cb_assignment"])

        # Split the function body for each role
        reader_slice = self._extract_kernel_slice(func, KernelRole.READER, dfg_metadata)
        compute_slice = self._extract_kernel_slice(func, KernelRole.COMPUTE, dfg_metadata)
        writer_slice = self._extract_kernel_slice(func, KernelRole.WRITER, dfg_metadata)

        # Create new functions
        reader_func = self._create_kernel_func(
            func, reader_slice, KernelRole.READER, func_name, cb_assignment
        )
        compute_func = self._create_kernel_func(
            func, compute_slice, KernelRole.COMPUTE, func_name, cb_assignment
        )
        writer_func = self._create_kernel_func(
            func, writer_slice, KernelRole.WRITER, func_name, cb_assignment
        )

        return reader_func, compute_func, writer_func

    def _extract_kernel_slice(self, func: "tir.PrimFunc", role: KernelRole,
                             dfg_metadata: Dict[str, Any]) -> KernelSlice:
        """Extract statements for a specific kernel role."""

        splitter = KernelSplitter(role, dfg_metadata)
        new_body = splitter.visit(func.body)

        # Determine runtime args needed
        runtime_args = self._determine_runtime_args(role, splitter.buffer_names)

        return KernelSlice(
            role=role,
            statements=[new_body],
            cb_names=splitter.cb_names,
            buffer_names=splitter.buffer_names,
            runtime_args=runtime_args
        )

    def _create_kernel_func(self, original_func: "tir.PrimFunc", kernel_slice: KernelSlice,
                           role: KernelRole, base_name: str, cb_assignment: Dict[str, int]) -> "tir.PrimFunc":
        """Create a new PrimFunc for a specific kernel role."""

        # Determine which parameters to keep
        params = []
        buffer_map = {}

        if role == KernelRole.READER:
            # Reader needs input buffer parameters
            for param in original_func.params:
                if param in original_func.buffer_map:
                    buffer = original_func.buffer_map[param]
                    if any(x in buffer.name.lower() for x in ["a", "b", "input", "weight"]):
                        params.append(param)
                        buffer_map[param] = buffer

        elif role == KernelRole.COMPUTE:
            # Compute typically doesn't need buffer parameters (works on CBs)
            # But keep the structure for now
            pass

        elif role == KernelRole.WRITER:
            # Writer needs output buffer parameters
            for param in original_func.params:
                if param in original_func.buffer_map:
                    buffer = original_func.buffer_map[param]
                    if any(x in buffer.name.lower() for x in ["c", "output", "result"]):
                        params.append(param)
                        buffer_map[param] = buffer

        # Create new function
        new_func = tir.PrimFunc(
            params=params,
            body=kernel_slice.statements[0] if kernel_slice.statements else tir.Evaluate(0),
            ret_type=original_func.ret_type,
            buffer_map=buffer_map,
            attrs=original_func.attrs
        )

        # Update attributes
        new_func = new_func.with_attr("tt.kernel_role", role.value)
        new_func = new_func.with_attr("tt.runtime_args", kernel_slice.runtime_args)

        # Assign CB indices for this kernel
        kernel_cb_assignment = {}
        for cb_name in kernel_slice.cb_names:
            if cb_name in cb_assignment:
                kernel_cb_assignment[cb_name] = cb_assignment[cb_name]

        new_func = new_func.with_attr("tt.cb_indices", kernel_cb_assignment)

        # Copy relevant metadata from original
        for key in ["tt.core_grid", "tt.core_ranges", "tt.partition_mode",
                   "tt.grid_tiles", "tt.work_partition"]:
            if original_func.attrs and key in original_func.attrs:
                new_func = new_func.with_attr(key, original_func.attrs[key])

        return new_func

    def _determine_runtime_args(self, role: KernelRole, buffer_names: Set[str]) -> List[str]:
        """Determine runtime arguments needed for a kernel role."""

        args = []

        if role == KernelRole.READER:
            # Reader needs addresses for input buffers
            for buffer_name in buffer_names:
                if any(x in buffer_name.lower() for x in ["a", "b", "input", "weight"]):
                    args.append(f"{buffer_name}_addr")

        elif role == KernelRole.COMPUTE:
            # Compute needs tile counts and iteration info
            args.extend(["Kt"])  # K tiles for GEMM

        elif role == KernelRole.WRITER:
            # Writer needs addresses for output buffers
            for buffer_name in buffer_names:
                if any(x in buffer_name.lower() for x in ["c", "output", "result"]):
                    args.append(f"{buffer_name}_addr")

        # All kernels need basic runtime args
        args.extend(["start_id", "count", "Mt", "Nt"])

        return args

    def _convert_to_dict(self, attr_value: Any) -> Dict[str, Any]:
        """Convert TVM attribute value to Python dict."""

        if isinstance(attr_value, dict):
            return attr_value

        # Handle TVM Map type
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
        elif isinstance(value, (int, float, str, bool, list, tuple)):
            return value
        else:
            return str(value)


# Module-level pass function for compatibility
def split_device_kernel(mod: IRModule) -> IRModule:
    """Apply SplitDeviceKernel pass to a module."""
    pass_instance = SplitDeviceKernel()
    return pass_instance(mod)


# Example usage and testing
if __name__ == "__main__":
    import tvm.script
    from tvm.script import tir as T

    # Create test module with protocol-less IR and DFG metadata
    @tvm.script.ir_module
    class TestModule:
        @T.prim_func
        def gemm(
            A: T.Buffer((256, 256), "float16"),
            B: T.Buffer((256, 256), "float16"),
            C: T.Buffer((256, 256), "float16")
        ):
            # Simulate output from C1-C3
            T.evaluate(T.call_extern("tt.alloc_cb", "cb_in0", [128, 32], "bf16"))
            T.evaluate(T.call_extern("tt.alloc_cb", "cb_in1", [32, 128], "bf16"))
            T.evaluate(T.call_extern("tt.alloc_cb", "cb_out", [128, 128], "bf16"))

            T.evaluate(T.call_extern("tt.read_to_cb", A[0:128, 0:32], "cb_in0"))
            T.evaluate(T.call_extern("tt.read_to_cb", B[0:32, 0:128], "cb_in1"))
            T.evaluate(T.call_extern("tt.mm.mma", "cb_in0", "cb_in1", 0, True))
            T.evaluate(T.call_extern("tt.write_from_cb", "cb_out", C[0:128, 0:128]))

    # Add simulated DFG metadata
    func = TestModule["gemm"]
    func = func.with_attr("tt.tile_dfg", {
        "nodes": {
            "A": {"type": "buffer", "kernel_role": "reader"},
            "B": {"type": "buffer", "kernel_role": "reader"},
            "C": {"type": "buffer", "kernel_role": "writer"},
            "cb_in0": {"type": "cb", "kernel_role": "reader/compute"},
            "cb_in1": {"type": "cb", "kernel_role": "reader/compute"},
            "cb_out": {"type": "cb", "kernel_role": "compute/writer"},
            "compute_matmul_0": {"type": "compute", "kernel_role": "compute"}
        },
        "kernel_roles": {
            "reader": ["A", "B", "cb_in0", "cb_in1"],
            "compute": ["cb_in0", "cb_in1", "cb_out", "compute_matmul_0"],
            "writer": ["cb_out", "C"]
        },
        "cb_reuse": {
            "cb_in0": {"read_count": 1, "write_count": 0},
            "cb_in1": {"read_count": 1, "write_count": 0},
            "cb_out": {"read_count": 0, "write_count": 1}
        }
    })
    func = func.with_attr("tt.cb_assignment", {
        "cb_in0": 0,
        "cb_in1": 1,
        "cb_out": 16
    })
    TestModule["gemm"] = func

    # Apply D1 pass
    pass_d1 = SplitDeviceKernel()
    result = pass_d1(TestModule)

    # Check results
    print("=== Split Kernels ===")
    for name, func in result.functions_items():
        print(f"\n{name}:")
        if func.attrs and "tt.kernel_role" in func.attrs:
            print(f"  Role: {func.attrs['tt.kernel_role']}")
            print(f"  Parameters: {[p.name if hasattr(p, 'name') else str(p) for p in func.params]}")
            if "tt.runtime_args" in func.attrs:
                print(f"  Runtime args: {func.attrs['tt.runtime_args']}")
            if "tt.cb_indices" in func.attrs:
                print(f"  CB indices: {func.attrs['tt.cb_indices']}")