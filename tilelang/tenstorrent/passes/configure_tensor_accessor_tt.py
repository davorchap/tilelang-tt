"""
Pass D2: ConfigureTensorAccessorTT (v5 Specification)
Version: 5.0
Date: 2025-10-15

Purpose: Bind abstract tensor accessors to runtime arguments after kernel split.
         Converts abstract accessors to concrete bound accessors.

Input: Split kernels with tt.tensor_accessor (abstract) and tt.runtime_args
Output: Updated tt.tensor_accessor with runtime binding (arg indices, sizes)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
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


class AccessorRole(Enum):
    """Role of tensor accessor in kernel"""
    INPUT = "input"      # Read by reader kernel
    OUTPUT = "output"    # Written by writer kernel
    WEIGHT = "weight"    # Read by reader (weights/parameters)
    INTERMEDIATE = "intermediate"  # Internal to compute


class TensorAccessorBinder:
    """Helper to bind tensor accessors to runtime arguments"""

    def __init__(self, kernel_role: str, runtime_args: List[str]):
        self.kernel_role = kernel_role
        self.runtime_args = runtime_args
        self.arg_index_map = {arg: idx for idx, arg in enumerate(runtime_args)}

    def bind_accessor(self, accessor: Dict[str, Any], buffer_name: str) -> Dict[str, Any]:
        """Bind an abstract accessor to runtime arguments"""

        # Copy accessor
        bound_accessor = dict(accessor)

        # Change type from abstract to bound
        bound_accessor["type"] = "bound"

        # Determine accessor role based on kernel and buffer
        accessor_role = self._determine_accessor_role(buffer_name, accessor)

        # Bind runtime argument indices based on role
        if self.kernel_role == "reader":
            bound_accessor = self._bind_reader_accessor(bound_accessor, buffer_name, accessor_role)
        elif self.kernel_role == "writer":
            bound_accessor = self._bind_writer_accessor(bound_accessor, buffer_name, accessor_role)
        elif self.kernel_role == "compute":
            # Compute kernel typically doesn't use tensor accessors (uses CBs)
            # But keep metadata for completeness
            bound_accessor["runtime_arg_idx"] = None
            bound_accessor["base_offset"] = None

        # Add binding metadata
        bound_accessor["kernel_role"] = self.kernel_role
        bound_accessor["binding_complete"] = True

        return bound_accessor

    def _determine_accessor_role(self, buffer_name: str, accessor: Dict[str, Any]) -> AccessorRole:
        """Determine the role of this accessor"""

        # Use access_pattern if available
        if "access_pattern" in accessor:
            pattern = accessor["access_pattern"]
            if pattern == "input":
                return AccessorRole.INPUT
            elif pattern == "output":
                return AccessorRole.OUTPUT
            elif pattern == "weight":
                return AccessorRole.WEIGHT

        # Fall back to name-based detection
        name_lower = buffer_name.lower()
        if any(x in name_lower for x in ["input", "a", "x"]):
            return AccessorRole.INPUT
        elif any(x in name_lower for x in ["output", "c", "result", "z"]):
            return AccessorRole.OUTPUT
        elif any(x in name_lower for x in ["weight", "w", "b", "kernel"]):
            return AccessorRole.WEIGHT
        else:
            return AccessorRole.INTERMEDIATE

    def _bind_reader_accessor(self, accessor: Dict[str, Any], buffer_name: str,
                             role: AccessorRole) -> Dict[str, Any]:
        """Bind accessor for reader kernel"""

        # Reader needs buffer address arguments
        addr_arg = f"{buffer_name}_addr"

        if addr_arg in self.arg_index_map:
            accessor["runtime_arg_idx"] = self.arg_index_map[addr_arg]
            accessor["base_offset"] = 0  # Base address is at arg[idx]
        else:
            # Try generic pattern
            if role == AccessorRole.INPUT and "input_addr" in self.arg_index_map:
                accessor["runtime_arg_idx"] = self.arg_index_map["input_addr"]
            elif role == AccessorRole.WEIGHT and "weight_addr" in self.arg_index_map:
                accessor["runtime_arg_idx"] = self.arg_index_map["weight_addr"]
            else:
                # Fallback: assign first available address arg
                for idx, arg in enumerate(self.runtime_args):
                    if "addr" in arg:
                        accessor["runtime_arg_idx"] = idx
                        break

        # Add tile iteration binding
        if "start_id" in self.arg_index_map:
            accessor["start_id_idx"] = self.arg_index_map["start_id"]
        if "count" in self.arg_index_map:
            accessor["count_idx"] = self.arg_index_map["count"]

        # Add dimension binding for address calculation
        if "Mt" in self.arg_index_map:
            accessor["Mt_idx"] = self.arg_index_map["Mt"]
        if "Nt" in self.arg_index_map:
            accessor["Nt_idx"] = self.arg_index_map["Nt"]
        if "Kt" in self.arg_index_map:
            accessor["Kt_idx"] = self.arg_index_map["Kt"]

        return accessor

    def _bind_writer_accessor(self, accessor: Dict[str, Any], buffer_name: str,
                              role: AccessorRole) -> Dict[str, Any]:
        """Bind accessor for writer kernel"""

        # Writer needs output buffer address
        addr_arg = f"{buffer_name}_addr"

        if addr_arg in self.arg_index_map:
            accessor["runtime_arg_idx"] = self.arg_index_map[addr_arg]
            accessor["base_offset"] = 0
        else:
            # Try generic pattern
            if "output_addr" in self.arg_index_map:
                accessor["runtime_arg_idx"] = self.arg_index_map["output_addr"]
            else:
                # Fallback: find any address arg
                for idx, arg in enumerate(self.runtime_args):
                    if "addr" in arg and "out" in arg.lower():
                        accessor["runtime_arg_idx"] = idx
                        break

        # Add tile iteration binding
        if "start_id" in self.arg_index_map:
            accessor["start_id_idx"] = self.arg_index_map["start_id"]
        if "count" in self.arg_index_map:
            accessor["count_idx"] = self.arg_index_map["count"]

        # Add dimension binding
        if "Mt" in self.arg_index_map:
            accessor["Mt_idx"] = self.arg_index_map["Mt"]
        if "Nt" in self.arg_index_map:
            accessor["Nt_idx"] = self.arg_index_map["Nt"]

        return accessor


class ConfigureTensorAccessorTT:
    """
    Pass to bind abstract tensor accessors to runtime arguments.

    This pass:
    1. Processes each split kernel (reader/compute/writer)
    2. Binds abstract accessors to runtime argument slots
    3. Adds index calculation metadata
    4. Marks accessors as "bound" for codegen
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
        """Process a single function to bind tensor accessors."""

        # Get kernel role
        kernel_role = None
        if func.attrs and "tt.kernel_role" in func.attrs:
            kernel_role = func.attrs["tt.kernel_role"]

        if not kernel_role or kernel_role == "monolithic":
            logger.debug("Skipping non-split kernel or monolithic kernel")
            return func

        # Get runtime arguments
        runtime_args = []
        if func.attrs and "tt.runtime_args" in func.attrs:
            runtime_args = self._convert_to_list(func.attrs["tt.runtime_args"])

        if not runtime_args:
            logger.warning(f"No runtime args found for {kernel_role} kernel")
            return func

        # Create binder for this kernel
        binder = TensorAccessorBinder(kernel_role, runtime_args)

        # Process each tensor accessor
        updated_attrs = dict(func.attrs) if func.attrs else {}

        for key in list(updated_attrs.keys()):
            if key.startswith("tt.tensor_accessor."):
                buffer_name = key.replace("tt.tensor_accessor.", "")
                accessor = self._convert_to_dict(updated_attrs[key])

                # Skip if already bound
                if accessor.get("type") == "bound":
                    continue

                # Bind the accessor
                bound_accessor = binder.bind_accessor(accessor, buffer_name)

                # Update the attribute
                updated_attrs[key] = tvm.runtime.convert(bound_accessor)

                logger.debug(f"Bound accessor for {buffer_name} in {kernel_role} kernel")

        # Create accessor binding summary
        binding_summary = self._create_binding_summary(updated_attrs, kernel_role, runtime_args)
        updated_attrs["tt.accessor_binding_summary"] = tvm.runtime.convert(binding_summary)

        # Update function with bound accessors
        for key, value in updated_attrs.items():
            func = func.with_attr(key, value)

        logger.info(f"Configured tensor accessors for {kernel_role} kernel")

        return func

    def _create_binding_summary(self, attrs: Dict[str, Any], kernel_role: str,
                               runtime_args: List[str]) -> Dict[str, Any]:
        """Create a summary of accessor bindings."""

        summary = {
            "kernel_role": kernel_role,
            "runtime_args": runtime_args,
            "bound_accessors": [],
            "total_accessors": 0,
            "binding_status": "complete"
        }

        # Count and list bound accessors
        for key in attrs:
            if key.startswith("tt.tensor_accessor."):
                summary["total_accessors"] += 1
                buffer_name = key.replace("tt.tensor_accessor.", "")
                accessor = self._convert_to_dict(attrs[key])

                if accessor.get("type") == "bound":
                    binding_info = {
                        "buffer": buffer_name,
                        "arg_idx": accessor.get("runtime_arg_idx"),
                        "access_pattern": accessor.get("access_pattern"),
                        "stride_mode": accessor.get("stride_mode")
                    }
                    summary["bound_accessors"].append(binding_info)

        # Check if all are bound
        if len(summary["bound_accessors"]) != summary["total_accessors"]:
            summary["binding_status"] = "partial"

        return summary

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

    def _convert_to_list(self, attr_value: Any) -> List[Any]:
        """Convert TVM attribute value to Python list."""

        if isinstance(attr_value, (list, tuple)):
            return list(attr_value)

        # Handle TVM Array type
        if hasattr(attr_value, "__iter__"):
            return [self._convert_value(x) for x in attr_value]

        return [attr_value]

    def _convert_value(self, value: Any) -> Any:
        """Convert TVM value to Python type."""

        if hasattr(value, "value"):
            return value.value
        elif isinstance(value, (int, float, str, bool)):
            return value
        else:
            return str(value)


# Module-level pass function for compatibility
def configure_tensor_accessor_tt(mod: IRModule) -> IRModule:
    """Apply ConfigureTensorAccessorTT pass to a module."""
    pass_instance = ConfigureTensorAccessorTT()
    return pass_instance(mod)


# Example usage and testing
if __name__ == "__main__":
    import tvm.script
    from tvm.script import tir as T

    # Create test module with split kernels
    @tvm.script.ir_module
    class TestModule:
        @T.prim_func
        def gemm_reader(
            A: T.Buffer((256, 256), "float16"),
            B: T.Buffer((256, 256), "float16")
        ):
            T.evaluate(0)  # Placeholder

        @T.prim_func
        def gemm_compute():
            T.evaluate(0)  # Placeholder

        @T.prim_func
        def gemm_writer(
            C: T.Buffer((256, 256), "float16")
        ):
            T.evaluate(0)  # Placeholder

    # Add kernel roles and runtime args (as if from D1)
    reader_func = TestModule["gemm_reader"]
    reader_func = reader_func.with_attr("tt.kernel_role", "reader")
    reader_func = reader_func.with_attr("tt.runtime_args",
                                       ["A_addr", "B_addr", "start_id", "count", "Mt", "Kt", "Nt"])

    # Add abstract accessors (as if from A3)
    reader_func = reader_func.with_attr("tt.tensor_accessor.A", {
        "type": "abstract",
        "buffer_name": "A",
        "layout_ref": "tt.buffer.A",
        "stride_mode": "tiled",
        "access_pattern": "input",
        "tile_dims": [32, 32],
        "tiles_per_dim": [8, 8],
        "memory": "DRAM",
        "layout_type": "interleaved",
        "base_offset": None,
        "runtime_arg_idx": None,
        "tile_size_bytes": 2048,
        "sharding": {"enabled": False}
    })

    reader_func = reader_func.with_attr("tt.tensor_accessor.B", {
        "type": "abstract",
        "buffer_name": "B",
        "layout_ref": "tt.buffer.B",
        "stride_mode": "tiled",
        "access_pattern": "input",
        "tile_dims": [32, 32],
        "tiles_per_dim": [8, 8],
        "memory": "DRAM",
        "layout_type": "interleaved",
        "base_offset": None,
        "runtime_arg_idx": None,
        "tile_size_bytes": 2048,
        "sharding": {"enabled": False}
    })

    TestModule["gemm_reader"] = reader_func

    # Setup compute kernel
    compute_func = TestModule["gemm_compute"]
    compute_func = compute_func.with_attr("tt.kernel_role", "compute")
    compute_func = compute_func.with_attr("tt.runtime_args", ["Kt"])
    TestModule["gemm_compute"] = compute_func

    # Setup writer kernel
    writer_func = TestModule["gemm_writer"]
    writer_func = writer_func.with_attr("tt.kernel_role", "writer")
    writer_func = writer_func.with_attr("tt.runtime_args",
                                       ["C_addr", "start_id", "count", "Mt", "Nt"])

    writer_func = writer_func.with_attr("tt.tensor_accessor.C", {
        "type": "abstract",
        "buffer_name": "C",
        "layout_ref": "tt.buffer.C",
        "stride_mode": "tiled",
        "access_pattern": "output",
        "tile_dims": [32, 32],
        "tiles_per_dim": [8, 8],
        "memory": "DRAM",
        "layout_type": "interleaved",
        "base_offset": None,
        "runtime_arg_idx": None,
        "tile_size_bytes": 2048,
        "sharding": {"enabled": False}
    })

    TestModule["gemm_writer"] = writer_func

    # Apply D2 pass
    pass_d2 = ConfigureTensorAccessorTT()
    result = pass_d2(TestModule)

    # Check results
    print("=== Tensor Accessor Binding Results ===\n")

    for name, func in result.functions_items():
        if func.attrs and "tt.kernel_role" in func.attrs:
            print(f"{name} ({func.attrs['tt.kernel_role']}):")

            # Check accessor bindings
            for key in func.attrs.keys():
                if key.startswith("tt.tensor_accessor."):
                    buffer_name = key.replace("tt.tensor_accessor.", "")
                    accessor = func.attrs[key]
                    print(f"  {buffer_name}:")
                    print(f"    Type: {accessor.get('type')}")
                    print(f"    Runtime arg idx: {accessor.get('runtime_arg_idx')}")
                    print(f"    Access pattern: {accessor.get('access_pattern')}")
                    print(f"    Binding complete: {accessor.get('binding_complete')}")

            # Check binding summary
            if "tt.accessor_binding_summary" in func.attrs:
                summary = func.attrs["tt.accessor_binding_summary"]
                print(f"  Binding Summary:")
                print(f"    Total accessors: {summary['total_accessors']}")
                print(f"    Bound accessors: {len(summary['bound_accessors'])}")
                print(f"    Status: {summary['binding_status']}")

            print()