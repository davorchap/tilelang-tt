"""
Pass A2: PropagateTTLayout (v5 Specification)
Version: 5.0
Date: 2025-10-15

Purpose: Derive conceptual CB geometry from layout descriptors
         created by A1: InferTTLayout.

Input: tt.buffer.* layout descriptors from A1
Output: tt.cb_desc with page_size, depth, data_format
"""

from __future__ import annotations
from typing import Dict, Any
import logging

try:
    import tvm
    from tvm import tir, IRModule
except ImportError:
    tvm = None
    tir = None
    IRModule = object

logger = logging.getLogger(__name__)


class PropagateTTLayout_v5:
    """
    Pass to propagate layout information and generate CB descriptors.

    This pass:
    1. Reads tt.buffer.* attributes from A1
    2. Generates CB descriptors with geometry info
    3. Calculates page_size from dtype and tile_shape
    4. Sets default depth for double-buffering
    """

    def __init__(self, default_cb_depth: int = 2) -> None:
        """
        Initialize with CB configuration defaults.

        Args:
            default_cb_depth: Default circular buffer depth (2 = double buffering)
        """
        self.default_cb_depth = default_cb_depth

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
        """Process a single function to generate CB descriptors."""

        # Collect all buffer layout descriptors from A1
        buffer_layouts = self._collect_buffer_layouts(func)

        if not buffer_layouts:
            logger.debug("No buffer layouts found, skipping CB generation")
            return func

        # Generate CB descriptors from layouts
        cb_descriptors = self._generate_cb_descriptors(buffer_layouts)

        # Attach CB descriptors to function
        func = func.with_attr("tt.cb_descriptors", tvm.runtime.convert(cb_descriptors))

        # Also propagate a summary of CB requirements
        cb_summary = self._generate_cb_summary(cb_descriptors)
        func = func.with_attr("tt.cb_summary", tvm.runtime.convert(cb_summary))

        logger.info(f"Generated {len(cb_descriptors)} CB descriptors")

        return func

    def _collect_buffer_layouts(self, func: "tir.PrimFunc") -> Dict[str, Dict[str, Any]]:
        """Collect all buffer layout descriptors from function attributes."""

        layouts = {}

        # Look for tt.buffer.* attributes
        if func.attrs:
            for key in func.attrs.keys():
                if key.startswith("tt.buffer."):
                    buffer_name = key.replace("tt.buffer.", "")
                    layout = self._convert_to_dict(func.attrs[key])
                    layouts[buffer_name] = layout
                    logger.debug(f"Found layout for buffer {buffer_name}: {layout}")

        return layouts

    def _generate_cb_descriptors(
            self, buffer_layouts: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Generate CB descriptors from buffer layouts."""

        cb_descriptors = {}
        cb_counter = {"input": 0, "output": 0, "intermediate": 0}

        for buffer_name, layout in buffer_layouts.items():
            # Determine CB type based on buffer name/usage
            if any(x in buffer_name.lower() for x in ["input", "a", "b"]):
                cb_type = "input"
                cb_index = cb_counter["input"]
                cb_counter["input"] += 1
                cb_name = f"cb_in{cb_index}"
            elif any(x in buffer_name.lower() for x in ["output", "c", "result"]):
                cb_type = "output"
                cb_index = cb_counter["output"]
                cb_counter["output"] += 1
                cb_name = f"cb_out{cb_index}"
            else:
                cb_type = "intermediate"
                cb_index = cb_counter["intermediate"]
                cb_counter["intermediate"] += 1
                cb_name = f"cb_intermed{cb_index}"

            # Generate CB descriptor
            cb_desc = self._create_cb_descriptor(layout, buffer_name)
            cb_desc["buffer_name"] = buffer_name  # Track source buffer
            cb_desc["cb_type"] = cb_type

            cb_descriptors[cb_name] = cb_desc

        return cb_descriptors

    def _create_cb_descriptor(self, layout: Dict[str, Any], buffer_name: str) -> Dict[str, Any]:
        """Create a CB descriptor from a buffer layout."""

        # Extract necessary information
        tile_shape = layout.get("tile_shape", [32, 32])
        dtype = layout.get("dtype", "bf16")
        memory = layout.get("memory", "DRAM")
        layout_type = layout.get("layout", "interleaved")

        # Calculate page size
        tile_elements = tile_shape[0] * tile_shape[1]
        bytes_per_elem = self._get_dtype_bytes(dtype)
        page_size = tile_elements * bytes_per_elem

        # Convert dtype to Metalium format
        data_format = self._dtype_to_metalium_format(dtype)

        # Determine depth based on usage pattern
        depth = self.default_cb_depth  # Default double-buffering

        # Adjust for specific patterns
        if memory == "L1" and layout_type == "sharded":
            # L1 sharded might need different depth
            depth = 1  # Single buffered for L1 resident
        elif "weight" in buffer_name.lower():
            # Weights might benefit from deeper buffering
            depth = 4  # Quad-buffering for weights

        # Create descriptor
        cb_descriptor = {
            "page_size": page_size,
            "depth": depth,
            "data_format": data_format,
            "tile_shape": tile_shape,
            "source_memory": memory,
            "source_layout": layout_type
        }

        # Add sharding info if present
        if "nd_shard" in layout:
            cb_descriptor["sharding"] = {
                "enabled": True,
                "grid": layout["nd_shard"].get("projected_grid", [1, 1]),
                "shard_tiles": layout["nd_shard"].get("projected_shard_tiles", [1, 1])
            }
        else:
            cb_descriptor["sharding"] = {"enabled": False}

        return cb_descriptor

    def _get_dtype_bytes(self, dtype: str) -> int:
        """Get number of bytes for a dtype string."""

        dtype_lower = dtype.lower()

        if "8" in dtype_lower:
            return 1
        elif "16" in dtype_lower or "bf16" in dtype_lower or "fp16" in dtype_lower:
            return 2
        elif "32" in dtype_lower:
            return 4
        elif "64" in dtype_lower:
            return 8
        else:
            # Default to 2 bytes (bf16)
            logger.warning(f"Unknown dtype {dtype}, defaulting to 2 bytes")
            return 2

    def _dtype_to_metalium_format(self, dtype: str) -> str:
        """Convert dtype string to Metalium data format."""

        dtype_lower = dtype.lower()

        # Map to Metalium formats
        if "bf16" in dtype_lower or "bfloat16" in dtype_lower:
            return "Float16_b"  # BFloat16
        elif "fp16" in dtype_lower or "float16" in dtype_lower:
            return "Float16"  # Float16
        elif "fp32" in dtype_lower or "float32" in dtype_lower:
            return "Float32"  # Float32
        elif "int8" in dtype_lower:
            return "Int8"  # Int8
        elif "uint8" in dtype_lower:
            return "UInt8"  # UInt8
        elif "int16" in dtype_lower:
            return "Int16"  # Int16
        elif "uint16" in dtype_lower:
            return "UInt16"  # UInt16
        elif "int32" in dtype_lower:
            return "Int32"  # Int32
        elif "uint32" in dtype_lower:
            return "UInt32"  # UInt32
        else:
            # Default to BFloat16
            logger.warning(f"Unknown dtype {dtype}, defaulting to Float16_b")
            return "Float16_b"

    def _generate_cb_summary(self, cb_descriptors: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of CB requirements."""

        total_cbs = len(cb_descriptors)
        total_l1_bytes = 0
        cb_types = {"input": 0, "output": 0, "intermediate": 0}

        for _cb_name, cb_desc in cb_descriptors.items():
            # Calculate L1 usage
            l1_usage = cb_desc["page_size"] * cb_desc["depth"]
            total_l1_bytes += l1_usage

            # Count types
            cb_type = cb_desc.get("cb_type", "intermediate")
            cb_types[cb_type] += 1

        summary = {
            "total_cbs": total_cbs,
            "total_l1_bytes": total_l1_bytes,
            "cb_counts": cb_types,
            "fits_in_l1":
                total_l1_bytes <= 1024 * 1024  # 1MB L1 typical
        }

        return summary

    def _convert_to_dict(self, attr_value: Any) -> Dict[str, Any]:
        """Convert TVM attribute value to Python dict."""

        if isinstance(attr_value, dict):
            return attr_value

        # Handle TVM Map type
        if hasattr(attr_value, "items"):
            result = {}
            for k, v in attr_value.items():
                # Recursively convert nested structures
                if hasattr(v, "items"):
                    result[str(k)] = self._convert_to_dict(v)
                elif isinstance(v, (list, tuple)):
                    result[str(k)] = [self._convert_value(x) for x in v]
                else:
                    result[str(k)] = self._convert_value(v)
            return result

        # Handle other types
        return {"value": self._convert_value(attr_value)}

    def _convert_value(self, value: Any) -> Any:
        """Convert TVM value to Python type."""

        if hasattr(value, "value"):
            # IntImm, FloatImm, etc.
            return value.value
        elif isinstance(value, (int, float, str, bool)):
            return value
        elif isinstance(value, (list, tuple)):
            # Already a Python list/tuple, return as-is
            return list(value)
        elif hasattr(value, "__iter__") and not isinstance(value, str):
            # TVM Array or other iterable - convert to list
            try:
                return [self._convert_value(x) for x in value]
            except (TypeError, AttributeError):
                return str(value)
        else:
            return str(value)


# Module-level pass function for compatibility
def propagate_tt_layout_v5(mod: IRModule) -> IRModule:
    """Apply PropagateTTLayout v5 pass to a module."""
    pass_instance = PropagateTTLayout_v5()
    return pass_instance(mod)


# Example usage and testing
if __name__ == "__main__":
    import tvm.script
    from tvm.script import tir as T

    # Create test module with layout attributes (as if A1 ran)
    @tvm.script.ir_module
    class TestModule:

        @T.prim_func
        def gemm(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                 C: T.Buffer((256, 256), "float16")):
            T.evaluate(0)  # Dummy body

    # Simulate A1 output
    func = TestModule["gemm"]

    # Add buffer layouts as if A1 ran
    func = func.with_attr("tt.buffer.A", {
        "memory": "DRAM",
        "layout": "interleaved",
        "tile_shape": [32, 32],
        "dtype": "bf16"
    })

    func = func.with_attr("tt.buffer.B", {
        "memory": "DRAM",
        "layout": "interleaved",
        "tile_shape": [32, 32],
        "dtype": "bf16"
    })

    func = func.with_attr(
        "tt.buffer.C", {
            "memory": "L1",
            "layout": "sharded",
            "tile_shape": [32, 32],
            "dtype": "bf16",
            "nd_shard": {
                "axes": ["M", "N"],
                "grid": [2, 4],
                "projected_grid": [2, 4],
                "projected_shard_tiles": [4, 2]
            }
        })

    TestModule["gemm"] = func

    # Apply A2 pass
    pass_a2 = PropagateTTLayout_v5()
    result = pass_a2(TestModule)

    # Check results
    func = result["gemm"]
    print("=== CB Descriptors ===")
    if "tt.cb_descriptors" in func.attrs:
        cb_descs = func.attrs["tt.cb_descriptors"]
        for cb_name, cb_desc in cb_descs.items():
            print(f"\n{cb_name}:")
            for key, value in cb_desc.items():
                print(f"  {key}: {value}")

    print("\n=== CB Summary ===")
    if "tt.cb_summary" in func.attrs:
        summary = func.attrs["tt.cb_summary"]
        for key, value in summary.items():
            print(f"  {key}: {value}")
