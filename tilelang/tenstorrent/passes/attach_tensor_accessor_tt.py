"""
Pass A3: AttachTensorAccessorTT (v5 Specification)
Version: 5.0
Date: 2025-10-15

Purpose: Attach abstract tensor accessor descriptors to buffers.
         These accessors will be used later for runtime binding.

Input: Buffer layout descriptors from A1
Output: tt.tensor_accessor attributes for each buffer (abstract, unbound)
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import logging

try:
    import tvm
    from tvm import tir, IRModule
except ImportError:
    tvm = None
    tir = None
    IRModule = object

logger = logging.getLogger(__name__)


class AttachTensorAccessorTT:
    """
    Pass to attach abstract tensor accessor metadata to buffers.

    This pass:
    1. Creates abstract accessor descriptors for each buffer
    2. Links accessors to layout descriptors
    3. Sets stride mode based on buffer layout
    4. Leaves runtime binding fields null (filled by D2 later)

    The accessors are "abstract" at this stage - they describe how to access
    the tensor but don't have concrete runtime argument indices yet.
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
        """Process a single function to add tensor accessors."""

        # Collect buffer layouts from A1
        buffer_layouts = self._collect_buffer_layouts(func)

        if not buffer_layouts:
            logger.debug("No buffer layouts found, skipping tensor accessor attachment")
            return func

        # Create accessor for each buffer
        for buffer_name, layout in buffer_layouts.items():
            accessor = self._create_tensor_accessor(buffer_name, layout, func)

            # Attach accessor to function
            accessor_key = f"tt.tensor_accessor.{buffer_name}"
            func = func.with_attr(accessor_key, tvm.runtime.convert(accessor))

            logger.debug(f"Attached abstract accessor for buffer {buffer_name}")

        # Also create a summary of all accessors
        from ..attrs import TT_ACCESSOR_SUMMARY
        accessor_summary = self._create_accessor_summary(buffer_layouts)
        func = func.with_attr(TT_ACCESSOR_SUMMARY, tvm.runtime.convert(accessor_summary))

        logger.info(f"Attached {len(buffer_layouts)} tensor accessors")

        return func

    def _collect_buffer_layouts(self, func: "tir.PrimFunc") -> Dict[str, Dict[str, Any]]:
        """Collect buffer layouts from function attributes (from A1)."""

        layouts = {}

        if func.attrs:
            from ..attrs import TT_BUFFER_PREFIX
            for key in func.attrs.keys():
                if key.startswith(TT_BUFFER_PREFIX):
                    buffer_name = key.replace(TT_BUFFER_PREFIX, "")
                    layout = self._convert_to_dict(func.attrs[key])
                    layouts[buffer_name] = layout

        return layouts

    def _create_tensor_accessor(self, buffer_name: str, layout: Dict[str, Any],
                                func: "tir.PrimFunc") -> Dict[str, Any]:
        """
        Create an abstract tensor accessor for a buffer.

        At this stage, the accessor is "abstract" - it describes the access pattern
        but doesn't have concrete runtime bindings yet.
        """

        # Get buffer info from function
        buffer_info = self._get_buffer_info(func, buffer_name)

        # Determine stride mode based on layout
        stride_mode = self._determine_stride_mode(layout)

        # Calculate tile-related parameters
        tile_params = self._calculate_tile_params(layout, buffer_info)

        # Create accessor descriptor
        accessor = {
            "type": "abstract",  # Will become "bound" in D2
            "buffer_name": buffer_name,
            "layout_ref": f"tt.buffer.{buffer_name}",  # Link to layout
            "stride_mode": stride_mode,

            # Access pattern info
            "access_pattern": self._determine_access_pattern(buffer_name),
            "tile_dims": tile_params["tile_dims"],
            "tiles_per_dim": tile_params["tiles_per_dim"],

            # Memory info from layout
            "memory": layout.get("memory", "DRAM"),
            "layout_type": layout.get("layout", "interleaved"),

            # Runtime binding fields (null at this stage, filled by D2)
            "base_offset": None,  # Will be runtime arg index
            "runtime_arg_idx": None,  # Index in runtime args array
            "tile_size_bytes": tile_params["tile_size_bytes"],

            # Sharding info if applicable
            "sharding": self._extract_sharding_info(layout)
        }

        return accessor

    def _get_buffer_info(self, func: "tir.PrimFunc", buffer_name: str) -> Optional[Dict[str, Any]]:
        """Get buffer information from function."""

        for param in func.params:
            buffer = func.buffer_map.get(param, None)
            if buffer and buffer.name == buffer_name:
                return {
                    "shape": [int(dim) if hasattr(dim, 'value') else dim for dim in buffer.shape],
                    "dtype": str(buffer.dtype),
                    "strides": buffer.strides if buffer.strides else None,
                    "elem_offset": buffer.elem_offset if hasattr(buffer, 'elem_offset') else 0
                }

        return None

    def _determine_stride_mode(self, layout: Dict[str, Any]) -> str:
        """
        Determine the stride mode based on layout.

        Stride modes:
        - "tiled": Data is stored in tiles (default for TT)
        - "linear": Traditional row-major/column-major
        - "sharded": Distributed across cores
        """

        layout_type = layout.get("layout", "interleaved")
        memory = layout.get("memory", "DRAM")

        if layout_type == "sharded":
            return "sharded"
        elif memory == "L1":
            # L1 is always tiled on TT
            return "tiled"
        else:
            # DRAM can be tiled or linear
            # Default to tiled for TT backend
            return "tiled"

    def _calculate_tile_params(self, layout: Dict[str, Any],
                               buffer_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate tile-related parameters."""

        tile_shape = layout.get("tile_shape", [32, 32])
        dtype = layout.get("dtype", "bf16")

        # Handle tile_shape conversion - could be string, list, or tuple
        if isinstance(tile_shape, str):
            # Convert string representation like "[32, 32]" to list
            import ast
            try:
                tile_shape = ast.literal_eval(tile_shape)
            except (ValueError, SyntaxError):
                logger.warning(
                    f"Could not parse tile_shape string: {tile_shape}, using default [32, 32]")
                tile_shape = [32, 32]

        # Ensure tile_shape elements are integers
        if isinstance(tile_shape, (list, tuple)):
            tile_shape = [int(x) if not isinstance(x, int) else x for x in tile_shape]

        # Calculate tile size in bytes
        bytes_per_elem = self._get_dtype_bytes(dtype)
        tile_elements = tile_shape[0] * tile_shape[1]
        tile_size_bytes = tile_elements * bytes_per_elem

        # Calculate tiles per dimension if buffer info available
        tiles_per_dim = []
        if buffer_info and "shape" in buffer_info:
            shape = buffer_info["shape"]
            if len(shape) >= 2 and len(tile_shape) >= 2:
                # Calculate tiles needed for each dimension
                import math
                tiles_per_dim = [
                    math.ceil(shape[-2] / tile_shape[0]),  # M tiles
                    math.ceil(shape[-1] / tile_shape[1])  # N tiles
                ]

        return {
            "tile_dims": tile_shape,
            "tile_size_bytes": tile_size_bytes,
            "tiles_per_dim": tiles_per_dim
        }

    def _determine_access_pattern(self, buffer_name: str) -> str:
        """
        Determine the access pattern based on buffer name/role.

        Patterns:
        - "input": Read-only access
        - "output": Write-only access
        - "read_write": Read-modify-write
        - "weight": Read-only, might be broadcast
        """

        name_lower = buffer_name.lower()

        if any(x in name_lower for x in ["input", "a", "x"]):
            return "input"
        elif any(x in name_lower for x in ["output", "c", "result", "z"]):
            return "output"
        elif any(x in name_lower for x in ["weight", "w", "b", "kernel"]):
            return "weight"
        else:
            # Default to read_write for unknown patterns
            return "read_write"

    def _extract_sharding_info(self, layout: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract sharding information if present."""

        if "nd_shard" in layout:
            nd_shard = layout["nd_shard"]

            # Parse axes if it's a string
            axes = nd_shard.get("axes", [])
            if isinstance(axes, str):
                import ast
                try:
                    axes = ast.literal_eval(axes)
                except (ValueError, SyntaxError):
                    logger.warning(f"Could not parse axes string: {axes}")
                    axes = []

            # Parse projected_grid if it's a string
            grid = nd_shard.get("projected_grid", [1, 1])
            if isinstance(grid, str):
                import ast
                try:
                    grid = ast.literal_eval(grid)
                except (ValueError, SyntaxError):
                    logger.warning(f"Could not parse grid string: {grid}")
                    grid = [1, 1]

            # Parse projected_shard_tiles if it's a string
            shard_tiles = nd_shard.get("projected_shard_tiles", [1, 1])
            if isinstance(shard_tiles, str):
                import ast
                try:
                    shard_tiles = ast.literal_eval(shard_tiles)
                except (ValueError, SyntaxError):
                    logger.warning(f"Could not parse shard_tiles string: {shard_tiles}")
                    shard_tiles = [1, 1]

            return {
                "enabled": True,
                "axes": axes,
                "grid": grid,
                "shard_tiles": shard_tiles,
                "order": nd_shard.get("order", "row_major")
            }

        return {"enabled": False}

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
            logger.warning(f"Unknown dtype {dtype}, defaulting to 2 bytes")
            return 2

    def _create_accessor_summary(self, buffer_layouts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of all tensor accessors."""

        summary = {
            "total_accessors": len(buffer_layouts),
            "access_patterns": {},
            "stride_modes": {},
            "memory_types": {}
        }

        # Count different patterns
        for buffer_name, layout in buffer_layouts.items():
            # Access pattern
            pattern = self._determine_access_pattern(buffer_name)
            summary["access_patterns"][pattern] = summary["access_patterns"].get(pattern, 0) + 1

            # Stride mode
            mode = self._determine_stride_mode(layout)
            summary["stride_modes"][mode] = summary["stride_modes"].get(mode, 0) + 1

            # Memory type
            memory = layout.get("memory", "DRAM")
            summary["memory_types"][memory] = summary["memory_types"].get(memory, 0) + 1

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

    def _convert_value(self, value: Any) -> Any:
        """Convert TVM value to Python type."""

        if hasattr(value, "value"):
            return value.value
        elif isinstance(value, (int, float, str, bool)):
            return value
        else:
            return str(value)


# Module-level pass function for compatibility
def attach_tensor_accessor_tt(mod: IRModule) -> IRModule:
    """Apply AttachTensorAccessorTT pass to a module."""
    pass_instance = AttachTensorAccessorTT()
    return pass_instance(mod)


# Example usage and testing
if __name__ == "__main__":
    import tvm.script
    from tvm.script import tir as T

    # Create test module
    @tvm.script.ir_module
    class TestModule:

        @T.prim_func
        def gemm(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                 C: T.Buffer((256, 256), "float16")):
            for i, j in T.grid(256, 256):
                C[i, j] = A[i, j] + B[i, j]

    func = TestModule["gemm"]

    # Simulate A1 output (buffer layouts)
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

    # Apply A3 pass
    pass_a3 = AttachTensorAccessorTT()
    result = pass_a3(TestModule)

    # Check results
    func = result["gemm"]
    print("=== Tensor Accessors ===")

    for buffer in ["A", "B", "C"]:
        key = f"tt.tensor_accessor.{buffer}"
        if key in func.attrs:
            accessor = func.attrs[key]
            print(f"\n{buffer}:")
            print(f"  Type: {accessor['type']}")
            print(f"  Stride mode: {accessor['stride_mode']}")
            print(f"  Access pattern: {accessor['access_pattern']}")
            print(f"  Memory: {accessor['memory']}")
            print(f"  Runtime arg idx: {accessor['runtime_arg_idx']} (will be set in D2)")
            if accessor['sharding']['enabled']:
                print(f"  Sharding: {accessor['sharding']['grid']}")

    print("\n=== Accessor Summary ===")
    if "tt.accessor_summary" in func.attrs:
        summary = func.attrs["tt.accessor_summary"]
        for key, value in summary.items():
            print(f"  {key}: {value}")
