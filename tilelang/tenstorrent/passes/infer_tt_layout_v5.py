"""
Pass A1: InferTTLayout (v5 Specification)
Version: 5.0
Date: 2025-10-15

Purpose: Normalize buffer defaults and explicit annotations
         following the v5 metadata schema.

Input: Buffer declarations with optional annotations
Output: Buffers with standardized tt.layout_desc attributes
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import logging

try:
    import tvm
    from tvm import tir, IRModule
except ImportError:
    tvm = None
    tir = None
    IRModule = object

logger = logging.getLogger(__name__)


class InferTTLayout_v5:
    """
    Pass to infer and attach layout descriptors following v5 specification.

    Defaults:
    - memory: DRAM
    - layout: interleaved
    - dtype: bf16
    - tile_shape: [32, 32]
    """

    def __init__(self, user_annotations: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize with optional user annotations.

        Args:
            user_annotations: User-provided layout annotations per buffer
                            Example: {"A": {"memory": "L1", "layout": "sharded"}}
        """
        self.user_annotations = user_annotations or {}

        # v5 defaults
        self.defaults = {
            "memory": "DRAM",
            "layout": "interleaved",
            "dtype": "bf16",
            "tile_shape": [32, 32]
        }

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
        """Process a single function to add layout descriptors."""

        # Extract core grid if needed (for validation)
        grid_x, grid_y = self._extract_grid_dimensions(func)

        # Get or create tt.layout_desc attribute dict
        layout_descs = {}

        # Process each buffer in the function
        for param in func.params:
            buffer = func.buffer_map.get(param, None)
            if not buffer:
                continue

            buffer_name = buffer.name

            # Create layout descriptor for this buffer
            layout = self._create_layout_descriptor(buffer, buffer_name)

            # Apply user annotations if provided
            if buffer_name in self.user_annotations:
                layout = self._apply_user_annotations(layout, self.user_annotations[buffer_name])

            # Validate the layout
            self._validate_layout(layout, buffer_name, grid_x, grid_y)

            # Store in the attribute dict
            layout_key = f"tt.buffer.{buffer_name}"
            layout_descs[layout_key] = layout

            logger.debug(f"Inferred v5 layout for buffer {buffer_name}: {layout}")

        # Attach all layout descriptors to function
        for key, desc in layout_descs.items():
            func = func.with_attr(key, tvm.runtime.convert(desc))

        # Also set core grid if extracted
        if grid_x > 0 and grid_y > 0:
            from tilelang.tenstorrent.attrs import TT_CORE_GRID
            func = func.with_attr(TT_CORE_GRID, [grid_x, grid_y])

        logger.info("Attached v5 layout descriptors to function")

        return func

    def _create_layout_descriptor(self, buffer: "tir.Buffer", name: str) -> Dict[str, Any]:
        """Create a v5-compliant layout descriptor for a buffer."""

        # Start with defaults
        layout = dict(self.defaults)

        # Extract dtype from buffer
        layout["dtype"] = str(buffer.dtype)

        # Infer better defaults based on buffer name/usage patterns
        if "output" in name.lower() or name == "C":
            # Output buffers might go to L1 for fast writeback
            # But keep default as DRAM unless user specifies
            pass
        elif "weight" in name.lower() or name == "B":
            # Weights might benefit from sharding
            # But keep as interleaved unless user specifies
            pass

        return layout

    def _apply_user_annotations(self, layout: Dict[str, Any],
                                annotations: Dict[str, Any]) -> Dict[str, Any]:
        """Apply user annotations to a layout descriptor."""

        # Override with user annotations
        for key, value in annotations.items():
            if key == "nd_shard":
                # Handle ND sharding metadata
                layout["nd_shard"] = self._process_nd_shard(value)
            else:
                layout[key] = value

        return layout

    def _process_nd_shard(self, nd_shard: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate ND sharding metadata."""

        required_fields = ["axes", "grid"]
        for field in required_fields:
            if field not in nd_shard:
                raise ValueError(f"ND shard missing required field: {field}")

        # Calculate derived fields if not provided
        if "shard_shape_elems" not in nd_shard:
            # This would need actual shape information
            nd_shard["shard_shape_elems"] = []

        if "order" not in nd_shard:
            nd_shard["order"] = "row_major"

        if "align_tiles" not in nd_shard:
            nd_shard["align_tiles"] = True

        # Calculate projected grid if not provided
        if "projected_grid" not in nd_shard:
            # Project ND grid to 2D compute plane
            grid = nd_shard["grid"]
            if len(grid) == 2:
                nd_shard["projected_grid"] = grid
            else:
                # Simple projection: last two dimensions
                nd_shard["projected_grid"] = grid[-2:] if len(grid) >= 2 else [1, grid[0]]

        # Calculate projected shard tiles if not provided
        if "projected_shard_tiles" not in nd_shard:
            # This needs shape and tile information
            nd_shard["projected_shard_tiles"] = [1, 1]  # Default

        return nd_shard

    def _validate_layout(self, layout: Dict[str, Any], buffer_name: str, grid_x: int, grid_y: int):
        """Validate a layout descriptor against v5 constraints."""

        # Validate memory type
        if layout["memory"] not in ["DRAM", "L1"]:
            raise ValueError(f"Invalid memory type for {buffer_name}: {layout['memory']}")

        # Validate layout type
        if layout["layout"] not in ["interleaved", "sharded"]:
            raise ValueError(f"Invalid layout type for {buffer_name}: {layout['layout']}")

        # L1-specific validation
        if layout["memory"] == "L1":
            if layout["layout"] == "sharded":
                # L1 sharded buffers must have nd_shard specification
                if "nd_shard" not in layout:
                    raise ValueError(
                        f"L1 sharded buffer {buffer_name} requires nd_shard specification")

                # Must be tile-aligned
                nd_shard = layout["nd_shard"]
                if not nd_shard.get("align_tiles", True):
                    raise ValueError(f"L1 sharded buffer {buffer_name} must be tile-aligned")

            # Check L1 capacity (simplified - would need actual size calculation)
            tile_shape = layout["tile_shape"]
            tile_size = tile_shape[0] * tile_shape[1]
            dtype_bytes = 2 if "16" in layout["dtype"] else 4
            page_size = tile_size * dtype_bytes

            if page_size > 2048:  # 2KB typical L1 page limit
                raise ValueError(
                    f"L1 tile size exceeds capacity for {buffer_name}: {page_size} bytes")

        # Reject halo metadata (not supported in v1)
        if "halo" in layout:
            raise ValueError(f"Halo metadata not supported for {buffer_name}")

        # Validate tile shape
        tile_shape = layout["tile_shape"]
        if len(tile_shape) != 2:
            raise ValueError(f"Tile shape must be 2D for {buffer_name}: {tile_shape}")
        if tile_shape[0] != 32 or tile_shape[1] != 32:
            logger.warning(f"Non-standard tile shape for {buffer_name}: {tile_shape}")

    def _extract_grid_dimensions(self, func: "tir.PrimFunc") -> tuple[int, int]:
        """Extract grid dimensions from the function."""

        # First check if already in attributes
        if func.attrs and "tt.core_grid" in func.attrs:
            grid = func.attrs["tt.core_grid"]
            return grid[0], grid[1]

        # Try to extract from IR (T.Kernel creates blockIdx bindings)
        grid_x, grid_y = 1, 1

        def extract_from_node(node):
            nonlocal grid_x, grid_y

            if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent" and hasattr(
                    node.node, "thread_tag"):
                tag = node.node.thread_tag
                extent = node.value

                # Extract integer value
                if hasattr(extent, "value"):
                    extent_val = int(extent.value)
                elif isinstance(extent, int):
                    extent_val = extent
                else:
                    try:
                        extent_val = int(extent)
                    except (ValueError, TypeError):
                        extent_val = 1

                if tag == "blockIdx.x":
                    grid_x = extent_val
                elif tag == "blockIdx.y":
                    grid_y = extent_val

            # Continue traversal
            if hasattr(node, "body"):
                extract_from_node(node.body)

        try:
            if hasattr(func, "body"):
                extract_from_node(func.body)
        except Exception as e:
            logger.debug(f"Failed to extract grid dimensions: {e}")

        return grid_x, grid_y


# Module-level pass function for compatibility
def infer_tt_layout_v5(mod: IRModule,
                       user_annotations: Optional[Dict[str, Any]] = None) -> IRModule:
    """Apply InferTTLayout v5 pass to a module."""
    pass_instance = InferTTLayout_v5(user_annotations)
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

    # Apply pass
    pass_v5 = InferTTLayout_v5()
    result = pass_v5(TestModule)

    # Check results
    func = result["gemm"]
    print("=== Function Attributes ===")
    for key in func.attrs.keys():
        if key.startswith("tt."):
            print(f"{key}: {func.attrs[key]}")

    # Test with user annotations
    user_annot = {
        "A": {
            "memory": "L1",
            "layout": "sharded",
            "nd_shard": {
                "axes": ["M", "N"],
                "grid": [2, 4]
            }
        }
    }

    pass_v5_with_annot = InferTTLayout_v5(user_annot)
    result2 = pass_v5_with_annot(TestModule)

    func2 = result2["gemm"]
    print("\n=== With User Annotations ===")
    for key in func2.attrs.keys():
        if key.startswith("tt."):
            print(f"{key}: {func2.attrs[key]}")
