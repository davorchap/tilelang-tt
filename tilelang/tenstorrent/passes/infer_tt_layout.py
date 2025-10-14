"""
InferTTLayout: stamps layout/shard metadata onto PrimFuncs.
This pass ensures all buffers have layout descriptors, applying defaults where needed.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import logging

try:
    import tvm
    from tvm import tir, IRModule
except ImportError:  # pragma: no cover
    tvm = None
    tir = None
    IRModule = object

from ..attrs import TT_LAYOUT_DESC, TT_CORE_GRID
from ..ir_sugar import with_core_grid

logger = logging.getLogger(__name__)


class InferTTLayout:
    """
    Pass to infer and attach layout descriptors to PrimFuncs.

    This pass ensures every buffer has a layout descriptor, applying
    sensible defaults based on buffer usage patterns.
    """

    def __init__(self, defaults: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the pass with optional default layout settings.

        Args:
            defaults: Default layout descriptors to apply when none exist.
                     Example: {"shard": "DRAM", "interleave": True}
        """
        self.defaults = defaults or {"shard": "DRAM", "interleave": False}

    def _extract_grid_dimensions(self, func: "tir.PrimFunc") -> tuple[int, int]:
        """Extract grid dimensions from blockIdx thread extents in the IR."""
        grid_x, grid_y = 1, 1

        # T.Kernel creates nested structure:
        # BlockRealize -> Block -> AttrStmt(thread_extent) chain
        def extract_from_node(node):
            nonlocal grid_x, grid_y

            if isinstance(node, tir.BlockRealize):
                # Extract from the Block's body
                if hasattr(node.block, "body"):
                    extract_from_node(node.block.body)

            elif isinstance(node, tir.AttrStmt):
                if node.attr_key == "thread_extent" and hasattr(node.node, "thread_tag"):
                    # Check if it's a blockIdx binding
                    tag = node.node.thread_tag
                    extent = node.value

                    # Get the integer value
                    if hasattr(extent, "value"):
                        extent_val = int(extent.value)
                    elif isinstance(extent, int):
                        extent_val = extent
                    else:
                        # It might be an IntImm
                        try:
                            extent_val = int(extent)
                        except Exception:
                            extent_val = 1

                    if tag == "blockIdx.x":
                        grid_x = extent_val
                    elif tag == "blockIdx.y":
                        grid_y = extent_val

                # Continue with the body
                if hasattr(node, "body"):
                    extract_from_node(node.body)

            elif hasattr(node, "body"):
                # Generic case - recursively visit body
                extract_from_node(node.body)

        if hasattr(func, "body"):
            try:
                extract_from_node(func.body)
            except Exception as e:
                logger.debug(f"Failed to extract grid dimensions: {e}")

        return grid_x, grid_y

    def __call__(self, mod: IRModule) -> IRModule:
        """Apply the pass to an IRModule."""
        if tvm is None:
            return mod

        new_funcs = {}
        for gvar, func in mod.functions_items():
            if not isinstance(func, tir.PrimFunc):
                new_funcs[gvar] = func
                continue

            # Extract and set grid dimensions if not already present
            if func.attrs is None or TT_CORE_GRID not in func.attrs:
                # First check if there are tl.grid_x/tl.grid_y attributes (from test setup)
                grid_x, grid_y = 1, 1
                if (func.attrs and "tl.grid_x" in func.attrs and "tl.grid_y" in func.attrs):
                    # Extract from tl.grid_x/tl.grid_y attributes
                    grid_x_attr = func.attrs["tl.grid_x"]
                    grid_y_attr = func.attrs["tl.grid_y"]

                    # Handle IntImm or int values
                    grid_x = (
                        int(grid_x_attr.value)
                        if hasattr(grid_x_attr, "value") else int(grid_x_attr))
                    grid_y = (
                        int(grid_y_attr.value)
                        if hasattr(grid_y_attr, "value") else int(grid_y_attr))
                    logger.debug(
                        f"Found grid dimensions from tl.grid_x/y attributes: ({grid_x}, {grid_y})")
                else:
                    # Otherwise extract from IR structure (T.Kernel)
                    grid_x, grid_y = self._extract_grid_dimensions(func)
                    if grid_x > 1 or grid_y > 1:
                        logger.debug(f"Extracted grid dimensions from IR: ({grid_x}, {grid_y})")

                # Set the core grid if we have valid dimensions
                if grid_x >= 1 and grid_y >= 1:
                    func = with_core_grid(func, grid_x, grid_y)
                    logger.debug(f"Set core grid dimensions: ({grid_x}, {grid_y})")

            # Get existing layout descriptors or initialize empty
            layouts = func.attrs.get(TT_LAYOUT_DESC, None) if func.attrs else None

            if layouts is None:
                layouts = {}

            # Ensure all buffers have layout descriptors
            for param in func.params:
                buffer = func.buffer_map.get(param, None)
                if buffer and buffer.name not in layouts:
                    # Apply defaults for this buffer
                    layout = dict(self.defaults)

                    # Infer better defaults based on buffer properties
                    if "output" in buffer.name.lower() or buffer.name == "C":
                        # Output buffers often go to L1 for fast writeback
                        layout["shard"] = "L1"
                        layout["tile_id_order"] = "row_major"
                    elif "weight" in buffer.name.lower() or buffer.name == "B":
                        # Weights might benefit from interleaving
                        layout["interleave"] = True

                    layouts[buffer.name] = layout
                    logger.debug(f"Inferred layout for buffer {buffer.name}: {layout}")

            # Attach the layouts to the function
            if layouts:
                func = func.with_attr(TT_LAYOUT_DESC, tvm.runtime.convert(layouts))
                logger.info(f"Attached layout descriptors to function {gvar}")

            new_funcs[gvar] = func

        return tvm.IRModule(new_funcs)
