"""
PropagateTTLayout: copies/normalizes tt.layout_desc across blocks/buffers.
This pass ensures layout consistency throughout the IR.

DEPRECATED: This pass is deprecated in favor of the v5 pipeline.
Use `propagate_tt_layout_v5` from `propagate_tt_layout_v5.py` instead.
"""

from __future__ import annotations
import logging
import warnings
from typing import Dict, Any

try:
    import tvm
    from tvm import tir, IRModule
except ImportError:  # pragma: no cover
    tvm = None
    tir = None
    IRModule = object

from ..attrs import TT_LAYOUT_DESC

logger = logging.getLogger(__name__)


class PropagateTTLayout:
    """
    Pass to propagate and normalize layout descriptors throughout the IR.

    This ensures that layout information is consistent and available
    at all levels where it's needed.

    .. deprecated::
        PropagateTTLayout is deprecated. Use `propagate_tt_layout_v5` or `run_pipeline()` instead.
    """

    def __call__(self, mod: IRModule) -> IRModule:
        """Apply the pass to an IRModule."""
        warnings.warn(
            "PropagateTTLayout is deprecated and will be removed in a future version. "
            "Use propagate_tt_layout_v5 from propagate_tt_layout_v5.py or run_pipeline() instead.",
            DeprecationWarning,
            stacklevel=2)
        if tvm is None:
            return mod

        new_funcs = {}
        for gvar, func in mod.functions_items():
            if not isinstance(func, tir.PrimFunc):
                new_funcs[gvar] = func
                continue

            # Get layout descriptors
            layouts = func.attrs.get(TT_LAYOUT_DESC, {}) if func.attrs else {}

            if layouts:
                # Normalize layout descriptors
                normalized_layouts = self._normalize_layouts(layouts)

                # Validate buffer references
                self._validate_buffer_refs(func, normalized_layouts)

                # Update function with normalized layouts
                func = func.with_attr(TT_LAYOUT_DESC, tvm.runtime.convert(normalized_layouts))
                logger.info(f"Propagated layouts for function {gvar}")

            new_funcs[gvar] = func

        return tvm.IRModule(new_funcs)

    def _normalize_layouts(self, layouts: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Normalize layout descriptors to ensure consistency."""
        normalized = {}

        for buffer_name, layout in layouts.items():
            norm_layout = dict(layout)

            # Ensure required fields have defaults
            if "shard" not in norm_layout:
                norm_layout["shard"] = "DRAM"

            # Normalize shard values
            if norm_layout["shard"].upper() in ["DRAM", "DDR"]:
                norm_layout["shard"] = "DRAM"
            elif norm_layout["shard"].upper() in ["L1", "SRAM"]:
                norm_layout["shard"] = "L1"

            # Set defaults for optional fields
            if "interleave" not in norm_layout:
                norm_layout["interleave"] = False

            if "tile_id_order" not in norm_layout:
                norm_layout["tile_id_order"] = "row_major"

            # Validate tile_id_order values
            valid_orders = ["row_major", "column_major", "match_shard", "z_order"]
            if norm_layout["tile_id_order"] not in valid_orders:
                logger.warning(
                    f"Unknown tile_id_order '{norm_layout['tile_id_order']}' for buffer {buffer_name}, defaulting to row_major"
                )
                norm_layout["tile_id_order"] = "row_major"

            normalized[buffer_name] = norm_layout

        return normalized

    def _validate_buffer_refs(self, func: "tir.PrimFunc", layouts: Dict[str, Dict[str,
                                                                                  Any]]) -> None:
        """Validate that all referenced buffers in layouts exist in the function."""
        buffer_names = set()

        # Collect all buffer names from the function
        for param in func.params:
            buffer = func.buffer_map.get(param, None)
            if buffer:
                buffer_names.add(buffer.name)

        # Check for layouts referencing non-existent buffers
        for buffer_name in layouts:
            if buffer_name not in buffer_names:
                logger.warning(f"Layout descriptor for non-existent buffer: {buffer_name}")
