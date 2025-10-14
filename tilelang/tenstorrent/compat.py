"""
Compatibility layer for the Tenstorrent backend.
This module provides backward compatibility for existing code that uses the legacy API.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional

try:
    import tvm
    from tvm import tir, IRModule
except ImportError:
    tvm = None
    tir = None
    IRModule = object

from .attrs import (
    TT_GRID, TT_BLOCK_SHAPE, TT_START_TILE, TT_RUNTIME_ARGS,
    TT_CORE_GRID, TT_LAYOUT_DESC, TT_WORK_PARTITION,
    CoreRange, WorkItem
)
from .ir_sugar import with_core_grid, with_layout_desc, with_work_partition
from .passes import build_tt_pipeline, run_pipeline

logger = logging.getLogger(__name__)

def legacy_to_new_attrs(func: "tir.PrimFunc") -> "tir.PrimFunc":
    """
    Convert legacy attributes to new format.
    
    This function maps old attribute names to new ones:
    - tt.grid → tt.core_grid
    - tt.block_shape → (used to infer work partition)
    - tt.start_tile → (used to infer work partition)
    """
    if not func.attrs:
        return func
    
    attrs = func.attrs
    
    # Convert tt.grid to tt.core_grid
    if TT_GRID in attrs and TT_CORE_GRID not in attrs:
        grid = attrs[TT_GRID]
        func = with_core_grid(func, int(grid[0]), int(grid[1]))
        logger.info(f"Converted tt.grid to tt.core_grid: {grid}")
    
    # Convert legacy runtime args to new work partition format
    if TT_RUNTIME_ARGS in attrs and TT_WORK_PARTITION not in attrs:
        runtime_args = attrs[TT_RUNTIME_ARGS]
        work_partition = _convert_runtime_args_to_work_partition(runtime_args)
        if work_partition:
            func = with_work_partition(func, work_partition)
            logger.info("Converted tt.runtime_args to tt.work_partition")
    
    # Ensure layout descriptors exist
    if TT_LAYOUT_DESC not in attrs:
        # Apply defaults based on common patterns
        layouts = _infer_default_layouts(func)
        if layouts:
            func = with_layout_desc(func, layouts)
            logger.info(f"Applied default layouts: {layouts}")
    
    return func

def _convert_runtime_args_to_work_partition(
    runtime_args: Dict[str, Any]
) -> Dict[str, List[WorkItem]]:
    """Convert legacy runtime_args format to new work_partition format."""
    work_partition = {}
    
    # Extract relevant fields from runtime args
    # This is a simplified conversion - real implementation would be more sophisticated
    if "core_assignments" in runtime_args:
        for core_str, assignments in runtime_args["core_assignments"].items():
            work_items = []
            for assignment in assignments:
                work_item = WorkItem(
                    io=assignment.get("m_idx", 0),
                    jo=assignment.get("n_idx", 0),
                    len_k=assignment.get("k_len", None)
                )
                work_items.append(work_item)
            work_partition[core_str] = work_items
    else:
        # Default single work item per core
        work_partition["(0,0)"] = [WorkItem(io=0, jo=0)]
    
    return work_partition

def _infer_default_layouts(func: "tir.PrimFunc") -> Dict[str, Dict[str, Any]]:
    """Infer default layout descriptors based on buffer names and usage."""
    layouts = {}
    
    for param in func.params:
        buffer = func.buffer_map.get(param, None)
        if buffer:
            name = buffer.name
            
            # Apply heuristics based on common naming patterns
            if name in ["A", "B", "input", "weight"]:
                layouts[name] = {"shard": "DRAM", "interleave": True}
            elif name in ["C", "output", "result"]:
                layouts[name] = {"shard": "L1", "tile_id_order": "row_major"}
            else:
                # Default fallback
                layouts[name] = {"shard": "DRAM"}
    
    return layouts

def apply_compatibility_transforms(mod: IRModule) -> IRModule:
    """
    Apply compatibility transforms to convert legacy IR to new format.
    
    This is the main entry point for backward compatibility.
    """
    if tvm is None:
        return mod
    
    new_funcs = {}
    for gvar, func in mod.functions_items():
        if isinstance(func, tir.PrimFunc):
            # Convert legacy attributes
            func = legacy_to_new_attrs(func)
        new_funcs[gvar] = func
    
    return tvm.IRModule(new_funcs)

# Legacy pass names mapped to new implementations
def apply_tt_metadata_passes(mod: IRModule) -> IRModule:
    """Legacy name for metadata inference passes."""
    logger.info("Using compatibility layer: apply_tt_metadata_passes → new pipeline")
    mod = apply_compatibility_transforms(mod)
    
    # Run only the metadata passes
    from .passes import InferTTLayout, PropagateTTLayout, TTTilesToCoreMap
    
    passes = [
        InferTTLayout(),
        PropagateTTLayout(),
        TTTilesToCoreMap(),
    ]
    
    for pass_inst in passes:
        mod = pass_inst(mod)
    
    return mod

def apply_layout_aware_metadata_passes(mod: IRModule) -> IRModule:
    """Legacy name for layout-aware passes."""
    logger.info("Using compatibility layer: apply_layout_aware_metadata_passes → new pipeline")
    return apply_tt_metadata_passes(mod)

def apply_tt_transform_passes(mod: IRModule, plan_path: str = "tt.plan.json") -> IRModule:
    """Legacy name for full transform pipeline."""
    logger.info("Using compatibility layer: apply_tt_transform_passes → new pipeline")
    mod = apply_compatibility_transforms(mod)
    return run_pipeline(mod, plan_path=plan_path)

# Export compatibility functions
__all__ = [
    "legacy_to_new_attrs",
    "apply_compatibility_transforms",
    "apply_tt_metadata_passes",
    "apply_layout_aware_metadata_passes",
    "apply_tt_transform_passes",
]
