"""
Tenstorrent-specific TIR passes for the new lowering pipeline.
"""

# Re-export pass constructors for convenience
from .infer_tt_layout import InferTTLayout
from .propagate_tt_layout import PropagateTTLayout
from .tt_tiles_to_core_map import TTTilesToCoreMap
from .lower_tt_tile_intrinsics import LowerTTTileIntrinsics
from .grid_to_persistent_tt import GridToPersistentTT
from .pipeline import build_tt_pipeline, run_pipeline

# Legacy compatibility imports (deprecated)
from ..compat import (
    apply_tt_metadata_passes,
    apply_layout_aware_metadata_passes,
    apply_tt_transform_passes,
    grid_to_persistent_tt,
    infer_default_tt_schedule,
    infer_default_tt_shard,
    tile_pad_tt,
    verify_tt_ir,
    lower_gemm_to_tt_intrinsics,
    memory_space_lower_tt,
    tt_tiles_to_core_map,
    infer_tt_layout,
    propagate_tt_layout,
    layout_aware_work_partition_tt,
)

__all__ = [
    # New pipeline passes
    "InferTTLayout",
    "PropagateTTLayout",
    "TTTilesToCoreMap",
    "LowerTTTileIntrinsics",
    "GridToPersistentTT",
    "build_tt_pipeline",
    "run_pipeline",
    # Legacy compatibility (deprecated)
    "apply_tt_metadata_passes",
    "apply_layout_aware_metadata_passes",
    "apply_tt_transform_passes",
    "grid_to_persistent_tt",
    "infer_default_tt_schedule",
    "infer_default_tt_shard",
    "tile_pad_tt",
    "verify_tt_ir",
    "lower_gemm_to_tt_intrinsics",
    "memory_space_lower_tt",
    "tt_tiles_to_core_map",
    "infer_tt_layout",
    "propagate_tt_layout",
    "layout_aware_work_partition_tt",
]
