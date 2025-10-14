"""Tenstorrent-specific TileLang utilities and helpers.

This package provides Tenstorrent-specific functionality including:
- Default annotation helpers for schedule and sharding (TT Defaults stage)
- Schedule and sharding inference passes (Metadata Inference stage)
- TT-specific transforms and utilities
- New metadata-driven lowering pipeline with mid-level IR representation
"""

# Core attribute definitions and dataclasses
from .attrs import (
    CoreRange,
    WorkItem,
    plan_dict,
    TT_CORE_GRID,
    TT_CORE_RANGES,
    TT_WORK_PARTITION,
    TT_LAYOUT_DESC,
)

# IR sugar helpers for metadata attachment
from .ir_sugar import (
    with_core_grid,
    with_core_ranges,
    with_work_partition,
    with_layout_desc,
)

# Runtime plan utilities
from .runtime_plan import (
    emit_tt_plan,
    extract_runtime_plan,
    load_tt_plan,
    validate_plan,
)

# New pass pipeline (from passes submodule)
from .passes import (
    InferTTLayout,
    PropagateTTLayout,
    TTTilesToCoreMap,
    LowerTTTileIntrinsics,
    GridToPersistentTT,
    build_tt_pipeline,
    run_pipeline,
)

# Legacy imports (kept for compatibility)
from .target import apply_tt_defaults
from .annotations import annotate_tt_layout, annotate_tt_schedule
from .passes import (
    infer_default_tt_schedule,
    infer_default_tt_shard,
    infer_tt_layout,
    layout_aware_work_partition_tt,
    lower_gemm_to_tt_intrinsics,
    memory_space_lower_tt,
    propagate_tt_layout,
    tile_pad_tt,
    tt_tiles_to_core_map,
    verify_tt_ir,
    apply_tt_metadata_passes,
    apply_layout_aware_metadata_passes,
    grid_to_persistent_tt,
    apply_tt_transform_passes,
)
from .codegen import (
    emit_tt_artifacts,
    write_artifacts_to_disk,
)
from . import intrin

__all__ = [
    # New exports
    "CoreRange",
    "WorkItem",
    "plan_dict",
    "TT_CORE_GRID",
    "TT_CORE_RANGES",
    "TT_WORK_PARTITION",
    "TT_LAYOUT_DESC",
    "with_core_grid",
    "with_core_ranges",
    "with_work_partition",
    "with_layout_desc",
    "emit_tt_plan",
    "extract_runtime_plan",
    "load_tt_plan",
    "validate_plan",
    "InferTTLayout",
    "PropagateTTLayout",
    "TTTilesToCoreMap",
    "LowerTTTileIntrinsics",
    "GridToPersistentTT",
    "build_tt_pipeline",
    "run_pipeline",
    # Legacy exports
    "apply_tt_defaults",
    "annotate_tt_layout",
    "annotate_tt_schedule",
    "infer_default_tt_schedule",
    "infer_default_tt_shard",
    "infer_tt_layout",
    "propagate_tt_layout",
    "layout_aware_work_partition_tt",
    "lower_gemm_to_tt_intrinsics",
    "memory_space_lower_tt",
    "tile_pad_tt",
    "apply_tt_metadata_passes",
    "apply_layout_aware_metadata_passes",
    "grid_to_persistent_tt",
    "tt_tiles_to_core_map",
    "verify_tt_ir",
    "apply_tt_transform_passes",
    "emit_tt_artifacts",
    "write_artifacts_to_disk",
    "intrin",
]
