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

# Target defaults
from .target import apply_tt_defaults

# Legacy annotations (will be deprecated)
from .annotations import annotate_tt_layout, annotate_tt_schedule

from .codegen import (
    emit_tt_artifacts,
    write_artifacts_to_disk,
)
from . import intrin

__all__ = [
    # Core attributes and dataclasses
    "CoreRange",
    "WorkItem",
    "plan_dict",
    "TT_CORE_GRID",
    "TT_CORE_RANGES",
    "TT_WORK_PARTITION",
    "TT_LAYOUT_DESC",
    # IR sugar helpers
    "with_core_grid",
    "with_core_ranges",
    "with_work_partition",
    "with_layout_desc",
    # Runtime plan utilities
    "emit_tt_plan",
    "extract_runtime_plan",
    "load_tt_plan",
    "validate_plan",
    # New pass pipeline
    "InferTTLayout",
    "PropagateTTLayout",
    "TTTilesToCoreMap",
    "LowerTTTileIntrinsics",
    "GridToPersistentTT",
    "build_tt_pipeline",
    "run_pipeline",
    # Target and code generation
    "apply_tt_defaults",
    "emit_tt_artifacts",
    "write_artifacts_to_disk",
    # Legacy annotations (temporary - will be removed)
    "annotate_tt_layout",
    "annotate_tt_schedule",
    # Intrinsics
    "intrin",
]
