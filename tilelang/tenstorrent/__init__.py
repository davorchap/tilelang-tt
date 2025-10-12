"""Tenstorrent-specific TileLang utilities and helpers.

This package provides Tenstorrent-specific functionality including:
- Default annotation helpers for schedule and sharding (TT Defaults stage)
- Schedule and sharding inference passes (Metadata Inference stage)
- TT-specific transforms and utilities
"""

from .target import apply_tt_defaults
from .passes import (
    annotate_tt_layout,
    annotate_tt_schedule,
    infer_default_tt_schedule,
    infer_default_tt_shard,
    infer_tt_layout,
    propagate_tt_layout,
    layout_aware_work_partition_tt,
    apply_tt_metadata_passes,
    apply_layout_aware_metadata_passes,
    grid_to_persistent_tt,
    tt_tiles_to_core_map,
    apply_tt_transform_passes,
)
from .codegen import (
    emit_tt_artifacts,
    write_artifacts_to_disk,
)
from . import intrin

__all__ = [
    "apply_tt_defaults",
    "annotate_tt_layout",
    "annotate_tt_schedule",
    "infer_default_tt_schedule",
    "infer_default_tt_shard",
    "infer_tt_layout",
    "propagate_tt_layout",
    "layout_aware_work_partition_tt",
    "apply_tt_metadata_passes",
    "apply_layout_aware_metadata_passes",
    "grid_to_persistent_tt",
    "tt_tiles_to_core_map",
    "apply_tt_transform_passes",
    "emit_tt_artifacts",
    "write_artifacts_to_disk",
    "intrin",
]
