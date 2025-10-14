"""Tenstorrent transformation pass entry points."""

from __future__ import annotations

from tilelang import tvm as tvm  # re-export for typing convenience

from .grid_to_persistent_tt import grid_to_persistent_tt
from .infer_default_tt_schedule import infer_default_tt_schedule
from .infer_default_tt_shard import infer_default_tt_shard
from .infer_tt_layout import infer_tt_layout
from .layout_aware_work_partition_tt import layout_aware_work_partition_tt
from .lower_gemm_to_tt_intrinsics import lower_gemm_to_tt_intrinsics
from .memory_space_lower_tt import memory_space_lower_tt
from .propagate_tt_layout import propagate_tt_layout
from .tile_pad_tt import tile_pad_tt
from .tt_tiles_to_core_map import tt_tiles_to_core_map
from .verify_tt_ir import verify_tt_ir


def apply_tt_metadata_passes(mod: tvm.IRModule) -> tvm.IRModule:
    """Run legacy Tenstorrent metadata inference passes."""
    mod = infer_default_tt_schedule(mod)
    mod = infer_default_tt_shard(mod)
    return mod


def apply_layout_aware_metadata_passes(mod: tvm.IRModule) -> tvm.IRModule:
    """Run the layout-aware metadata inference pipeline."""
    mod = infer_tt_layout(mod)
    mod = propagate_tt_layout(mod)
    mod = layout_aware_work_partition_tt(mod)
    return mod


def apply_tt_transform_passes(mod: tvm.IRModule) -> tvm.IRModule:
    """Run the Tenstorrent persistent transform pipeline."""
    mod = grid_to_persistent_tt(mod)
    mod = tt_tiles_to_core_map(mod)
    mod = memory_space_lower_tt(mod)
    mod = tile_pad_tt(mod)
    mod = lower_gemm_to_tt_intrinsics(mod)
    mod = verify_tt_ir(mod)
    return mod


__all__ = [
    "apply_layout_aware_metadata_passes",
    "apply_tt_metadata_passes",
    "apply_tt_transform_passes",
    "grid_to_persistent_tt",
    "infer_default_tt_schedule",
    "infer_default_tt_shard",
    "infer_tt_layout",
    "layout_aware_work_partition_tt",
    "lower_gemm_to_tt_intrinsics",
    "memory_space_lower_tt",
    "propagate_tt_layout",
    "tile_pad_tt",
    "tt_tiles_to_core_map",
    "verify_tt_ir",
]
