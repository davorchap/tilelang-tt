"""Tenstorrent transformation pass entry points."""

from __future__ import annotations

from tilelang import tvm as tvm  # re-export for typing convenience

from .grid_to_persistent_tt import grid_to_persistent_tt
from .infer_default_tt_schedule import infer_default_tt_schedule
from .infer_default_tt_shard import infer_default_tt_shard
from .infer_tt_layout import infer_tt_layout
from .layout_aware_work_partition_tt import layout_aware_work_partition_tt
from .lower_gemm_to_tt_intrinsics import lower_gemm_to_tt_intrinsics
from .lower_to_sfpu import lower_to_sfpu
from .memory_space_lower_tt import memory_space_lower_tt
from .propagate_tt_layout import propagate_tt_layout
from .tile_pad_tt import tile_pad_tt
from .tt_tiles_to_core_map import tt_tiles_to_core_map
from .verify_tt_ir import verify_tt_ir


def _log_pass_output(mod: tvm.IRModule, *, pass_name: str) -> tvm.IRModule:
    """Print the TIR module produced by a Tenstorrent pass."""

    header = f"--- [Tenstorrent] After {pass_name} ---"
    print(header)
    # Include metadata in the textual IR so downstream debugging has full context.
    print(mod.script(show_meta=True))
    return mod


def apply_tt_metadata_passes(mod: tvm.IRModule) -> tvm.IRModule:
    """Run legacy Tenstorrent metadata inference passes."""
    mod = _log_pass_output(infer_default_tt_schedule(mod), pass_name="InferDefaultTTSchedule")
    mod = _log_pass_output(infer_default_tt_shard(mod), pass_name="InferDefaultTTShard")
    return mod


def apply_layout_aware_metadata_passes(mod: tvm.IRModule) -> tvm.IRModule:
    """Run the layout-aware metadata inference pipeline."""
    mod = _log_pass_output(infer_tt_layout(mod), pass_name="InferTTLayout")
    mod = _log_pass_output(propagate_tt_layout(mod), pass_name="PropagateTTLayout")
    mod = _log_pass_output(
        layout_aware_work_partition_tt(mod), pass_name="LayoutAwareWorkPartitionTT")
    return mod


def apply_tt_transform_passes(mod: tvm.IRModule) -> tvm.IRModule:
    """Run the Tenstorrent persistent transform pipeline."""
    mod = _log_pass_output(grid_to_persistent_tt(mod), pass_name="GridToPersistentTT")
    mod = _log_pass_output(lower_to_sfpu(mod), pass_name="LowerToSFPU")
    mod = _log_pass_output(tt_tiles_to_core_map(mod), pass_name="TTTilesToCoreMap")
    mod = _log_pass_output(memory_space_lower_tt(mod), pass_name="MemorySpaceLowerTT")
    mod = _log_pass_output(tile_pad_tt(mod), pass_name="TilePadTT")
    mod = _log_pass_output(lower_gemm_to_tt_intrinsics(mod), pass_name="LowerGemmToTTIntrinsics")
    mod = _log_pass_output(verify_tt_ir(mod), pass_name="VerifyTTIR")
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
    "lower_to_sfpu",
    "memory_space_lower_tt",
    "propagate_tt_layout",
    "tile_pad_tt",
    "tt_tiles_to_core_map",
    "verify_tt_ir",
]
