"""
Attribute keys and simple dataclasses for the Tenstorrent TileLang backend.
This module is pure-Python and has no heavy deps.
Centralized attribute definitions to prevent drift between passes and host code.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

"""Central registry of Tenstorrent attribute keys used across passes/codegen.

All code should import keys from this module instead of hardcoding strings.
Legacy aliases are provided where needed for back-compat.
"""

# Attribute keys (single source of truth)
TT_CORE_GRID = "tt.core_grid"
TT_CORE_RANGES = "tt.core_ranges"  # List[CoreRange]
TT_CORE_RANGE = "tt.core_range"  # Back-compat single-range
TT_WORK_PARTITION = "tt.work_partition"  # Dict[str, List[WorkItem]]
TT_LAYOUT_DESC = "tt.layout_desc"  # Dict[str, Dict[str, Any]] per buffer
TT_GRID_TILES = "tt.grid_tiles"
TT_PARTITION_MODE = "tt.partition_mode"
TT_SHARD_GRID = "tt.shard_grid"
TT_LOCAL_SHAPE_TILES = "tt.local_shape_tiles"

# Kernel metadata keys
TT_KERNEL_ROLE = "tt.kernel_role"  # reader | compute | writer | monolithic
TT_RUNTIME_ARGS = "tt.runtime_args"
TT_RUNTIME_ARGS_FINALIZED = "tt.runtime_args_finalized"

# Protocol/phase markers
TT_CB_PROTOCOL_INSERTED = "tt.cb_protocol_inserted"
TT_COMPUTE_INIT_INSERTED = "tt.compute_init_inserted"
TT_COMPUTE_INIT_INFO = "tt.compute_init_info"
TT_DST_MANAGEMENT_INSERTED = "tt.dst_management_inserted"

# CB- and DFG-related metadata
TT_CB_DESCRIPTORS = "tt.cb_descriptors"
TT_CB_SUMMARY = "tt.cb_summary"
TT_CB_INDICES = "tt.cb_indices"
TT_TILE_DFG = "tt.tile_dfg"
TT_CB_ASSIGNMENT = "tt.cb_assignment"
TT_ACCESSOR_SUMMARY = "tt.accessor_summary"
TT_CORE_MAP_X = "tt.core_map_x"
TT_CORE_MAP_Y = "tt.core_map_y"
TT_TRANSFORMED_TO_CORE = "tt.transformed_to_core"
TT_RUNTIME_ARGS_INFO = "tt.runtime_args_info"
TT_PERSISTENT_CONFIG = "tt.persistent_config"
TT_RUNTIME_ARG_NAMES = "tt.runtime_arg_names"
TT_RUNTIME_CONSTANTS = "tt.runtime_constants"
TT_CORE_RUNTIME_ARGS = "tt_core_runtime_args"
TT_CONCEPTUAL_CBS = "tt.conceptual_cbs"
TT_SHARED_TO_CB_MAP = "tt.shared_to_cb_map"

# User hints (annotation helpers)
TT_USER_LAYOUT = "tt.user_layout"
TT_USER_SCHEDULE = "tt.user_schedule"

# Prefix helpers
TT_BUFFER_PREFIX = "tt.buffer."

# Legacy attributes (for compatibility)
TT_GRID = "tt.grid"
TT_BLOCK_SHAPE = "tt.block_shape"
TT_START_TILE = "tt.start_tile"
TT_DEVICE_MESH = "tt.device_mesh"
TT_PARTITION_MODE_LEGACY = "partition_mode"


@dataclass(frozen=True)
class CoreRange:
    """Represents a rectangular range of cores."""

    start: Tuple[int, int]  # (sx, sy)
    extent: Tuple[int, int]  # (ex, ey)

    def to_json(self) -> Dict[str, Any]:
        return {"start": list(self.start), "extent": list(self.extent)}

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "CoreRange":
        return cls(tuple(data["start"]), tuple(data["extent"]))


@dataclass(frozen=True)
class WorkItem:
    """Represents a unit of work assigned to a core."""

    io: int
    jo: int
    len_k: Optional[int] = None
    tile_order: Optional[str] = None  # e.g., "row_major", "match_shard"

    def to_json(self) -> Dict[str, Any]:
        d = {"io": self.io, "jo": self.jo}
        if self.len_k is not None:
            d["len_k"] = self.len_k
        if self.tile_order:
            d["tile_order"] = self.tile_order
        return d

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "WorkItem":
        return cls(**data)


def plan_dict(
    core_grid: Tuple[int, int],
    core_ranges: List[CoreRange],
    work_partition: Dict[str, List[WorkItem]],
    layouts: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a plan dictionary for tt.plan.json."""
    return {
        "core_grid": list(core_grid),
        "core_ranges": [cr.to_json() for cr in core_ranges],
        "work_partition": {
            k: [wi.to_json() for wi in v] for k, v in work_partition.items()
        },
        "layouts": layouts,
    }
