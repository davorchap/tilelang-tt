"""
Attribute keys and simple dataclasses for the Tenstorrent TileLang backend.
This module is pure-Python and has no heavy deps.
Centralized attribute definitions to prevent drift between passes and host code.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional

# Attribute keys (single source of truth)
TT_CORE_GRID = "tt.core_grid"
TT_CORE_RANGES = "tt.core_ranges"          # List[CoreRange]
TT_CORE_RANGE = "tt.core_range"            # Back-compat single-range
TT_WORK_PARTITION = "tt.work_partition"    # Dict[str, List[WorkItem]]
TT_LAYOUT_DESC = "tt.layout_desc"          # Dict[str, Dict[str, Any]] per buffer

# Legacy attributes (for compatibility)
TT_GRID = "tt.grid"
TT_BLOCK_SHAPE = "tt.block_shape"
TT_START_TILE = "tt.start_tile"
TT_RUNTIME_ARGS = "tt.runtime_args"
TT_DEVICE_MESH = "tt.device_mesh"
TT_PARTITION_MODE = "partition_mode"

@dataclass(frozen=True)
class CoreRange:
    """Represents a rectangular range of cores."""
    start: Tuple[int, int]    # (sx, sy)
    extent: Tuple[int, int]   # (ex, ey)

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

def plan_dict(core_grid: Tuple[int, int],
              core_ranges: List[CoreRange],
              work_partition: Dict[str, List[WorkItem]],
              layouts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Build a plan dictionary for tt.plan.json."""
    return {
        "core_grid": list(core_grid),
        "core_ranges": [cr.to_json() for cr in core_ranges],
        "work_partition": {k: [wi.to_json() for wi in v] for k, v in work_partition.items()},
        "layouts": layouts,
    }
