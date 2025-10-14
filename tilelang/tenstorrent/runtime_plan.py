"""
tt.plan.json emitter and runtime plan utilities.
Reads attributes from a PrimFunc and writes a single JSON file that serves
as the single source of truth for host/device coordination.
"""

from __future__ import annotations
import json
from typing import Any, Dict, List
import logging

try:
    import tvm
    from tvm import tir
except ImportError:  # pragma: no cover
    tvm = None
    tir = None

from .attrs import (
    TT_CORE_GRID,
    TT_CORE_RANGES,
    TT_CORE_RANGE,
    TT_WORK_PARTITION,
    TT_LAYOUT_DESC,
    CoreRange,
    WorkItem,
    plan_dict,
)

logger = logging.getLogger(__name__)


def _as_py(obj):
    """Convert TVM objects to Python primitives."""
    if hasattr(obj, "value"):
        return int(obj)
    if isinstance(obj, (list, tuple)):
        return [_as_py(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _as_py(v) for k, v in obj.items()}

    # Handle TVM container types - check by class name
    if tvm and hasattr(obj, "__class__"):
        class_name = obj.__class__.__name__
        module_name = (
            obj.__class__.__module__ if hasattr(obj.__class__, "__module__") else ""
        )

        # Map type - has items() method
        if "Map" in class_name and hasattr(obj, "items"):
            return {str(k): _as_py(v) for k, v in obj.items()}

        # Array type - can iterate
        if "Array" in class_name:
            try:
                return [_as_py(x) for x in obj]
            except (TypeError, AttributeError):
                pass

        # String type
        if "String" in class_name:
            return str(obj)

    # Handle dataclass instances (CoreRange, WorkItem)
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _as_py(getattr(obj, k)) for k in obj.__dataclass_fields__}

    return obj


def _read_core_ranges(attrs) -> List[CoreRange]:
    """Read core ranges from attributes, handling both new and legacy formats."""
    if TT_CORE_RANGES in attrs:
        arr = attrs[TT_CORE_RANGES]
        return [
            CoreRange(tuple(_as_py(x["start"])), tuple(_as_py(x["extent"])))
            for x in arr
        ]
    if TT_CORE_RANGE in attrs:
        x = attrs[TT_CORE_RANGE]
        return [CoreRange(tuple(_as_py(x["start"])), tuple(_as_py(x["extent"])))]
    return []


def extract_runtime_plan(func: "tir.PrimFunc") -> Dict[str, Any]:
    """Extract runtime plan from a PrimFunc's attributes."""
    if tvm is None:
        return {}

    attrs = func.attrs or {}

    # Extract core grid
    grid = tuple(_as_py(attrs.get(TT_CORE_GRID, [1, 1])))

    # Extract core ranges
    ranges = _read_core_ranges(attrs)
    if not ranges:
        # Default to full grid if no ranges specified
        ranges = [CoreRange((0, 0), grid)]

    # Extract work partition
    work_raw = attrs.get(TT_WORK_PARTITION, {})
    work = {}
    for k, v in work_raw.items():
        work_items = []
        for wi in v:
            wi_dict = _as_py(wi)
            work_items.append(WorkItem(**wi_dict))
        work[k] = work_items

    # Extract layout descriptors
    layouts = _as_py(attrs.get(TT_LAYOUT_DESC, {}))

    return plan_dict(
        core_grid=grid, core_ranges=ranges, work_partition=work, layouts=layouts
    )


def emit_tt_plan(
    func: "tir.PrimFunc", out_path: str = "tt.plan.json"
) -> Dict[str, Any]:
    """
    Emit a tt.plan.json file from a PrimFunc's attributes.
    This serves as the single source of truth for host/device coordination.
    """
    plan = extract_runtime_plan(func)

    # Write to JSON file
    with open(out_path, "w") as f:
        json.dump(plan, f, indent=2)

    logger.info(f"Emitted TT runtime plan to {out_path}")
    return plan


def load_tt_plan(path: str) -> Dict[str, Any]:
    """Load a tt.plan.json file."""
    with open(path, "r") as f:
        return json.load(f)


def validate_plan(plan: Dict[str, Any]) -> List[str]:
    """
    Validate a runtime plan for consistency.
    Returns a list of validation errors (empty if valid).
    """
    errors = []

    # Check required fields
    required_fields = ["core_grid", "core_ranges", "work_partition", "layouts"]
    for field in required_fields:
        if field not in plan:
            errors.append(f"Missing required field: {field}")

    if "core_grid" in plan:
        grid = plan["core_grid"]
        if not isinstance(grid, list) or len(grid) != 2:
            errors.append("core_grid must be a list of 2 integers")
        else:
            gx, gy = grid

            # Validate core ranges
            if "core_ranges" in plan:
                for i, cr in enumerate(plan["core_ranges"]):
                    if "start" not in cr or "extent" not in cr:
                        errors.append(f"core_ranges[{i}] missing start or extent")
                    else:
                        sx, sy = cr["start"]
                        ex, ey = cr["extent"]
                        if sx < 0 or sy < 0:
                            errors.append(f"core_ranges[{i}] has negative start")
                        if sx + ex > gx or sy + ey > gy:
                            errors.append(f"core_ranges[{i}] exceeds grid bounds")

    # Validate work partition keys match core coordinates
    if "work_partition" in plan:
        for key in plan["work_partition"]:
            if not key.startswith("(") or not key.endswith(")"):
                errors.append(f"Invalid work_partition key format: {key}")

    return errors
