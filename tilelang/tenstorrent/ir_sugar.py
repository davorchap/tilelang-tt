"""
IR sugar and helpers for annotating PrimFuncs with Tenstorrent-specific metadata.
Provides a light scaffolding for attaching TT attributes without deep TVMScript parser changes.
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Tuple

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
    TT_GRID,
    TT_BLOCK_SHAPE,
    TT_START_TILE,
    TT_RUNTIME_ARGS,
)


def with_core_grid(func, gx: int, gy: int):
    """Attach tt.core_grid=(gx,gy) to a PrimFunc."""
    if tvm is None:
        return func
    return func.with_attr(TT_CORE_GRID, tvm.runtime.convert([int(gx), int(gy)]))


def with_core_ranges(func, ranges: Iterable[CoreRange]):
    """Attach tt.core_ranges=[...]. Also mirrors single-range key for back-compat if len==1."""
    if tvm is None:
        return func
    arr = [tvm.runtime.convert({"start": list(r.start), "extent": list(r.extent)}) for r in ranges]
    func = func.with_attr(TT_CORE_RANGES, tvm.runtime.convert(arr))
    if len(arr) == 1:
        func = func.with_attr(TT_CORE_RANGE, arr[0])
    return func


def with_work_partition(func, work: Dict[str, List[WorkItem]]):
    """Attach tt.work_partition mapping string key '(cx,cy)' -> list of WorkItem."""
    if tvm is None:
        return func
    mp = {k: [tvm.runtime.convert(wi.to_json()) for wi in v] for k, v in work.items()}
    return func.with_attr(TT_WORK_PARTITION, tvm.runtime.convert(mp))


def with_layout_desc(func, layouts: Dict[str, Dict[str, Any]]):
    """Attach tt.layout_desc per buffer name."""
    if tvm is None:
        return func
    return func.with_attr(TT_LAYOUT_DESC, tvm.runtime.convert(layouts))


# Legacy compatibility helpers
def with_tt_grid(func, grid: Tuple[int, int]):
    """Legacy: attach tt.grid attribute."""
    if tvm is None:
        return func
    return func.with_attr(TT_GRID, tvm.runtime.convert(list(grid)))


def with_tt_block_shape(func, shape: Tuple[int, int]):
    """Legacy: attach tt.block_shape attribute."""
    if tvm is None:
        return func
    return func.with_attr(TT_BLOCK_SHAPE, tvm.runtime.convert(list(shape)))


def with_tt_start_tile(func, tile: Tuple[int, int]):
    """Legacy: attach tt.start_tile attribute."""
    if tvm is None:
        return func
    return func.with_attr(TT_START_TILE, tvm.runtime.convert(list(tile)))


def with_tt_runtime_args(func, args: Dict[str, Any]):
    """Legacy: attach tt.runtime_args attribute."""
    if tvm is None:
        return func
    return func.with_attr(TT_RUNTIME_ARGS, tvm.runtime.convert(args))


# Optional: very light sugar resembling launch_core (no TVMScript parser mods).
# This simply returns a Var and records intent via attributes on the function later.
def make_core_var(name: str = "coreIdx.x", dtype: str = "int32"):
    """Create a core index variable for use in mid-level IR."""
    if tvm is None:
        return (None, name)
    return tir.Var(name, dtype)
