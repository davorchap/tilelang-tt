"""Tenstorrent metadata annotation helpers."""

from __future__ import annotations

from typing import Any, Dict

from tilelang import tvm as tvm
from tvm.runtime import convert


def annotate_tt_layout(func: tvm.tir.PrimFunc, layout: Dict[str, Any]) -> tvm.tir.PrimFunc:
    """Attach user-specified layout metadata to a PrimFunc."""
    if not isinstance(func, tvm.tir.PrimFunc):
        raise TypeError("annotate_tt_layout expects a tvm.tir.PrimFunc")
    return func.with_attr("tt.user_layout", convert(layout))


def annotate_tt_schedule(func: tvm.tir.PrimFunc, schedule: Dict[str, Any]) -> tvm.tir.PrimFunc:
    """Attach user-specified schedule metadata to a PrimFunc."""
    if not isinstance(func, tvm.tir.PrimFunc):
        raise TypeError("annotate_tt_schedule expects a tvm.tir.PrimFunc")
    return func.with_attr("tt.user_schedule", convert(schedule))


__all__ = ["annotate_tt_layout", "annotate_tt_schedule"]
