"""Tenstorrent metadata annotation helpers."""

from __future__ import annotations

from typing import Any, Dict

from tilelang import tvm as tvm
from .attrs import TT_USER_LAYOUT, TT_USER_SCHEDULE


def annotate_tt_layout(func: tvm.tir.PrimFunc, layout: Dict[str, Any]) -> tvm.tir.PrimFunc:
    """Attach user-specified layout metadata to a PrimFunc."""
    if not isinstance(func, tvm.tir.PrimFunc):
        raise TypeError("annotate_tt_layout expects a tvm.tir.PrimFunc")
    # Use tvm.runtime.convert when available, otherwise pass dict directly
    # TVM's with_attr should handle dict conversion internally
    return func.with_attr(TT_USER_LAYOUT, layout)


def annotate_tt_schedule(func: tvm.tir.PrimFunc, schedule: Dict[str, Any]) -> tvm.tir.PrimFunc:
    """Attach user-specified schedule metadata to a PrimFunc."""
    if not isinstance(func, tvm.tir.PrimFunc):
        raise TypeError("annotate_tt_schedule expects a tvm.tir.PrimFunc")
    # Use tvm.runtime.convert when available, otherwise pass dict directly
    # TVM's with_attr should handle dict conversion internally
    return func.with_attr(TT_USER_SCHEDULE, schedule)


__all__ = ["annotate_tt_layout", "annotate_tt_schedule"]
