"""Python entry point for InferDefaultTTSchedule."""

from __future__ import annotations

from tilelang import tvm as tvm


def infer_default_tt_schedule(mod: tvm.IRModule) -> tvm.IRModule:
    """Infer default Tenstorrent schedule metadata."""
    pass_func = tvm.ffi.get_global_func("tl.transform.InferDefaultTTSchedule")
    return pass_func()(mod)


__all__ = ["infer_default_tt_schedule"]
