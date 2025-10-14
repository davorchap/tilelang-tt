"""Python entry point for the GridToPersistentTT transform."""

from __future__ import annotations

from tilelang import tvm as tvm


def grid_to_persistent_tt(mod: tvm.IRModule) -> tvm.IRModule:
    """Transform grid-style kernels into Tenstorrent persistent loops."""
    pass_func = tvm.ffi.get_global_func("tl.transform.GridToPersistentTT")
    return pass_func()(mod)


__all__ = ["grid_to_persistent_tt"]
