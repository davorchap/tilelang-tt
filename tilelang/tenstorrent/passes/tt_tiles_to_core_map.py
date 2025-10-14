"""Python entry point for TTTilesToCoreMap."""

from __future__ import annotations

from tilelang import tvm as tvm


def tt_tiles_to_core_map(mod: tvm.IRModule) -> tvm.IRModule:
    """Map tile assignments to physical Tenstorrent core coordinates."""
    pass_func = tvm.ffi.get_global_func("tl.transform.TTTilesToCoreMap")
    return pass_func()(mod)


__all__ = ["tt_tiles_to_core_map"]
