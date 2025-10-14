"""Python entry point for TilePadTT."""

from __future__ import annotations

from tilelang import tvm as tvm


def tile_pad_tt(mod: tvm.IRModule) -> tvm.IRModule:
    """Attach padding metadata for non-tile-aligned buffers."""
    pass_func = tvm.ffi.get_global_func("tl.transform.TilePadTT")
    return pass_func()(mod)


__all__ = ["tile_pad_tt"]
