"""Python entry point for MemorySpaceLowerTT."""

from __future__ import annotations

from tilelang import tvm as tvm


def memory_space_lower_tt(mod: tvm.IRModule) -> tvm.IRModule:
    """Record circular-buffer metadata for tile-local buffers."""
    pass_func = tvm.ffi.get_global_func("tl.transform.MemorySpaceLowerTT")
    return pass_func()(mod)


__all__ = ["memory_space_lower_tt"]
