"""Python entry point for InferDefaultTTShard."""

from __future__ import annotations

from tilelang import tvm as tvm


def infer_default_tt_shard(mod: tvm.IRModule) -> tvm.IRModule:
    """Infer default Tenstorrent sharding metadata."""
    pass_func = tvm.ffi.get_global_func("tl.transform.InferDefaultTTShard")
    return pass_func()(mod)


__all__ = ["infer_default_tt_shard"]
