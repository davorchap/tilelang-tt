"""Python entry point for VerifyTTIR."""

from __future__ import annotations

from tilelang import tvm as tvm


def verify_tt_ir(mod: tvm.IRModule) -> tvm.IRModule:
    """Validate Tenstorrent-transformed IR metadata."""
    pass_func = tvm.ffi.get_global_func("tl.transform.VerifyTTIR")
    return pass_func()(mod)


__all__ = ["verify_tt_ir"]
