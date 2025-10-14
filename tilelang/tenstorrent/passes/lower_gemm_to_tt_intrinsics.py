"""Python entry point for LowerGemmToTTIntrinsics."""

from __future__ import annotations

from tilelang import tvm as tvm


def lower_gemm_to_tt_intrinsics(mod: tvm.IRModule) -> tvm.IRModule:
    """Lower `tl.gemm` intrinsics into Tenstorrent tile intrinsics."""
    pass_func = tvm.ffi.get_global_func("tl.transform.LowerGemmToTTIntrinsics")
    return pass_func()(mod)


__all__ = ["lower_gemm_to_tt_intrinsics"]
