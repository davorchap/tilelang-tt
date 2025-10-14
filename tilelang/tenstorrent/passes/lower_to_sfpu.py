"""Python entry point for the LowerToSFPU transform."""

from __future__ import annotations

from tilelang import tvm as tvm


def lower_to_sfpu(mod: tvm.IRModule) -> tvm.IRModule:
    """Lower threadIdx constructs to SFPU/SIMD operations.

    For now, this is a placeholder that errors out when threadIdx is detected,
    since SFPU lowering is not yet implemented.
    """
    pass_func = tvm.ffi.get_global_func("tl.transform.LowerToSFPU")
    return pass_func()(mod)


__all__ = ["lower_to_sfpu"]
