"""Tenstorrent lowering entry point.

This module provides the Tenstorrent backend lowering implementation. It applies
default TT annotations and will integrate the full lowering pipeline in future
workstreams.
"""

from __future__ import annotations

from typing import List, Optional, Union

from tvm.target import Target

from tilelang import tvm as tvm
from tilelang.engine.param import CompiledArtifact, KernelParam
from tilelang.tt import apply_tt_defaults


def lower(
    mod: tvm.IRModule,
    params: Optional[List[KernelParam]],
    target: Union[str, Target],
    target_host: Optional[Union[str, Target]],
    *,
    runtime_only: bool,
    enable_host_codegen: bool,
    enable_device_compile: bool,
) -> CompiledArtifact:
    """Lower the given module for the Tenstorrent backend.

    This implementation validates the target, applies default TT annotations,
    and prepares the module for lowering. The full lowering pipeline (passes,
    codegen, runtime) will be implemented in future workstreams.

    Args:
        mod: The TVM IRModule to lower
        params: Optional list of kernel parameters (unused in current implementation)
        target: The target (should be Tenstorrent target)
        target_host: Optional host target (unused in current implementation)
        runtime_only: Whether to generate runtime-only code (unused in current implementation)
        enable_host_codegen: Whether to enable host code generation (unused in current implementation)
        enable_device_compile: Whether to enable device compilation (unused in current implementation)

    Raises:
        ValueError: If the target is not a Tenstorrent target
        NotImplementedError: The full lowering pipeline is not yet implemented

    Returns:
        CompiledArtifact: Will be returned when full lowering pipeline is complete
    """
    from tilelang.engine.lower import get_target_kind
    from tilelang.utils.target import TENSTORRENT_TARGET

    # Validate that we're actually targeting Tenstorrent
    target_kind = get_target_kind(target)
    if target_kind != TENSTORRENT_TARGET:
        raise ValueError(f"Tenstorrent lowering called with invalid target: {target_kind}. "
                         f"Expected: {TENSTORRENT_TARGET}")

    # Apply default TT annotations if not already present
    # This ensures backward compatibility - GPU-style kernels can run on TT
    # with sensible defaults (contiguous schedule, row-major order, DRAM interleaved layout)
    mod = apply_tt_defaults(mod)

    # Unused parameters in this stub implementation - will be used in full implementation
    _ = params
    _ = target_host
    _ = runtime_only
    _ = enable_host_codegen
    _ = enable_device_compile

    # TODO: Implement full lowering pipeline:
    # 1. Run TT-specific passes (GridToPersistentTT, TTShardToCoreMap, etc.)
    # 2. Generate TT kernels (reader/compute/writer)
    # 3. Generate host code
    # 4. Return CompiledArtifact
    raise NotImplementedError(
        "Tenstorrent backend lowering is not yet implemented. "
        "Default annotations have been applied to the module, but the full "
        "lowering pipeline (passes, codegen, runtime) will be added in future workstreams.")
