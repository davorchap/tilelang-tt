"""Tenstorrent lowering entry point.

This module provides the Tenstorrent backend lowering implementation with proper
IR lowering pipeline following the TileLang architecture.
"""

from __future__ import annotations

from typing import List, Optional, Union, Tuple

from tvm.target import Target

from tilelang import tvm as tvm
from tilelang.engine.param import CompiledArtifact, KernelParam
from tilelang.tt import apply_tt_defaults


def LowerAndLegalizeTT(mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    """Frontend lowering phase - shared with CUDA backend.

    This will call the shared LowerAndLegalize pipeline in Task 2.
    For now, it's a placeholder that will be implemented properly.

    Args:
        mod: The TVM IRModule to lower
        target: The target (Tenstorrent)

    Returns:
        Lowered and legalized IRModule
    """
    # TODO Task 2: Import and call shared LowerAndLegalize from engine.phase
    # from tilelang.engine.phase import LowerAndLegalize
    # return LowerAndLegalize(mod, target)

    # Placeholder - just return mod for now
    return mod


def OptimizeForTargetTT(mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    """TT-specific optimization phase.

    This will integrate WS2/WS3 passes in Tasks 3-4.
    For now, it's a placeholder.

    Args:
        mod: The TVM IRModule to optimize
        target: The target (Tenstorrent)

    Returns:
        Optimized IRModule with TT-specific transforms
    """
    # TODO Task 3-4: Add WS2/WS3 passes
    # from tilelang.tt.passes import (
    #     infer_default_tt_schedule,
    #     infer_default_tt_shard,
    #     grid_to_persistent_tt,
    #     tt_shard_to_core_map,
    #     memory_space_lower_tt,
    #     tile_pad_tt,
    #     tensorize_tt,
    #     verify_tt_ir,
    # )
    # mod = infer_default_tt_schedule(mod)
    # mod = infer_default_tt_shard(mod)
    # mod = grid_to_persistent_tt(mod)
    # ... etc

    # Placeholder - just return mod for now
    return mod


def SplitTTKernels(mod: tvm.IRModule) -> Tuple[tvm.IRModule, tvm.IRModule]:
    """Split TT module into device kernels and host wrapper.

    This will implement 3-kernel architecture in Task 6.
    For now, it's a placeholder that returns mod for both device and host.

    Args:
        mod: The TVM IRModule to split

    Returns:
        Tuple of (device_mod, host_mod)
    """
    # TODO Task 6: Implement 3-kernel splitting
    # - Create reader/compute/writer kernels
    # - Create host wrapper

    # Placeholder - return same mod for both device and host
    return mod, mod


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

    This implements the proper IR lowering pipeline following the TileLang
    architecture:
    1. Frontend lowering (LowerAndLegalize) - shared with CUDA
    2. Target-specific optimization (OptimizeForTargetTT) - TT-specific
    3. Device splitting (SplitTTKernels) - TT 3-kernel architecture
    4. Codegen and runtime module creation

    Args:
        mod: The TVM IRModule to lower
        params: Optional list of kernel parameters
        target: The target (should be Tenstorrent target)
        target_host: Optional host target
        runtime_only: Whether to generate runtime-only code
        enable_host_codegen: Whether to enable host code generation
        enable_device_compile: Whether to enable device compilation

    Raises:
        ValueError: If the target is not a Tenstorrent target

    Returns:
        CompiledArtifact: Compiled kernel with host/device modules and source
    """
    from tilelang.engine.lower import get_target_kind
    from tilelang.utils.target import TENSTORRENT_TARGET

    # Validate that we're actually targeting Tenstorrent
    target_kind = get_target_kind(target)
    if target_kind != TENSTORRENT_TARGET:
        raise ValueError(f"Tenstorrent lowering called with invalid target: {target_kind}. "
                         f"Expected: {TENSTORRENT_TARGET}")

    # Convert target to Target object if it's a string
    if isinstance(target, str):
        target = tvm.target.Target(target)

    # === Phase 1: Apply TT defaults (WS1) ===
    # This ensures backward compatibility - GPU-style kernels can run on TT
    # with sensible defaults (contiguous schedule, row-major order, DRAM interleaved layout)
    mod = apply_tt_defaults(mod)

    # === Phase 2: Frontend lowering (shared with CUDA) ===
    # TODO Task 2: Uncomment when LowerAndLegalize is enabled
    # mod = LowerAndLegalizeTT(mod, target)

    # === Phase 3: TT-specific optimizations ===
    # TODO Task 3-4: Uncomment when OptimizeForTargetTT is implemented
    # mod = OptimizeForTargetTT(mod, target)

    # === Phase 4: Device splitting (3-kernel architecture) ===
    # TODO Task 6: Uncomment when SplitTTKernels is implemented
    # device_mod, host_mod = SplitTTKernels(mod)

    # Placeholder - use mod for both device and host
    device_mod = mod
    host_mod = mod

    # === Phase 5: Generate kernel source ===
    # For now, generate a placeholder source string
    # TODO: Integrate with emit_tt_artifacts when codegen is ready
    kernel_source = "// TT kernel placeholder - codegen integration pending\n"

    # === Phase 6: Create CompiledArtifact ===
    return CompiledArtifact(
        host_mod=host_mod,
        device_mod=device_mod,
        params=params or [],
        kernel_source=kernel_source,
        rt_mod=None,  # Runtime module will be created during JIT compilation
    )
