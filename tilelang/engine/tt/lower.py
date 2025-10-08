"""Tenstorrent lowering entry point.

This module provides the Tenstorrent backend lowering implementation with proper
IR lowering pipeline following the TileLang architecture.
"""

from __future__ import annotations

from typing import List, Optional, Union, Tuple

from tvm.target import Target

import tilelang
from tilelang import tvm as tvm
from tilelang.engine.param import CompiledArtifact, KernelParam
from tilelang.engine.phase import LowerAndLegalize
from tilelang.tt import apply_tt_defaults
from tilelang.tt.passes import (
    infer_default_tt_schedule,
    infer_default_tt_shard,
    grid_to_persistent_tt,
    tt_shard_to_core_map,
    memory_space_lower_tt,
    tile_pad_tt,
    tensorize_tt,
    verify_tt_ir,
)


def LowerAndLegalizeTT(mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    """Frontend lowering phase - shared with CUDA backend.

    This calls the shared LowerAndLegalize pipeline which applies:
    - LetInline
    - AddWrapperForSingleBufStore
    - InjectAssumes
    - Simplify
    - LayoutReducer
    - LayoutInference
    - LowerTileOp
    - LowerL2Persistent
    - LegalizeVectorizedLoop
    - LegalizeSafeMemoryAccess
    - LoopVectorizeDynamic

    Args:
        mod: The TVM IRModule to lower
        target: The target (Tenstorrent)

    Returns:
        Lowered and legalized IRModule
    """
    return LowerAndLegalize(mod, target)


def OptimizeForTargetTT(mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    """TT-specific optimization phase.

    This phase transforms TIR into TT-ready IR with:
    - Schedule inference (compute per-core tile ranges)
    - Sharding inference (DRAM layout descriptors)
    - Persistent loop structure (grid → persistent cores)
    - Core topology mapping (shard → core coordinates)
    - Circular buffer allocations (memory space lowering)
    - Tile padding (align buffers to 32×32 tiles)
    - Tensorization (map to TT intrinsics like matmul_tiles)
    - Common optimizations (buffer flattening, loop unrolling, etc.)
    - IR validation (verify TT constraints)

    This integrates WS2 and WS3 transformation passes, plus common
    backend-agnostic optimizations shared with CUDA.

    Args:
        mod: The TVM IRModule to optimize
        target: The target (Tenstorrent)

    Returns:
        Optimized IRModule with TT-specific transforms
    """
    # === WS2: Schedule and Sharding Inference ===
    # WS2 Phase 1: Schedule inference
    # Compute per-core tile ranges from grid dimensions
    mod = infer_default_tt_schedule(mod)

    # WS2 Phase 2: Sharding inference
    # Generate DRAM layout descriptors (tiled, interleaved)
    mod = infer_default_tt_shard(mod)

    # === WS3: TT-Specific TIR Transformations ===
    # WS3 Phase 1: Grid to persistent transformation
    # Transform GPU-style grid kernel to TT persistent loop model
    mod = grid_to_persistent_tt(mod)

    # WS3 Phase 2: Core topology and memory
    # Map shards to core coordinates (x, y) on NOC grid
    mod = tt_shard_to_core_map(mod)

    # Lower memory spaces (DRAM → L1 circular buffers)
    mod = memory_space_lower_tt(mod)

    # Pad buffers to tile-aligned dimensions (multiple of 32)
    mod = tile_pad_tt(mod)

    # WS3 Phase 3: Tensorization
    # Map high-level ops (gemm, etc.) to TT intrinsics (matmul_tiles, etc.)
    mod = tensorize_tt(mod)

    # === Common Optimizations (Shared with CUDA) ===
    # These are backend-agnostic passes that work on TIR

    # Flatten multi-dimensional buffers to 1D for easier codegen
    mod = tilelang.transform.FlattenBuffer()(mod)

    # Optimize index computation bitwidth (must come after FlattenBuffer)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)

    # Simplify expressions after transformations
    mod = tvm.tir.transform.Simplify()(mod)

    # Vectorize loops where possible
    # TT supports vectorization with 32-element tiles
    pass_ctx = tilelang.transform.get_pass_context()
    from tilelang.engine.phase import allow_vectorize
    mod = tilelang.transform.VectorizeLoop(
        enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)

    # Rewrite storage allocations for better memory usage
    # Should work with TT's circular buffer model
    mod = tilelang.transform.StorageRewrite()(mod)

    # Unroll loops for better performance
    mod = tvm.tir.transform.UnrollLoop()(mod)

    # Normalize split patterns after unrolling
    mod = tvm.tir.transform.RenormalizeSplitPattern()(mod)

    # Simplify again after optimizations
    mod = tvm.tir.transform.Simplify()(mod)

    # Remove no-op statements
    mod = tvm.tir.transform.RemoveNoOp()(mod)

    # Rewrite unsafe select operations
    mod = tvm.tir.transform.RewriteUnsafeSelect()(mod)

    # Hoist if-then-else out of loops
    mod = tvm.tir.transform.HoistIfThenElse()(mod)

    # === Verification ===
    # Verify memory accesses are correct
    mod = tvm.tir.transform.VerifyMemory()(mod)

    # TT-specific IR verification (grid size, CB counts, etc.)
    mod = verify_tt_ir(mod)

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

    # Convert target to Target object if it's a string and create composite target
    # This matches CUDA backend behavior and provides proper target context for passes
    if isinstance(target, str):
        target = tvm.target.Target(target)

    # Create composite target with host (matches CUDA backend)
    # This is needed for passes like LayoutInference that query target information
    if target_host is not None:
        if isinstance(target_host, str):
            target_host = tvm.target.Target(target_host)
        target = tvm.target.Target(target, target_host)
    else:
        # Use target directly if no host target specified
        target = tvm.target.Target(target)

    # === Phase 1: Apply TT defaults (WS1) ===
    # This ensures backward compatibility - GPU-style kernels can run on TT
    # with sensible defaults (contiguous schedule, row-major order, DRAM interleaved layout)
    mod = apply_tt_defaults(mod)

    # === Phase 2: Frontend lowering (shared with CUDA) ===
    # Wrap in target context for passes that need it (e.g., LayoutInference)
    with target:
        mod = LowerAndLegalizeTT(mod, target)

    # === Phase 3: TT-specific optimizations ===
    # Apply WS3 transformation passes
    mod = OptimizeForTargetTT(mod, target)

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
