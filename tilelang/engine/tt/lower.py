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
    apply_layout_aware_metadata_passes,
    grid_to_persistent_tt,
    tt_tiles_to_core_map,
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

    This integrates Metadata Inference stage and Persistent Transform stage transformation passes, plus common
    backend-agnostic optimizations shared with CUDA.

    Args:
        mod: The TVM IRModule to optimize
        target: The target (Tenstorrent)

    Returns:
        Optimized IRModule with TT-specific transforms
    """
    # === Metadata Inference stage: Schedule and Sharding Inference ===
    # Metadata Inference stage Phase 1: Schedule inference
    # Compute per-core tile ranges from grid dimensions
    mod = infer_default_tt_schedule(mod)

    # Metadata Inference stage Phase 2: Sharding inference
    # Generate DRAM layout descriptors (tiled, interleaved)
    mod = infer_default_tt_shard(mod)

    # Layout-aware metadata (buffer layouts, partitioning, runtime arg schema)
    mod = apply_layout_aware_metadata_passes(mod)

    # === Persistent Transform stage: TT-Specific TIR Transformations ===
    # Persistent Transform stage: Grid to persistent transformation
    # Transform GPU-style grid kernel to TT persistent loop model
    mod = grid_to_persistent_tt(mod)

    # Persistent Transform stage: Core topology and memory
    # Map scheduled tiles to core coordinates (x, y) on NOC grid
    mod = tt_tiles_to_core_map(mod)

    # Lower memory spaces (DRAM → L1 circular buffers)
    mod = memory_space_lower_tt(mod)

    # Pad buffers to tile-aligned dimensions (multiple of 32)
    mod = tile_pad_tt(mod)

    # Persistent Transform stage Phase 3: Tensorization
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
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)

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
    """Prepare TT module for 3-kernel codegen (reader/compute/writer).

    Unlike CUDA's SplitHostDevice which splits into separate IR functions,
    TT's 3-kernel architecture is implemented during **codegen** (Artifact Generation stage-6).
    This function prepares the module by:
    1. Annotating device regions (marks device code)
    2. Keeping the module intact for codegen to process

    The actual split into reader/compute/writer happens in:
    - TTReaderCodegenVisitor (Reader/Writer Generation stage): Generates reader kernel
    - TTComputeCodegenVisitor (Artifact Generation stage): Generates compute kernel
    - TTWriterCodegenVisitor (Reader/Writer Generation stage): Generates writer kernel
    - EmitTTHostProgram (Host Program stage): Generates host wrapper

    Args:
        mod: The TVM IRModule to prepare for codegen

    Returns:
        Tuple of (device_mod, host_mod):
        - device_mod: Module with annotated device regions for codegen
        - host_mod: Same module (host wrapper generated during codegen)

    See Also:
        - docs/tenstorrent/workstream4/Artifact Generation stage_STATUS.md (compute kernel)
        - docs/tenstorrent/workstream5/Reader/Writer Generation stage_STATUS.md (reader/writer kernels)
        - docs/tenstorrent/workstream6/Host Program stage_STATUS.md (host program)
    """
    # Annotate device regions (marks code that runs on device)
    # This is similar to what CUDA does before SplitHostDevice
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)

    # NOTE: We do NOT call SplitHostDevice here!
    # TT's 3-kernel split happens during codegen, not IR transformation.
    # The codegen visitors (reader/compute/writer) will generate 3 separate
    # kernel files from this single module.

    # Return the same module for both device and host
    # - device_mod: Will be processed by codegen visitors (Artifact Generation stage-6)
    # - host_mod: Will be used to generate host wrapper (Host Program stage)
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

    # === Phase 1: Apply TT defaults (TT Defaults stage) ===
    # This ensures backward compatibility - GPU-style kernels can run on TT
    # with sensible defaults (contiguous schedule, row-major order, DRAM interleaved layout)
    mod = apply_tt_defaults(mod)

    # === Phase 2: Frontend lowering (shared with CUDA) ===
    # Wrap in target context for passes that need it (e.g., LayoutInference)
    with target:
        mod = LowerAndLegalizeTT(mod, target)

    # === Phase 3: TT-specific optimizations ===
    # Apply Metadata Inference stage/Persistent Transform stage transformation passes + common optimizations
    mod = OptimizeForTargetTT(mod, target)

    # === Phase 4: Device splitting (3-kernel architecture) ===
    # Annotate device regions and prepare for 3-kernel codegen
    # Note: Actual split into reader/compute/writer happens during codegen (Artifact Generation stage-6)
    device_mod, host_mod = SplitTTKernels(mod)

    # === Phase 5: Generate kernel source ===
    # Use emit_tt_artifacts to generate reader/compute/writer kernels
    import tilelang.tt as tt
    artifacts = tt.emit_tt_artifacts(device_mod)

    # Convert artifacts dict to JSON string for kernel_source field
    import json
    kernel_source = json.dumps(artifacts)

    # === Phase 6: Create CompiledArtifact ===
    return CompiledArtifact(
        host_mod=host_mod,
        device_mod=device_mod,
        params=params or [],
        kernel_source=kernel_source,
        rt_mod=None,  # Runtime module will be created during JIT compilation
    )
