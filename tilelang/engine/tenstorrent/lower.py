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
from tilelang.tenstorrent import apply_tt_defaults
from tilelang.tenstorrent.passes import run_pipeline


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
    """TT-specific optimization phase using the new metadata-driven pipeline.

    This phase transforms TIR into TT-ready IR using the new 5-pass pipeline:
    1. InferTTLayout - Infer buffer layouts and metadata
    2. PropagateTTLayout - Propagate and normalize layout info
    3. TTTilesToCoreMap - Compute core mapping and work partition
    4. LowerTTTileIntrinsics - Lower tile ops to device intrinsics
    5. GridToPersistentTT - Final lowering to persistent kernels

    The pipeline also emits a runtime plan (tt.plan.json) for host-device coordination.

    Args:
        mod: The TVM IRModule to optimize
        target: The target (Tenstorrent)

    Returns:
        Optimized IRModule with TT-specific transforms
    """
    # Import the new pipeline

    # Extract device type from target if possible
    target_device = "grayskull"  # Default
    if hasattr(target, "attrs") and "device" in target.attrs:
        target_device = target.attrs["device"]

    # Run the new metadata-driven pipeline
    mod = run_pipeline(
        mod,
        plan_path="tt.plan.json",
        target_device=target_device,
        partition_strategy="row_major",
        enable_double_buffer=True,
        enable_prefetch=True,
        verbose=False,
    )

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

    # Note: TT-specific IR verification is now integrated into the pipeline passes

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
    import tilelang.tenstorrent as tt

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
