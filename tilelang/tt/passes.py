"""Tenstorrent backend compiler passes.

This module provides Python bindings for TT-specific TVM passes that inject
schedule and sharding metadata into IRModules.
"""

from __future__ import annotations

from typing import Optional

from tilelang import tvm as tvm


def infer_default_tt_schedule(mod: tvm.IRModule) -> tvm.IRModule:
    """Infer default Tenstorrent schedule metadata.

    This pass analyzes the kernel grid dimensions (from T.Kernel) and computes
    contiguous per-core tile ranges for the Tenstorrent backend. It attaches
    schedule metadata including:

    - Number of tiles (grid_x * grid_y * grid_z)
    - Grid dimensions
    - Per-core tile assignments (start_tile, tile_count)
    - Number of active cores

    The pass implements a contiguous, row-major tile distribution strategy
    where tiles are assigned sequentially to cores.

    Args:
        mod: The TVM IRModule to process

    Returns:
        A new IRModule with schedule metadata attached to PrimFuncs

    Example:
        >>> import tilelang.language as T
        >>> from tilelang.tt import apply_tt_defaults, infer_default_tt_schedule
        >>>
        >>> @T.prim_func
        >>> def gemm(A, B, C):
        >>>     with T.Kernel(8, 8) as (bx, by):
        >>>         # ... kernel implementation ...
        >>>         pass
        >>>
        >>> mod = tvm.IRModule.from_expr(gemm)
        >>> mod = apply_tt_defaults(mod)  # TT defaults stage: Add default annotations
        >>> mod = infer_default_tt_schedule(mod)  # metadata inference stage: Infer schedule
    """
    # Call C++ pass via FFI
    pass_func = tvm.ffi.get_global_func("tl.transform.InferDefaultTTSchedule")
    return pass_func()(mod)


def infer_default_tt_shard(mod: tvm.IRModule) -> tvm.IRModule:
    """Infer default Tenstorrent sharding metadata.

    This pass analyzes buffer parameters and generates DRAM interleaved layout
    descriptors. It attaches sharding metadata including:

    - Layout type (dram_interleaved)
    - Tile shape (32×32)
    - Number of tiles per dimension
    - Padding requirements for non-tile-aligned dimensions
    - Padded shape (if padding needed)

    The pass detects buffers with dimensions that are not multiples of 32 and
    marks them for padding. Actual padding insertion and TensorAccessor
    configuration are deferred to later passes (persistent transform stage/Artifact Generation stage).

    Args:
        mod: The TVM IRModule to process

    Returns:
        A new IRModule with sharding metadata attached to buffer parameters

    Example:
        >>> import tilelang.language as T
        >>> from tilelang.tt import apply_tt_defaults, infer_default_tt_shard
        >>>
        >>> @T.prim_func
        >>> def gemm(A: T.Buffer[(256, 256), "float16"],
        >>>          B: T.Buffer[(256, 256), "float16"],
        >>>          C: T.Buffer[(256, 256), "float16"]):
        >>>     # ... kernel implementation ...
        >>>     pass
        >>>
        >>> mod = tvm.IRModule.from_expr(gemm)
        >>> mod = apply_tt_defaults(mod)  # TT defaults stage: Add default annotations
        >>> mod = infer_default_tt_shard(mod)  # metadata inference stage: Infer sharding
    """
    # Call C++ pass via FFI
    pass_func = tvm.ffi.get_global_func("tl.transform.InferDefaultTTShard")
    return pass_func()(mod)


def apply_tt_metadata_passes(mod: tvm.IRModule) -> tvm.IRModule:
    """Apply all Workstream 2 passes (schedule + sharding inference).

    This is a convenience function that applies both schedule and sharding
    inference passes in the correct order.

    Args:
        mod: The TVM IRModule to process (should already have TT defaults stage defaults)

    Returns:
        A new IRModule with both schedule and sharding metadata

    Example:
        >>> from tilelang.tt import apply_tt_defaults, apply_tt_metadata_passes
        >>>
        >>> mod = create_tilelang_kernel()
        >>> mod = apply_tt_defaults(mod)  # TT defaults stage
        >>> mod = apply_tt_metadata_passes(mod)  # metadata inference stage
    """
    mod = infer_default_tt_schedule(mod)
    mod = infer_default_tt_shard(mod)
    return mod


# ============================================================================
# Workstream 3: TIR Transform Pipeline
# ============================================================================


def grid_to_persistent_tt(mod: tvm.IRModule) -> tvm.IRModule:
    """Transform grid-style kernel to persistent per-core loop.

    This pass converts GPU-style grid kernels to Tenstorrent's persistent
    execution model. Each core runs a persistent loop iterating over its
    assigned tiles, recovering block indices from the static schedule. It also
    appends scalar parameters (`tt_start_tile`, `tt_tile_count`) and emits the
    `tt_runtime_args` map describing iteration order and grid shape.

    Args:
        mod: The TVM IRModule to process (should have metadata inference stage schedule metadata)

    Returns:
        A new IRModule with persistent loop structure

    Example:
        >>> from tilelang.tt import apply_tt_metadata_passes, grid_to_persistent_tt
        >>>
        >>> mod = create_tilelang_kernel()
        >>> mod = apply_tt_defaults(mod)  # TT defaults stage
        >>> mod = apply_tt_metadata_passes(mod)  # metadata inference stage
        >>> mod = grid_to_persistent_tt(mod)  # persistent transform stage
    """
    pass_func = tvm.ffi.get_global_func("tl.transform.GridToPersistentTT")
    return pass_func()(mod)


def tt_tiles_to_core_map(mod: tvm.IRModule) -> tvm.IRModule:
    """Map tile assignments to physical core coordinates.

    This pass converts logical tile-to-core assignments from metadata inference stage into physical
    CoreRangeSet topology for Tenstorrent devices. It generates:

    - tt_core_ranges: Physical core topology as [start_x, start_y, end_x, end_y, start_tile, count]
    - tt_core_runtime_args: Per-core runtime args as [start_tile, num_tiles]

    For Grayskull/Wormhole 8×8 core grids, uses row-major layout where
    core_id maps to (x, y) = (core_id % 8, core_id / 8).

    Args:
        mod: The TVM IRModule to process (should have metadata inference stage tt_tiles_per_core metadata)

    Returns:
        A new IRModule with physical core topology metadata

    Example:
        >>> from tilelang.tt import apply_tt_metadata_passes, tt_tiles_to_core_map
        >>>
        >>> mod = create_tilelang_kernel()
        >>> mod = apply_tt_defaults(mod)  # TT defaults stage
        >>> mod = apply_tt_metadata_passes(mod)  # metadata inference stage
        >>> mod = tt_tiles_to_core_map(mod)  # persistent transform stage Phase 2
    """
    pass_func = tvm.ffi.get_global_func("tl.transform.TTTilesToCoreMap")
    return pass_func()(mod)


def memory_space_lower_tt(mod: tvm.IRModule) -> tvm.IRModule:
    """Record circular-buffer metadata for tile-local buffers.

    This pass scans `DeclBuffer` nodes created by `T.alloc_fragment`, heuristically
    identifies tile-sized temporaries, assigns circular-buffer IDs, and emits
    configuration metadata (`tt_circular_buffers`, `tt_num_cbs`) for codegen to consume.
    The underlying buffers are left unchanged; TT codegen materialises the actual CB
    allocations when emitting C++.

    Args:
        mod: The TVM IRModule to process (should have TT defaults stage TT defaults)

    Returns:
        A new IRModule with TT circular-buffer metadata attached
    """
    pass_func = tvm.ffi.get_global_func("tl.transform.MemorySpaceLowerTT")
    return pass_func()(mod)


def tile_pad_tt(mod: tvm.IRModule) -> tvm.IRModule:
    """Attach padding metadata for non-tile-aligned buffers.

    The pass consults metadata inference stage shard metadata (`tt_buffer_*_needs_padding`) and, for each
    buffer that requires padding, records the original shape, padded shape, and
    padding amounts under `tt_padding_info`. The IR itself is unchanged—codegen reads
    the metadata to handle edge tiles.

    Args:
        mod: The TVM IRModule to process (should have metadata inference stage padding flags)

    Returns:
        A new IRModule with padding metadata for codegen
    """
    pass_func = tvm.ffi.get_global_func("tl.transform.TilePadTT")
    return pass_func()(mod)


def tensorize_tt(mod: tvm.IRModule) -> tvm.IRModule:
    """Tag frontend GEMM markers with TT matmul annotations.

    The current implementation wraps `AttrStmt` nodes labelled `"pragma_gemm"`,
    `"tl.gemm"`, or `"gemm_operation"` with a TT-specific `"tt.matmul_intrinsic"`
    attribute and tracks the number of matmul regions (`tt_num_matmuls`,
    `tt_has_tensorize`). More advanced pattern detection (manual loops, element-wise
    ops) is still TODO.

    Args:
        mod: The TVM IRModule to process (should contain GEMM pragmas)

    Returns:
        A new IRModule with TT matmul annotations
    """
    pass_func = tvm.ffi.get_global_func("tl.transform.TensorizeTT")
    return pass_func()(mod)


def verify_tt_ir(mod: tvm.IRModule) -> tvm.IRModule:
    """Validate TT-transformed IR metadata.

    This pass performs comprehensive validation of transformed IR to ensure it's
    ready for Tenstorrent codegen. It verifies:

    - TT defaults stage: tt_schedule_policy, tt_layout_type, tt_tile_* dimensions
    - metadata inference stage: tt_grid_*, tt_num_tiles, tt_tiles_per_core, tt_num_cores
    - persistent transform stage: tt_persistent_loop, tt_core_ranges, tt_circular_buffers, tt_padding_info

    The pass logs errors and warnings but does not modify the IR. It attaches
    validation results as metadata:

    - tt_ir_validated: Bool indicating if validation passed
    - tt_validation_error_count: Number of errors found
    - tt_validation_warning_count: Number of warnings found

    Args:
        mod: The TVM IRModule to validate (should have TT defaults stage-3 metadata)

    Returns:
        A new IRModule with validation result metadata attached

    Example:
        >>> from tilelang.tt import apply_tt_transform_passes, verify_tt_ir
        >>>
        >>> mod = create_tilelang_kernel()
        >>> mod = apply_tt_defaults(mod)  # TT defaults stage
        >>> mod = apply_tt_metadata_passes(mod)  # metadata inference stage
        >>> mod = apply_tt_transform_passes(mod)  # persistent transform stage (includes verify_tt_ir)
    """
    pass_func = tvm.ffi.get_global_func("tl.transform.VerifyTTIR")
    return pass_func()(mod)


def apply_tt_transform_passes(mod: tvm.IRModule) -> tvm.IRModule:
    """Apply all Workstream 3 TIR transform passes.

    This is a convenience function that applies all persistent transform stage transforms in the
    correct order to produce TT-ready IR.

    Args:
        mod: The TVM IRModule to process (should have metadata inference stage metadata)

    Returns:
        A new IRModule with transformed TIR ready for codegen

    Example:
        >>> from tilelang.tt import apply_tt_defaults, apply_tt_metadata_passes, apply_tt_transform_passes
        >>>
        >>> mod = create_tilelang_kernel()
        >>> mod = apply_tt_defaults(mod)  # TT defaults stage
        >>> mod = apply_tt_metadata_passes(mod)  # metadata inference stage
        >>> mod = apply_tt_transform_passes(mod)  # persistent transform stage
    """
    # persistent transform stage Transform Pipeline
    mod = grid_to_persistent_tt(mod)
    mod = tt_tiles_to_core_map(mod)
    mod = memory_space_lower_tt(mod)
    mod = tile_pad_tt(mod)
    mod = tensorize_tt(mod)
    mod = verify_tt_ir(mod)

    return mod
