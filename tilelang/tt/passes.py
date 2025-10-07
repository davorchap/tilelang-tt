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
    - Per-core tile assignments (start_id, count)
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
        >>> mod = apply_tt_defaults(mod)  # WS1: Add default annotations
        >>> mod = infer_default_tt_schedule(mod)  # WS2: Infer schedule
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
    configuration are deferred to later passes (WS3/WS4).

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
        >>> mod = apply_tt_defaults(mod)  # WS1: Add default annotations
        >>> mod = infer_default_tt_shard(mod)  # WS2: Infer sharding
    """
    # Call C++ pass via FFI
    pass_func = tvm.ffi.get_global_func("tl.transform.InferDefaultTTShard")
    return pass_func()(mod)


def apply_ws2_passes(mod: tvm.IRModule) -> tvm.IRModule:
    """Apply all Workstream 2 passes (schedule + sharding inference).

    This is a convenience function that applies both schedule and sharding
    inference passes in the correct order.

    Args:
        mod: The TVM IRModule to process (should already have WS1 defaults)

    Returns:
        A new IRModule with both schedule and sharding metadata

    Example:
        >>> from tilelang.tt import apply_tt_defaults, apply_ws2_passes
        >>>
        >>> mod = create_tilelang_kernel()
        >>> mod = apply_tt_defaults(mod)  # WS1
        >>> mod = apply_ws2_passes(mod)  # WS2
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
    assigned tiles, recovering block indices from the static schedule.

    Args:
        mod: The TVM IRModule to process (should have WS2 schedule metadata)

    Returns:
        A new IRModule with persistent loop structure

    Example:
        >>> from tilelang.tt import apply_ws2_passes, grid_to_persistent_tt
        >>>
        >>> mod = create_tilelang_kernel()
        >>> mod = apply_tt_defaults(mod)  # WS1
        >>> mod = apply_ws2_passes(mod)  # WS2
        >>> mod = grid_to_persistent_tt(mod)  # WS3
    """
    pass_func = tvm.ffi.get_global_func("tl.transform.GridToPersistentTT")
    return pass_func()(mod)


def tt_shard_to_core_map(mod: tvm.IRModule) -> tvm.IRModule:
    """Map tile assignments to physical core coordinates.

    This pass converts logical tile-to-core assignments from WS2 into physical
    CoreRangeSet topology for Tenstorrent devices. It generates:

    - tt_core_ranges: Physical core topology as [start_x, start_y, end_x, end_y, start_tile, count]
    - tt_core_runtime_args: Per-core runtime args as [start_tile, num_tiles]

    For Grayskull/Wormhole 8×8 core grids, uses row-major layout where
    core_id maps to (x, y) = (core_id % 8, core_id / 8).

    Args:
        mod: The TVM IRModule to process (should have WS2 tt_tiles_per_core metadata)

    Returns:
        A new IRModule with physical core topology metadata

    Example:
        >>> from tilelang.tt import apply_ws2_passes, tt_shard_to_core_map
        >>>
        >>> mod = create_tilelang_kernel()
        >>> mod = apply_tt_defaults(mod)  # WS1
        >>> mod = apply_ws2_passes(mod)  # WS2
        >>> mod = tt_shard_to_core_map(mod)  # WS3 Phase 2
    """
    pass_func = tvm.ffi.get_global_func("tl.transform.TTShardToCoreMap")
    return pass_func()(mod)


def memory_space_lower_tt(mod: tvm.IRModule) -> tvm.IRModule:
    """Lower abstract buffer allocations to TT circular buffers.

    This pass transforms TileLang's alloc_fragment buffer allocations into
    Tenstorrent circular buffer (CB) configurations in L1 memory. It:

    - Identifies tile-sized buffers (typically 32×32)
    - Assigns circular buffer IDs (CB0, CB1, CB2, ...)
    - Configures num_pages (1 for accumulator, 2 for inputs/outputs)
    - Stamps storage_scope = "tt.l1"
    - Attaches tt_circular_buffers metadata for codegen

    Args:
        mod: The TVM IRModule to process (should have WS1 TT defaults)

    Returns:
        A new IRModule with circular buffer annotations

    Example:
        >>> from tilelang.tt import apply_ws3_passes, memory_space_lower_tt
        >>>
        >>> mod = create_tilelang_kernel()
        >>> mod = apply_tt_defaults(mod)  # WS1
        >>> mod = apply_ws2_passes(mod)  # WS2
        >>> mod = grid_to_persistent_tt(mod)  # WS3
        >>> mod = memory_space_lower_tt(mod)  # WS3 Phase 2
    """
    pass_func = tvm.ffi.get_global_func("tl.transform.MemorySpaceLowerTT")
    return pass_func()(mod)


def apply_ws3_passes(mod: tvm.IRModule) -> tvm.IRModule:
    """Apply all Workstream 3 TIR transform passes.

    This is a convenience function that applies all WS3 transforms in the
    correct order to produce TT-ready IR.

    Args:
        mod: The TVM IRModule to process (should have WS2 metadata)

    Returns:
        A new IRModule with transformed TIR ready for codegen

    Example:
        >>> from tilelang.tt import apply_tt_defaults, apply_ws2_passes, apply_ws3_passes
        >>>
        >>> mod = create_tilelang_kernel()
        >>> mod = apply_tt_defaults(mod)  # WS1
        >>> mod = apply_ws2_passes(mod)  # WS2
        >>> mod = apply_ws3_passes(mod)  # WS3
    """
    # WS3 Transform Pipeline
    mod = grid_to_persistent_tt(mod)
    mod = tt_shard_to_core_map(mod)
    mod = memory_space_lower_tt(mod)

    # TODO(WS3): Add remaining transforms when implemented
    # mod = tile_pad_tt(mod)
    # mod = tensorize_tt(mod)
    # mod = verify_tt_ir(mod)

    return mod
