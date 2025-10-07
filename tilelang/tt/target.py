"""Tenstorrent target-specific default annotation helper.

This module provides utilities to stamp default Tenstorrent schedule/sharding
metadata when user code omits TT-specific annotations.
"""

from __future__ import annotations

from typing import Dict

from tilelang import tvm as tvm


def apply_tt_defaults(mod: tvm.IRModule) -> tvm.IRModule:
    """Apply default Tenstorrent annotations to an IRModule.

    This function adds default schedule and sharding attributes to PrimFuncs
    in the module when they are not already present. This allows GPU-style
    kernels to run on Tenstorrent with minimal changes by providing sensible
    defaults.

    Default annotations applied:
    - **Schedule policy**: "contiguous" with "row_major" order
      - Tiles are assigned to cores in row-major contiguous order
      - Each core processes a contiguous range of tiles

    - **Layout**: Row-major 32×32 DRAM interleaved tilization
      - Tensors are stored in DRAM with 32×32 tile granularity
      - Tiles are interleaved across DRAM banks

    - **Sharding**: DRAM interleaved tensors via TensorAccessor
      - Default sharding strategy for input/output tensors
      - Compatible with default tilization layout

    The helper ensures idempotency - if a function already has TT-specific
    attributes, they are preserved and not overwritten.

    Args:
        mod: The TVM IRModule to process

    Returns:
        A new IRModule with default TT attributes applied to all PrimFuncs
        that don't already have them

    Example:
        >>> import tilelang.language as T
        >>> from tilelang.tt import apply_tt_defaults
        >>>
        >>> @T.prim_func
        >>> def gemm(A, B, C):
        >>>     # ... kernel implementation ...
        >>>     pass
        >>>
        >>> mod = tvm.IRModule.from_expr(gemm)
        >>> mod_with_defaults = apply_tt_defaults(mod)
    """
    # Default schedule attributes
    DEFAULT_SCHEDULE_ATTRS = {
        "tt_schedule_policy": "contiguous",
        "tt_schedule_order": "row_major",
    }

    # Default sharding/layout attributes
    DEFAULT_LAYOUT_ATTRS = {
        "tt_layout_type": "dram_interleaved",
        "tt_tile_height": 32,
        "tt_tile_width": 32,
    }

    # Combine all default attributes
    default_attrs = {**DEFAULT_SCHEDULE_ATTRS, **DEFAULT_LAYOUT_ATTRS}

    # Process each function in the module
    updated_functions: Dict[str, tvm.tir.PrimFunc] = {}

    for global_var, base_func in mod.functions.items():
        # Only process PrimFuncs (not relay functions or other types)
        if not isinstance(base_func, tvm.tir.PrimFunc):
            updated_functions[global_var] = base_func
            continue

        func = base_func

        # Check if any TT-specific attributes are already present
        has_tt_attrs = any(
            key.startswith("tt_") for key in (func.attrs.keys() if func.attrs else []))

        # If no TT attributes present, add defaults
        if not has_tt_attrs:
            # Add each default attribute to the function
            for attr_name, attr_value in default_attrs.items():
                func = func.with_attr(attr_name, attr_value)

        updated_functions[global_var] = func

    # Create new module with updated functions
    return tvm.IRModule(updated_functions)
