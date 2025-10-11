"""Tenstorrent backend compiler passes.

This module provides Python bindings for TT-specific TVM passes that inject
schedule and sharding metadata into IRModules.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from tilelang import tvm as tvm
from tvm.runtime import DataType, convert
from tvm.tir import IntImm, FloatImm
from tvm.tir.analysis import simplify


def annotate_tt_layout(func: tvm.tir.PrimFunc, layout: Dict[str, Any]) -> tvm.tir.PrimFunc:
    """Attach user-specified layout metadata to a PrimFunc.

    The layout dictionary should be keyed by buffer name and contain arbitrary
    metadata that `InferTTLayout` can read. The helper simply stores the raw
    payload under the `tt.user_layout` attribute.
    """

    if not isinstance(func, tvm.tir.PrimFunc):
        raise TypeError("annotate_tt_layout expects a tvm.tir.PrimFunc")

    return func.with_attr("tt.user_layout", convert(layout))


def annotate_tt_schedule(func: tvm.tir.PrimFunc, schedule: Dict[str, Any]) -> tvm.tir.PrimFunc:
    """Attach user-specified schedule metadata to a PrimFunc.

    The schedule dictionary should follow the schema expected by layout-aware
    passes. The helper stores the payload under `tt.user_schedule`.
    """

    if not isinstance(func, tvm.tir.PrimFunc):
        raise TypeError("annotate_tt_schedule expects a tvm.tir.PrimFunc")

    return func.with_attr("tt.user_schedule", convert(schedule))


def _dtype_num_bytes(dtype_str: str) -> int:
    dt = DataType(dtype_str)
    if dt.bits % 8 != 0:
        raise ValueError(f"Unsupported dtype bit-width for {dtype_str}")
    return (dt.bits // 8) * dt.lanes


def _infer_data_format(dtype_str: str) -> str:
    mapping = {
        "float16": "Float16_b",
        "float32": "Float32",
        "bfloat16": "BFloat16_b",
        "int8": "Int8",
        "uint8": "UInt8",
    }
    return mapping.get(dtype_str, dtype_str)


def _to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, tvm.tir.IntImm):
        return int(value.value)
    return int(value)


def _array_to_int_list(array_obj: Any) -> Optional[list[int]]:
    if array_obj is None:
        return None
    if isinstance(array_obj, tvm.runtime.Array):
        return [int(x) for x in array_obj]
    return None


def _map_to_python(map_obj: Any) -> Dict[str, Any]:
    if map_obj is None or not isinstance(map_obj, tvm.runtime.Map):
        return {}
    return {str(key): value for key, value in map_obj.items()}


def _map_of_maps_to_python(map_obj: Any) -> Dict[str, Dict[str, Any]]:
    if map_obj is None or not isinstance(map_obj, tvm.runtime.Map):
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for key, value in map_obj.items():
        if isinstance(value, tvm.runtime.Map):
            result[str(key)] = {
                str(inner_key): _convert_to_python(inner_value)
                for inner_key, inner_value in value.items()
            }
        else:
            result[str(key)] = {"value": _convert_to_python(value)}
    return result


def _convert_to_python(obj: Any) -> Any:
    if isinstance(obj, tvm.runtime.Map):
        return {str(k): _convert_to_python(v) for k, v in obj.items()}
    if isinstance(obj, tvm.runtime.Array):
        return [_convert_to_python(x) for x in obj]
    if isinstance(obj, IntImm):
        return int(obj)
    if isinstance(obj, FloatImm):
        return float(obj)
    if isinstance(obj, tvm.tir.PrimExpr):
        simplified = simplify(obj)
        if isinstance(simplified, IntImm):
            return int(simplified)
        if isinstance(simplified, FloatImm):
            return float(simplified)
        return simplified
    return obj


def _get_attr(attrs: Optional[tvm.ir.container.Map], key: str, default: Any = None) -> Any:
    if attrs is None:
        return default
    if key in attrs:
        return attrs[key]
    return default


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
    """Apply the legacy metadata inference passes (schedule + sharding).

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
# TT Transform Pipeline
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


def infer_tt_layout(mod: tvm.IRModule) -> tvm.IRModule:
    """Infer canonical layout metadata (`tt.buffer.*`) for each PrimFunc buffer."""

    def transform(func: tvm.tir.PrimFunc, *_):
        if not isinstance(func, tvm.tir.PrimFunc):
            return func

        new_func = func
        attrs = func.attrs

        user_layout_map = _map_of_maps_to_python(_get_attr(attrs, "tt.user_layout"))
        shard_map_map = _map_of_maps_to_python(_get_attr(attrs, "tt_shard"))

        for _, buffer in func.buffer_map.items():
            buffer_name = buffer.name

            metadata: Dict[str, Any] = {}
            if buffer_name in user_layout_map:
                metadata.update(user_layout_map[buffer_name])
            elif buffer_name in shard_map_map:
                metadata.update(shard_map_map[buffer_name])

            metadata = {k: _convert_to_python(v) for k, v in metadata.items()}

            layout_kind = str(metadata.get("layout", "interleaved"))
            layout_kind = layout_kind.lower()
            if layout_kind in ("dram_interleaved", "interleaved"):
                layout_kind = "interleaved"
            elif layout_kind == "sharded":
                layout_kind = "sharded"
            else:
                layout_kind = "interleaved"

            dtype_str = str(metadata.get("dtype", buffer.dtype))

            tile_shape = metadata.get("tile_shape")
            if not isinstance(tile_shape, (list, tuple)) or len(tile_shape) != 2:
                attr_tile = _get_attr(attrs, f"tt_buffer_{buffer_name}_tile_shape")
                tile_shape = _array_to_int_list(attr_tile)
            if not tile_shape:
                tile_shape = [32, 32]
            tile_shape = [int(tile_shape[0]), int(tile_shape[1])]

            buffer_meta: Dict[str, Any] = {
                "memory": str(metadata.get("memory", "DRAM")),
                "layout": layout_kind,
                "tile_shape": tile_shape,
                "dtype": dtype_str,
            }

            needs_padding_value = metadata.get("needs_padding")
            if needs_padding_value is None:
                attr_padding = _get_attr(attrs, f"tt_buffer_{buffer_name}_needs_padding")
                if attr_padding is not None:
                    needs_padding_value = bool(int(attr_padding))
            if needs_padding_value is not None:
                buffer_meta["needs_padding"] = bool(needs_padding_value)

            padded_shape_value = metadata.get("padded_shape")
            if padded_shape_value is None:
                attr_padded = _get_attr(attrs, f"tt_buffer_{buffer_name}_padded_shape")
                padded_shape_value = _array_to_int_list(attr_padded)
            if isinstance(padded_shape_value, (list, tuple)) and padded_shape_value:
                buffer_meta["padded_shape"] = [int(x) for x in padded_shape_value]

            nd_shard = metadata.get("nd_shard")
            if isinstance(nd_shard, dict) and nd_shard:
                axes = [str(ax) for ax in nd_shard.get("axes", [])]
                grid = nd_shard.get("grid")
                shard_shape_elems = nd_shard.get("shard_shape_elems")

                if not axes or grid is None or shard_shape_elems is None:
                    raise ValueError(
                        f"nd_shard metadata for buffer '{buffer_name}' must include 'axes', 'grid', and 'shard_shape_elems'"
                    )

                grid = [int(x) for x in grid]
                shard_shape_elems = [int(x) for x in shard_shape_elems]

                if "M" not in axes or "N" not in axes:
                    raise ValueError(
                        f"nd_shard metadata for buffer '{buffer_name}' must include axes 'M' and 'N'"
                    )

                idx_m = axes.index("M")
                idx_n = axes.index("N")
                grid_m = grid[idx_m]
                grid_n = grid[idx_n]
                shard_m = shard_shape_elems[idx_m]
                shard_n = shard_shape_elems[idx_n]

                projected_grid = [grid_m, grid_n]
                projected_tiles = [
                    max(1, math.ceil(shard_m / tile_shape[0])),
                    max(1, math.ceil(shard_n / tile_shape[1])),
                ]

                if buffer_meta["memory"].upper() == "L1" and (shard_m % tile_shape[0] != 0 or
                                                              shard_n % tile_shape[1] != 0):
                    raise ValueError(f"L1 shard for buffer '{buffer_name}' must be tile-aligned")

                nd_shard_processed = dict(nd_shard)
                nd_shard_processed["projected_grid"] = projected_grid
                nd_shard_processed["projected_shard_tiles"] = projected_tiles
                buffer_meta["nd_shard"] = nd_shard_processed

            if buffer_name in shard_map_map:
                legacy_entry = {
                    k: _convert_to_python(v) for k, v in shard_map_map[buffer_name].items()
                }
                buffer_meta["legacy_shard"] = legacy_entry

            new_func = new_func.with_attr(f"tt.buffer.{buffer_name}", convert(buffer_meta))

        return new_func

    pass_obj = tvm.tir.transform.prim_func_pass(transform, opt_level=0, name="tl.InferTTLayout")
    return pass_obj(mod)


def propagate_tt_layout(mod: tvm.IRModule) -> tvm.IRModule:
    """Propagate buffer layout metadata into circular-buffer (`tt.cb.*`) attributes."""

    def transform(func: tvm.tir.PrimFunc, *_):
        if not isinstance(func, tvm.tir.PrimFunc):
            return func

        new_func = func
        for _, buffer in func.buffer_map.items():
            buffer_name = buffer.name
            buffer_attr_key = f"tt.buffer.{buffer_name}"
            buffer_meta_obj = _get_attr(func.attrs, buffer_attr_key)
            if buffer_meta_obj is None:
                continue

            buffer_meta = _convert_to_python(buffer_meta_obj)

            dtype_str = str(buffer_meta.get("dtype", buffer.dtype))
            tile_shape = buffer_meta.get("tile_shape")
            if not isinstance(tile_shape, (list, tuple)) or len(tile_shape) != 2:
                tile_shape = [32, 32]
            tile_shape = [int(tile_shape[0]), int(tile_shape[1])]

            try:
                element_bytes = _dtype_num_bytes(dtype_str)
            except ValueError:
                element_bytes = 0

            page_size = tile_shape[0] * tile_shape[1] * element_bytes
            depth = int(buffer_meta.get("cb_depth", 2))
            data_format = buffer_meta.get("data_format", _infer_data_format(dtype_str))

            cb_entry = {
                "page_size": page_size,
                "depth": depth,
                "data_format": str(data_format),
            }

            new_func = new_func.with_attr(f"tt.cb.{buffer_name}", convert(cb_entry))

        return new_func

    pass_obj = tvm.tir.transform.prim_func_pass(transform, opt_level=0, name="tl.PropagateTTLayout")
    return pass_obj(mod)


def layout_aware_work_partition_tt(mod: tvm.IRModule) -> tvm.IRModule:
    """Emit shard-aware partition metadata (`tt.partition_mode`, `tt.core_ranges`, etc.)."""

    def transform(func: tvm.tir.PrimFunc, *_):
        if not isinstance(func, tvm.tir.PrimFunc):
            return func

        attrs = func.attrs
        tiles_per_core = _get_attr(attrs, "tt_tiles_per_core")
        if tiles_per_core is None:
            return func

        grid_x = _to_int(_get_attr(attrs, "tt_grid_x"), 1)
        grid_y = _to_int(_get_attr(attrs, "tt_grid_y"), 1)
        grid_z = _to_int(_get_attr(attrs, "tt_grid_z"), 1)
        num_cores = _to_int(_get_attr(attrs, "tt_num_cores"), grid_x * grid_y)

        Mt_default = max(grid_y * grid_z, 1)
        Nt_default = max(grid_x, 1)

        # Detect sharded buffers to auto-enable local_shard mode
        has_l1_shard = False
        detected_grid: Optional[list[int]] = None
        detected_tiles: Optional[list[int]] = None
        for _, buffer in func.buffer_map.items():
            meta_obj = _get_attr(attrs, f"tt.buffer.{buffer.name}")
            if meta_obj is None:
                continue
            meta = _convert_to_python(meta_obj)
            layout_kind = str(meta.get("layout", "interleaved")).lower()
            memory_kind = str(meta.get("memory", "DRAM")).upper()
            nd_shard = meta.get("nd_shard")
            if layout_kind == "sharded" and nd_shard:
                pg = nd_shard.get("projected_grid")
                pt = nd_shard.get("projected_shard_tiles")
                if pg and pt:
                    detected_grid = [int(pg[0]), int(pg[1])]
                    detected_tiles = [int(pt[0]), int(pt[1])]
            if layout_kind == "sharded" and memory_kind == "L1" and detected_grid and detected_tiles:
                has_l1_shard = True

        schedule_raw = _get_attr(attrs, "tt.user_schedule")
        partition_mode = "local_shard" if has_l1_shard else "global"
        grid_tiles = [Mt_default, Nt_default]
        shard_grid = [1, 1]
        local_tiles = grid_tiles[:]
        runtime_arg_names: Optional[list[str]] = None

        if isinstance(schedule_raw, tvm.runtime.Map):
            if schedule_raw.get("partition_mode", None) is not None:
                partition_mode = str(schedule_raw["partition_mode"])
            if schedule_raw.get("grid_tiles", None) is not None:
                grid_tiles = [int(x) for x in schedule_raw["grid_tiles"]]
            if schedule_raw.get("shard_grid", None) is not None:
                shard_grid = [int(x) for x in schedule_raw["shard_grid"]]
            if schedule_raw.get("local_shape_tiles", None) is not None:
                local_tiles = [int(x) for x in schedule_raw["local_shape_tiles"]]
            if schedule_raw.get("runtime_args", None) is not None:
                runtime_arg_names = [str(x) for x in schedule_raw["runtime_args"]]

        if partition_mode == "local_shard":
            if detected_grid is None or detected_tiles is None:
                raise ValueError("layout-aware partitioning requires sharded buffer metadata")
            shard_grid = detected_grid
            local_tiles = detected_tiles

        if runtime_arg_names is None:
            if partition_mode == "local_shard":
                runtime_arg_names = [
                    "tt_start_tile",
                    "tt_tile_count",
                    "Mt",
                    "Kt",
                    "Nt",
                    "Sm",
                    "Sn",
                    "Gy",
                    "Gx",
                    "tt_shard_coord_y",
                    "tt_shard_coord_x",
                ]
            else:
                runtime_arg_names = ["tt_start_tile", "tt_tile_count", "Mt", "Kt", "Nt"]

        Mt = int(grid_tiles[0])
        Nt = int(grid_tiles[1])
        Sm = int(local_tiles[0])
        Sn = int(local_tiles[1])
        Gy = int(shard_grid[0])
        Gx = int(shard_grid[1])

        runtime_constants: Dict[str, int] = {"Mt": Mt, "Nt": Nt, "Kt": 1}
        if partition_mode == "local_shard":
            runtime_constants.update({"Sm": Sm, "Sn": Sn, "Gy": Gy, "Gx": Gx})

        mesh_width = int(round(num_cores**0.5))
        if mesh_width * mesh_width != num_cores:
            mesh_width = grid_x if grid_x > 0 else max(num_cores, 1)

        core_ranges = []
        core_runtime_args = []
        for core_id, assignment in enumerate(tiles_per_core):
            start = _to_int(assignment[0], 0)
            count = _to_int(assignment[1], 0)
            x = core_id % mesh_width
            y = core_id // mesh_width

            if partition_mode == "local_shard":
                if y < Gy and x < Gx:
                    start = 0
                    count = Sm * Sn
                    shard_sy = y
                    shard_sx = x
                else:
                    start = 0
                    count = 0
                    shard_sy = 0
                    shard_sx = 0

                core_ranges.append([x, y, x, y, start, count])
                core_runtime_args.append([
                    start,
                    count,
                    Mt,
                    1,
                    Nt,
                    Sm,
                    Sn,
                    Gy,
                    Gx,
                    shard_sy,
                    shard_sx,
                ])
            else:
                core_ranges.append([x, y, x, y, start, count])
                core_runtime_args.append([start, count, Mt, 1, Nt])

        new_func = func.with_attr("tt.partition_mode", convert(partition_mode))
        new_func = new_func.with_attr("tt.grid_tiles", convert(grid_tiles))
        new_func = new_func.with_attr("tt.local_shape_tiles", convert(local_tiles))
        new_func = new_func.with_attr("tt.shard_grid", convert(shard_grid))
        new_func = new_func.with_attr("tt.runtime_arg_names", convert(runtime_arg_names))
        new_func = new_func.with_attr("tt.runtime_constants", convert(runtime_constants))
        new_func = new_func.with_attr("tt.core_ranges", convert(core_ranges))
        new_func = new_func.with_attr("tt_core_runtime_args", convert(core_runtime_args))

        return new_func

    pass_obj = tvm.tir.transform.prim_func_pass(
        transform, opt_level=0, name="tl.LayoutAwareWorkPartitionTT")
    return pass_obj(mod)


def apply_layout_aware_metadata_passes(mod: tvm.IRModule) -> tvm.IRModule:
    """Convenience wrapper that runs the layout-aware metadata passes in order."""

    mod = infer_tt_layout(mod)
    mod = propagate_tt_layout(mod)
    mod = layout_aware_work_partition_tt(mod)
    return mod


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
    """Apply all TT-specific TIR transform passes.

    This is a convenience function that applies the persistent-transform
    pipeline in the correct order to produce TT-ready IR.

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
