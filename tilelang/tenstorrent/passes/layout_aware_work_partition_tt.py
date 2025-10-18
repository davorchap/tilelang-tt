"""Python implementation of LayoutAwareWorkPartitionTT."""

from __future__ import annotations

from typing import Dict, Optional

from tilelang import tvm as tvm

from ._common import convert_dict_for_ffi, convert_to_python, get_attr, to_int
from ..attrs import (
    TT_PARTITION_MODE,
    TT_GRID_TILES,
    TT_LOCAL_SHAPE_TILES,
    TT_SHARD_GRID,
    TT_CORE_RANGES,
    TT_RUNTIME_ARG_NAMES,
    TT_RUNTIME_CONSTANTS,
    TT_CORE_RUNTIME_ARGS,
)


def layout_aware_work_partition_tt(mod: tvm.IRModule) -> tvm.IRModule:
    """Emit shard-aware partition metadata (partition mode, runtime args, core ranges)."""

    def transform(func: tvm.tir.PrimFunc, *_):
        if not isinstance(func, tvm.tir.PrimFunc):
            return func

        attrs = func.attrs
        tiles_per_core = get_attr(attrs, "tt_tiles_per_core")
        if tiles_per_core is None:
            return func

        grid_x = to_int(get_attr(attrs, "tt_grid_x"), 1)
        grid_y = to_int(get_attr(attrs, "tt_grid_y"), 1)
        grid_z = to_int(get_attr(attrs, "tt_grid_z"), 1)
        num_cores = to_int(get_attr(attrs, "tt_num_cores"), grid_x * grid_y)

        Mt_default = max(grid_y * grid_z, 1)
        Nt_default = max(grid_x, 1)

        has_l1_shard = False
        detected_grid: Optional[list[int]] = None
        detected_tiles: Optional[list[int]] = None
        for _, buffer in func.buffer_map.items():
            meta_obj = get_attr(attrs, f"tt.buffer.{buffer.name}")
            if meta_obj is None:
                continue
            meta = convert_to_python(meta_obj)
            layout_kind = str(meta.get("layout", "interleaved")).lower()
            memory_kind = str(meta.get("memory", "DRAM")).upper()
            nd_shard = meta.get("nd_shard")
            if layout_kind == "sharded" and nd_shard:
                pg = nd_shard.get("projected_grid")
                pt = nd_shard.get("projected_shard_tiles")
                if pg and pt:
                    detected_grid = [int(pg[0]), int(pg[1])]
                    detected_tiles = [int(pt[0]), int(pt[1])]
            if (layout_kind == "sharded" and memory_kind == "L1" and detected_grid and
                    detected_tiles):
                has_l1_shard = True

        schedule_raw = get_attr(attrs, "tt.user_schedule")
        partition_mode = "local_shard" if has_l1_shard else "global"
        grid_tiles = [Mt_default, Nt_default]
        shard_grid = [1, 1]
        local_tiles = grid_tiles[:]
        runtime_arg_names: Optional[list[str]] = None

        if isinstance(schedule_raw, tvm.ir.Map):
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

        runtime_constants_ffi = convert_dict_for_ffi(runtime_constants)

        mesh_width = int(round(num_cores**0.5))
        if mesh_width * mesh_width != num_cores:
            mesh_width = grid_x if grid_x > 0 else max(num_cores, 1)

        core_ranges = []
        core_runtime_args = []
        for core_id, assignment in enumerate(tiles_per_core):
            start = to_int(assignment[0], 0)
            count = to_int(assignment[1], 0)
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
                core_runtime_args.append(
                    [start, count, Mt, 1, Nt, Sm, Sn, Gy, Gx, shard_sy, shard_sx])
            else:
                core_ranges.append([x, y, x, y, start, count])
                core_runtime_args.append([start, count, Mt, 1, Nt])

        new_func = func.with_attr(TT_PARTITION_MODE, partition_mode)
        new_func = new_func.with_attr(TT_GRID_TILES, grid_tiles)
        new_func = new_func.with_attr(TT_LOCAL_SHAPE_TILES, local_tiles)
        new_func = new_func.with_attr(TT_SHARD_GRID, shard_grid)
        new_func = new_func.with_attr(TT_RUNTIME_ARG_NAMES, runtime_arg_names)
        new_func = new_func.with_attr(TT_RUNTIME_CONSTANTS, runtime_constants_ffi)

        core_ranges_ffi = [convert_dict_for_ffi({"vals": r})["vals"] for r in core_ranges]
        core_runtime_args_ffi = [
            convert_dict_for_ffi({"vals": r})["vals"] for r in core_runtime_args
        ]
        new_func = new_func.with_attr(TT_CORE_RANGES, core_ranges_ffi)
        new_func = new_func.with_attr(TT_CORE_RUNTIME_ARGS, core_runtime_args_ffi)

        return new_func

    pass_obj = tvm.tir.transform.prim_func_pass(
        transform, opt_level=0, name="tl.LayoutAwareWorkPartitionTT")
    return pass_obj(mod)


__all__ = ["layout_aware_work_partition_tt"]
