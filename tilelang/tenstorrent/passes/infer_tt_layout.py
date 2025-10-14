"""Python implementation of the InferTTLayout pass."""

from __future__ import annotations

import math
from typing import Any, Dict

from tilelang import tvm as tvm

from ._common import (
    array_to_int_list,
    convert_dict_for_ffi,
    convert_to_python,
    get_attr,
    map_of_maps_to_python,
)


def infer_tt_layout(mod: tvm.IRModule) -> tvm.IRModule:
    """Infer canonical `tt.buffer.*` layout metadata for each PrimFunc buffer."""

    def transform(func: tvm.tir.PrimFunc, *_):
        if not isinstance(func, tvm.tir.PrimFunc):
            return func

        new_func = func
        attrs = func.attrs

        user_layout_map = map_of_maps_to_python(get_attr(attrs, "tt.user_layout"))
        shard_map_map = map_of_maps_to_python(get_attr(attrs, "tt_shard"))

        for _, buffer in func.buffer_map.items():
            buffer_name = buffer.name

            metadata: Dict[str, Any] = {}
            if buffer_name in user_layout_map:
                metadata.update(user_layout_map[buffer_name])
            elif buffer_name in shard_map_map:
                metadata.update(shard_map_map[buffer_name])

            metadata = {k: convert_to_python(v) for k, v in metadata.items()}

            layout_kind = str(metadata.get("layout", "interleaved")).lower()
            if layout_kind in ("dram_interleaved", "interleaved"):
                layout_kind = "interleaved"
            elif layout_kind == "sharded":
                layout_kind = "sharded"
            else:
                layout_kind = "interleaved"

            dtype_str = str(metadata.get("dtype", buffer.dtype))

            tile_shape = metadata.get("tile_shape")
            if not isinstance(tile_shape, (list, tuple)) or len(tile_shape) != 2:
                attr_tile = get_attr(attrs, f"tt_buffer_{buffer_name}_tile_shape")
                tile_shape = array_to_int_list(attr_tile)
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
                attr_padding = get_attr(attrs, f"tt_buffer_{buffer_name}_needs_padding")
                if attr_padding is not None:
                    needs_padding_value = bool(int(attr_padding))
            if needs_padding_value is not None:
                buffer_meta["needs_padding"] = bool(needs_padding_value)

            padded_shape_value = metadata.get("padded_shape")
            if padded_shape_value is None:
                attr_padded = get_attr(attrs, f"tt_buffer_{buffer_name}_padded_shape")
                padded_shape_value = array_to_int_list(attr_padded)
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
                    k: convert_to_python(v) for k, v in shard_map_map[buffer_name].items()
                }
                buffer_meta["legacy_shard"] = legacy_entry

            buffer_meta_ffi = convert_dict_for_ffi(buffer_meta)
            new_func = new_func.with_attr(f"tt.buffer.{buffer_name}", buffer_meta_ffi)

        return new_func

    pass_obj = tvm.tir.transform.prim_func_pass(transform, opt_level=0, name="tl.InferTTLayout")
    return pass_obj(mod)


__all__ = ["infer_tt_layout"]
