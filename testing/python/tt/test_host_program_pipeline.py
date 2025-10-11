"""
Host program metadata summary tests.

These tests verify that the Tenstorrent host program generator emits the
layout-aware metadata tables (runtime arguments, constants, tensor-accessor
payloads) and guardrails required by the new pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List

import tvm
from tvm import tir

import tilelang.tt as tt


def _convert_dict_for_ffi(d: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Python dict to be FFI-compatible (helper for tests)."""
    result = {}
    for key, value in d.items():
        if isinstance(value, bool):
            result[key] = tvm.tir.IntImm("int32", 1 if value else 0)
        elif isinstance(value, int):
            result[key] = tvm.tir.IntImm("int32", value)
        elif isinstance(value, (list, tuple)):
            converted_list = []
            for elem in value:
                if isinstance(elem, bool):
                    converted_list.append(tvm.tir.IntImm("int32", 1 if elem else 0))
                elif isinstance(elem, int):
                    converted_list.append(tvm.tir.IntImm("int32", elem))
                else:
                    converted_list.append(elem)
            result[key] = converted_list
        elif isinstance(value, dict):
            result[key] = _convert_dict_for_ffi(value)
        else:
            result[key] = value
    return result


def _make_tt_module(partition_mode: str = "global") -> tvm.IRModule:
    """Construct a minimal PrimFunc with layout-aware metadata attached."""

    # Buffers are necessary so codegen emits TensorAccessorArgs bindings.
    A = tir.decl_buffer((256, 256), "float16", name="A")
    B = tir.decl_buffer((256, 256), "float16", name="B")
    C = tir.decl_buffer((256, 256), "float16", name="C")

    # Kernel body is irrelevant for host metadata tests.
    func = tir.PrimFunc(params=[A, B, C], body=tir.Evaluate(0), ret_type=None)

    if partition_mode == "local_shard":
        grid_x = 4
        grid_y = 4
        shard_grid = [2, 2]
        local_tiles = [2, 2]
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
        runtime_constants: Dict[str, int] = {
            "Mt": 4,
            "Kt": 1,
            "Nt": 4,
            "Sm": 2,
            "Sn": 2,
            "Gy": 2,
            "Gx": 2
        }
        core_runtime_args: List[List[int]] = [
            [0, 4, 4, 1, 4, 2, 2, 2, 2, 0, 0],
            [0, 4, 4, 1, 4, 2, 2, 2, 2, 0, 1],
            [0, 4, 4, 1, 4, 2, 2, 2, 2, 1, 0],
            [0, 4, 4, 1, 4, 2, 2, 2, 2, 1, 1],
        ]
        tiles_per_core: List[List[int]] = [[0, 4]] * len(core_runtime_args)
        buffer_meta = {
            "A": {
                "memory": "L1",
                "layout": "sharded",
                "tile_shape": [32, 32],
                "nd_shard": {
                    "projected_grid": [2, 2],
                    "projected_shard_tiles": [2, 2]
                },
            },
            "B": {
                "memory": "L1",
                "layout": "sharded",
                "tile_shape": [32, 32],
                "nd_shard": {
                    "projected_grid": [2, 2],
                    "projected_shard_tiles": [2, 2]
                },
            },
            "C": {
                "memory": "L1",
                "layout": "sharded",
                "tile_shape": [32, 32],
                "nd_shard": {
                    "projected_grid": [2, 2],
                    "projected_shard_tiles": [2, 2]
                },
            },
        }
    else:
        grid_x = 8
        grid_y = 8
        shard_grid = [1, 1]
        local_tiles = [grid_y, grid_x]
        runtime_arg_names = ["tt_start_tile", "tt_tile_count", "Mt", "Kt", "Nt"]
        runtime_constants = {"Mt": grid_y, "Kt": 1, "Nt": grid_x}
        total_tiles = grid_x * grid_y
        core_runtime_args = [[0, total_tiles, grid_y, 1, grid_x]]
        tiles_per_core = [[0, total_tiles]]
        buffer_meta = {
            "A": {
                "memory": "DRAM",
                "layout": "interleaved",
                "tile_shape": [32, 32]
            },
            "B": {
                "memory": "DRAM",
                "layout": "interleaved",
                "tile_shape": [32, 32]
            },
            "C": {
                "memory": "DRAM",
                "layout": "interleaved",
                "tile_shape": [32, 32]
            },
        }

    num_tiles = grid_x * grid_y
    num_cores = len(core_runtime_args)

    # Convert all nested structures to FFI-compatible format
    tiles_per_core_ffi = []
    for start, count in tiles_per_core:
        tiles_per_core_ffi.append([tvm.tir.IntImm("int32", start), tvm.tir.IntImm("int32", count)])

    runtime_constants_ffi = _convert_dict_for_ffi(runtime_constants)
    core_runtime_args_ffi = [_convert_dict_for_ffi({"vals": r})["vals"] for r in core_runtime_args]

    func = func.with_attrs({
        "global_symbol": "main",
        "tt_grid_x": tvm.tir.IntImm("int32", grid_x),
        "tt_grid_y": tvm.tir.IntImm("int32", grid_y),
        "tt_grid_z": tvm.tir.IntImm("int32", 1),
        "tt_num_tiles": tvm.tir.IntImm("int32", num_tiles),
        "tt_num_cores": tvm.tir.IntImm("int32", num_cores),
        "tt_tiles_per_core": tiles_per_core_ffi,
        "tt.partition_mode": tvm.runtime.convert(partition_mode),
        "tt.grid_tiles": tvm.runtime.convert([grid_y, grid_x]),
        "tt.local_shape_tiles": tvm.runtime.convert(local_tiles),
        "tt.shard_grid": tvm.runtime.convert(shard_grid),
        "tt.runtime_constants": runtime_constants_ffi,
        "tt.runtime_arg_names": tvm.runtime.convert(runtime_arg_names),
        "tt_core_runtime_args": core_runtime_args_ffi,
    })

    for name, meta in buffer_meta.items():
        meta_ffi = _convert_dict_for_ffi(meta)
        func = func.with_attr(f"tt.buffer.{name}", meta_ffi)

    mod = tvm.IRModule({"main": func})
    return mod


def test_host_program_reports_partition_and_tensor_accessors():
    """Global layout: guardrails and TensorAccessor payloads are emitted."""

    mod = _make_tt_module(partition_mode="global")
    host_cpp = tt.emit_tt_artifacts(mod)["main.cpp"]

    assert 'constexpr const char* kPartitionMode = "global";' in host_cpp
    assert 'TensorAccessorArgs::Create("A", "DRAM", "interleaved", 32, 32, 1, 1)' in host_cpp
    assert 'TensorAccessorArgs::Create("C", "DRAM", "interleaved", 32, 32, 1, 1)' in host_cpp
    assert "TensorAccessorArgs must be created via TensorAccessorArgs::Create" in host_cpp


def test_host_program_runtime_args_schema_global():
    """Global layout: canonical runtime arg schema and per-core payloads."""

    mod = _make_tt_module(partition_mode="global")
    host_cpp = tt.emit_tt_artifacts(mod)["main.cpp"]

    assert 'kRuntimeArgNames = {{"tt_start_tile", "tt_tile_count", "Mt", "Kt", "Nt"}};' in host_cpp
    assert "{{0, 64, 8, 1, 8}}" in host_cpp
    assert '{"Mt", 8}' in host_cpp and '{"Kt", 1}' in host_cpp and '{"Nt", 8}' in host_cpp


def test_host_program_runtime_args_schema_local_shard():
    """Local shard layout: expanded runtime args include shard geometry."""

    mod = _make_tt_module(partition_mode="local_shard")
    host_cpp = tt.emit_tt_artifacts(mod)["main.cpp"]

    assert 'constexpr const char* kPartitionMode = "local_shard";' in host_cpp
    assert (
        'kRuntimeArgNames = {{"tt_start_tile", "tt_tile_count", "Mt", "Kt", "Nt", "Sm", "Sn", "Gy", '
        '"Gx", "tt_shard_coord_y", "tt_shard_coord_x"}};') in host_cpp
    assert "{{0, 4, 4, 1, 4, 2, 2, 2, 2, 0, 0}}" in host_cpp
    assert "{{0, 4, 4, 1, 4, 2, 2, 2, 2, 1, 1}}" in host_cpp
    assert 'TensorAccessorArgs::Create("A", "L1", "sharded", 32, 32, 2, 2)' in host_cpp
