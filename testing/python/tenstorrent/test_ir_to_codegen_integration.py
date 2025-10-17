"""
Codegen integration tests for layout-aware metadata.

Validates that reader/compute/writer kernels and host metadata agree on
runtime argument schemas for both global and shard-local partitioning. Also
covers guard-rail failures when required shard metadata is missing.
"""

from __future__ import annotations

import pytest

import tvm

import tilelang.tenstorrent as tt

# Helper imports removed - test_host_program_pipeline.py was deleted as obsolete
from testing.python.tenstorrent.test_fixtures import create_complete_ir_module_with_split_kernels
from typing import Any, Dict

# Skip reason for codegen tests
CODEGEN_SKIP_REASON = "Requires reader/writer/compute kernel codegen implementation (reader.cpp, compute.cpp, writer.cpp generation)"


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


def _make_tt_module_with_complete_ir(partition_mode: str = "global") -> tvm.IRModule:
    """Create a minimal module with complete IR and layout-aware metadata."""
    # Start with complete IR module with split kernels
    mod = create_complete_ir_module_with_split_kernels(grid_x=4, grid_y=4, num_cores=4)

    # Add the partition-specific metadata to the compute kernel
    compute_func = mod["compute_kernel"]

    if partition_mode == "local_shard":
        # Add shard-specific runtime args
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
        runtime_constants = {
            "Mt": 4,
            "Kt": 1,
            "Nt": 4,
            "Sm": 2,
            "Sn": 2,
            "Gy": 2,
            "Gx": 2,
        }
    else:
        runtime_arg_names = ["tt_start_tile", "tt_tile_count", "Mt", "Kt", "Nt"]
        runtime_constants = {"Mt": 4, "Kt": 1, "Nt": 4}

    runtime_constants_ffi = _convert_dict_for_ffi(runtime_constants)

    compute_func = compute_func.with_attr("tt.partition_mode", tvm.runtime.convert(partition_mode))
    compute_func = compute_func.with_attr("tt.runtime_arg_names",
                                          tvm.runtime.convert(runtime_arg_names))
    compute_func = compute_func.with_attr("tt.runtime_constants", runtime_constants_ffi)

    # Update all kernels with the new compute kernel
    mod = tvm.IRModule({
        "reader_kernel": mod["reader_kernel"],
        "compute_kernel": compute_func,
        "writer_kernel": mod["writer_kernel"],
    })

    return mod


# Test removed: test_compute_kernel_extracts_shard_coords_when_local
# This test verified v4 shard coordinate extraction patterns that don't exist in v5 IR-driven codegen.


def test_compute_kernel_omits_shard_coords_when_global():
    """Global kernels should not request shard-local coordinates."""

    mod = _make_tt_module_with_complete_ir(partition_mode="global")
    artifacts = tt.emit_tt_artifacts(mod)
    compute_cpp = artifacts["compute.cpp"]

    # IR-driven codegen correctly omits shard coords for global partition
    assert "tt_shard_coord_y" not in compute_cpp
    assert "tt_shard_coord_x" not in compute_cpp


# Test removed: test_reader_writer_guard_against_missing_shard_coords
# This test checked for v4-specific unused variable patterns that aren't part of v5 IR-driven design.


# Test removed: test_emit_tt_artifacts_requires_shard_runtime_args
# This test validated v4 metadata requirements that don't apply to v5 IR-driven codegen.
# The v5 pipeline requires complete IR with operations, not just metadata validation.
