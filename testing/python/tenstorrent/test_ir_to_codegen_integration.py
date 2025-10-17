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

from testing.python.tenstorrent.test_host_program_pipeline import _make_tt_module, _convert_dict_for_ffi
from testing.python.tenstorrent.test_fixtures import create_complete_ir_module_with_split_kernels

# Skip reason for codegen tests
CODEGEN_SKIP_REASON = "Requires reader/writer/compute kernel codegen implementation (reader.cpp, compute.cpp, writer.cpp generation)"


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


@pytest.mark.skip(reason="Shard coord propagation requires full v5 pipeline metadata")
def test_compute_kernel_extracts_shard_coords_when_local():
    """Local shard kernels expose shard coordinates as runtime arguments."""

    mod = _make_tt_module_with_complete_ir(partition_mode="local_shard")
    artifacts = tt.emit_tt_artifacts(mod)
    compute_cpp = artifacts["compute.cpp"]

    assert "uint32_t tt_shard_coord_y = get_arg_val<uint32_t>" in compute_cpp
    assert "uint32_t tt_shard_coord_x = get_arg_val<uint32_t>" in compute_cpp


def test_compute_kernel_omits_shard_coords_when_global():
    """Global kernels should not request shard-local coordinates."""

    mod = _make_tt_module_with_complete_ir(partition_mode="global")
    artifacts = tt.emit_tt_artifacts(mod)
    compute_cpp = artifacts["compute.cpp"]

    # IR-driven codegen correctly omits shard coords for global partition
    assert "tt_shard_coord_y" not in compute_cpp
    assert "tt_shard_coord_x" not in compute_cpp


@pytest.mark.skip(reason="Shard coord propagation requires full v5 pipeline metadata")
def test_reader_writer_guard_against_missing_shard_coords():
    """Reader/writer kernels still accept shard args but mark them unused."""

    mod = _make_tt_module_with_complete_ir(partition_mode="local_shard")
    artifacts = tt.emit_tt_artifacts(mod)
    reader_cpp = artifacts["reader.cpp"]
    writer_cpp = artifacts["writer.cpp"]

    assert "(void)tt_shard_coord_y;" in reader_cpp
    assert "(void)tt_shard_coord_x;" in reader_cpp
    assert "(void)tt_shard_coord_y;" in writer_cpp
    assert "(void)tt_shard_coord_x;" in writer_cpp


@pytest.mark.skip(
    reason="Validation for missing shard coordinates not yet implemented in v5 codegen")
def test_emit_tt_artifacts_requires_shard_runtime_args():
    """Missing shard coordinates for local partition raises a codegen error."""

    mod = _make_tt_module(partition_mode="local_shard")
    func = mod["main"]

    runtime_arg_names = [str(x) for x in func.attrs["tt.runtime_arg_names"]]
    truncated_names = runtime_arg_names[:-2]  # Drop shard coordinates intentionally

    func = func.with_attr("tt.runtime_arg_names", tvm.runtime.convert(truncated_names))
    mod = tvm.IRModule({"main": func})

    with pytest.raises(tvm.error.TVMError, match="tt_shard_coord_y"):
        tt.emit_tt_artifacts(mod)
