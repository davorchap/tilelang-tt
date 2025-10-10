"""
Codegen integration tests for layout-aware metadata.

Validates that reader/compute/writer kernels and host metadata agree on
runtime argument schemas for both global and shard-local partitioning. Also
covers guard-rail failures when required shard metadata is missing.
"""

from __future__ import annotations

import pytest

import tvm

import tilelang.tt as tt

from testing.python.tt.test_host_program_pipeline import _make_tt_module


def test_compute_kernel_extracts_shard_coords_when_local():
    """Local shard kernels expose shard coordinates as runtime arguments."""

    mod = _make_tt_module(partition_mode="local_shard")
    artifacts = tt.emit_tt_artifacts(mod)
    compute_cpp = artifacts["compute.cpp"]

    assert "uint32_t tt_shard_coord_y = get_arg_val<uint32_t>" in compute_cpp
    assert "uint32_t tt_shard_coord_x = get_arg_val<uint32_t>" in compute_cpp


def test_compute_kernel_omits_shard_coords_when_global():
    """Global kernels should not request shard-local coordinates."""

    mod = _make_tt_module(partition_mode="global")
    artifacts = tt.emit_tt_artifacts(mod)
    compute_cpp = artifacts["compute.cpp"]

    assert "tt_shard_coord_y" not in compute_cpp
    assert "tt_shard_coord_x" not in compute_cpp


def test_reader_writer_guard_against_missing_shard_coords():
    """Reader/writer kernels still accept shard args but mark them unused."""

    mod = _make_tt_module(partition_mode="local_shard")
    artifacts = tt.emit_tt_artifacts(mod)
    reader_cpp = artifacts["reader.cpp"]
    writer_cpp = artifacts["writer.cpp"]

    assert "(void)tt_shard_coord_y;" in reader_cpp
    assert "(void)tt_shard_coord_x;" in reader_cpp
    assert "(void)tt_shard_coord_y;" in writer_cpp
    assert "(void)tt_shard_coord_x;" in writer_cpp


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
