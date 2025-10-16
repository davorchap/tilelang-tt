#!/usr/bin/env python3
"""Test shard coordinate handling"""

import sys
sys.path.insert(0, '.')

import tilelang.tenstorrent as tt
from testing.python.tenstorrent.test_host_program_pipeline import _make_tt_module

# Test with local_shard partition mode
mod = _make_tt_module(partition_mode="local_shard")
artifacts = tt.emit_tt_artifacts(mod)

print("=== COMPUTE.CPP ===")
compute_cpp = artifacts["compute.cpp"]
print(compute_cpp[:1500])
print("\n---Checking for shard coords in compute---")
print(f"Has tt_shard_coord_y: {'tt_shard_coord_y' in compute_cpp}")
print(f"Has tt_shard_coord_x: {'tt_shard_coord_x' in compute_cpp}")
print(f"Has get_arg_val<uint32_t>(9): {'get_arg_val<uint32_t>(9)' in compute_cpp}")
print(f"Has get_arg_val<uint32_t>(10): {'get_arg_val<uint32_t>(10)' in compute_cpp}")

print("\n=== READER.CPP ===")
reader_cpp = artifacts["reader.cpp"]
print(reader_cpp[:1500])
print("\n---Checking for shard coords in reader---")
print(f"Has (void)tt_shard_coord_y: {'(void)tt_shard_coord_y' in reader_cpp}")
print(f"Has (void)tt_shard_coord_x: {'(void)tt_shard_coord_x' in reader_cpp}")