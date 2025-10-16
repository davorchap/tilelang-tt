#!/usr/bin/env python3
"""Detailed check of what the skipped tests expect"""

import sys
sys.path.insert(0, '.')

import tilelang.tenstorrent as tt
from testing.python.tenstorrent.test_codegen_pipeline import create_tt_module_with_metadata

# Create a test module
mod = create_tt_module_with_metadata(grid_x=8, grid_y=8)

# Generate artifacts
artifacts = tt.emit_tt_artifacts(mod)

print("=== READER.CPP ===")
print(artifacts.get("reader.cpp", "NOT GENERATED")[:1000])

print("\n=== COMPUTE.CPP ===")
print(artifacts.get("compute.cpp", "NOT GENERATED")[:1000])

print("\n=== WRITER.CPP ===")
print(artifacts.get("writer.cpp", "NOT GENERATED")[:1000])

print("\n=== MAIN.CPP ===")
print(artifacts.get("main.cpp", "NOT GENERATED")[:1000])