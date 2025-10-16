#!/usr/bin/env python3
"""Quick test to see what's currently being generated"""

import sys
sys.path.insert(0, '.')

import tilelang.tenstorrent as tt
from testing.python.tenstorrent.test_codegen_pipeline import create_tt_module_with_metadata

# Create a test module
mod = create_tt_module_with_metadata(grid_x=8, grid_y=8)

# Generate artifacts
artifacts = tt.emit_tt_artifacts(mod)

# Print what we got
print("Generated artifacts:")
for filename in sorted(artifacts.keys()):
    print(f"  - {filename}")

# Check if we have the expected files
expected_files = ["compute.cpp", "reader.cpp", "writer.cpp", "main.cpp", "tt.plan.json"]
for filename in expected_files:
    if filename in artifacts:
        print(f"✓ {filename} exists ({len(artifacts[filename])} bytes)")
    else:
        print(f"✗ {filename} MISSING!")

# Check some key content
print("\n=== Checking compute.cpp ===")
compute = artifacts.get("compute.cpp", "")
checks = [
    ("MAIN function", "void MAIN()" in compute),
    ("Runtime args", "get_arg_val" in compute),
    ("CB indices", "CBIndex" in compute or "cb_in0" in compute),
]
for check_name, result in checks:
    print(f"  {check_name}: {'✓' if result else '✗'}")

if "reader.cpp" in artifacts:
    print("\n=== Checking reader.cpp ===")
    reader = artifacts["reader.cpp"]
    checks = [
        ("kernel_main or MAIN", "void kernel_main()" in reader or "void MAIN()" in reader),
        ("CB reserve", "cb_reserve_back" in reader),
        ("NOC read", "noc_async_read" in reader),
    ]
    for check_name, result in checks:
        print(f"  {check_name}: {'✓' if result else '✗'}")