#!/usr/bin/env python3
"""
Verify that generated code uses real Metalium kernel APIs.
This test runs the v5 pipeline with split_device_kernel.
"""

import sys
import tilelang.tenstorrent as tt
from tilelang.tenstorrent.passes import (
    infer_tt_layout_v5,
    split_device_kernel,
)
from tilelang import tvm
from tvm.script import tir as T
import tvm.script

print("=== Verifying Real Metalium Kernel API Usage ===")


# Create a proper test module with compute operations
@tvm.script.ir_module
class TestModule:

    @T.prim_func
    def gemm(A: T.Buffer((64, 64), "float16"), B: T.Buffer((64, 64), "float16"), C: T.Buffer(
        (64, 64), "float16")):
        T.func_attr({"global_symbol": "gemm"})
        for i, j in T.grid(64, 64):
            C[i, j] = T.float16(0)
            for k in T.serial(64):
                C[i, j] = C[i, j] + A[i, k] * B[k, j]


mod = TestModule

# Run minimal v5 pipeline to get split kernels
print("Running v5 pipeline...")
mod = infer_tt_layout_v5(mod)  # A1: Add layout metadata
mod = split_device_kernel(mod)  # D1: Split into reader/compute/writer

# Generate code from split kernels
artifacts = tt.emit_tt_artifacts(mod)

# Verify kernel files use real Metalium API headers
# (Note: host program uses metadata summary format regardless of SDK mode)
compute_cpp = artifacts.get("compute.cpp", "")
reader_cpp = artifacts.get("reader.cpp", "")
writer_cpp = artifacts.get("writer.cpp", "")

# Check for real kernel API includes in compute kernel
kernel_api_markers = [
    '#include "ckernel_include.h"',
    '#include "compute_kernel_api',
]

dataflow_markers = [
    '#include "dataflow_api.h"',
]

missing = []

# Check compute kernel for real APIs
for marker in kernel_api_markers:
    if marker not in compute_cpp:
        missing.append(f"compute.cpp: {marker}")

# Check reader/writer kernels for dataflow APIs
for marker in dataflow_markers:
    if marker not in reader_cpp:
        missing.append(f"reader.cpp: {marker}")
    if marker not in writer_cpp:
        missing.append(f"writer.cpp: {marker}")

if missing:
    print(f"❌ Missing real Metalium kernel API markers: {missing}")
    print("\nCompute kernel snippet:")
    print(compute_cpp[:300])
    sys.exit(1)
else:
    print("✅ Generated kernels use real Metalium APIs")
    print("  - Compute kernel uses ckernel_include.h and compute_kernel_api headers")
    print("  - Reader/Writer kernels use dataflow_api.h")

print("\n=== Verification Complete ===")
