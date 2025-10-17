#!/usr/bin/env python3
"""Test script to check actual intrinsic op names in IR"""

import tvm
from tvm import tir, IRModule
import tilelang
import tilelang.language as T
from tilelang.utils.target import TENSTORRENT_TARGET
from tilelang.engine.tenstorrent.lower import LowerAndLegalizeTT
from tilelang.tenstorrent import apply_tt_defaults
from tilelang.tenstorrent.passes import (
    infer_tt_layout_v5,
    propagate_tt_layout_v5,
    attach_tensor_accessor_tt,
    layout_aware_work_partition_tt_v5,
    grid_to_core_grid_v5,
    lower_shared_to_cb_v5,
)

# Create GEMM function
@T.prim_func
def gemm(
        A: T.Tensor((256, 256), "float16"),
        B: T.Tensor((256, 256), "float16"),
        C: T.Tensor((256, 256), "float16"),
):
    with T.Kernel(T.ceildiv(256, 32), T.ceildiv(256, 32), threads=128) as (bx, by):
        A_shared = T.alloc_shared((32, 32), "float16")
        B_shared = T.alloc_shared((32, 32), "float16")
        C_local = T.alloc_fragment((32, 32), "float")

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(256, 32), num_stages=3):
            T.copy(A[by * 32, k * 32], A_shared)
            T.copy(B[k * 32, bx * 32], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * 32, bx * 32])

# Get IR and process through stages
mod = IRModule({"main": gemm})
target = tvm.target.Target(TENSTORRENT_TARGET)

# Apply frontend lowering
with target:
    mod = LowerAndLegalizeTT(mod, target)

# Apply v5 passes up to C1
mod = infer_tt_layout_v5(mod)
mod = propagate_tt_layout_v5(mod)
mod = attach_tensor_accessor_tt(mod)
mod = layout_aware_work_partition_tt_v5(mod)
mod = grid_to_core_grid_v5(mod)
mod = lower_shared_to_cb_v5(mod)

print("="*80)
print("BEFORE Pass C2: Checking actual op.name values")
print("="*80)

# Find intrinsic calls
for gvar, func in mod.functions.items():
    if isinstance(func, tir.PrimFunc):
        print(f"\nFunction: {gvar.name_hint}")

        def visit_stmt(stmt):
            if isinstance(stmt, tir.Evaluate):
                if isinstance(stmt.value, tir.Call):
                    call = stmt.value
                    if hasattr(call, 'op'):
                        print(f"\nCall found:")
                        print(f"  call.op type: {type(call.op)}")
                        print(f"  call.op: {call.op}")
                        if hasattr(call.op, 'name'):
                            print(f"  call.op.name: '{call.op.name}'")
                        print(f"  str(call.op): '{str(call.op)}'")
                        print(f"  repr(call.op): {repr(call.op)}")
                        print(f"  args count: {len(call.args)}")

                        # Check if matches patterns
                        if hasattr(call.op, 'name'):
                            op_name = call.op.name
                            if "fill" in op_name:
                                print(f"  ✅ MATCHES fill pattern: '{op_name}'")
                            elif "copy" in op_name:
                                print(f"  ✅ MATCHES copy pattern: '{op_name}'")
                            elif "gemm" in op_name:
                                print(f"  ✅ MATCHES gemm pattern: '{op_name}'")

        tvm.tir.stmt_functor.post_order_visit(func.body, visit_stmt)

print("\n" + "="*80)
print("Now applying Pass C2 and checking again...")
print("="*80)

from tilelang.tenstorrent.passes.lower_tt_tile_intrinsics_v5 import lower_tt_tile_intrinsics_v5
mod = lower_tt_tile_intrinsics_v5(mod)

print("\n" + "="*80)
print("AFTER Pass C2: Checking op names")
print("="*80)

for gvar, func in mod.functions.items():
    if isinstance(func, tir.PrimFunc):
        print(f"\nFunction: {gvar.name_hint}")

        calls = []
        def visit_stmt(stmt):
            if isinstance(stmt, tir.Evaluate):
                if isinstance(stmt.value, tir.Call):
                    call = stmt.value
                    # Check for regular op.name format
                    if hasattr(call, 'op') and hasattr(call.op, 'name'):
                        op_name = call.op.name
                        # For call_extern, extract the function name from args[1]
                        if op_name == "tir.call_extern" and len(call.args) > 1:
                            if isinstance(call.args[1], tir.StringImm):
                                calls.append(f"{call.args[1].value}")
                        else:
                            calls.append(op_name)

        tvm.tir.stmt_functor.post_order_visit(func.body, visit_stmt)

        for i, op_name in enumerate(calls[:10], 1):  # Show first 10
            print(f"  {i}. {op_name}")
        print(f"\nTotal calls found: {len(calls)}")
