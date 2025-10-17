#!/usr/bin/env python3
"""Debug script to dump IR at various stages of GEMM compilation."""

import os
import tilelang
import tilelang.language as T
from tilelang.utils.target import TENSTORRENT_TARGET

# Create the GEMM function
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

# Get initial IR module
from tvm import IRModule
mod = IRModule({"main": gemm})

print("="*80)
print("INITIAL IR (after TileLang frontend)")
print("="*80)
print(mod)

# Now apply TT defaults
from tilelang.tenstorrent import apply_tt_defaults
mod = apply_tt_defaults(mod)

print("\n" + "="*80)
print("AFTER apply_tt_defaults")
print("="*80)
print(mod)

# Now do frontend lowering
from tilelang.engine.tenstorrent.lower import LowerAndLegalizeTT
import tvm
target = tvm.target.Target(TENSTORRENT_TARGET)

with target:
    mod = LowerAndLegalizeTT(mod, target)

print("\n" + "="*80)
print("AFTER LowerAndLegalizeTT (frontend lowering)")
print("="*80)
print(mod)

# Now dump IR before Pass C2 by running just Stage A and B
from tilelang.tenstorrent.passes import (
    infer_tt_layout_v5,
    propagate_tt_layout_v5,
    attach_tensor_accessor_tt,
    layout_aware_work_partition_tt_v5,
    grid_to_core_grid_v5,
    lower_shared_to_cb_v5,
)

# Stage A
mod = infer_tt_layout_v5(mod)
print("\n" + "="*80)
print("AFTER Stage A1: infer_tt_layout_v5")
print("="*80)
print(mod)

mod = propagate_tt_layout_v5(mod)
print("\n" + "="*80)
print("AFTER Stage A2: propagate_tt_layout_v5")
print("="*80)
print(mod)

mod = attach_tensor_accessor_tt(mod)
print("\n" + "="*80)
print("AFTER Stage A3: attach_tensor_accessor_tt")
print("="*80)
print(mod)

# Stage B
mod = layout_aware_work_partition_tt_v5(mod)
print("\n" + "="*80)
print("AFTER Stage B1: layout_aware_work_partition_tt_v5")
print("="*80)
print(mod)

mod = grid_to_core_grid_v5(mod)
print("\n" + "="*80)
print("AFTER Stage B2: grid_to_core_grid_v5")
print("="*80)
print(mod)

# Stage C1
mod = lower_shared_to_cb_v5(mod)
print("\n" + "="*80)
print("AFTER Stage C1: lower_shared_to_cb_v5")
print("="*80)
print(mod)

print("\n" + "="*80)
print("BEFORE Stage C2: lower_tt_tile_intrinsics_v5")
print("This is the critical point - what do the intrinsics look like?")
print("="*80)

# Look specifically for T.clear, T.copy, T.gemm patterns
for gvar, func in mod.functions.items():
    if isinstance(func, tvm.tir.PrimFunc):
        print(f"\nFunction: {gvar.name_hint}")
        print(f"Body type: {type(func.body)}")

        # Walk the IR to find intrinsic calls
        intrinsics = []

        def visit_stmt(stmt):
            if isinstance(stmt, tvm.tir.Evaluate):
                if isinstance(stmt.value, tvm.tir.Call):
                    intrinsics.append({
                        'name': stmt.value.op.name if hasattr(stmt.value.op, 'name') else str(stmt.value.op),
                        'args': len(stmt.value.args),
                        'call': str(stmt.value)
                    })

        tvm.tir.stmt_functor.post_order_visit(func.body, visit_stmt)

        print(f"\nFound {len(intrinsics)} intrinsic calls:")
        for i, intr in enumerate(intrinsics[:10]):  # Show first 10
            print(f"  {i+1}. {intr['name']} (args={intr['args']})")
            print(f"      {intr['call'][:200]}")  # Show first 200 chars

print("\n" + "="*80)
print("This shows what intrinsics are present before Pass C2 should lower them")
print("="*80)
