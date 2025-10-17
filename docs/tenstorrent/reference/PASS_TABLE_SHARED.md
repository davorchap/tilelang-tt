# TileLang Shared Pass Reference

**Document Version:** 1.0
**Date:** 2025-10-08
**Status:** Complete

## Overview

This reference catalogs TileLang IR transformation passes that are shared across GPU and Tenstorrent backends. Target-specific passes are covered separately in `PASS_TABLE_GPU.md` for GPU and in [passes/README.md](../passes/README.md) for Tenstorrent.

## Pass Categories

- **Frontend**: High-level TileLang DSL → TVM TIR
- **Legalization**: Make IR legal for downstream passes
- **Optimization**: Performance optimizations
- **Memory**: Memory allocation and layout
- **Device**: Device-specific lowering
- **Codegen Prep**: Prepare IR for code generation

---

## Phase 1: Frontend Lowering (Shared)

Applied by both GPU and TT targets via `LowerAndLegalize()`.

| Pass | Category | Input IR | Output IR | Purpose | File |
|------|----------|----------|-----------|---------|------|
| **BindTarget** | Frontend | Unbound IR | Target-bound IR | Bind target info to module | TVM built-in |
| **LetInline** | Frontend | Let expr/stmt | Inlined expr | Inline let expressions | `tilelang/transform/simplify.py` |
| **AddWrapperForSingleBufStore** | Frontend | Single BufferStore | Wrapped BufferStore | Add wrapper for correctness | `tilelang/transform/add_bufstore_wrapper.py` |
| **InjectAssumes** | Frontend | TIR | TIR + assumes | Inject assumes for prover | `src/transform/inject_assumes.cc` |
| **Simplify** | Optimization | Complex expr | Simplified expr | Simplify expressions | `src/transform/simplify.cc` |
| **LayoutReducer** | Memory | Reducers | Reducers + layout | Set layouts for reducers | `src/transform/layout_reducer.cc` |
| **LayoutInference** | Memory | High-level ops | Ops + memory layout | Infer fragment/shared layouts | `src/transform/layout_inference.cc` |
| **LowerTileOp** | Frontend | High-level tile ops | Loops + buffer ops | Lower T.copy, T.fill, etc. | `src/transform/lower_tile_op.cc` |
| **LowerL2Persistent** | Memory | L2 persistent anno | L2 memory ops | Lower L2 persistent map | `src/transform/lower_l2_persistent_annotation.cc` |
| **LegalizeVectorizedLoop** | Legalization | Vector loops | Legal vector loops | Ensure valid vectorization | `src/transform/legalize_vectorized_loop.cc` |
| **LegalizeSafeMemoryAccess** | Legalization | Unsafe mem access | Safe mem access + checks | Add safety checks | `src/transform/legalize_safe_memory_access.cc` |
| **LoopVectorizeDynamic** | Optimization | Dynamic loops | Vectorized loops | Vectorize dynamic shapes | `src/transform/loop_vectorize_dynamic.cc` |

**Total:** 12 passes (shared)

**Input:** TileLang DSL functions (T.Kernel, T.copy, T.gemm, etc.)

**Output:** Legalized TIR with memory layouts inferred

**Key Transforms:**
- `T.copy(A, B)` → `for i, j: B[i,j] = A[i,j]`
- Fragment allocations get memory scope annotations
- Safety checks for dynamic indexing

---

## Shared Optimization Passes

### Common Optimizations (Shared with GPU)

| Pass | Category | Input IR | Output IR | Purpose | File |
|------|----------|----------|-----------|---------|------|
| **FlattenBuffer** | Memory | Multi-dim buffers | 1D buffers | (Same as GPU) | `src/transform/flatten_buffer.cc` (assumed) |
| **ConfigIndexBitwidth** | Optimization | Index exprs | Optimized index | (Same as GPU) | `src/transform/config_index_bitwidth.cc` |
| **Simplify** | Optimization | Complex expr | Simplified expr | (Same as GPU) | `src/transform/simplify.cc` |
| **VectorizeLoop** | Optimization | Scalar loops | Vector loops | (Same as GPU) | `src/transform/vectorize_loop.cc` |
| **StorageRewrite** | Memory | Buffer allocs | Rewritten storage | (Same as GPU) | `src/transform/storage_rewrite.cc` |
| **UnrollLoop** | Optimization | Loops | Unrolled loops | (Same as GPU) | TVM built-in |
| **RenormalizeSplitPattern** | Optimization | Split patterns | Normalized splits | (Same as GPU) | TVM built-in |
| **RemoveNoOp** | Optimization | IR + no-ops | IR without no-ops | (Same as GPU) | TVM built-in |
| **RewriteUnsafeSelect** | Legalization | Unsafe select | Safe select | (Same as GPU) | TVM built-in |
| **HoistIfThenElse** | Optimization | If in loops | Hoisted if | (Same as GPU) | TVM built-in |
| **VerifyMemory** | Verification | IR | Verified IR | (Same as GPU) | TVM built-in |

## Transform vs Codegen Responsibilities

| Responsibility | Transform Passes | Codegen |
|----------------|------------------|---------|
| **Pattern Detection** | ✅ Yes (LowerGemmToTTIntrinsics, InferFragment) | ❌ No |
| **Annotation** | ✅ Yes (AttrStmt with metadata) | ❌ No |
| **IR Optimization** | ✅ Yes (simplify, loop unroll, etc.) | ❌ No |
| **Memory Planning** | ✅ Yes (storage rewrite, buffer merge) | ❌ No |
| **Intrinsic Emission** | ❌ No | ✅ Yes (read annotations, emit code) |
| **Code Generation** | ❌ No | ✅ Yes (emit C++/CUDA/PTX) |

**Key Principle:** Transform passes are "smart" (detect patterns, optimize), codegen is "dumb" (emit based on annotations).

---

## Quick Reference

**Shared Pass Inventory:**
- Phase 1 front-end lowering: 12 passes
- Common post-target optimizations: 11 passes

**Key Shared Files:**
- `tilelang/engine/phase.py` (`LowerAndLegalize`)
- `src/transform/*.cc` (shared TVM/TileLang transforms)
- `tilelang/transform/*.py` (frontend helpers)
