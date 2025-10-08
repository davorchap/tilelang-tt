# TileLang IR Transformation Pass Reference

**Document Version:** 1.0
**Date:** 2025-10-08
**Status:** Complete

## Overview

This document provides a comprehensive reference of all IR transformation passes in TileLang, organized by:
- **Shared passes**: Used by both GPU (CUDA/ROCm) and Tenstorrent
- **GPU-specific passes**: Only used for CUDA/ROCm targets
- **TT-specific passes**: Only used for Tenstorrent target

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

## Phase 2A: GPU-Specific Optimization

Applied only for CUDA/ROCm targets via `OptimizeForTarget()`.

### Memory and Synchronization

| Pass | Category | Input IR | Output IR | Purpose | File |
|------|----------|----------|-----------|---------|------|
| **LowerSharedBarrier** | Device | Barrier.arrive | Specific init slot | Lower barrier to device API | `src/transform/lower_shared_barrier.cc` |
| **LowerSharedTmem** | Device | Shared.tmem | Specific init slot | Lower shared tmem | `src/transform/lower_shared_tmem.cc` |

### Hopper-Specific Path (H100 with TMA)

| Pass | Category | Input IR | Output IR | Purpose | File |
|------|----------|----------|-----------|---------|------|
| **IfStmtBinding** | Optimization | If stmt | Bound if stmt | Bind if statements | `src/transform/if_stmt_binding.cc` |
| **MultiVersionBuffer** | Optimization | Single buffer | Multi-version buf | Create buffer versions | `src/transform/multi_version_buffer_rewriter.cc` |
| **WarpSpecialized** | Device | Generic warps | Specialized warps | Warp specialization | `src/transform/warp_specialized_rewriter.cc` |
| **InjectTmaBarrier** | Device | TMA ops | TMA + barrier | Add TMA barriers | `src/transform/inject_tma_barrier.cc` |
| **AnnotateWarpGroupRegAlloc** | Device | Warp groups | Warp group + reg | Annotate reg allocation | `src/transform/annotate_warp_group_reg_alloc.cc` |
| **PipelinePlanning** | Optimization | Sequential | Pipelined | Plan software pipeline | `src/transform/pipeline_planning.cc` |
| **InjectSoftwarePipeline** | Optimization | Planned pipeline | Pipeline + prologue/epilogue | Inject pipeline | `src/transform/inject_pipeline.cc` |
| **LowerOpaqueBlock** | Legalization | Opaque blocks | Lowered blocks | Lower opaque blocks | `src/transform/lower_opaque_block.cc` |
| **MergeIfStmt** | Optimization | Multiple if | Merged if | Merge if statements | `src/transform/merge_if_stmt.cc` |
| **RewriteWgmmaSync** | Device | Wgmma ops | Hopper wgmma | Rewrite for Hopper | `src/transform/wgmma_sync_rewriter.cc` |
| **InjectFenceProxy** | Device | Async ops | Async + fence | Inject fence proxy | `src/transform/inject_fence_proxy.cc` |

### Non-Hopper Path (Ampere, etc.)

| Pass | Category | Input IR | Output IR | Purpose | File |
|------|----------|----------|-----------|---------|------|
| **IfStmtBinding** | Optimization | If stmt | Bound if stmt | (Same as Hopper) | `src/transform/if_stmt_binding.cc` |
| **PlanAndUpdateBufferAllocationLocation** | Memory | Buffer alloc | Planned alloc | Plan buffer locations | TVM built-in |
| **PipelinePlanning** | Optimization | Sequential | Pipelined | (Same as Hopper) | `src/transform/pipeline_planning.cc` |
| **InjectSoftwarePipeline** | Optimization | Planned pipeline | Pipeline impl | (Same as Hopper) | `src/transform/inject_pipeline.cc` |
| **MergeIfStmt** | Optimization | Multiple if | Merged if | (Same as Hopper) | `src/transform/merge_if_stmt.cc` |
| **InjectFenceProxy** | Device | Async ops | Async + fence | (Same as Hopper) | `src/transform/inject_fence_proxy.cc` |

### Common GPU Optimizations

| Pass | Category | Input IR | Output IR | Purpose | File |
|------|----------|----------|-----------|---------|------|
| **NarrowDataType** | Optimization | Wide types | Narrow types | Narrow data types (32-bit) | TVM built-in |
| **FlattenBuffer** | Memory | Multi-dim buffers | 1D buffers | Flatten multi-dim buffers | `src/transform/flatten_buffer.cc` (assumed) |
| **ConfigIndexBitwidth** | Optimization | Index exprs | Optimized index | Config index bitwidth | `src/transform/config_index_bitwidth.cc` |
| **VectorizeLoop** | Optimization | Scalar loops | Vector loops | Vectorize loops | `src/transform/vectorize_loop.cc` |
| **StorageRewrite** | Memory | Buffer allocs | Rewritten storage | Rewrite storage allocations | `src/transform/storage_rewrite.cc` |
| **UnrollLoop** | Optimization | Loops | Unrolled loops | Unroll small loops | TVM built-in |
| **RenormalizeSplitPattern** | Optimization | Split patterns | Normalized splits | Normalize splits | TVM built-in |
| **RemoveNoOp** | Optimization | IR + no-ops | IR without no-ops | Remove no-op statements | TVM built-in |
| **RewriteUnsafeSelect** | Legalization | Unsafe select | Safe select | Rewrite unsafe selects | TVM built-in |
| **HoistIfThenElse** | Optimization | If in loops | Hoisted if | Hoist if out of loops | TVM built-in |
| **VerifyMemory** | Verification | IR | Verified IR | Verify memory accesses | TVM built-in |
| **AnnotateEntryFunc** | Codegen Prep | Func | Annotated func | Annotate entry function | TVM built-in |

### TensorCore and Threading

| Pass | Category | Input IR | Output IR | Purpose | File |
|------|----------|----------|-----------|---------|------|
| **InferFragment** | Device | TensorCore intrinsics | Intrinsics + metadata | Infer TensorCore fragments | TVM: `tensorcore_infer_fragment.cc` |
| **LowerThreadAllreduce** | Device | Thread allreduce | Lowered allreduce | Lower thread allreduce | `src/transform/lower_thread_allreduce.cc` |
| **LowerHopperIntrin** | Device | Hopper intrinsics | Lowered intrinsics | Lower Hopper intrinsics | `src/transform/lower_hopper_intrin.cc` |
| **ThreadSync** | Device | Parallel threads | Threads + sync | Insert thread sync | `src/transform/thread_storage_sync.cc` |

### Host/Device Splitting

| Pass | Category | Input IR | Output IR | Purpose | File |
|------|----------|----------|-----------|---------|------|
| **AnnotateDeviceRegions** | Codegen Prep | IR | IR + device annos | Annotate device regions | `src/transform/annotate_device_regions.cc` |
| **SplitHostDevice** | Codegen Prep | Mixed IR | Host IR + Device IR | Split host and device | TVM built-in |
| **MergeSharedMemoryAllocations** | Memory | Multiple allocs | Merged alloc | Merge shared memory | `src/transform/merge_shared_memory_allocations.cc` |
| **InjectPTXAsyncCopy** | Device | Copies | PTX async copy | Inject PTX async copy | `src/transform/inject_ptx_async_copy.cc` |
| **MakePackedAPI** | Codegen Prep | Func | Packed API func | Create packed API | `src/transform/make_packed_api.cc` |
| **LowerDeviceKernelLaunch** | Codegen Prep | Kernel launch | Lowered launch | Lower kernel launch | `src/transform/lower_device_kernel_launch.cc` |
| **PersistThreadblock** | Device | Threadblock | Persistent threadblock | Make persistent | `src/transform/persist_threadblock.cc` |

**Total:** ~35 GPU-specific passes

**Input:** Legalized TIR from Phase 1

**Output:** Host IR + Device IR ready for CUDA codegen

**Key Transforms:**
- Fragment inference: `wmma.matrix_a` → shape metadata
- Warp specialization: Assign roles to warp groups
- Pipeline injection: Add prologue/epilogue for async copies
- Host/Device split: Separate host and device functions

---

## Phase 2B: Tenstorrent-Specific Optimization

Applied only for Tenstorrent target via `OptimizeForTargetTT()`.

### WS2: Schedule and Sharding Inference

| Pass | Category | Input IR | Output IR | Purpose | File |
|------|----------|----------|-----------|---------|------|
| **infer_default_tt_schedule** | Device | Grid kernel | Kernel + schedule | Compute per-core tile ranges | `src/transform/tt/infer_tt_schedule.cc` |
| **infer_default_tt_shard** | Memory | Buffers | Buffers + shard | Generate DRAM layout descriptors | `src/transform/tt/infer_tt_shard.cc` |

**Annotations Added:**
```python
# Schedule annotation
"tt.schedule": {
  "policy": "contiguous",
  "order": "row_major",
  "assignments": [(core_id, start_tile, count), ...]
}

# Sharding annotation
"tt.shard": {
  "layout": "tiled",
  "interleaved": true,
  "tile_shape": [32, 32]
}
```

### WS3: TIR Transformations

| Pass | Category | Input IR | Output IR | Purpose | File |
|------|----------|----------|-----------|---------|------|
| **grid_to_persistent_tt** | Device | GPU grid kernel | TT persistent kernel | Transform grid → persistent loop | `src/transform/tt/grid_to_persistent_tt.cc` |
| **tt_shard_to_core_map** | Device | Shard IDs | Core (x, y) coords | Map shards to NOC grid | `src/transform/tt/tt_shard_to_core_map.cc` |
| **memory_space_lower_tt** | Memory | DRAM buffers | L1 circular buffers | Lower DRAM → L1 CB | `src/transform/tt/memory_space_lower_tt.cc` |
| **tile_pad_tt** | Memory | Arbitrary shapes | Tile-aligned shapes | Pad to 32×32 tiles | `src/transform/tt/tile_pad_tt.cc` |
| **tensorize_tt** | Device | Loops | Loops + intrinsic annos | Detect patterns, annotate | `src/transform/tt/tensorize_tt.cc` |
| **verify_tt_ir** | Verification | TT IR | Verified TT IR | Verify TT constraints | `src/transform/tt/verify_tt_ir.cc` |

**Example Transform (grid_to_persistent_tt):**

```python
# Before
@T.prim_func
def kernel(...):
  with T.Kernel(8, 8) as (bx, by):  # 64 blocks
    # Compute for block (bx, by)
    C[bx*32:(bx+1)*32, by*32:(by+1)*32] = ...

# After
@T.prim_func
def kernel(...):
  core_id = get_core_id()
  start_tile, tile_count = get_tile_assignment(core_id)

  for tile_id in range(start_tile, start_tile + tile_count):
    bx = tile_id // 8  # Recover bx from tile_id
    by = tile_id % 8   # Recover by from tile_id

    # Original computation
    C[bx*32:(bx+1)*32, by*32:(by+1)*32] = ...
```

**Example Transform (tensorize_tt):**

```python
# Before
for kt in T.serial(Kt):
  for i, j in T.Parallel(32, 32):
    C[m, n] += A[m, kt] * B[kt, n]

# After (annotated)
AttrStmt("tt.matmul_k_loop", kt):
  for kt in T.serial(Kt):
    AttrStmt("tt.input_buffers", [A, B]):
      AttrStmt("tt.output_buffer", C):
        # Original loop body (not emitted by codegen)
        for i, j: C[m, n] += A[m, kt] * B[kt, n]
```

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

**Total:** 6 TT-specific + 11 shared = 17 passes

**Input:** Legalized TIR from Phase 1 + TT defaults

**Output:** TT-ready IR with persistent loops, CB allocations, intrinsic annotations

**Key Transforms:**
- Grid kernel → Persistent kernel with tile assignments
- DRAM buffers → L1 circular buffers
- Manual matmul loops → Annotated with tt.matmul_k_loop
- Buffers padded to 32×32 tile boundaries

---

## Phase 3: Device Codegen

### GPU Device Codegen

| Pass | Category | Input IR | Output IR | Purpose | File |
|------|----------|----------|-----------|---------|------|
| **LowerDeviceStorageAccessInfo** | Codegen Prep | Device IR | Lowered storage | Lower device storage info | `src/transform/lower_device_storage_access_info.cc` |
| **LowerIntrin** | Codegen Prep | TVM intrinsics | Target intrinsics | Lower intrinsics to PTX | TVM built-in |
| **tilelang_cuda / tilelang_hip** | Codegen | TIR | CUDA/HIP source | Generate CUDA/HIP code | `src/target/codegen_cuda.cc`, `codegen_hip.cc` |

**Output:** CUDA source code (`.cu`) or HIP source code (`.cpp`)

### TT Device Codegen

| Component | Input IR | Output Artifacts | Purpose | File |
|-----------|----------|------------------|---------|------|
| **TTReaderCodegenVisitor** | Annotated TIR | `reader.cpp` | Generate reader kernel | `src/target/tt/codegen_tt_reader_visitor.cc` |
| **TTComputeCodegenVisitor** | Annotated TIR | `compute.cpp` | Generate compute kernel | `src/target/tt/codegen_tt_compute_visitor.cc` |
| **TTWriterCodegenVisitor** | Annotated TIR | `writer.cpp` | Generate writer kernel | `src/target/tt/codegen_tt_writer_visitor.cc` |
| **EmitTTHostProgram** | Annotated TIR | `main.cpp` | Generate host program | `src/target/tt/codegen_tt.cc` |
| **EmitTTPlanJSON** | Schedule metadata | `tt.plan.json` | Generate execution plan | `src/target/tt/codegen_tt.cc` |

**Output:** 5 TT artifacts (reader, compute, writer, host, plan)

**Note:** TT codegen is **IR-driven** - visitors walk the annotated TIR and emit code based on annotations (not templates).

---

## Pass Dependencies

### GPU Pipeline Dependencies

```
Phase 1 (Shared)
  ↓
LowerAndLegalize (12 passes)
  ↓
Phase 2A (GPU-Specific)
  ↓
OptimizeForTarget (35 passes)
  ├─ Hopper path (11 passes)
  └─ Non-Hopper path (6 passes)
  ↓
Common optimizations (12 passes)
  ↓
TensorCore & Threading (4 passes)
  ↓
Host/Device Split (7 passes)
  ↓
Phase 3 (Codegen)
  ↓
LowerIntrin → tilelang_cuda
  ↓
CUDA source (.cu)
```

### TT Pipeline Dependencies

```
Phase 0 (TT Defaults)
  ↓
apply_tt_defaults (WS1)
  ↓
Phase 1 (Shared)
  ↓
LowerAndLegalize (12 passes)
  ↓
Phase 2B (TT-Specific)
  ↓
WS2: Schedule/Shard Inference (2 passes)
  ↓
WS3: TIR Transformations (4 passes)
  ↓
WS3: Tensorization (1 pass) ⭐
  ↓
Common optimizations (11 passes)
  ↓
Verification (1 pass)
  ↓
Phase 3 (Codegen)
  ↓
3 Kernel Visitors + Host + Plan
  ↓
5 TT artifacts (.cpp + .json)
```

---

## Transform vs Codegen Responsibilities

| Responsibility | Transform Passes | Codegen |
|----------------|------------------|---------|
| **Pattern Detection** | ✅ Yes (tensorize_tt, InferFragment) | ❌ No |
| **Annotation** | ✅ Yes (AttrStmt with metadata) | ❌ No |
| **IR Optimization** | ✅ Yes (simplify, loop unroll, etc.) | ❌ No |
| **Memory Planning** | ✅ Yes (storage rewrite, buffer merge) | ❌ No |
| **Intrinsic Emission** | ❌ No | ✅ Yes (read annotations, emit code) |
| **Code Generation** | ❌ No | ✅ Yes (emit C++/CUDA/PTX) |

**Key Principle:** Transform passes are "smart" (detect patterns, optimize), codegen is "dumb" (emit based on annotations).

---

## Current TT Gaps

| Gap | Current Behavior | Expected Behavior | Fix |
|-----|------------------|-------------------|-----|
| **K-loop detection** | Codegen heuristics (variable name) | Transform pass annotation | Extend `tensorize_tt.cc` |
| **Intrinsic emission** | Raw array operations emitted | Metalium intrinsics emitted | Update compute visitor |
| **Element-wise ops** | Manual pattern in codegen | Transform pass annotation | Extend `tensorize_tt.cc` |
| **T.gemm() support** | Layout inference fails | Full T.gemm() support | Implement layout inference for TT |

See [IR Lowering Tasks](./IR_LOWERING_TASKS.md) for implementation plan.

---

## Quick Reference

**Total Passes:**
- Shared (Phase 1): 12 passes
- GPU-specific (Phase 2A): ~35 passes
- TT-specific (Phase 2B): 6 passes
- Common optimizations: 11 passes (shared by both)

**Key Files:**
- GPU lowering: `tilelang/engine/phase.py` (LowerAndLegalize, OptimizeForTarget)
- TT lowering: `tilelang/engine/tt/lower.py` (LowerAndLegalizeTT, OptimizeForTargetTT)
- Transform passes: `src/transform/*.cc` and `src/transform/tt/*.cc`
- GPU codegen: `src/target/codegen_cuda.cc`, `codegen_hip.cc`
- TT codegen: `src/target/tt/codegen_tt*.cc`

**References:**
- [IR Lowering Analysis](./IR_LOWERING_ANALYSIS.md) - GPU vs TT architecture comparison
- [IR Lowering Tasks](./IR_LOWERING_TASKS.md) - Implementation tasks for TT gaps
- [Unified Matmul MVP Plan](./UNIFIED_MATMUL_MVP_PLAN.md) - MVP status and history
