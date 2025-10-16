# TileLang GPU Pass Reference

**Document Version:** 1.0
**Date:** 2025-10-08
**Status:** Complete

## Overview

This document describes CUDA and ROCm specific passes in TileLang. Shared passes live in `PASS_TABLE_SHARED.md` and must be applied before the phases captured here.

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

## Phase 3: GPU Device Codegen

### GPU Device Codegen

| Pass | Category | Input IR | Output IR | Purpose | File |
|------|----------|----------|-----------|---------|------|
| **LowerDeviceStorageAccessInfo** | Codegen Prep | Device IR | Lowered storage | Lower device storage info | `src/transform/lower_device_storage_access_info.cc` |
| **LowerIntrin** | Codegen Prep | TVM intrinsics | Target intrinsics | Lower intrinsics to PTX | TVM built-in |
| **tilelang_cuda / tilelang_hip** | Codegen | TIR | CUDA/HIP source | Generate CUDA/HIP code | `src/target/codegen_cuda.cc`, `codegen_hip.cc` |

**Output:** CUDA source code (`.cu`) or HIP source code (`.cpp`)

## GPU Pipeline Dependencies

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

## Quick Reference

**GPU-Specific Pass Count:** ~35 (Phase 2A)

**Key GPU Files:**
- `tilelang/engine/phase.py` (OptimizeForTarget)
- `src/transform/*.cc` (GPU transforms)
- `src/target/codegen_cuda.cc`, `src/target/codegen_hip.cc` (device codegen)
