# IR Lowering Implementation Tasks

**Document Version:** 2.0
**Date:** 2025-10-09
**Status:** Active

---

## Overview

This document tracks high-level implementation tasks for completing the Tenstorrent backend pattern detection and tensorization.

**Problem**: TT codegen currently uses heuristics for pattern detection instead of relying on transform pass annotations.

**Solution**: Enhance `tensorize_tt` transform pass to detect and annotate patterns, making codegen "dumb" (just emit based on annotations).

---

## Quick Status

| Component | Status | Documentation |
|-----------|--------|---------------|
| **InferTTLayout** | 🚧 Planned | [📄 passes/infer_layout_tt.md](./passes/infer_layout_tt.md) |
| **PropagateTTLayout** | 🚧 Planned | [📄 passes/propagate_layout_tt.md](./passes/propagate_layout_tt.md) |
| **LayoutAwareWorkPartitionTT** | 🚧 Planned | [📄 passes/layout_aware_partition_tt.md](./passes/layout_aware_partition_tt.md) |
| **grid_to_persistent_tt** | 🛠️ Update required | [📄 passes/grid_to_persistent_tt.md](./passes/grid_to_persistent_tt.md) |
| **memory_space_lower_tt** | ✅ Complete (consume new metadata) | [📄 passes/memory_space_lower_tt.md](./passes/memory_space_lower_tt.md) |
| **tensorize_tt** | 🟡 Partial | [📄 passes/tensorize_tt.md](./passes/tensorize_tt.md) |
| **verify_tt_ir** | ✅ Complete | [📄 passes/verify_tt_ir.md](./passes/verify_tt_ir.md) |
| **infer_default_tt_schedule** | 🟡 Legacy | [📄 passes/infer_default_tt_schedule.md](./passes/infer_default_tt_schedule.md) |
| **infer_default_tt_shard** | 🟡 Legacy | [📄 passes/infer_default_tt_shard.md](./passes/infer_default_tt_shard.md) |
| **tt_tiles_to_core_map** | 🟡 Legacy | [📄 passes/tt_tiles_to_core_map.md](./passes/tt_tiles_to_core_map.md) |

---

## Current Problem

**Generated Code Issues:**
- K-loop structure detected ✅
- Scaffolding emitted (mm_init, tile_regs_acquire) ✅
- **Body still has raw array operations** ❌
- Missing intrinsic calls (cb_wait_front, matmul_tiles, cb_pop_front) ❌

**Root Cause**: Pattern detection happens in codegen visitor instead of transform pass.

---

## Implementation Priority

### Priority 1: Layout-Aware Metadata (HIGH) 🔴

**What**: Implement the three new passes that anchor buffer/PrimFunc attributes (`InferTTLayout`, `PropagateTTLayout`, `LayoutAwareWorkPartitionTT`) and ensure existing passes consume them.

**Why**: Enables sharding schema (DRAM & L1), TensorAccessor correctness, and shard-aware persistent lowering.

**Status**: 🚧 Planned (no implementation yet)

**Tasks**:
1. Add Python annotations (`annotate_tt_layout`, `annotate_tt_schedule`) that supply metadata.
2. Implement `InferTTLayout` with validation, projection helpers, and diagnostics.
3. Implement `PropagateTTLayout` to stamp `tt.cb.*` metadata from the inferred layout.
4. Implement `LayoutAwareWorkPartitionTT` to choose partition mode, populate `tt.core_ranges`, `tt.grid_tiles`, and runtime args.

**Estimated Effort**: 3-4 days

### Priority 2: Update GridToPersistent & Codegen (HIGH) 🔴

**What**: Teach `grid_to_persistent_tt` to handle `local_shard` partitioning and update `EmitTTKernels` to pass shard geometry + guardrails.

**Why**: Ensures runtime argument order stays consistent and kernels can reconstruct `(m, n)` coordinates correctly.

**Status**: 🛠️ Spec drafted (see pass docs)

**Tasks**:
1. Extend persistent lowering to emit shard-aware `(m, n)` math.
2. Update codegen to build TensorAccessor compile args, enforce guardrail, and set runtime args.
3. Refresh reader/writer kernel templates to use new runtime args and TensorAccessor usage.

**Estimated Effort**: 2-3 days

### Priority 3: Tensorize TT Pass (MEDIUM) 🟡

**What**: Extend `tensorize_tt.cc` to detect manual matmul and element-wise patterns

**Why**: Enables codegen to emit correct intrinsics based on annotations

**Status**: 🟡 Partial (only T.gemm() intrinsic calls detected)

**Details**: See [passes/tensorize_tt.md](./passes/tensorize_tt.md)

**Tasks**:
1. Implement matmul pattern matcher (3-nested loop with accumulation)
2. Implement element-wise pattern matcher (T.grid operations)
3. Add IR annotations (AttrStmt nodes)
4. Update tests

**Estimated Effort**: 2-3 days

---

### Priority 4: Add Integration Tests (MEDIUM) 🟡

**What**: End-to-end tests for annotated IR → correct codegen

**Status**: ⏳ Pending Task 1-2 completion

**Test Cases**:
1. Manual matmul loop → matmul_tiles intrinsic
2. Element-wise add → add_tiles intrinsic
3. Mixed patterns in single kernel
4. Verify no heuristics remain in codegen

**File**: `testing/python/tt/test_ir_to_codegen_integration.py` (new)

**Estimated Effort**: 1 day

---

### Priority 5: Update Example Matmul (LOW) 🟢

**What**: Update `examples/tenstorrent/example_matmul_tt_poc.py` to use real TileLang operations

**Status**: ⏳ Pending Task 1-3 completion

**Current**: Uses placeholder operations
**Target**: Uses actual `T.gemm()` or manual loops

**File**: `examples/tenstorrent/example_matmul_tt_poc.py`

**Estimated Effort**: 0.5 days

---

## Success Criteria

**Task 1 (Layout-Aware Metadata)**:
- [ ] `InferTTLayout` emits `tt.buffer.*` for all tensors and enforces diagnostics.
- [ ] `PropagateTTLayout` attaches `tt.cb.*` with correct page size/depth/format.
- [ ] `LayoutAwareWorkPartitionTT` stamps `tt.partition_mode`, `tt.core_ranges`, and canonical `tt.runtime_args`.

**Task 2 (Persistent + Codegen Updates)**:
- [ ] `GridToPersistentTT` recovers `(m, n)` for both `global` and `local_shard` modes.
- [ ] Host codegen builds TensorAccessor compile args from actual buffers.
- [ ] Runtime args include shard geometry when required; guardrail prevents default TA usage.

**Task 3 (Tensorize TT)**:
- [ ] Detects manual matmul loops and element-wise patterns.
- [ ] Emits attr-based annotations consumed by compute codegen.

**Task 4 (Integration Tests)**:
- [ ] Layout-aware feature matrix covered (DRAM/L1, interleaved/sharded).
- [ ] Negative tests assert diagnostics (halo, L1 overflow, guardrail).
- [ ] Regression suite remains green.

**Task 5 (Example Update)**:
- [ ] Example uses real TileLang ops with new annotations.
- [ ] Generated Metalium code validates layout-aware pathways.

---

## Timeline

| Task | Estimated | Dependencies |
|------|-----------|--------------|
| Layout-aware metadata passes | 3-4 days | Python annotations |
| Persistent + codegen updates | 2-3 days | Metadata passes |
| Tensorize extensions | 2-3 days | None |
| Integration tests | 1 day | Metadata + codegen |
| Example refresh | 0.5 days | All of the above |
| **Total** | **8.5-11.5 days** | Staged |

---

## Detailed Pass Documentation

For detailed specifications, implementation notes, and tests for each transform pass, see:

- **Metadata Inference**:
  - [infer_layout_tt.md](./passes/infer_layout_tt.md) - Buffer schema, ND sharding
  - [propagate_layout_tt.md](./passes/propagate_layout_tt.md) - CB metadata propagation
  - [layout_aware_partition_tt.md](./passes/layout_aware_partition_tt.md) - Core ranges & runtime args
  - [infer_default_tt_schedule.md](./passes/infer_default_tt_schedule.md) - Legacy defaults
  - [infer_default_tt_shard.md](./passes/infer_default_tt_shard.md) - Legacy shard metadata

- **Transform Pipeline**:
  - [grid_to_persistent_tt.md](./passes/grid_to_persistent_tt.md) - Grid → persistent loop
  - [tt_tiles_to_core_map.md](./passes/tt_tiles_to_core_map.md) - Tile assignments → NOC coordinates
  - [memory_space_lower_tt.md](./passes/memory_space_lower_tt.md) - DRAM → L1 circular buffers
  - [tile_pad_tt.md](./passes/tile_pad_tt.md) - Tile alignment (32×32)
  - [tensorize_tt.md](./passes/tensorize_tt.md) - Pattern detection & annotation ⭐
  - [verify_tt_ir.md](./passes/verify_tt_ir.md) - Constraint verification

---

## Related Documents

- [IR Lowering Analysis](./IR_LOWERING_ANALYSIS.md) - Detailed analysis of GPU vs TT
- [PASS_TABLE.md](./PASS_TABLE.md) - Complete pass reference (60+ passes)
- [TT_ARCHITECTURE.md](./TT_ARCHITECTURE.md) - Complete TT backend architecture

---

**Last Updated**: 2025-10-09
