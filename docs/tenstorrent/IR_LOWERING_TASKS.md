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
| **infer_default_tt_schedule** | ✅ Complete | [📄 passes/infer_default_tt_schedule.md](./passes/infer_default_tt_schedule.md) |
| **infer_default_tt_shard** | ✅ Complete | [📄 passes/infer_default_tt_shard.md](./passes/infer_default_tt_shard.md) |
| **grid_to_persistent_tt** | ✅ Complete | [📄 passes/grid_to_persistent_tt.md](./passes/grid_to_persistent_tt.md) |
| **tt_tiles_to_core_map** | ✅ Complete | [📄 passes/tt_tiles_to_core_map.md](./passes/tt_tiles_to_core_map.md) |
| **memory_space_lower_tt** | ✅ Complete | [📄 passes/memory_space_lower_tt.md](./passes/memory_space_lower_tt.md) |
| **tile_pad_tt** | ✅ Complete | [📄 passes/tile_pad_tt.md](./passes/tile_pad_tt.md) |
| **tensorize_tt** | 🟡 Partial | [📄 passes/tensorize_tt.md](./passes/tensorize_tt.md) |
| **verify_tt_ir** | ✅ Complete | [📄 passes/verify_tt_ir.md](./passes/verify_tt_ir.md) |

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

### Priority 1: Tensorize TT Pass (HIGH) 🔴

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

### Priority 2: Update Codegen to Read Annotations (MEDIUM) 🟡

**What**: Modify `codegen_tt_compute_visitor.cc` to read annotations instead of using heuristics

**Status**: ⏳ Pending Task 1 completion

**Current Approach** (heuristics):
```cpp
// BAD: Detects K-loop via variable name
if (loop_var_name.find("kt") != std::string::npos) {
  // Assume it's a K-loop for matmul
}
```

**Target Approach** (annotation-driven):
```cpp
// GOOD: Reads annotation from IR
if (HasAttribute(loop, "tt.matmul_k_loop")) {
  // Read buffer names from annotations
  auto input_bufs = GetAttribute(loop, "tt.input_buffers");
  auto output_buf = GetAttribute(loop, "tt.output_buffer");

  // Emit intrinsics based on annotations
  EmitMatmulIntrinsics(input_bufs, output_buf);
}
```

**File**: `src/target/tt/codegen_tt_compute_visitor.cc`

**Estimated Effort**: 1-2 days

---

### Priority 3: Add Integration Tests (MEDIUM) 🟡

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

### Priority 4: Update Example Matmul (LOW) 🟢

**What**: Update `examples/tenstorrent/example_matmul_tt_poc.py` to use real TileLang operations

**Status**: ⏳ Pending Task 1-3 completion

**Current**: Uses placeholder operations
**Target**: Uses actual `T.gemm()` or manual loops

**File**: `examples/tenstorrent/example_matmul_tt_poc.py`

**Estimated Effort**: 0.5 days

---

## Success Criteria

**Task 1 (Tensorize TT)**:
- [ ] Detects manual matmul loops (3-nested with accumulation)
- [ ] Detects element-wise operations (T.grid patterns)
- [ ] Generates correct nested annotations
- [ ] Existing tests pass + new pattern detection tests pass

**Task 2 (Update Codegen)**:
- [ ] Codegen reads annotations (no heuristics)
- [ ] Emits `matmul_tiles()` for annotated matmul
- [ ] Emits `add_tiles()` for annotated element-wise
- [ ] Emits correct CB operations (wait/pop)
- [ ] Generated code matches Metalium examples

**Task 3 (Integration Tests)**:
- [ ] IR → codegen pipeline tested end-to-end
- [ ] All patterns covered
- [ ] No regressions in existing tests

**Task 4 (Example Update)**:
- [ ] Example uses real operations
- [ ] Generates correct Metalium code
- [ ] Demonstrates full pipeline

---

## Timeline

| Task | Estimated | Dependencies |
|------|-----------|--------------|
| Task 1: Extend tensorize_tt | 2-3 days | None |
| Task 2: Update codegen | 1-2 days | Task 1 |
| Task 3: Integration tests | 1 day | Tasks 1-2 |
| Task 4: Update example | 0.5 days | Tasks 1-3 |
| **Total** | **4.5-6.5 days** | Sequential |

---

## Detailed Pass Documentation

For detailed specifications, implementation notes, and tests for each transform pass, see:

- **Metadata Inference**:
  - [infer_default_tt_schedule.md](./passes/infer_default_tt_schedule.md) - Per-core tile assignment
  - [infer_default_tt_shard.md](./passes/infer_default_tt_shard.md) - DRAM layout descriptors

- **Transform Pipeline**:
  - [grid_to_persistent_tt.md](./passes/grid_to_persistent_tt.md) - Grid → persistent loop
  - [tt_tiles_to_core_map.md](./passes/tt_tiles_to_core_map.md) - Tile assignments → NOC coordinates
  - [memory_space_lower_tt.md](./passes/memory_space_lower_tt.md) - Circular-buffer metadata
  - [tile_pad_tt.md](./passes/tile_pad_tt.md) - Padding metadata (32×32)
  - [tensorize_tt.md](./passes/tensorize_tt.md) - Matmul annotation scaffold ⭐
  - [verify_tt_ir.md](./passes/verify_tt_ir.md) - Constraint verification

---

## Related Documents

- [IR Lowering Analysis](./IR_LOWERING_ANALYSIS.md) - Detailed analysis of GPU vs TT
- [PASS_TABLE.md](./PASS_TABLE.md) - Complete pass reference (60+ passes)
- [TT_ARCHITECTURE.md](./TT_ARCHITECTURE.md) - Complete TT backend architecture

---

**Last Updated**: 2026-02-20
