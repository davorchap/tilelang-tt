# IR Lowering Implementation Tasks

**Document Version:** 2.0
**Date:** 2025-10-09
**Status:** Active

---

## Overview

This document tracks high-level implementation tasks for completing the Tenstorrent backend pattern detection and tensorization.

**Problem**: The TT pipeline still relies on legacy metadata defaults and ad-hoc heuristics (especially for tensorization). Layout-aware metadata, shard-aware lowering, and the associated documentation/test story are unfinished, keeping codegen brittle.

**Solution**: Land the layout-aware metadata passes, refit persistent/codegen stages to consume them, finish tensorization annotations, and refresh documentation/tests so the new flow becomes the default. Then retire legacy compatibility code paths.

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

### Priority 1: Land Layout-Aware Metadata Core (HIGH) 🔴

**What**: Implement the new metadata passes (`InferTTLayout`, `PropagateTTLayout`, `LayoutAwareWorkPartitionTT`) and their Python annotations so PrimFuncs/buffers gain canonical layout+runtime attributes.

**Why**: These passes are the prerequisite for shard-aware lowering, TensorAccessor correctness, and eventual retirement of the legacy metadata inference path.

**Status**: 🟡 Partial – helpers + N-D projection landed; halo diagnostics and advanced validation still pending

**Tasks**:
1. Add `annotate_tt_layout` / `annotate_tt_schedule` helpers with input validation. ✅ (basic helpers merged; extra validation still TODO)
2. Implement `InferTTLayout` with diagnostics, projection helpers, and N-D shard normalization. ✅ (projection + tile-alignment checks shipped; halo/capacity diagnostics still pending)
3. Implement `PropagateTTLayout` to emit `tt.cb.*` metadata per buffer. ⚠️ Emits defaults (depth=2) but lacks role-aware policy
4. Implement `LayoutAwareWorkPartitionTT` to stamp `tt.partition_mode`, `tt.core_ranges`, `tt.runtime_args`, etc. ⚠️ Global mode only; shard-aware path TODO
5. Update docs (`README`, `TT_ARCHITECTURE`, `PASS_TABLE`) to describe the shipped behavior. ⚠️ PASS_TABLE partially updated; remaining docs pending

**Estimated Effort**: 3-4 days

### Priority 2: Shard-Aware Persistent Lowering & Codegen (HIGH) 🔴

**What**: Teach `grid_to_persistent_tt`, host codegen, and TT kernels to consume the new metadata, emit shard-local `(m,n)` math, and enforce TensorAccessor guardrails.

**Why**: Without shard-aware lowering the new metadata is unused; codegen must rely on the canonical runtime args for determinism.

**Status**: 🟡 Partial – persistent pass handles global + basic local_shard; host/runtime wiring still pending

**Tasks**:
1. Extend persistent lowering to branch on `tt_partition_mode` and recover shard-local/global indices. ✅ (local_shard math emitted; more validation still needed)
2. Update host/kernel generation to plumb the expanded runtime arg payload, enforce TA guardrails, and refresh templates. ⛔ Not started
3. Document the final runtime argument contract (architecture + pass docs). ⚠️ PASS_TABLE + pass doc updated; architecture doc pending

**Estimated Effort**: 2-3 days

### Priority 3: Complete Tensorize TT Annotations (MEDIUM) 🟡

**What**: Extend `tensorize_tt.cc` to detect manual matmul and element-wise loops, emitting AttrStmt annotations consumed by compute codegen.

**Why**: Removes heuristic detection and unlocks intrinsic emission driven by metadata.

**Status**: 🟡 Partial (T.gemm intrinsic path only)

**Tasks**:
1. Implement matmul loop matcher (3-nested accumulation) and annotate buffers.
2. Implement element-wise matcher for `T.grid` loops.
3. Strip heuristic paths from compute codegen; rely on annotations.
4. Update `tensorize_tt` documentation with the final matcher matrix.

**Estimated Effort**: 2-3 days

### Priority 4: Integration Test Suite (MEDIUM) 🟡

**What**: Add `testing/python/tt/test_ir_to_codegen_integration.py` covering DRAM vs L1, global vs local-shard, and negative diagnostics aligned with the architecture test matrix.

**Why**: Validates the end-to-end pipeline and prevents regressions as metadata and codegen evolve.

**Status**: ⏳ Blocked on Tasks 1-2

**Tasks**:
1. Build positive coverage for DRAM interleaved, DRAM sharded, and L1 shard scenarios.
2. Add negative tests for halo hints, L1 overflows, and TensorAccessor guardrails.
3. Document the new test matrix in `TT_ARCHITECTURE.md`.

**Estimated Effort**: 1 day

### Priority 5: Refresh Matmul Example & Guides (LOW) 🟢

**What**: Update `examples/tenstorrent/example_matmul_tt_poc.py` and associated documentation to demonstrate the layout-aware pipeline.

**Why**: Developers need accurate reference material once the new flow ships.

**Status**: ⏳ Pending Tasks 1-3

**Tasks**:
1. Replace placeholder loops with annotated TileLang ops mirroring the new metadata flow.
2. Regenerate doc snippets (README, example docs) to match the updated sample.

**Estimated Effort**: 0.5 days

### Priority 6: Deprecate Legacy Metadata Passes (MEDIUM) 🟡

**What**: Gate `infer_default_tt_schedule`, `infer_default_tt_shard`, and `tt_tiles_to_core_map` behind compatibility flags, add deprecation warnings, and plan removal once layout-aware defaults are stable.

**Why**: Cleanly exiting the legacy pipeline avoids double-maintenance and simplifies future codegen logic.

**Status**: ⏳ Pending Tasks 1-2

**Tasks**:
1. Detect/analyse when annotations are absent and fall back gracefully to the legacy defaults.
2. Emit user-facing warnings + documentation updates marking passes as deprecated.
3. After bake-in, remove the legacy passes, update tests, and scrub docs.

**Estimated Effort**: 2 days (post Tasks 1-2)

### Priority 7: Refresh Analysis & Architecture Docs (LOW) 🟢

**What**: Rewrite `IR_LOWERING_ANALYSIS.md`, `PASS_TABLE.md`, and `TT_ARCHITECTURE.md` sections to describe the final layout-aware pipeline instead of “planned” language.

**Why**: Keeps long-form documentation authoritative and prevents confusion for new contributors.

**Status**: ⏳ Pending Tasks 1-6

**Tasks**:
1. Update pipeline diagrams and stage descriptions in `IR_LOWERING_ANALYSIS.md`.
2. Sync pass tables/architecture docs with the new default flow.
3. Remove remaining references to workstreams/legacy-only behavior.

**Estimated Effort**: 1 day

---

## Success Criteria

- [x] `InferTTLayout` emits `tt.buffer.*` for all tensors (now includes N-D projection + L1 alignment checks; halo/capacity diagnostics pending).
- [x] `PropagateTTLayout` attaches `tt.cb.*` with default page size/depth for each buffer (policy tuning pending).
- [x] `LayoutAwareWorkPartitionTT` stamps `tt.partition_mode`, `tt.core_ranges`, and canonical runtime arg names for global mode (local_shard TODO).
- [ ] Documentation updated (`README`, `TT_ARCHITECTURE`) to reflect shipped behavior (PASS_TABLE updated; remaining docs pending).

**Task 2 (Persistent + Codegen Updates)**:
- [x] `GridToPersistentTT` recovers `(m, n)` for both `global` and `local_shard` modes.
- [ ] Host codegen builds TensorAccessor compile args from actual buffers.
- [ ] Runtime args include shard geometry when required; guardrail prevents default TA usage.
- [ ] Runtime argument contract documented for host + kernels.

**Task 3 (Tensorize TT)**:
- [ ] Detects manual matmul loops and element-wise patterns.
- [ ] Emits attr-based annotations consumed by compute codegen.
- [ ] Heuristic paths removed from compute visitor.

**Task 4 (Integration Tests)**:
- [ ] Layout-aware feature matrix covered (DRAM/L1, interleaved/sharded).
- [ ] Negative tests assert diagnostics (halo, L1 overflow, guardrail).
- [ ] Regression suite remains green.
- [ ] Test matrix documented in architecture guide.

**Task 5 (Example Update)**:
- [ ] Example uses real TileLang ops with new annotations.
- [ ] Generated Metalium code validates layout-aware pathways.
- [ ] Example documentation refreshed.

**Task 6 (Legacy Pass Deprecation)**:
- [ ] Legacy passes gated behind compatibility flag with warnings.
- [ ] Documentation notes deprecation path.
- [ ] Legacy pass code/tests removed after bake-in.

**Task 7 (Docs Refresh)**:
- [ ] IR lowering analysis updated to describe new pipeline.
- [ ] Architecture/pass documentation free of “planned” language.
- [ ] References to deprecated passes removed.

---

## Timeline

| Task | Estimated | Dependencies |
|------|-----------|--------------|
| Layout-aware metadata passes | 3-4 days | Python annotations |
| Shard-aware persistent + codegen updates | 2-3 days | Metadata passes |
| Tensorize extensions | 2-3 days | Metadata + codegen |
| Integration tests | 1 day | Tasks 1-2 |
| Example refresh | 0.5 days | Tasks 1-3 |
| Legacy pass deprecation | 2 days | Tasks 1-2 |
| Docs refresh | 1 day | Tasks 1-6 |
| **Total** | **11.5-15.5 days** | Sequential with staging gates |

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
