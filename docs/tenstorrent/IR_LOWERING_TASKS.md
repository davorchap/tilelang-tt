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
| **InferTTLayout** | ✅ Complete | [📄 passes/infer_layout_tt.md](./passes/infer_layout_tt.md) |
| **PropagateTTLayout** | ✅ Complete | [📄 passes/propagate_layout_tt.md](./passes/propagate_layout_tt.md) |
| **LayoutAwareWorkPartitionTT** | ✅ Complete | [📄 passes/layout_aware_partition_tt.md](./passes/layout_aware_partition_tt.md) |
| **grid_to_persistent_tt** | 🟡 Follow-up diagnostics | [📄 passes/grid_to_persistent_tt.md](./passes/grid_to_persistent_tt.md) |

> For a holistic view of the metadata + transform pipeline, refer to
> [TT_ARCHITECTURE.md](TT_ARCHITECTURE.md). This table is deliberately terse so that future edits can
> stay in sync by updating the architecture doc once and linking from strategy trackers like this one.
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

### Priority 1: Layout-Aware Metadata Core (COMPLETE)

**What**: Ship the metadata passes (`InferTTLayout`, `PropagateTTLayout`, `LayoutAwareWorkPartitionTT`) and Python helpers so PrimFuncs/buffers expose canonical layout + runtime attributes. The canonical pipeline is documented in [TT_ARCHITECTURE.md](TT_ARCHITECTURE.md).

**Status**: ✅ Complete – core functionality and guardrails landed; remaining follow-ups are targeted diagnostics (halo hints, role-aware CB policy) tracked under future tasks.

**Next follow-ups**:
1. Strengthen halo/L1 capacity diagnostics (see Priority 4).
2. Triage CB depth policy improvements once reuse/multicast design is finalized.
3. Keep documentation in sync by pointing to shared sources instead of duplicating snippets (this file now references the architecture doc directly).

### Priority 2: Shard-Aware Persistent Lowering & Codegen (HIGH) 🟢

**What**: Teach `grid_to_persistent_tt`, host codegen, and TT kernels to consume the new metadata, emit shard-local `(m,n)` math, and enforce TensorAccessor guardrails.

**Why**: Without shard-aware lowering the new metadata is unused; codegen must rely on the canonical runtime args for determinism.

**Status**: 🟢 Active – persistent pass handles global + local_shard, and host/runtime wiring with TensorAccessor guardrails has landed (see the runtime metadata section in [TT_ARCHITECTURE.md](TT_ARCHITECTURE.md#host--kernel-responsibilities)).

**Tasks**:
1. Extend persistent lowering to branch on `tt_partition_mode` and recover shard-local/global indices. ✅ (local_shard math emitted; more validation still needed)
2. Update host/kernel generation to plumb the expanded runtime arg payload, enforce TA guardrails, and refresh templates. ✅ Complete (host metadata summary replaces legacy mock)
3. Document the final runtime argument contract (architecture + pass docs). ✅ Covered in [TT_ARCHITECTURE.md](TT_ARCHITECTURE.md#host--kernel-responsibilities) and linked throughout this tracker.

**Estimated Effort**: 2-3 days

### Priority 3: Tensorize TT Intrinsic Injection (HIGH) 🟡

**What**: Upgrade `tensorize_tt.cc` to inject Tenstorrent tile intrinsics directly into TIR and simplify compute codegen to faithfully emit those intrinsics, mirroring the mature CUDA flow.

**Why**: Moves pattern detection and lowering into the transform pipeline so codegen no longer reverse-engineers loops. This aligns TT tensorization with `tir.transform.InferFragment` on GPU and unlocks deterministic reader/compute/writer emission.

**Status**: 🟡 Partial – GEMM pragmas are detected but we only stamp `tt.matmul_intrinsic` metadata; no intrinsic replacement yet.

**Tasks**:
1. **Define TT intrinsic calls in TIR** ✅  
   - Introduce `call_intrin` helpers (or extern handles) for `mm_init`, `matmul_tiles`, `cb_wait_front`, `cb_pop_front`, `tile_regs_*`, `pack_tile`, `binary_op_init_common`, etc. *(Implemented via `src/target/tt/tt_intrin.cc` and `tilelang/tt/intrin.py`)*  
   - Ensure intrinsics carry CB indices and accumulate flags as explicit operands.
2. **Extend pattern matcher** ✅  
   - Handle both `T.gemm()` AttrStmt and raw K-loop nests (`for kk in range(Kt)`) with reduction semantics. *(Implemented via `MatmulPatternCollector` in `tensorize_tt.cc`, emitting `tt_matmul_patterns` metadata)*  
   - Capture operand buffers, CB IDs, accumulation state, and tile indices.
3. **Inject intrinsic sequence**  
   - Replace matched loop bodies with ordered intrinsic calls (emit `Evaluate(call_intrin("tt.mm_init", ...))`, etc.), preserving persistent loop scaffolding.  
   - Attach buffer metadata (`tt.input_buffers`, `tt.output_buffer`, `tt.cb_roles`) on the enclosing PrimFunc.
4. **Simplify compute codegen**  
   - Update `codegen_tt_compute_visitor.cc` to detect TT intrinsic calls and serialize them verbatim instead of using heuristic loop detection.  
   - Remove legacy pattern state (`current_pattern_`, `elementwise_init_emitted_`, etc.) once intrinsics drive emission.
5. **Element-wise + tilize coverage**  
   - Extend matcher/injection for `T.grid(32, 32)` element-wise loops and tilize/untilize regions using `add_tiles`, `mul_tiles`, `tilize`, `untilize`.
6. **Verification and tests**  
   - Add TIR-level unit tests to assert the intrinsic sequence after `tensorize_tt`.  
   - Update integration tests to confirm compute codegen mirrors the injected operations.  
   - Refresh `tensorize_tt` documentation with intrinsic tables and matcher matrix.

**Estimated Effort**: 3-4 days

### Priority 4: Integration Test Suite (MEDIUM) 🟢

**What**: Add `testing/python/tt/test_ir_to_codegen_integration.py` covering DRAM vs L1, global vs local-shard, and negative diagnostics aligned with the architecture test matrix.

**Why**: Validates the end-to-end pipeline and prevents regressions as metadata and codegen evolve.

**Status**: 🟢 Active – baseline coverage for global vs local shards and guardrail diagnostics in place (halo/L1 overflow coverage still TODO)

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
- [x] Documentation centralized; refer to `README.md` for summary and `docs/tenstorrent/TT_ARCHITECTURE.md` for canonical pipeline (this file now links instead of duplicating details).

**Task 2 (Persistent + Codegen Updates)**:
- [x] `GridToPersistentTT` recovers `(m, n)` for both `global` and `local_shard` modes.
- [x] Host codegen builds TensorAccessor compile args from actual buffers.
- [x] Runtime args include shard geometry when required; guardrail prevents default TA usage.
- [ ] Runtime argument contract documented for host + kernels.

**Task 3 (Tensorize TT)**:
- [ ] Matched regions are rewritten into explicit TT intrinsics (`tt.mm_init`, `tt.matmul_tiles`, `tt.cb_wait_front`, etc.).
- [x] PrimFuncs carry buffer role metadata (`tt_matmul_patterns` with buffer roles/indices, loop vars, reduction var).
- [ ] Compute codegen serializes injected intrinsics without heuristic pattern detection.

**Task 4 (Integration Tests)**:
- [x] Layout-aware feature matrix covered (DRAM/L1, interleaved/sharded).
- [ ] Negative tests assert diagnostics (halo, L1 overflow, guardrail). *(Guardrail path covered; halo/overflow still pending)*
- [x] Regression suite remains green.
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
| Tensorize intrinsic injection | 3-4 days | Metadata + codegen |
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
