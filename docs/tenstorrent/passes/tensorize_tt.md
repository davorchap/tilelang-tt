# TensorizeTT Pass

**Status**: 🟡 Partial  
**Priority**: HIGH  
**File**: `src/transform/tt/tensorize_tt.cc`

---

## Purpose

Bridge between TileLang's high level matmul pragmas and Tenstorrent codegen by annotating TIR with TT-specific intrinsic metadata.

---

## Current Behavior

- Walks the TIR, looking for `AttrStmt` markers inserted by the frontend (`"pragma_gemm"`, `"tl.gemm"`, `"gemm_operation"`).
- Rewrites matched regions into TT intrinsics (`tt.tile_regs_acquire`, `tt.mm_init`, `tt.matmul_tiles`, CB wait/pop, `tt.pack_tile`) using circular-buffer IDs resolved from `tt_circular_buffers` metadata (falls back to canonical `c0/c1/c16` when missing).
- Attaches convenience attributes to the `PrimFunc`:
  - `tt_num_matmuls`
  - `tt_has_tensorize`
- Collects matmul metadata—including buffer roles, indices, loop vars, reduction vars, and resolved CB IDs—in `tt_matmul_patterns` so downstream passes/codegen can reason about operand placement.
- `codegen_tt_compute_visitor.cc` now streams the injected TT intrinsics directly instead of reinventing pattern detection; the legacy `tt.matmul_intrinsic` wrapper has been retired.

This minimal functionality allows codegen visitors to enumerate matmul regions without relying on brittle name-based heuristics.

---

## Missing Work

- Element-wise pattern detection.
- Integration with reader / writer kernels for non-matmul intrinsics.

These items remain on the roadmap; the current pass provides scaffolding without modifying user loops.

---

## Tests

Covered indirectly via persistent transform stage integration tests that assert `tt_has_tensorize` metadata when `T.gemm()` is used.

---

## Dependencies

**Depends On**: None  
**Depended On By**: `codegen_tt_compute_visitor.cc` (emits the injected TT intrinsics)

---

## Success Criteria (current milestone)

- [x] Translate frontend GEMM pragmas into TT-specific annotations
- [x] Count and label matmul regions
- [x] Detect unannotated loop patterns (manual K-loop matcher)
- [ ] Handle element-wise tensorization (future)

---

**Last Updated**: 2026-02-20
