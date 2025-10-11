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
- Wraps each matched region with a new `AttrStmt` tagged `"tt.matmul_intrinsic"` and assigns a unique matmul ID.
- Attaches convenience attributes to the `PrimFunc`:
  - `tt_num_matmuls`
  - `tt_has_tensorize`
- Rewrites matched reduction loops into TT intrinsics (`tt.tile_regs_acquire`, `tt.mm_init`, `tt.matmul_tiles`, CB wait/pop, `tt.pack_tile`) using default CB indices. Legacy visitors still see a `tt.matmul_intrinsic` wrapper for each injected block.
- Collects matmul metadata—including buffer roles, indices, loop vars, and reduction vars—in `tt_matmul_patterns` so downstream passes/codegen can reason about operand placement.

This minimal functionality allows codegen visitors to enumerate matmul regions without relying on brittle name-based heuristics.

---

## Missing Work

- Pattern matching for handwritten loop nests (no `pragma_gemm`).
- Buffer operand annotations (`tt.input_buffers`, `tt.output_buffer`) and real CB index wiring derived from layout metadata.
- Element-wise pattern detection.
- Integration with reader / writer kernels for non-matmul intrinsics.

These items remain on the roadmap; the current pass provides scaffolding without modifying user loops.

---

## Tests

Covered indirectly via persistent transform stage integration tests that assert `tt_has_tensorize` metadata when `T.gemm()` is used.

---

## Dependencies

**Depends On**: None  
**Depended On By**: `codegen_tt_compute_visitor.cc` (consumes `tt.matmul_intrinsic` annotations)

---

## Success Criteria (current milestone)

- [x] Translate frontend GEMM pragmas into TT-specific annotations
- [x] Count and label matmul regions
- [ ] Detect unannotated loop patterns (future)
- [ ] Handle element-wise tensorization (future)

---

**Last Updated**: 2026-02-20
