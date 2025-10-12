# LowerGemmToTTIntrinsics Pass

**Status**: 🟡 Partial  
**Priority**: HIGH  
**File**: `src/transform/tt/lower_gemm_to_tt_intrinsics.cc`

---

## Purpose

Bridge TileLang's `tl.gemm` intrinsics to the Tenstorrent tile-intrinsic sequence consumed by TT codegen.

---

## Current Behavior

- Drops the legacy loop-pattern matcher and consumes frontend `tl.gemm` intrinsics directly (the backend relies on TileLang lowering to supply the GEMM call).
- Strips the enclosing pragma wrapper, expands each `tl.gemm` evaluate node into the TT intrinsic sequence (`tt.tile_regs_acquire`, `tt.mm_init`, `tt.cb_wait_front`, `tt.matmul_tiles`, `tt.cb_pop_front`, `tt.tile_regs_commit`, `tt.cb_reserve_back`, `tt.pack_tile`, `tt.cb_push_back`, `tt.tile_regs_release`).
- Resolves circular-buffer IDs from the layout-aware metadata (`tt_circular_buffers`) with canonical fallbacks (`c0`, `c1`, `c16`) where data is missing.
- Attaches convenience attributes to the `PrimFunc`:
  - `tt_num_matmuls`
  - `tt_has_tensorize`
  - `tt_matmul_patterns` (records basic bookkeeping such as CB IDs and the emitted signature)
- `codegen_tt_compute_visitor.cc` streams the injected TT intrinsics directly instead of re-running pattern detection.

This keeps the TT backend aligned with the CUDA pipeline (`InferFragment` also consumes the same `tl.gemm` intrinsic) while avoiding duplicate loop heuristics.

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

- [x] Translate frontend `tl.gemm` intrinsics into TT-specific annotations
- [x] Count and label matmul regions
- [ ] Propagate richer metadata (loop/reduction info) alongside the intrinsic lowering
- [ ] Handle element-wise tensorization (future)

---

**Last Updated**: 2026-02-20
