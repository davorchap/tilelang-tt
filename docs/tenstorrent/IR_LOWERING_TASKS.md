# IR Lowering Implementation Tasks

**Document Version:** 2.1
**Date:** 2025-10-11
**Status:** Active (Consolidation Plan adopted)

---

## Overview

This document tracks high-level implementation tasks for completing the Tenstorrent backend pattern detection and tensorization.

**Problem**: The TT pipeline still relies on legacy metadata defaults and ad-hoc heuristics (especially for tensorization). Layout-aware metadata, shard-aware lowering, and the associated documentation/test story are unfinished, keeping codegen brittle.

**Solution**: Land the layout-aware metadata passes, refit persistent/codegen stages to consume them, finish tensorization annotations, and refresh documentation/tests so the new flow becomes the default. Then retire legacy compatibility code paths. This tracker now mirrors the 2025-10-11 consolidation proposal in `TT_BACKEND_CONSOLIDATION_PLAN.md` and folds its feedback into the pass/task status below.

---

## Quick Status

| Component | Status | Documentation |
|-----------|--------|---------------|
| **InferTTLayout** | 🟡 Python impl; C++ port pending (Phase 2) | [📄 passes/infer_layout_tt.md](./passes/infer_layout_tt.md) |
| **PropagateTTLayout** | 🟡 Python impl; C++ port pending (Phase 2) | [📄 passes/propagate_layout_tt.md](./passes/propagate_layout_tt.md) |
| **LayoutAwareWorkPartitionTT** | 🟡 Python driver; C++ port pending (Phase 2) | [📄 passes/layout_aware_partition_tt.md](./passes/layout_aware_partition_tt.md) |
| **grid_to_persistent_tt** | 🟡 Consumes new runtime args; diagnostics refresh queued | [📄 passes/grid_to_persistent_tt.md](./passes/grid_to_persistent_tt.md) |

> For a holistic view of the metadata + transform pipeline, refer to
> [TT_ARCHITECTURE.md](TT_ARCHITECTURE.md). This table is deliberately terse so that future edits can
> stay in sync by updating the architecture doc once and linking from strategy trackers like this one.

| **memory_space_lower_tt** | 🟡 Heuristic CB sizing; Phase 2 rework | [📄 passes/memory_space_lower_tt.md](./passes/memory_space_lower_tt.md) |
| **tensorize_tt** | 🟡 Phase 1 focus: `T.gemm` path | [📄 passes/tensorize_tt.md](./passes/tensorize_tt.md) |
| **verify_tt_ir** | 🟡 Needs `T.gemm` schema update | [📄 passes/verify_tt_ir.md](./passes/verify_tt_ir.md) |
| **infer_default_tt_schedule** | 🟡 Legacy (removal tracked in Phase 2) | [📄 passes/infer_default_tt_schedule.md](./passes/infer_default_tt_schedule.md) |
| **infer_default_tt_shard** | 🟡 Legacy (removal tracked in Phase 2) | [📄 passes/infer_default_tt_shard.md](./passes/infer_default_tt_shard.md) |
| **tt_tiles_to_core_map** | 🟡 Legacy (removal tracked in Phase 2) | [📄 passes/tt_tiles_to_core_map.md](./passes/tt_tiles_to_core_map.md) |

---

## Current Focus

- `tensorize_tt` still depends on circular buffer heuristics (`_tile` inflection) and does not yet export a fully deterministic `T.gemm` lowering path.
- `VerifyTTIR` reports against the legacy schema, leaving the layout-aware `tt.runtime_args` unchecked.
- Metadata inference resides in Python helpers; the consolidation feedback requests a C++ port before we delete the legacy pipeline.

---

## Consolidation Roadmap

The 2025-10-11 consolidation plan organizes the work into three phases. This tracker mirrors that structure so engineering tasks and documentation stay aligned.

### Phase 1 – Solidify the `T.gemm` Lowering Path (🚧 In Progress)

*Goal*: Deliver an end-to-end stable compilation path for explicit `T.gemm`, free of CB and runtime heuristics.

- [ ] `refactor-tensorize-tt` — Rescope `tensorize_tt.cc` around explicit `T.gemm` lowering; remove residual heuristic pattern detection.
- [ ] `bug-fix-cb-id-resolution` — Drop the `_tile` suffix fallback and require CB IDs from `tt_circular_buffers`.
- [ ] `testing-update-verify-tt-ir` — Update `VerifyTTIR` to validate the `T.gemm`-specific IR and metadata emitted by the refactored pass.
- [ ] `testing-add-jit-entry-point-tests` — Add `@tilelang.jit(target="tenstorrent")` coverage that inspects generated artifacts for a `T.gemm` kernel.

*Feedback highlights*:
- Ensure all runtime metadata emitted by `tensorize_tt` is canonical so compute codegen becomes a straightforward printer.
- Keep mock-mode JIT output inspection focused on artifact correctness, not numerical execution.

### Phase 2 – Core Infrastructure and Legacy Cleanup (🟡 Planned)

*Goal*: Move layout-aware metadata to C++, consume it across persistent/codegen passes, and eliminate legacy helpers.

- [ ] `refactor-metadata-passes-to-cpp` — Port `infer_tt_layout`, `propagate_tt_layout`, and `layout_aware_work_partition_tt` into `src/transform/tt/`.
- [ ] `refactor-memory-space-lower-tt` — Rework CB sizing to rely solely on `tt.cb.*` attributes.
- [ ] `refactor-grid-to-persistent-tt` — Consume `tt.runtime_arg_names`/`tt.runtime_args` directly inside the pass and drop reinvention logic.
- [ ] `cleanup-deprecate-legacy-passes` — Remove `infer_default_tt_schedule`, `infer_default_tt_shard`, `tt_tiles_to_core_map` after the C++ path is validated.
- [ ] `improvement-strengthen-diagnostics` — Harden layout-aware diagnostics (N-D sharding, L1 capacity, halo hints).
- [ ] `testing-expand-integration-tests` — Grow pytest coverage to exercise the C++ metadata pipeline across shard/layout permutations.

*Feedback highlights*:
- Align the host metadata summary in `main.cpp` with the runtime schema so both sides share a single source of truth.
- Use the new layout-aware metadata to eliminate CB sizing heuristics; treat persistent guardrails as enforced invariants.

### Phase 3 – Documentation and Finalization (🟡 Planned)

*Goal*: Publish the consolidated pipeline and update canonical examples.

- [ ] `docs-refresh-all` — Refresh Tenstorrent docs to present the layout-aware path as the canonical workflow.
- [ ] `docs-update-examples` — Revise `examples/tenstorrent/example_matmul_tt_poc.py` to demonstrate the new annotations and JIT wiring.

*Feedback highlights*:
- Remove “planned” language throughout the doc set once Phases 1-2 land.
- Surface the artifact inspection flow from the mock-mode CI story so contributors know how to validate kernels locally.

---

## Reference Validation Target

- `examples/gemm/example_gemm.py` remains the smoke test for consolidation. A successful Phase 1 pass compiles it via `@tilelang.jit(target="tenstorrent")`, emitting correct C++ and `tt.plan.json` artifacts without manual edits.

---

## Success Criteria

- [ ] `T.gemm` kernels compile end-to-end with deterministic CB/runtime metadata (`tensorize_tt`, `memory_space_lower_tt`, `VerifyTTIR`).
- [ ] Legacy passes (`infer_default_tt_schedule`, `infer_default_tt_shard`, `tt_tiles_to_core_map`) are deleted after bake-in.
- [ ] Tensorization removes CB/memory heuristics and relies exclusively on canonical metadata.
- [ ] Layout-aware metadata inference lives in C++.
- [ ] Mock-mode CI validates the `@tilelang.jit` entry point for `T.gemm` kernels and guards against regressions.

---

## Timeline & Next Milestones

| Phase | Target Window | Dependencies | Notes |
|-------|---------------|--------------|-------|
| Phase 1 | 1.5–2 weeks | Existing Python metadata + current tensorization | Requires agreement on CB ID contract and VerifyTTIR updates. |
| Phase 2 | 2–3 weeks | Phase 1 complete | Blocks on C++ metadata port and CB sizing refactor. |
| Phase 3 | 1 week | Phases 1-2 complete | Documentation refresh and example updates after backend stabilizes. |

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
- [PASS_TABLE_SHARED.md](./PASS_TABLE_SHARED.md), [PASS_TABLE_GPU.md](./PASS_TABLE_GPU.md), [PASS_TABLE_TT.md](./PASS_TABLE_TT.md) - Complete pass references by target (shared, GPU, Tenstorrent)
- [TT_ARCHITECTURE.md](./TT_ARCHITECTURE.md) - Complete TT backend architecture

---

**Last Updated**: 2025-10-11
