# Tenstorrent Backend Tasks

**Document Version:** 3.0
**Date:** 2025-10-11
**Status:** Active

---

## Overview

This document captures the end-to-end task tracker for consolidating the TileLang Tenstorrent (TT) backend. The focus is on finalizing the layout-aware metadata pipeline, eliminating legacy heuristics, and stabilizing the `@tilelang.jit(target="tenstorrent")` entry point. The work mirrors the backend flow that runs from Python orchestration helpers (`tilelang/tenstorrent`) through TT-specific transforms (`src/transform/tenstorrent/`) into IR-driven visitors under `src/target/tenstorrent/`, with host codegen now emitting a metadata summary aligned with runtime argument schemas.

### Out of Scope
- Handling tensor dimensions that are not wholly tile-aligned (32×32) — tracked separately once the consolidated path ships.
- Heuristic matmul pattern detection in `LowerGemmToTTIntrinsics`; the consolidation prioritizes explicit `T.gemm` lowering.
- Supporting multiple matmuls per reduction loop inside `LowerGemmToTTIntrinsics`; this extension follows after the primary path is stable.

### Architecture Snapshot
- **V5 metadata-driven pipeline**: 14 Python passes in stages A-E provide canonical `tt.buffer.*`, `tt.cb.*`, and `tt.runtime_args` metadata that downstream passes consume without heuristics.
- **Shard-aware persistent lowering**: Grid-to-core transformation converts tiled kernels into persistent loops using the canonical runtime schema; host codegen emits a metadata summary in `main.cpp` that mirrors runtime argument payloads.
- **Python implementation**: All TT backend passes are implemented in Python for maintainability and rapid iteration. No C++ migration planned.
- **Mock-mode validation**: Mock CI remains the primary validation path; real SDK (`--with-metalium`) flows are unchanged but now depend on the consolidated metadata emitted by the host/runtime stack.

---

## Quick Status

| Component | Status | Documentation |
|-----------|--------|---------------|
| **InferTTLayout** | 🟡 Python impl; C++ port pending (Phase 2) | [📄 passes/infer_tt_layout.md](./passes/infer_tt_layout.md) |
| **PropagateTTLayout** | 🟡 Python impl; C++ port pending (Phase 2) | [📄 passes/propagate_tt_layout.md](./passes/propagate_tt_layout.md) |
| **LayoutAwareWorkPartitionTT** | 🟡 Python driver; C++ port pending (Phase 2) | [📄 passes/layout_aware_work_partition_tt.md](./passes/layout_aware_work_partition_tt.md) |
| **grid_to_persistent_tt** | 🟡 Consumes new runtime args; diagnostics refresh queued | [📄 passes/grid_to_persistent_tt.md](./passes/grid_to_persistent_tt.md) |

> For a holistic view of the metadata + transform pipeline, refer to [TT_ARCHITECTURE.md](TT_ARCHITECTURE.md). This table stays terse so that future edits maintain a single source of truth.

| **memory_space_lower_tt** | 🟡 Heuristic CB sizing; Phase 2 rework | [📄 passes/memory_space_lower_tt.md](./passes/memory_space_lower_tt.md) |
| **LowerGemmToTTIntrinsics** | 🟡 Phase 1 focus: `T.gemm` path | [📄 passes/lower_gemm_to_tt_intrinsics.md](./passes/lower_gemm_to_tt_intrinsics.md) |
| **verify_tt_ir** | 🟡 Needs `T.gemm` schema update | [📄 passes/verify_tt_ir.md](./passes/verify_tt_ir.md) |
| **infer_default_tt_schedule** | 🟡 Legacy (removal tracked in Phase 2) | [📄 passes/infer_default_tt_schedule.md](./passes/infer_default_tt_schedule.md) |
| **infer_default_tt_shard** | 🟡 Legacy (removal tracked in Phase 2) | [📄 passes/infer_default_tt_shard.md](./passes/infer_default_tt_shard.md) |
| **tt_tiles_to_core_map** | 🟡 Legacy (removal tracked in Phase 2) | [📄 passes/tt_tiles_to_core_map.md](./passes/tt_tiles_to_core_map.md) |

---

## Immediate Focus

- `LowerGemmToTTIntrinsics` now targets explicit `T.gemm` regions; it still falls back to `_tile` CB heuristics until `bug-fix-cb-id-resolution` lands.
- `VerifyTTIR` reports against the legacy schema, leaving the layout-aware `tt.runtime_args` unchecked.
- Metadata inference currently lives in Python helpers; a C++ port is required before deleting legacy pipelines.

---

## Phased Consolidation Plan

The work is organized into three sequential phases with clear responsibilities. The tables below capture the authoritative task list.

### Phase 1 – Solidify the `T.gemm` Lowering Path (🚧 In Progress)

*Goal*: Achieve a stable, end-to-end compilation path for the explicit `T.gemm` operator.

| Task ID | Description |
|---------|-------------|
| `refactor-tensorize-tt` | ✅ **Refactor `lower_gemm_to_tt_intrinsics.cc` to Prioritize `T.gemm` Lowering**: Explicit GEMM markers drive tensorization; next iteration will consume `tl.gemm` intrinsics directly and drop the residual matcher. |
| `bug-fix-cb-id-resolution` | **Remove CB ID Heuristics**: Delete `_tile` suffix fallbacks so `LowerGemmToTTIntrinsics` relies strictly on `tt_circular_buffers` metadata. |
| `testing-update-verify-tt-ir` | **Update `VerifyTTIR` for `T.gemm`**: Extend the verifier to match the IR and metadata generated by the new lowering path. |
| `testing-add-jit-entry-point-tests` | **Add JIT Entry Point Tests for `T.gemm`**: Validate `@tilelang.jit(target="tenstorrent")` artifact generation for representative kernels. |

*Feedback highlights*:
- Ensure all runtime metadata emitted by `LowerGemmToTTIntrinsics` is canonical so compute codegen remains a straightforward printer.
- Keep mock-mode JIT validation focused on artifact correctness rather than numerical execution.

### Phase 2 – Core Infrastructure and Legacy Cleanup (🟡 Outdated - v5 completed)

**NOTE**: This phase description is outdated. The v5 Python pipeline has been completed and is now the default. See `planning/TT_Pass_Status.md` for current status.

*Original Goal*: Eliminate legacy helpers and remove circular buffer heuristics.

| Task ID | Description | Status |
|---------|-------------|--------|
| `cleanup-deprecate-legacy-passes` | **Deprecate and Remove Legacy Passes**: Retire old pass implementations | ✅ Complete (PR #135) |
| `improvement-strengthen-diagnostics` | **Strengthen Metadata Diagnostics**: Improve error messages for N-D sharding, halo hints, and L1 capacity checks | 🟡 Ongoing |
| `testing-expand-integration-tests` | **Expand Integration Test Suite**: Add pytest coverage for sharding/layout permutations | 🟡 Ongoing |

*Implementation Note*:
- **All TT backend passes remain in Python** - No C++ migration planned. Python implementation provides better maintainability and rapid iteration.

### Phase 3 – Documentation and Finalization (🟡 Planned)

*Goal*: Present the layout-aware pipeline as the canonical workflow and refresh examples.

| Task ID | Description |
|---------|-------------|
| `docs-refresh-all` | **Refresh All Backend Documentation**: Update Tenstorrent docs to remove “planned” language and describe the consolidated pipeline. |
| `docs-update-examples` | **Update Matmul Example**: Revise `examples/tenstorrent/example_matmul_tt_poc.py` to showcase the new annotations and JIT flow. |

*Feedback highlights*:
- Remove "planned" wording after Phases 1–2 land.
- Document the artifact inspection flow used by mock-mode CI so contributors can validate kernels locally.

---

## Reference Validation Target

`examples/gemm/example_gemm.py` remains the smoke-test kernel. Consolidation is complete when it compiles via `@tilelang.jit(target="tenstorrent")`, producing correct C++ artifacts and a `tt.plan.json` without manual edits.

---

## Success Criteria

- [ ] `T.gemm` kernels compile end-to-end with deterministic CB/runtime metadata (`LowerGemmToTTIntrinsics`, `memory_space_lower_tt`, `VerifyTTIR`).
- [ ] Legacy passes (`infer_default_tt_schedule`, `infer_default_tt_shard`, `tt_tiles_to_core_map`) are deleted after bake-in.
- [ ] The `T.gemm` lowering path is heuristic-free, relying solely on canonical metadata.
- [ ] Layout-aware metadata inference lives in C++.
- [ ] Mock-mode CI validates the `@tilelang.jit` entry point and guards against regressions.

---

## Timeline & Next Milestones

| Phase | Target Window | Dependencies | Notes |
|-------|---------------|--------------|-------|
| Phase 1 | 1.5–2 weeks | Existing Python metadata + current tensorization | Requires agreement on CB ID contract and VerifyTTIR updates. |
| Phase 2 | 2–3 weeks | Phase 1 complete | Blocks on C++ metadata port and CB sizing refactor. |
| Phase 3 | 1 week | Phases 1–2 complete | Documentation refresh and example updates after backend stabilizes. |

---

## Detailed Pass Documentation

For specifications, implementation notes, and tests for individual passes, see:

- **Metadata Inference**:
  - [infer_tt_layout.md](./passes/infer_tt_layout.md) – Buffer schema, N-D sharding.
  - [propagate_tt_layout.md](./passes/propagate_tt_layout.md) – CB metadata propagation.
  - [layout_aware_work_partition_tt.md](./passes/layout_aware_work_partition_tt.md) – Core ranges & runtime args.
  - [infer_default_tt_schedule.md](./passes/infer_default_tt_schedule.md) – Legacy defaults.
  - [infer_default_tt_shard.md](./passes/infer_default_tt_shard.md) – Legacy shard metadata.

- **Transform Pipeline**:
  - [grid_to_persistent_tt.md](./passes/grid_to_persistent_tt.md) – Grid → persistent loop.
  - [tt_tiles_to_core_map.md](./passes/tt_tiles_to_core_map.md) – Tile assignments → NOC coordinates.
  - [memory_space_lower_tt.md](./passes/memory_space_lower_tt.md) – DRAM → L1 circular buffers.
  - [tile_pad_tt.md](./passes/tile_pad_tt.md) – Tile alignment (32×32).
  - [lower_gemm_to_tt_intrinsics.md](./passes/lower_gemm_to_tt_intrinsics.md) – Pattern detection & annotation ⭐.
  - [verify_tt_ir.md](./passes/verify_tt_ir.md) – Constraint verification.

---

## Related Documents

- [IR_LOWERING_ANALYSIS.md](./IR_LOWERING_ANALYSIS.md) – GPU vs TT analysis.
- [PASS_TABLE_SHARED.md](./PASS_TABLE_SHARED.md), [PASS_TABLE_GPU.md](./PASS_TABLE_GPU.md), [PASS_TABLE_TT.md](./PASS_TABLE_TT.md) – Pass references by target.
- [TT_ARCHITECTURE.md](./TT_ARCHITECTURE.md) – Comprehensive TT backend architecture.

---

**Last Updated**: 2025-10-11
