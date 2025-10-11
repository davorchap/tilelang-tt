# Tenstorrent Backend Consolidation Plan

**Date**: 2025-10-11
**Status**: Proposed

## 1. Overview

This document outlines a focused plan to consolidate the TileLang Tenstorrent (TT) backend, prioritize stability, and finalize the transition to the new layout-aware metadata pipeline. The primary goal is to clean up the existing implementation, remove legacy code, and refactor key components to create a robust and maintainable foundation.

### Out of Scope for This Plan
- **Handling of Non-Tile-Aligned Dimensions**: Support for tensor dimensions that are not perfectly divisible by the 32x32 tile size is a follow-up task. The focus is on tile-aligned inputs.
- **Heuristic Matmul Pattern Detection**: The `tensorize_tt` pass has logic to detect matmul patterns from raw loops. This is not a priority. The focus is on correctly lowering the explicit `T.gemm` operator.
- **Multiple Matmuls Per Loop**: The `tensorize_tt` pass currently only supports a single matmul pattern per reduction loop. Extending this is a future enhancement, not part of this consolidation.

---

## 2. Phased Consolidation Plan

The work is organized into three sequential phases, each with a clear goal.

### Phase 1: Solidify the `T.gemm` Lowering Path
*Goal: Achieve a stable, end-to-end compilation path for the explicit `T.gemm` operator.*

| Task ID | Description |
|---|---|
| `refactor-tensorize-tt` | **Refactor `tensorize_tt.cc` to Prioritize `T.gemm` Lowering**: The current implementation in `tensorize_tt.cc` handles both explicit `T.gemm` lowering and heuristic pattern detection. This task focuses on refactoring the pass to create a clean, robust, and well-tested path for lowering the `T.gemm` operator. |
| `bug-fix-cb-id-resolution` | **Remove CB ID Heuristics**: In `tensorize_tt.cc`, the logic for resolving Circular Buffer IDs includes a heuristic fallback (e.g., appending a `_tile` suffix). This must be removed. The pass must strictly rely on the `tt_circular_buffers` metadata. |
| `testing-update-verify-tt-ir` | **Update `VerifyTTIR` for `T.gemm`**: Ensure the `VerifyTTIR` pass can validate the specific IR produced by the `T.gemm` lowering path. |
| `testing-add-jit-entry-point-tests`| **Add JIT Entry Point Tests for `T.gemm`**: Create initial tests to confirm the `@tilelang.jit` flow produces the expected artifacts for a `T.gemm` kernel. |

### Phase 2: Core Infrastructure and Legacy Cleanup
*Goal: Migrate core logic to C++ and remove the old pipeline now that the main `T.gemm` path is stable.*

| Task ID | Description |
|---|---|
| `refactor-metadata-passes-to-cpp` | **Refactor Metadata Passes to C++**: With the `T.gemm` path stable, undertake the larger refactoring of the Python layout-aware metadata passes (`infer_tt_layout`, `propagate_tt_layout`, `layout_aware_work_partition_tt`) into C++ within `src/transform/tt/`. |
| `refactor-memory-space-lower-tt` | **Rework `MemorySpaceLowerTT`**: This pass still uses heuristics for sizing CBs. It should be reworked to consume the canonical `tt.cb.*` attributes, making CB configuration explicit and deterministic. |
| `refactor-grid-to-persistent-tt` | **Update `GridToPersistentTT`**: Ensure the pass strictly consumes the canonical `tt.runtime_arg_names` and `tt.runtime_args` metadata. Any legacy logic for generating runtime arguments should be removed. |
| `cleanup-deprecate-legacy-passes` | **Deprecate and Remove Legacy Passes**: Once the new C++ metadata pipeline is validated, the old passes (`infer_default_tt_schedule`, `infer_default_tt_shard`, `tt_tiles_to_core_map`) can be safely removed. |
| `improvement-strengthen-diagnostics` | **Strengthen Metadata Diagnostics**: Enhance the newly refactored C++ layout-aware passes with stronger validation and clearer error messages, especially for N-D sharding and L1 memory capacity. |
| `testing-expand-integration-tests`| **Expand Integration Test Suite**: Add more tests to cover the newly refactored C++ pipeline and its handling of different sharding and layout scenarios. |


### Phase 3: Documentation and Finalization
*Goal: Update all documentation and examples to reflect the consolidated state of the backend.*

| Task ID | Description |
|---|---|
| `docs-refresh-all` | **Refresh All Backend Documentation**: Update all Tenstorrent-related documentation (`docs/tenstorrent/`, `README.md`) to remove "planned" language and present the layout-aware pipeline as the single, canonical workflow. |
| `docs-update-examples` | **Update Matmul Example**: Update `examples/tenstorrent/example_matmul_tt_poc.py` to use the new layout-aware annotations, providing a clean and current reference for developers. |

---

## 3. Reference Use Case: `examples/gemm/example_gemm.py`

To ground this consolidation effort, the standard `examples/gemm/example_gemm.py` will serve as a key validation test case. Successfully compiling this example through the Tenstorrent backend without modification is a primary goal.

This example relies on features that test the core of the TT backend pipeline:
- **Shared Memory (`T.alloc_shared`)**: This must be correctly lowered to L1 Circular Buffers by a deterministic `memory_space_lower_tt` pass. This highlights the need for **`refactor-memory-space-lower-tt`**.
- **Explicit `T.gemm` Operator**: The `tensorize_tt` pass must correctly lower the explicit `T.gemm` call within the pipelined loop into the appropriate data movement and compute intrinsics. This requires the work in **`refactor-tensorize-tt`** and **`bug-fix-cb-id-resolution`**.
- **Default GPU-Style Kernel**: The example is a generic GPU-style kernel. Making it work on Tenstorrent without TT-specific annotations is the central promise of the backend. This validates the entire default pipeline, from metadata inference to codegen.

**Note on Execution**: For the TT backend, the "execution" of a JIT-compiled kernel in mock mode involves generating source artifacts (`*.cpp`, `*.json`), not performing numerical computation. The validation for this use case would involve inspecting the generated files for correctness, rather than comparing numerical outputs as is done in the example's `main()` function.

---

## 4. Success Criteria

This consolidation effort will be considered complete when the following criteria are met:

- [ ] **`T.gemm` End-to-End**: A TileLang program using the `T.gemm` operator (such as `example_gemm.py`) can be successfully compiled via `@tilelang.jit(target="tenstorrent")`, generating complete and correct C++ artifacts and a `tt.plan.json`.
- [ ] **Legacy Passes Removed**: The legacy passes (`infer_default_tt_schedule`, `infer_default_tt_shard`, `tt_tiles_to_core_map`) have been fully removed from the codebase and CI.
- [ ] **Heuristics Eliminated for `T.gemm`**: The `T.gemm` lowering path is free of heuristics for CB ID resolution and memory management, relying instead on the canonical metadata pipeline.
- [ ] **Core Metadata Passes in C++**: The core layout-aware metadata inference logic lives in C++ (`src/transform/tt/`), not Python.
- [ ] **CI Validation**: The CI includes dedicated tests that validate the `@tilelang.jit` entry point and the correctness of generated artifacts for a `T.gemm` kernel, preventing future regressions.
