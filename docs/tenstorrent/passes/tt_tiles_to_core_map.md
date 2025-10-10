# TTTilesToCoreMap Pass

**Status**: 🟡 Legacy (compatibility shim)  
**Priority**: MEDIUM  
**File**: `src/transform/tt/tt_tiles_to_core_map.cc`

---

## Purpose

Translate the legacy tile scheduling metadata (`tt_tiles_per_core`) into the physical
topology required by the Tenstorrent runtime (core range sets and per-core runtime arguments).
When layout-aware metadata is available, `LayoutAwareWorkPartitionTT` emits the canonical
`tt.core_ranges` and `tt.runtime_args`, making this pass a fallback for unannotated kernels.

---

## Why Needed

Tenstorrent devices address cores by `(x, y)` coordinates on an 8 × 8 mesh (for Grayskull/Wormhole).
The legacy metadata inference path produces linear tile assignments per core; the runtime expects:
- Physical `CoreRange` descriptors describing which cores participate.
- An ordered list of per-core runtime arguments (`start_id`, `count`).

This pass performs the conversion in the legacy pipeline.

---

## Implementation

1. Read `tt_tiles_per_core` and `tt_num_cores` (produced by `infer_default_tt_schedule`).
2. For each core ID:
   - Convert the linear index to `(x, y)` using row-major mapping.
   - Create a single-core `CoreRange` entry `[start_x, start_y, end_x, end_y, start_id, count]`.
   - Emit runtime arguments `[start_id, count]`.
3. Attach metadata:
   ```python
   "tt_core_ranges" = [
     [0, 0, 0, 0, 0, 1],   # core (0,0) gets tile 0, count 1
     [1, 0, 1, 0, 1, 1],   # core (1,0) gets tile 1, count 1
     ...
   ]
   "tt_core_runtime_args" = [
     [0, 1],
     [1, 1],
     ...
   ]
   ```

Future optimisations may merge adjacent cores into rectangular ranges. In the layout-aware
pipeline this responsibility moves to `LayoutAwareWorkPartitionTT`, which can emit dense
`CoreRangeSet` descriptors directly from shard geometry.

---

## Tests

**File**: `testing/python/tt/test_tt_tiles_to_core_map.py`  
**Status**: ✅ Covers basic mapping, coordinate validation, runtime arg structure, and legacy metadata integration.

---

## Dependencies

**Depends On**:
- `infer_default_tt_schedule.cc` (provides `tt_tiles_per_core` when annotations are missing)
- `LayoutAwareWorkPartitionTT` *(preferred)* — supersedes this pass for annotated pipelines

**Depended On By**:
- TT codegen visitors (reader / compute / writer) when the layout-aware metadata is unavailable
  and legacy defaults must be honoured.

---

## Success Criteria

- [x] Converts legacy tile assignments into physical coordinates
- [x] Emits per-core runtime arguments aligned with the persistent loop contract
- [x] Leaves inactive cores with zero tiles
- [x] Validated by unit tests

---

**Last Updated**: 2025-10-10
