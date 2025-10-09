# TTTilesToCoreMap Pass

**Status**: ✅ Complete  
**Priority**: MEDIUM  
**File**: `src/transform/tt/tt_tiles_to_core_map.cc`

---

## Purpose

Translate the logical tile scheduling metadata from Stage 2 (`tt_tiles_per_core`) into the physical topology required by the Tenstorrent runtime (core range sets and per-core runtime arguments).

---

## Why Needed

Tenstorrent devices address cores by `(x, y)` coordinates on an 8 × 8 mesh (for Grayskull/Wormhole). metadata inference stage produces linear tile assignments per core; the runtime expects:
- Physical CoreRange descriptors describing which cores participate.
- An ordered list of per-core runtime arguments (`start_tile`, `tile_count`).

This pass performs the conversion.

---

## Implementation

1. Read `tt_tiles_per_core` and `tt_num_cores` (produced by `infer_default_tt_schedule`).
2. For each core ID:
   - Convert the linear index to `(x, y)` using row-major mapping.
   - Create a single-core `CoreRange` entry `[start_x, start_y, end_x, end_y, start_tile, count]`.
   - Emit runtime arguments `[start_tile, count]`.
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

Future optimisations may merge adjacent cores into rectangular ranges, but the baseline feature set emits one range per active core.

---

## Tests

**File**: `testing/python/tt/test_tt_tiles_to_core_map.py`  
**Status**: ✅ Covers basic mapping, coordinate validation, runtime arg structure, and metadata inference stage integration.

---

## Dependencies

**Depends On**:
- `infer_default_tt_schedule.cc` (provides `tt_tiles_per_core`)

**Depended On By**:
- TT codegen visitors (reader / compute / writer) when materialising per-core launch parameters

---

## Success Criteria

- [x] Converts metadata inference stage tile assignments into physical coordinates
- [x] Emits per-core runtime arguments aligned with the persistent loop contract
- [x] Leaves inactive cores with zero tiles
- [x] Validated by unit tests

---

**Last Updated**: 2026-02-20
