# InferDefaultTTSchedule Pass

**Status**: ✅ Complete
**Priority**: CRITICAL
**File**: `src/transform/tt/infer_tt_schedule.cc`

---

## Purpose

Compute per-core tile assignments from grid dimensions, determining which tiles each Tensix core will process.

---

## Why Needed

Tenstorrent uses a **persistent loop model** where each core iterates over multiple tiles, unlike GPU's one-block-per-tile model. This pass maps the `T.Kernel(grid_x, grid_y)` grid to static tile assignments.

---

## Implementation

**Input**: Grid kernel with dimensions
```python
with T.Kernel(8, 8) as (bx, by):  # 64 total tiles
    # kernel body
```

**Output**: Schedule metadata attached to IR
```python
tt_num_tiles = 64
tt_grid_x = 8
tt_grid_y = 8
tt_grid_z = 1
tt_tiles_per_core = [[0, 1], [1, 1], ...]
tt_schedule = {
  "policy": "contiguous",
  "order": "row_major",
  "grid_shape": [8, 8, 1],
  "assignments": [
    {"core_id": 0, "start_tile": 0, "tile_count": 1},
    {"core_id": 1, "start_tile": 1, "tile_count": 1},
    ...
  ]
}
```

**Algorithm**:
1. Determine `grid_x/y/z` from `T.Kernel` metadata or blockIdx extents.
2. Compute total tiles: `num_tiles = grid_x * grid_y * grid_z`.
3. Partition tiles contiguously across 64 cores (row-major enumeration).
4. Emit legacy scalar attributes plus consolidated `tt_schedule` map.

---

## Configuration

The pass currently supports the default **contiguous / row-major** policy that is seeded by
`apply_tt_defaults()`. Alternate policies can be introduced by front-end annotations, but the
core algorithm always produces a contiguous slice per core today.

`tt_schedule["policy"]` and `tt_schedule["order"]` are carried through from user or default
attributes so downstream components can branch if additional policies are added.

---

## Tests

**File**: Covered indirectly in `testing/python/tt/test_ws3_grid_to_persistent.py`
**Status**: ✅ Runtime metadata verified alongside persistent transform stage transforms

---

## Dependencies

**Depends On**:
- `apply_tt_defaults()` - Provides default policy if not annotated

**Depended On By**:
- `grid_to_persistent_tt.cc` - Uses schedule metadata to build persistent loops
- Runtime code generation relies on `tt_tiles_per_core` and `tt_schedule`

---

## Related Files

- `src/transform/tt/infer_tt_schedule.cc` - Implementation
- `tilelang/tt/passes.py` - Python binding
- `testing/python/tt/test_passes.py` - Tests

---

## Success Criteria

- [x] Infers tile assignments from grid dimensions
- [x] Emits both legacy scalar attrs and consolidated `tt_schedule` map
- [x] Compatible with downstream persistent transform stage passes

---

**Last Updated**: 2026-02-20
