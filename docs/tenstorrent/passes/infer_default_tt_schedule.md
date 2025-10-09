# InferDefaultTTSchedule Pass

**Status**: 🟡 Legacy (superseded by layout-aware pipeline)  
**Priority**: P1 (compatibility)
**File**: `src/transform/tt/infer_tt_schedule.cc`

---

## Purpose

Compute per-core tile assignments from grid dimensions, determining which tiles each Tensix core will process. In the layout-aware roadmap this pass becomes a compatibility shim that seeds defaults when users do not provide explicit annotations. Downstream passes (`LayoutAwareWorkPartitionTT`) will override the metadata with buffer-derived shard geometry.

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
{
  "policy": "contiguous",  # or "block_cyclic"
  "order": "row_major",    # or "column_major"
  "assignments": [
    {"core_id": 0, "start_tile": 0, "count": 1},
    {"core_id": 1, "start_tile": 1, "count": 1},
    ...
  ]
}
```
Legacy attributes (`tt.schedule`) are maintained for backward compatibility; new passes will translate them into the canonical `tt.partition_mode`, `tt.core_ranges`, and `tt.runtime_args`.

**Algorithm**:
1. Calculate total tiles: `num_tiles = grid_x * grid_y`
2. For contiguous policy:
   - Divide tiles evenly across cores
   - Assign consecutive tiles to each core
3. Attach as `tt.schedule` attribute

---

## Configuration

**Policies**:
- `contiguous` (default): Core 0 gets tiles 0-N, core 1 gets N+1-2N, etc.
- `block_cyclic`: Tiles distributed in round-robin fashion

**Order**:
- `row_major` (default): Tiles numbered left-to-right, top-to-bottom
- `column_major`: Tiles numbered top-to-bottom, left-to-right

---

## Tests

**File**: `testing/python/tt/test_passes.py`
**Status**: ✅ 7 tests passing

Tests cover:
- Default schedule inference
- Different grid sizes
- Policy variations
- Metadata structure validation

---

## Dependencies

**Depends On**:
- `apply_tt_defaults()` - Provides default policy if not annotated

**Depended On By**:
- `LayoutAwareWorkPartitionTT` (only when explicit layout annotations are missing).
- `grid_to_persistent_tt.cc` - Uses schedule to seed tile counts until replaced by layout-aware metadata.
- Legacy code paths in host codegen.

---

## Related Files

- `src/transform/tt/infer_tt_schedule.cc` - Implementation
- `tilelang/tt/passes.py` - Python binding
- `testing/python/tt/test_passes.py` - Tests

---

## Success Criteria

- [x] Infers tile assignments from grid dimensions
- [x] Supports contiguous and block-cyclic policies
- [x] Attaches metadata to IR module
- [x] All tests passing (7/7)

---

**Last Updated**: 2025-10-09
