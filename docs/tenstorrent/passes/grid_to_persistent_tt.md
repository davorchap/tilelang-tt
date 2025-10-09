# GridToPersistentTT Pass

**Status**: ✅ Complete
**Priority**: CRITICAL
**File**: `src/transform/tt/grid_to_persistent_tt.cc`

---

## Purpose

Transform GPU-style grid kernels (`T.Kernel(grid_x, grid_y)`) into Tenstorrent's persistent loop model where each core iterates over assigned tiles.

---

## Why Needed

**GPU model**: Launch N threadblocks, each processes one tile
**TT model**: Launch N cores, each iterates over M tiles

This fundamental difference requires IR transformation to convert block indices `(bx, by)` into tile iteration with dynamic index recovery.

---

## Transformation

**Before** (GPU-style):
```python
with T.Kernel(8, 8) as (bx, by):  # bx, by are block indices
    C[bx*32:(bx+1)*32, by*32:(by+1)*32] = ...
```

**After** (TT persistent loop):
```python
core_id = get_core_id()  # 0 to num_cores-1
start_tile, count = get_tile_assignment(core_id)  # from schedule metadata

for tile_id in range(start_tile, start_tile + count):
    # Recover block indices from tile_id
    bx = tile_id // grid_y
    by = tile_id % grid_y

    # Original kernel body (unchanged)
    C[bx*32:(bx+1)*32, by*32:(by+1)*32] = ...
```

---

## Implementation Details

**Steps**:
1. Read `tt.schedule` metadata (from infer_default_tt_schedule)
2. Replace `T.Kernel` with persistent for-loop
3. Insert tile index recovery logic
4. Preserve original kernel body
5. Add core ID and tile assignment getters

**IR Nodes**:
- `KernelLaunch` → `For` (persistent loop)
- `blockIdx.x/y` → Computed from `tile_id`

---

## Tests

**File**: `testing/python/tt/test_grid_to_persistent_tt.py`
**Status**: ✅ 12 tests passing

Tests cover:
- Basic grid transformation
- Index recovery correctness
- Different grid sizes (4×4, 8×8, 16×16)
- Metadata preservation
- Multi-dimensional indexing

---

## Dependencies

**Depends On**:
- `infer_default_tt_schedule.cc` - Requires schedule metadata

**Depended On By**:
- All subsequent TT transforms expect persistent loop model

---

## Related Files

- `src/transform/tt/grid_to_persistent_tt.cc` - Implementation
- `tilelang/tt/passes.py` - Python binding
- `testing/python/tt/test_grid_to_persistent_tt.py` - Tests

---

## Success Criteria

- [x] Transforms T.Kernel to persistent loop
- [x] Correctly recovers block indices
- [x] Preserves kernel semantics
- [x] All tests passing (12/12)

---

**Last Updated**: 2025-10-09
