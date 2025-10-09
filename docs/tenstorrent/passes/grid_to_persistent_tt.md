# GridToPersistentTT Pass

**Status**: ✅ Complete  
**Priority**: CRITICAL  
**File**: `src/transform/tt/grid_to_persistent_tt.cc`

---

## Purpose

Convert GPU-style grid kernels (`T.Kernel(...)`) into Tenstorrent's persistent execution model. Each Tensix core receives a contiguous range of tiles and iterates over them with a persistent loop.

---

## Why Needed

- **GPU model**: launch N threadblocks, each handles exactly one tile.  
- **TT model**: launch N cores, each core loops over *M* tiles determined at compile time.

Bridging the two requires removing `thread_extent` annotations, introducing explicit runtime parameters for tile ranges, and replacing `blockIdx` uses with computed indices.

---

## Transformation

**Before**
```python
with T.Kernel(8, 8) as (bx, by):
    C[bx, by] = ...
```

**After**
```python
@T.prim_func
def kernel(..., tt_start_tile: T.int32, tt_tile_count: T.int32):
    for tt_tile_iter in T.serial(tt_tile_count):
        tile_id = tt_start_tile + tt_tile_iter
        bx = tile_id % 8
        by = (tile_id // 8) % 8
        C[bx, by] = ...
```

Key additions:
- New scalar parameters `tt_start_tile` and `tt_tile_count`
- Persistent `for` loop over `tt_tile_iter`
- Computed replacements for `blockIdx.{x,y,z}`

The pass supports 1D, 2D, and 3D grids. `tt_runtime_args["iteration_ndims"]` advertises how many dimensions are non-trivial, and `tt_runtime_args["iteration_symbols"]` lists the recovered block indices (`["bx"]`, `["bx", "by"]`, or `["bx", "by", "bz"]`).

---

## Metadata Emitted

- `tt_runtime_args`: map describing runtime parameters
  - `start_tile` / `tile_count`: `{name, dtype, semantic}`
  - `grid_shape`: `[grid_x, grid_y, grid_z]`
  - `iteration_ndims`, `iteration_symbols`, `param_order`
- `tt_persistent_loop`: `True`
- `tt_persistent_iteration_ndims`: mirrors `iteration_ndims`

The pass does **not** materialize runtime helpers (`get_core_id`, etc.); codegen consumes the metadata to marshal per-core arguments.

---

## Implementation Notes

1. Read grid dimensions (`tt_grid_{x,y,z}`) from metadata inference stage.
2. Append the two runtime scalar parameters to the `PrimFunc`.
3. Remove `thread_extent` annotations for `blockIdx.x/y/z`.
4. Wrap the body with a `For` loop over `tt_tile_iter`.
5. Replace `blockIdx` variable uses with expressions derived from `tile_id = tt_start_tile + tt_tile_iter`.
6. Attach the metadata listed above.

---

## Tests

**File**: `testing/python/tt/test_ws3_grid_to_persistent.py`  
**Status**: ✅ Covers 1D, 2D, and 3D kernels, metadata validation, and parameter ordering.

---

## Dependencies

**Depends On**:
- `infer_default_tt_schedule.cc` (provides grid shape & tile assignments)

**Depended On By**:
- All TT-specific passes that assume persistent loop form (`tt_tiles_to_core_map`, `memory_space_lower_tt`, etc.)

---

## Success Criteria

- [x] Rewrites grid kernels into persistent loops
- [x] Emits runtime metadata + parameters consumed by codegen
- [x] Supports 1D / 2D / 3D tile ranges
- [x] Verified via persistent transform stage regression tests

---

**Last Updated**: 2026-02-20
