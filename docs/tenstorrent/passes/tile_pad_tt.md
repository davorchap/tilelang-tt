# TilePadTT Pass

**Status**: ✅ Complete
**Priority**: MEDIUM
**File**: `src/transform/tt/tile_pad_tt.cc`

---

## Purpose

Pad buffer dimensions to multiples of 32 (tile size), ensuring all operations work on complete 32×32 tiles.

---

## Why Needed

Tenstorrent hardware operates on **32×32 tiles** exclusively. Arbitrary dimensions (e.g., 100×100) must be padded to the next tile boundary (128×128 = 4×4 tiles).

---

## Transformation

**Before** (arbitrary dimensions):
```python
A = T.Buffer((100, 100), "float16")  # Not tile-aligned
```

**After** (tile-aligned):
```python
A_padded = T.Buffer((128, 128), "float16")  # 4×4 tiles
# Original: 100×100
# Padding: +28 rows, +28 columns
```

**Padding Strategy**:
- Rows: `ceil(100 / 32) * 32 = 128`
- Columns: `ceil(100 / 32) * 32 = 128`
- Padding filled with zeros

---

## Implementation

**Algorithm**:
1. For each buffer dimension:
   - If `dim % 32 != 0`:
     - Pad to next multiple of 32
     - Add padding metadata to IR
2. Update buffer shapes in IR
3. Insert padding initialization (fill with 0)

**IR Attribute**:
```python
"tt.tile_padding": {
  "original_shape": [100, 100],
  "padded_shape": [128, 128],
  "padding": [28, 28]
}
```

---

## Codegen Impact

Codegen must:
- Allocate padded size in DRAM
- Only write valid data (avoid padding regions)
- Handle padding in result validation

---

## Tests

**File**: `testing/python/tt/test_tile_pad_tt.py`
**Status**: ✅ 6 tests passing

Tests cover:
- Already aligned (no padding needed)
- Single dimension padding
- Both dimensions padding
- Large padding requirements
- Metadata correctness

---

## Dependencies

**Depends On**:
- None (can run early in pipeline)

**Depended On By**:
- `memory_space_lower_tt.cc` - Expects tile-aligned shapes
- All code generation passes

---

## Related Files

- `src/transform/tt/tile_pad_tt.cc` - Implementation
- `tilelang/tt/passes.py` - Python binding
- `testing/python/tt/test_tile_pad_tt.py` - Tests

---

## Success Criteria

- [x] Pads all dimensions to multiples of 32
- [x] Preserves semantics (padding is transparent)
- [x] Metadata tracks original vs padded shapes
- [x] All tests passing (6/6)

---

**Last Updated**: 2025-10-09
