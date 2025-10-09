# TTShardToCoreMap Pass

**Status**: ✅ Complete
**Priority**: MEDIUM
**File**: `src/transform/tt/tt_shard_to_core_map.cc`

---

## Purpose

Map shard IDs to physical NOC (Network-on-Chip) grid coordinates (x, y) for Tensix cores.

---

## Why Needed

Tenstorrent uses a 2D mesh interconnect (NOC) where cores are addressed by `(x, y)` coordinates. This pass converts logical shard IDs to physical core coordinates for data routing.

---

## Mapping

**Input**: Shard ID (logical)
```python
shard_id = 0, 1, 2, ...
```

**Output**: NOC coordinates (physical)
```python
core_coords = [
  (0, 0),  # shard 0 → core (0, 0)
  (0, 1),  # shard 1 → core (0, 1)
  (1, 0),  # shard 2 → core (1, 0)
  ...
]
```

**Grid Layout** (Grayskull example):
```
     y=0  y=1  y=2  y=3  y=4  y=5
x=0  [0]  [1]  [2]  [3]  [4]  [5]
x=1  [6]  [7]  [8]  [9]  [10] [11]
```

Row-major mapping by default.

---

## Implementation

**Algorithm**:
1. Read device grid dimensions (e.g., 12×9 for Grayskull)
2. Map shard IDs to (x, y):
   - `x = shard_id / grid_width`
   - `y = shard_id % grid_width`
3. Attach as `tt.core_map` metadata

**Configurable**: Supports custom mappings for advanced topologies

---

## Tests

**File**: `testing/python/tt/test_tt_shard_to_core_map.py`
**Status**: ✅ 5 tests passing

Tests cover:
- Row-major mapping
- Different grid sizes
- Coordinate correctness
- Metadata structure

---

## Dependencies

**Depends On**:
- `infer_default_tt_shard.cc` - Requires shard metadata

**Depended On By**:
- `codegen_tt_reader_visitor.cc` - Uses coordinates for NOC transfers
- `codegen_tt_writer_visitor.cc` - Uses coordinates for NOC transfers

---

## Related Files

- `src/transform/tt/tt_shard_to_core_map.cc` - Implementation
- `tilelang/tt/passes.py` - Python binding
- `testing/python/tt/test_tt_shard_to_core_map.py` - Tests

---

## Success Criteria

- [x] Maps shards to NOC coordinates
- [x] Supports different grid topologies
- [x] Metadata correct for codegen
- [x] All tests passing (5/5)

---

**Last Updated**: 2025-10-09
