# InferDefaultTTShard Pass

**Status**: ✅ Complete
**Priority**: CRITICAL
**File**: `src/transform/tt/infer_tt_shard.cc`

---

## Purpose

Generate DRAM layout descriptors (sharding metadata) for buffers, describing how data is laid out in Tenstorrent's tiled DRAM format.

---

## Why Needed

Tenstorrent DRAM is organized in **32×32 tiles**, not linear arrays. This pass infers the shard configuration (layout, interleaving, tile shape) for each buffer.

---

## Implementation

**Input**: Buffer allocations
```python
A = T.Buffer((256, 256), "float16")  # DRAM buffer
```

**Output**: Shard metadata
```python
{
  "layout": "tiled",        # Data organized as 32×32 tiles
  "interleaved": true,      # Tiles interleaved across DRAM banks
  "tile_shape": [32, 32],   # Tile dimensions
  "shard_strategy": "height" # or "width", "block"
}
```

**Algorithm**:
1. For each buffer in the IR:
   - Determine if it's a DRAM buffer (vs L1)
   - Calculate tile grid: `(height/32, width/32)`
   - Set default tiled layout with 32×32 tiles
2. Attach as `tt.shard` attribute

---

## Shard Strategies

- `height`: Shard along height dimension
- `width`: Shard along width dimension
- `block`: 2D block sharding
- `replicate`: Full copy on each core

Default: `height` for row-major layouts

---

## Tests

**File**: `testing/python/tt/test_passes.py`
**Status**: ✅ 7 tests passing (shared with schedule inference)

Tests cover:
- Default shard inference
- Different buffer shapes
- Tile alignment
- Metadata structure

---

## Dependencies

**Depends On**:
- `apply_tt_defaults()` - Provides default shard config

**Depended On By**:
- `tt_shard_to_core_map.cc` - Maps shards to NOC coordinates
- `memory_space_lower_tt.cc` - Creates circular buffers based on shard info

---

## Related Files

- `src/transform/tt/infer_tt_shard.cc` - Implementation
- `tilelang/tt/passes.py` - Python binding
- `testing/python/tt/test_passes.py` - Tests

---

## Success Criteria

- [x] Infers shard metadata for all DRAM buffers
- [x] Handles arbitrary buffer shapes (pads to 32×32)
- [x] Attaches correct metadata to IR
- [x] All tests passing

---

**Last Updated**: 2025-10-09
