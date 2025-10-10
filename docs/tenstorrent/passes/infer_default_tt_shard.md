# InferDefaultTTShard Pass

**Status**: ✅ Complete  
**Priority**: CRITICAL  
**File**: `src/transform/tt/infer_tt_shard.cc`

---

## Purpose

Emit DRAM tilization metadata for every buffer, describing how data is arranged in Tenstorrent's 32×32 tiled memory layout and whether padding is required.

---

## Why Needed

Tenstorrent DRAM is organized in 32×32 tiles rather than linear rows. Downstream passes need tile counts, padding information, and layout descriptors to plan circular buffers, DMA descriptors, and NOC routing.

---

## Implementation

**Input**: Buffer bindings (function parameters / buffer_map entries)
```python
A = T.Buffer((256, 256), "float16")  # DRAM buffer
```

**Output**: Shard metadata emitted as legacy scalar attrs plus a consolidated map
```python
# Legacy attributes (per buffer)
"tt_buffer_A_layout" = "dram_interleaved"
"tt_buffer_A_tile_shape" = [32, 32]
"tt_buffer_A_num_tiles_height" = 8
"tt_buffer_A_num_tiles_width" = 8
"tt_buffer_A_needs_padding" = False

# Aggregated metadata
"tt_shard" = {
  "A": {
    "layout": "dram_interleaved",
    "tile_shape": [32, 32],
    "tiles_height": 8,
    "tiles_width": 8,
    "needs_padding": False
  }
}
```

**Algorithm**:
1. Iterate over every buffer in `PrimFunc.buffer_map`.
2. Compute tile coverage with ceil-division by the default 32×32 tile shape.
3. Detect whether padding is necessary and record the padded shape if so.
4. Attach both the existing scalar attributes (`tt_buffer_*`) for backward compatibility and a new `tt_shard` map keyed by buffer name.

---

## Tests

**File**: Covered indirectly via `testing/python/tt/test_persistent_lowering.py`  
**Status**: ✅ Metadata shape validated alongside persistent transform stage transforms

---

## Dependencies

**Depends On**:
- `apply_tt_defaults()` for default layout/tile annotations

**Depended On By**:
- `tt_tiles_to_core_map.cc` (consumes tile counts for runtime arg planning)
- `memory_space_lower_tt.cc` (derives circular-buffer sizing)

---

## Related Files

- `src/transform/tt/infer_tt_shard.cc` - Implementation
- `tilelang/tt/passes.py` - Python binding

---

## Success Criteria

- [x] Emits tilization metadata for all DRAM buffers
- [x] Records padding requirements when shapes are not tile-aligned
- [x] Provides both legacy and consolidated annotations for downstream passes

---

**Last Updated**: 2026-02-20
