# TilePadTT Pass

**Status**: ✅ Complete  
**Priority**: MEDIUM  
**File**: `src/transform/tenstorrent/tile_pad_tt.cc`

---

## Purpose

Capture padding requirements for buffers whose shapes are not multiples of the Tenstorrent tile size (32 × 32). The pass does not mutate buffers; it records the padding plan for codegen.

---

## Why Needed

Hardware tensor operations expect tile-aligned shapes. When user buffers have trailing elements that do not fill a full tile, the runtime must handle edge tiles carefully. Recording the intended padded shape up front allows codegen to configure DMA descriptors and guard stores.

---

## Metadata Emitted

For each buffer that requires padding, the pass emits:

```python
"tt_padding_info" = {
  "A": {
    "needs_padding": True,
    "original_shape": [100, 100],
    "padded_shape": [128, 128],
    "padding_amount": [28, 28],
  },
  ...
}
```

Buffers that are already tile aligned are skipped.

---

## Implementation

1. Look up sharding metadata produced by `infer_default_tt_shard` (`tt_buffer_*_needs_padding`, etc.).
2. For every buffer flagged as needing padding:
   - Reconstruct the original and padded shapes.
   - Compute per-dimension padding amounts.
3. Aggregate the metadata under `tt_padding_info`.

The pass leaves the underlying TIR untouched—no additional buffers or loops are inserted at this stage. Padding logic is implemented later in TT codegen.

---

## Tests

**File**: Pending (covered indirectly by persistent transform stage integration tests)

---

## Dependencies

**Depends On**:
- `infer_default_tt_shard.cc` (provides per-buffer padding flags)

**Depended On By**:
- TT codegen visitors (reader / writer) when configuring DMA strides

---

## Success Criteria

- [x] Emits padding metadata for every buffer that needs it
- [x] Leaves already aligned buffers untouched
- [x] Does not rewrite user IR—pure metadata pass

---

**Last Updated**: 2026-02-20
