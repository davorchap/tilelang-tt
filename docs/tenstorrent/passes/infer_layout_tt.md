# InferTTLayout Pass

**Status**: 🟡 Partial (defaults only)  
**Priority**: P0  
**File**: Python helper in `tilelang/tt/passes.py`

---

## Purpose

Normalize user-provided layout annotations into canonical `tt.buffer.<name>`
attributes. The current Python implementation establishes sensible defaults,
bridging `annotate_tt_layout` hints to backend metadata:

- Records memory space, layout kind, dtype, and tile shape for every buffer.
- Accepts explicit overrides when the user supplies `tt.user_layout`.
- Defers N‑D shard projection, L1-capacity checks, and halo diagnostics to
  future iterations.

---

## Metadata Written

Each buffer currently receives a simple dictionary:

```json
{
  "memory": "DRAM",
  "layout": "interleaved",
  "tile_shape": [32, 32],
  "dtype": "float16"
}
```

Notes:
- Tile shape is hard-coded to the TT default (32×32) until tile inference is wired in.
- Sharded metadata (axes, projected grids, shard tile counts) is not emitted yet.
- Unsupported hints (e.g. halo) are silently ignored for now; diagnostics should
  be added once projection logic lands.

---

## High-Level Algorithm

1. Read `tt.user_layout` (if present) for each buffer, otherwise fall back to
   defaults (`memory="DRAM"`, `layout="interleaved"`).
2. Derive dtype from the buffer type; set `tile_shape=[32,32]`.
3. Attach the resulting dictionary as `tt.buffer.<buffer_name>` on the PrimFunc.
4. TODO: add validation for halo hints, incomplete shard metadata, and L1
   capacity alignment once those features are implemented.

---

## Diagnostics

Currently none. Future revisions should surface:
- `halo unsupported`
- `nd_shard incomplete`
- `L1 shard exceeds capacity` / `requires tile alignment`

---

## Tests

Covered by `testing/python/tt/test_layout_aware_metadata.py` (default DRAM
scenario). Sharded/L1-focused cases remain to be added alongside the missing
functionality.

---

## Dependencies

- Python helper `annotate_tt_layout` (exported via `tilelang.tt`).
- Tile shape constants and dtype → byte-size helper reused across TT passes.

---

## Downstream Consumers

- `PropagateTTLayout` consumes `tt.buffer.*` when emitting CB metadata.
- `LayoutAwareWorkPartitionTT` reads layout fields today and will use shard
  projections once they exist.
- Host/codegen passes eventually rely on this metadata to choose Metalium buffer
  configurations.
