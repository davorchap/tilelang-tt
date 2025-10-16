# InferTTLayout Pass

**Status**: 🟡 Partial (defaults + N-D projection)  
**Priority**: P0  
**File**: Python helper in `tilelang/tenstorrent/passes/infer_tt_layout.py`

---

## Purpose

Normalize user-provided layout annotations into canonical `tt.buffer.<name>`
attributes. The pass now:

- Records memory space, layout kind, dtype, and tile shape for every buffer.
- Accepts explicit overrides via `annotate_tt_layout`.
- Projects N‑D sharding metadata onto the 2‑D compute plane (`projected_grid`
  and `projected_shard_tiles`).
- Enforces basic L1 shard guardrails (tile alignment) and rejects unsupported
  hints such as halo regions.

---

## Metadata Written

Each buffer receives a dictionary of the form:

```json
{
  "memory": "DRAM" | "L1",
  "layout": "interleaved" | "sharded",
  "tile_shape": [32, 32],
  "dtype": "float16",
  "nd_shard": {
    "axes": ["B", "M", "N"],
    "grid": [1, 2, 4],
    "shard_shape_elems": [1, 64, 128],
    "projected_grid": [2, 4],
    "projected_shard_tiles": [2, 4]
  }
}
```

Notes:
- Tile shape remains fixed at 32×32 until tensor-level inference is added.
- `projected_*` helper fields are only present when sharding metadata is
  supplied.
- L1 shards must be tile-aligned; misaligned shard sizes raise a `ValueError`.

---

## High-Level Algorithm

1. Read `tt.user_layout` (if present) for each buffer; otherwise fall back to
   defaults (`memory="DRAM"`, `layout="interleaved"`).
2. Derive dtype and tile shape (32×32).
3. When `nd_shard` metadata is present:
   - Validate required keys (`axes`, `grid`, `shard_shape_elems`).
   - Ensure axes include `M` and `N` and compute projected grid/tile counts.
   - Enforce tile alignment for L1 shards.
4. Attach the resulting dictionary as `tt.buffer.<buffer_name>` on the PrimFunc.
5. Reject unsupported hints (e.g. halo).

---

## Diagnostics

- `nd_shard metadata for buffer ... must include 'axes', 'grid', 'shard_shape_elems'`
- `nd_shard metadata ... must include axes 'M' and 'N'`
- `L1 shard for buffer ... must be tile-aligned`
- `halo unsupported` (future TODO when halo metadata is surfaced)

---

## Tests

See `testing/python/tenstorrent/test_layout_aware_metadata.py` for coverage of the default
and sharded cases, plus negative tests for L1 misalignment.

---

## Dependencies

- `annotate_tt_layout` helper (exported via `tilelang.tenstorrent`).
- Tile shape constants and dtype → byte-size helper reused across TT passes.

---

## Downstream Consumers

- `PropagateTTLayout` consumes `tt.buffer.*` to stamp CB metadata.
- `LayoutAwareWorkPartitionTT` reads the projected grid/tile information.
- Host/codegen layers will eventually use this metadata to choose Metalium
  buffer configurations.
