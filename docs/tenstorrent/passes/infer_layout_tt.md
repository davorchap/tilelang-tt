# InferTTLayout Pass

**Status**: 🚧 Planned  
**Priority**: P0  
**File**: `src/transform/tt/infer_layout_tt.cc`

---

## Purpose

Normalize user-provided layout annotations into a single source of truth on every buffer and stamp the derived metadata onto the TIR. This pass bridges Python-surface hints (e.g. `annotate_tt_layout`) with the backend by:

- Recording `tt.buffer.<name>` attributes that describe memory space, layout, dtype, and tile shape.
- Projecting N‑D sharding strategies onto the compute tile plane to obtain a 2‑D shard grid and local shard dimensions.
- Validating opt-in L1 sharding (capacity, tile alignment) while keeping DRAM sharding first-class.
- Rejecting features that are explicitly out-of-scope (halo regions, auto-promotion).

---

## Metadata Written

Each buffer participating in the PrimFunc receives a JSON-like dict attribute:

```json
{
  "memory": "DRAM" | "L1",
  "layout": "interleaved" | "sharded",
  "tile_shape": [32, 32],
  "dtype": "bf16" | "fp16" | "fp32" | "...",
  "nd_shard": {
    "axes": ["B", "H", "M", "N"],
    "grid": [gB, gH, gM, gN],
    "shard_shape_elems": [sB, sH, sM, sN],
    "order": "row_major" | "block_linear(k)",
    "align_tiles": true,
    "projected_grid": [Gy, Gx],
    "projected_shard_tiles": [Sm, Sn]
  }
}
```

Notes:
- `nd_shard` is only present when `layout == "sharded"`.
- `projected_grid` maps the N‑D grid onto the compute plane (cores on y/x).
- `projected_shard_tiles` expresses per-core output tiles on the compute plane after tilization.
- For L1 shards, `align_tiles` must be true (enforced here).

---

## High-Level Algorithm

1. Scan all buffers in the function and fetch `tt.layout` annotations (if any). Apply defaults `memory="DRAM"`, `layout="interleaved"`, `order="row_major"`.
2. Derive dtype and tile shape (currently fixed at 32×32) directly from buffer type and target defaults.
3. If the layout is sharded:
   - Validate that `axes`, `grid`, and `shard_shape_elems` are present.
   - Project the N‑D sharding definition onto the 2‑D compute plane (`ProjectShardToPlane`). The method selects the loop axes that map to M and N tiles, multiplies shard sizes by tile dimensions, and creates `[Gy, Gx]`, `[Sm, Sn]`.
   - Run `ValidateL1Capacity` when `memory == "L1"` to ensure `Sm × Sn × tile_bytes × cb_depth` fits available L1.
4. Attach the fully-populated `DictAttrs` as `tt.buffer.<buffer_name>` on the owning PrimFunc.
5. Emit diagnostics and fail if:
   - Halo metadata is encountered (`halo` field set by the user).
   - L1 shard violates capacity or tile alignment.
   - N‑D metadata is incomplete.

---

## Diagnostics

- `halo unsupported` when halo regions are requested.
- `nd_shard incomplete` if required keys are missing.
- `L1 shard exceeds capacity` or `L1 shard requires tile alignment` for opt-in L1 sharding failures.

---

## Tests

Target new coverage in `tests/tt/test_layouts.py`:

- Default DRAM + interleaved buffer stamps attributes with defaults.
- DRAM sharded tensor projects to 2‑D grid and records shard tiles.
- L1 sharded tensor triggers capacity enforcement and demands `align_tiles=true`.
- Negative cases for halo presence, missing N‑D metadata, and L1 overflow.

---

## Dependencies

- Python helper `annotate_tt_layout` (to be added in `python/tilelang_tt/annotations.py`).
- Tile shape constants and dtype → byte-size helper (shared with other TT passes).

---

## Downstream Consumers

- `PropagateTTLayout` uses `tt.buffer.*` to stamp CB metadata.
- `LayoutAwareWorkPartitionTT` reads memory/layout fields and `projected_grid`/`projected_shard_tiles`.
- `EmitTTKernels` relies on buffer metadata to pick the correct Metalium buffer configuration at host-generation time.
