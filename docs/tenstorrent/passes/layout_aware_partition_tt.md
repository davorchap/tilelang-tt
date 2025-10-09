# LayoutAwareWorkPartitionTT Pass

**Status**: 🚧 Planned  
**Priority**: P0  
**File**: `src/transform/tt/layout_aware_partition_tt.cc`

---

## Purpose

Choose per-core work assignments that respect buffer residency and sharding. This pass replaces heuristic schedule inference with a layout-driven policy:

- When any tensor is L1-resident and sharded, map each shard to a unique core and emit local runtime ranges.
- Otherwise, fall back to the global tile grid inferred from tensor shapes and user annotations.
- Stamp function-level attributes describing partition mode, core range selection, and shard-local tile geometry.

---

## Metadata Written

| Attribute | Type | Description |
|-----------|------|-------------|
| `tt.partition_mode` | `String` | `"global"` or `"local_shard"` |
| `tt.core_ranges` | `Array` | CoreRangeSet encoded as `[y0,x0,y1,x1,...]` |
| `tt.grid_tiles` | `Array<Integer>` | `[Mt, Nt]` global tile grid (only when global) |
| `tt.shard_grid` | `Array<Integer>` | `[Gy, Gx]` projection of shard grid onto compute plane (local shard) |
| `tt.local_shape_tiles` | `Array<Integer>` | `[Sm, Sn]` tiles per shard on compute plane |
| `tt.runtime_args` | `Array<String>` | Ordered list of runtime argument names |

Per-core `(start_id, count)` ranges remain encoded via existing schedule metadata or via new helper structures emitted into `tt.runtime_args`.

---

## High-Level Algorithm

1. Inspect `tt.buffer.*` metadata for all buffers in the PrimFunc.
2. If any buffer has `memory="L1"` and `layout="sharded"`:
   - Extract `projected_grid = [Gy, Gx]` and `projected_shard_tiles = [Sm, Sn]`.
   - Set `tt.partition_mode = "local_shard"`.
   - Materialize `tt.core_ranges` that exactly match the shard grid footprint.
   - Emit `tt.shard_grid` and `tt.local_shape_tiles`.
   - For every core, assign tile IDs `0 .. Sm*Sn-1` in shard-local space; record runtime args `(start_id, count, Sm, Sn, Gy, Gx, sy, sx)` for codegen.
3. Otherwise:
   - Compute global tile counts `[Mt, Nt]` from buffer metadata (or fallback to existing schedule).
   - Retain user/default `CoreRangeSet`.
   - Set `tt.partition_mode = "global"` and populate `tt.grid_tiles`.
   - Assign contiguous `(start_id, count)` in global tile ID space.
4. Preserve order annotations (`row_major`, `match_shard`, `block_linear(k)`) for downstream traversal.

---

## Diagnostics

- Fail if L1 sharded buffers disagree on projected shard geometry (mismatched `[Gy, Gx]` or `[Sm, Sn]`).
- Fail if requested shard grid exceeds available cores.
- Flag inconsistent order annotations (e.g., `block_linear(k)` incompatible with current pipeline) with actionable messages.

---

## Tests

- `local_shard` happy path: ensure core ranges equal shard grid, runtime args include shard geometry.
- Mixed DRAM/L1 scenario: L1 shard forces local partition while DRAM buffers remain accessible via TensorAccessor.
- Pure DRAM: fallback to global contiguous schedule with `[Mt, Nt]` inferred.
- Negative cases for mismatched shard grids and over-subscribed cores.

All coverage to live in `tests/tt/test_layouts.py`.

---

## Dependencies

- Requires `InferTTLayout` metadata (`tt.buffer.*` with `projected_grid`).
- Runs before `GridToPersistentTT` and any rasterization/order rewrites.

---

## Downstream Consumers

- `GridToPersistentTT` reads `tt.partition_mode`, `tt.grid_tiles`, `tt.local_shape_tiles`, and shard grid info to generate tile-id math.
- Host codegen leverages `tt.core_ranges` to select launch topology.
