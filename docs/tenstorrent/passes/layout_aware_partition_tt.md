# LayoutAwareWorkPartitionTT Pass

**Status**: 🟡 Partial (global mode only)  
**Priority**: P0  
**File**: Python helper in `tilelang/tt/passes.py`

---

## Purpose

Translate buffer/layout metadata into function-level partition descriptors the
rest of the pipeline can consume. The current implementation focuses on the
legacy global schedule while paving the way for shard-aware execution:

- Reads `tt.buffer.*` and `tt.user_schedule` metadata to determine partition
  mode (`global` vs `local_shard`, defaulting to `global`).
- Emits canonical runtime-argument names (`tt_start_tile`, `tt_tile_count`,
  `Mt`, `Kt`, `Nt`, …) so persistent lowering and host codegen can agree on
  ordering.
- Preserves existing `tt_tiles_per_core` ranges; still relies on them for core
  mapping.
- Leaves shard-local math (`local_shard`) and per-core shard coordinates marked
  as TODO.

---

## Metadata Written

| Attribute | Type | Description |
|-----------|------|-------------|
| `tt.partition_mode` | `String` | Currently always `"global"` unless user overrides. |
| `tt.grid_tiles` | `Array<Integer>` | `[Mt, Nt]` global tile grid (defaults to grid_y×grid_z, grid_x). |
| `tt.local_shape_tiles` | `Array<Integer>` | `[Sm, Sn]` shard-local tiles (defaults to `[Mt, Nt]`). |
| `tt.shard_grid` | `Array<Integer>` | `[Gy, Gx]` shard grid (defaults to `[1, 1]`). |
| `tt.runtime_arg_names` | `Array<String>` | Canonical runtime argument ordering used by persistent lowering (`tt_start_tile`, `tt_tile_count`, `Mt`, `Kt`, `Nt`, …). |
| `tt.runtime_constants` | `Dict` | Constant payload exposed to runtime (`Mt`, `Nt`, `Kt`, with placeholders for future shard fields). |
| `tt.core_ranges` | `Array<Array<Integer>>` | Core ranges derived from legacy schedule metadata. |
| `tt_core_runtime_args` | `Array<Array<Integer>>` | Per-core `[start_id, count]` pairs. |

---

## High-Level Algorithm

1. Read grid metadata (`tt_grid_x/y/z`, `tt_tiles_per_core`, `tt_num_cores`).
2. Merge user overrides from `tt.user_schedule` when available.
3. Populate default runtime constants and arg-name list matching the global
   schedule contract.
4. Reuse the legacy contiguous core mapping to populate `tt.core_ranges` and
   `tt_core_runtime_args`.
5. TODO: introduce shard-local partitioning (`local_shard`) once shard metadata
   is available and persistent/codegen understand the larger runtime payload.

---

## Diagnostics

None yet. Planned follow-ups include:
- Validating that user-provided `tt.user_schedule` values are consistent with
  the available core count.
- Detecting malformed overrides (e.g., missing `grid_tiles` entries).

---

## Tests

Covered indirectly by:
- `testing/python/tt/test_layout_aware_metadata.py` (ensures metadata is stamped).
- `testing/python/tt/test_ws3_grid_to_persistent.py` (verifies persistent pass
  consumes the runtime argument names).

Additional shard-aware tests will be required when `local_shard` support lands.

---

## Dependencies

- Runs after `InferTTLayout` / `PropagateTTLayout` (for buffer metadata).
- Uses legacy schedule metadata (`tt_tiles_per_core`, `tt_num_cores`) until
  shard-aware planning replaces it.

---

## Downstream Consumers

- `grid_to_persistent_tt` reads `tt.runtime_arg_names`, `tt.grid_tiles`, and
  `tt.partition_mode` when constructing persistent loops.
- Host/codegen layers will consume the same metadata once their runtime payloads
  are expanded.
