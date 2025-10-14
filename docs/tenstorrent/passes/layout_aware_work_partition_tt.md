# LayoutAwareWorkPartitionTT Pass

**Status**: 🟡 Partial (global mode only)  
**Priority**: P0  
**File**: Python helper in `tilelang/tenstorrent/passes/layout_aware_work_partition_tt.py`

---

## Purpose

Translate buffer/layout metadata into function-level partition descriptors.
Current behaviour reflects the legacy global schedule while capturing canonical
runtime argument naming:

- Emits `tt.partition_mode` (defaults to `global`) and preserves the legacy
  contiguous `tt_tiles_per_core` mapping.
- Stamps canonical runtime argument names (`tt_start_tile`, `tt_tile_count`,
  `Mt`, `Kt`, `Nt`, …) and exposes them via both `tt.runtime_arg_names` and
  `tt.runtime_constants`.
- Records helper attributes (`tt.grid_tiles`, `tt.local_shape_tiles`,
  `tt.shard_grid`) so downstream passes can transition to shard-aware execution.
- Leaves shard-local partitioning (`local_shard`) as future work pending
  shard metadata and persistent/codegen support.

---

## Metadata Written

| Attribute | Type | Description |
|-----------|------|-------------|
| `tt.partition_mode` | `String` | Currently always `"global"` unless user overrides. |
| `tt.grid_tiles` | `Array<Integer>` | `[Mt, Nt]` global tile grid (defaults to `grid_y * grid_z` / `grid_x`). |
| `tt.local_shape_tiles` | `Array<Integer>` | `[Sm, Sn]` (defaults to `[Mt, Nt]`). |
| `tt.shard_grid` | `Array<Integer>` | `[Gy, Gx]` (defaults to `[1, 1]`). |
| `tt.runtime_arg_names` | `Array<String>` | Canonical runtime arguments used by persistent lowering. |
| `tt.runtime_constants` | `Dict` | `{ "Mt": Mt, "Nt": Nt, "Kt": 1 }` (placeholders for shard fields). |
| `tt.core_ranges` | `Array<Array<Integer>>` | Core ranges derived from legacy tile scheduling. |
| `tt_core_runtime_args` | `Array<Array<Integer>>` | Per-core `[start_id, count]` pairs. |

---

## High-Level Algorithm

1. Read grid metadata (`tt_grid_x/y/z`, `tt_tiles_per_core`, `tt_num_cores`).
2. Merge user overrides from `tt.user_schedule` when provided.
3. Populate runtime argument names/constants using the canonical ordering.
4. Reuse the legacy contiguous mapping to populate `tt.core_ranges` and
   `tt_core_runtime_args`.
5. TODO: incorporate shard-aware partitioning once `InferTTLayout` supplies
   projected shard geometry and persistent lowering/host codegen consume it.

---

## Diagnostics

None at present. Future work should validate user overrides against available
cores and report inconsistencies.

---

## Tests

- `testing/python/tenstorrent/test_layout_aware_metadata.py` ensures metadata is present.
- `testing/python/tenstorrent/test_persistent_lowering.py` verifies persistent
  lowering consumes the canonical runtime argument list.

Additional shard-aware tests will arrive alongside the `local_shard` feature.

---

## Dependencies

- Runs after `InferTTLayout` / `PropagateTTLayout` (buffer metadata).
- Uses legacy schedule metadata (`tt_tiles_per_core`) until shard-aware planning
  replaces it.

---

## Downstream Consumers

- `grid_to_persistent_tt` reads the emitted metadata when constructing
  persistent loops.
- Host/codegen will use the canonical runtime argument ordering once shard-aware
  launches are implemented.
