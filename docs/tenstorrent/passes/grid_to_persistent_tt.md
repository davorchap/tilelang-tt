# GridToPersistentTT Pass

**Status**: 🚧 Update in progress  
**Priority**: CRITICAL  
**File**: `src/transform/tt/grid_to_persistent_tt.cc`

---

## Purpose

Transform GPU-style grid kernels (`T.Kernel(...)`) into Tenstorrent's persistent loop model where each core iterates over assigned tiles. The updated pass consumes the layout-aware metadata emitted by `InferTTLayout` and `LayoutAwareWorkPartitionTT` to decide whether indices should be recovered globally (`partition_mode="global"`) or within shard-local coordinates (`partition_mode="local_shard"`).

---

## Why Needed

**GPU model**: Launch N threadblocks, each processes one tile
**TT model**: Launch N cores, each iterates over M tiles

This fundamental difference requires IR transformation to convert block indices `(bx, by)` into tile iteration with dynamic index recovery.

---

## Transformation

**Before** (GPU-style):
```python
with T.Kernel(8, 8) as (bx, by):  # bx, by are block indices
    C[bx*32:(bx+1)*32, by*32:(by+1)*32] = ...
```

**After** (TT persistent loop):
```python
core_id = get_core_id()  # 0 .. active_cores-1
start, count, mode = get_runtime_args(core_id)

if mode == "global":
    for offset in range(count):
        tid = start + offset
        m = tid // Nt
        n = tid % Nt
        body(m, n)
else:  # local_shard
    sy, sx = shard_coords(core_id)           # from shard grid Gy×Gx
    for offset in range(count):
        tid = start + offset                 # shard-local id
        m_l = tid // Sn
        n_l = tid % Sn
        m = sy * Sm + m_l
        n = sx * Sn + n_l
        body(m, n)
```

---

## Implementation Details

**Steps**:
1. Read core topology and partition metadata (`tt.partition_mode`, `tt.grid_tiles`, `tt.local_shape_tiles`, `tt.shard_grid`) produced by preceding passes.
2. Replace `T.Kernel` with a persistent `for` loop that iterates over per-core `(start, count)`.
3. Emit tile-ID recovery logic:
   - Global mode: convert `tid` into `(m, n)` using `Nt`.
   - Local shard mode: derive shard coordinates `(sy, sx)` from core position and compute global `(m, n)` from shard-local `(m_l, n_l)`, `Sm`, `Sn`.
4. Respect the requested traversal `order` (currently `row_major`; shard-aware and `block_linear(k)` will be layered via `RasterizationTT`).
5. Append runtime argument descriptors to `tt.runtime_args` in canonical order so that host codegen and kernels agree on indices.

**Runtime Arguments**:

| Index | Name | Description |
|-------|------|-------------|
| 0 | `start_id` | Global or shard-local starting tile id |
| 1 | `count` | Number of tiles for this core |
| 2 | `Mt` | Global tiles in M dimension |
| 3 | `Kt` | Tiles in reduction dimension (if needed by compute) |
| 4 | `Nt` | Global tiles in N dimension |
| 5.. | `Sm`, `Sn`, `Gy`, `Gx`, `sy`, `sx` | Present only for `local_shard` mode |

The pass is responsible for ordering these fields and recording the argument names in `tt.runtime_args`.

---

## Tests

**File**: `testing/python/tt/test_grid_to_persistent_tt.py`
**Status**: ✅ 12 tests passing

Tests cover:
- Basic grid transformation
- Index recovery correctness
- Different grid sizes (4×4, 8×8, 16×16)
- Metadata preservation
- Multi-dimensional indexing

---

## Dependencies

**Depends On**:
- `InferTTLayout` → supplies tile geometry.
- `LayoutAwareWorkPartitionTT` → supplies partition mode, core ranges, and shard shapes.
- `PropagateTTLayout` → optional (for CB metadata in later passes).

**Depended On By**:
- Host/kernel codegen (`EmitTTKernels`) which assumes persistent loop semantics and consistent runtime arg ordering.

---

## Related Files

- `src/transform/tt/grid_to_persistent_tt.cc` - Implementation
- `tilelang/tt/passes.py` - Python binding
- `testing/python/tt/test_grid_to_persistent_tt.py` - Tests

---

## Success Criteria

- [x] Transforms T.Kernel to persistent loop
- [x] Correctly recovers block indices
- [x] Preserves kernel semantics
- [x] All tests passing (12/12)

---

**Last Updated**: 2025-10-09
