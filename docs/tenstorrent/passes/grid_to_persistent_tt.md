# GridToPersistentTT Pass

**Status**: 🟡 Partial (global mode)  
**Priority**: CRITICAL  
**File**: `src/transform/tenstorrent/grid_to_persistent_tt.cc`

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
# TODO: local_shard recovery (Sy/Sx + shard-local iteration) to be implemented once
# layout-aware partitioning emits shard coordinates per core.
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
5. Append runtime argument descriptors to `tt.runtime_args` / `tt.runtime_arg_names` in canonical order so that host codegen and kernels agree on indices. For `local_shard`, shard coordinates are materialised as additional scalar parameters.

**Runtime Arguments**:

| Index | Name | Description |
|-------|------|-------------|
| 0 | `start_id` | Global or shard-local starting tile id (runtime param) |
| 1 | `count` | Number of tiles for this core (runtime param) |
| 2 | `Mt` | Global tiles in M dimension (recorded in `tt.runtime_constants`) |
| 3 | `Kt` | Tiles in reduction dimension (placeholder = 1 until tensorization drives it) |
| 4 | `Nt` | Global tiles in N dimension (`tt.runtime_constants["Nt"]`) |
| 5.. | `Sm`, `Sn`, `Gy`, `Gx`, `tt_shard_coord_y`, `tt_shard_coord_x` | Present when `partition_mode="local_shard"` |

The first two entries (`tt_start_tile`, `tt_tile_count`) are always appended as
PrimFunc parameters. When `partition_mode="local_shard"`, two additional
parameters (`tt_shard_coord_y`, `tt_shard_coord_x`) carry the shard coordinates
for the executing core; the remaining values are kept as constants/metadata.

The pass is responsible for ordering these fields and recording the argument names in `tt.runtime_args`.

---

## IR Structure Changes: Root Block Normalization

When `GridToPersistentTT` removes GPU threading constructs, TVM normalizes the block structure, making the root block explicit with `T.reads()` / `T.writes()` annotations.

**Before (Implicit Root Block)**:
```python
# with T.block("root"):  # ← Commented = implicit/hidden
bx = T.launch_thread("blockIdx.x", 4)
by = T.launch_thread("blockIdx.y", 4)
tx = T.launch_thread("threadIdx.x", 128)
ty = T.launch_thread("threadIdx.y", 1)
tz = T.launch_thread("threadIdx.z", 1)
with T.block("tilelang_root"):
    T.reads(A[bx * 32, by * 32])
    T.writes()
    T.evaluate(A[bx * 32, by * 32])
```

**After (Explicit Root Block)**:
```python
for tt_tile_iter in range(tt_tile_count):
    tt_tile_id: T.int32 = tt_start_tile + tt_tile_iter
    with T.block("root"):  # ← Now explicit
        T.reads()
        T.writes()
        with T.block("tilelang_root"):
            T.reads(A[tt_tile_id % 4 * 32, tt_tile_id // 4 % 4 * 32])
            T.writes()
            T.evaluate(A[tt_tile_id % 4 * 32, tt_tile_id // 4 % 4 * 32])
```

**Why This Happens**:

1. **Implicit → Explicit**: The `# with T.block("root")` comment means the root block exists but is **implicit** (hidden from display). When `GridToPersistentTT` removes the thread launch constructs and restructures the IR, TVM makes the root block **explicit**.

2. **Thread constructs were wrapping the body**: The `launch_thread` calls created `AttrStmt` nodes that wrapped the inner blocks. When we remove them (lines 135-137 and 142-144 in grid_to_persistent_tt.cc), we're left with the raw block structure.

3. **Block annotations are required**: TVM requires all blocks to have `T.reads()` and `T.writes()` annotations for:
   - Dependency analysis
   - Scheduling validation
   - Buffer access tracking

4. **Empty annotations on root block**: The root block has empty `T.reads()` / `T.writes()` because:
   - It's just a **container/scope** for the persistent loop
   - The actual memory access happens in the inner `"tilelang_root"` block
   - Root blocks typically don't perform operations themselves

This is **correct and expected behavior** - it's how TVM normalizes block structure after IR transformations. The root block becomes explicit to properly represent the nesting structure once the thread launch constructs are removed.

**GPU Threading Construct Removal**:

As of the latest update, `GridToPersistentTT` removes **both** `blockIdx.*` and `threadIdx.*` constructs:
- `blockIdx.x/y/z` are **replaced** with tile ID recovery expressions (e.g., `tt_tile_id % 4`)
- `threadIdx.x/y/z` are **removed entirely** (TT uses persistent cores, not ephemeral threads)

---

## Tests

**File**: `testing/python/tenstorrent/test_grid_to_persistent_tt.py`
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

- `src/transform/tenstorrent/grid_to_persistent_tt.cc` - Implementation
- `tilelang/tenstorrent/passes/grid_to_persistent_tt.py` - Python binding
- `testing/python/tenstorrent/test_grid_to_persistent_tt.py` - Tests

---

## Success Criteria

- [x] Transforms T.Kernel to persistent loop
- [x] Correctly recovers block indices
- [x] Preserves kernel semantics
- [x] All tests passing (12/12)

---

**Last Updated**: 2025-10-09
