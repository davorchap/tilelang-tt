# TIR Basics & Tenstorrent Lowering Plan

This document is a practical primer on **TensorIR (TIR)** and the **Tenstorrent (TT) grid→persistent lowering** we plan to implement with an AI agent.

---

## Part I — TIR Basics (Quick Reference)

### 1) PrimFunc & Buffers
- `@T.prim_func` declares a lowerable function with a static signature.
- `T.Buffer(shape, dtype)` describes a typed memory region; indexing `A[i, j]` corresponds to loads/stores in TIR.

### 2) Launch & Thread Bindings
- `T.launch_thread(tag, extent)` binds loop domains to hardware/logical threads (e.g., `blockIdx.x`, `threadIdx.x`).
- Even if unused in index math, these bindings define the **realization domain** (SPMD instances).

### 3) Blocks (`with T.block("name"):`)
- Smallest schedulable/dataflow unit; declares **access regions** and contains body statements.
- Common block clauses:
  ```python
  with T.block("name"):
      T.reads(...)
      T.writes(...)
      T.where(optional_pred)
      T.init()                 # reduction init, optional
      # body: loads/stores/intrinsics
  ```
- Blocks can be **0‑D** (no explicit axes) but still depend on outer symbols.

### 4) Access Region Annotations
- `T.reads(...)`: read set; `T.writes(...)`: write set (use `T.writes()` for **empty write set**).
- These enable legality checks and optimization (reorder/fuse/vectorize).

### 5) Axes — Spatial vs Reduce
- `T.axis.spatial(M, idx)`: non-reducing dimension.
- `T.axis.reduce(R, idx)`: "reducing" dimension, pairs with `T.init()`.
- **Example:** sum over `k`
  ```python
  @T.prim_func
  def matvec(A: T.Buffer((M, K), "float16"),
             x: T.Buffer((K,), "float16"),
             y: T.Buffer((M,), "float32")):
      for i0 in range(M):
          with T.block("y"):
              i = T.axis.spatial(M, i0)
              T.reads(A[i, 0:K], x[0:K])
              T.writes(y[i])
              T.init():
                  y[i] = T.float32(0)
              for k0 in range(K):
                  with T.block("y.update"):
                      k = T.axis.reduce(K, k0)
                      y[i] = y[i] + T.cast(A[i, k], "float32") * T.cast(x[k], "float32")
  ```

### 6) `T.evaluate(expr)` & Implicit Root
- `T.evaluate(expr)` forces side‑effectful evaluation (e.g., keep a load alive); value is discarded.
- Pretty‑printers often show `# with T.block("root"):` as a comment; the **root** block is implicit and not schedulable.

### 7) Tile‑Level Intrinsics (Conceptual)
Backends expose tile‑granular ops:
- tile load/store: move tiles across memory spaces
- elementwise on tiles: add/mul/fma/cast/activation
- **GEMM/MMA**: matrix‑engine tile multiply‑accumulate (`T.gemm(...)`)
- sync/barriers & double‑buffering helpers

**Tiled GEMM sketch:**
```python
@T.prim_func
def tiled_gemm(A: T.Buffer((M, K), "float16"),
               B: T.Buffer((K, N), "float16"),
               C: T.Buffer((M, N), "float32")):
    bx = T.launch_thread("blockIdx.x", M // TM)
    by = T.launch_thread("blockIdx.y", N // TN)
    with T.block("C.tile"):
        io = T.axis.spatial(M // TM, bx)
        jo = T.axis.spatial(N // TN, by)
        T.reads(A[io*TM:(io+1)*TM, 0:K], B[0:K, jo*TN:(jo+1)*TN])
        T.writes(C[io*TM:(io+1)*TM, jo*TN:(jo+1)*TN])
        T.init(): pass
        for ko in range(0, K, TK):
            a_tile = T.tile_load(A, io*TM, ko, shape=(TM, TK))
            b_tile = T.tile_load(B, ko, jo*TN, shape=(TK, TN))
            T.gemm(a_tile, b_tile)
        # T.tile_store(C, io*TM, jo*TN, accum)
```

**Mental model:** map `blockIdx → tile‑level (matrix engine)`; `threadIdx → intra‑tile (SIMD)`.

---

## Part II — Tenstorrent Lowering Plan (Grid → Persistent)

### Goals
- Keep tiling, regions, and scheduling analyzable in TileLang/TIR.
- Model core ownership & work partition explicitly.
- Lower to TT **persistent kernels** as a **late** pass.
- Make `CoreRange`/worklists a **single source of truth** for host & device.

### Stage A — Mid‑Level Core Form (target‑agnostic)
Introduce a **core domain** (IR sugar) + metadata; do **not** materialize persistent loops yet.

- IR sugar:
  - `cx = T.launch_core("coreIdx.x", extent=gx)`
  - `cy = T.launch_core("coreIdx.y", extent=gy)`
- Attributes on `PrimFunc` (or top block):
  - `tt.core_grid = (gx, gy)`
  - `tt.core_ranges = [CoreRange(...), ...]`  (disjoint ok)
  - `tt.work_partition = { "(cx,cy)": [ {io, jo, len_k?, tile_order?}, ... ] }`
  - `tt.layout_desc[buffer] = { shard, interleave, stride, tile_id_order, ... }`

**Canonical mid‑level example:**
```python
@T.prim_func
def mm_core_tiles(A: T.Buffer((M, K), "float16"),
                  B: T.Buffer((K, N), "float16"),
                  C: T.Buffer((M, N), "float32")):
    cx = T.launch_core("coreIdx.x", extent = M_tiles // CORES_X)
    cy = T.launch_core("coreIdx.y", extent = N_tiles // CORES_Y)
    with T.block("C.tile"):
        io = T.axis.spatial(M_tiles, T.attr("tt.core_map_i", cx, cy))
        jo = T.axis.spatial(N_tiles, T.attr("tt.core_map_j", cx, cy))
        T.reads(A[io*TM:(io+1)*TM, 0:K], B[0:K, jo*TN:(jo+1)*TN])
        T.writes(C[io*TM:(io+1)*TM, jo*TN:(jo+1)*TN])
        T.init(): pass
        for ko in range(0, K, TK):
            a_tile = T.tile_load(A, io*TM, ko, shape=(TM, TK))
            b_tile = T.tile_load(B, ko, jo*TN, shape=(TK, TN))
            T.gemm(a_tile, b_tile)
```

### Stage B — Late Pass `GridToPersistentTT`
Consumes the mid‑level form and **injects** TT persistent loops, staging, barriers, and host artifacts.

**Responsibilities:**
1. Build **per‑core worklists** from `tt.work_partition` & `tt.core_ranges`.
2. Insert **persistent outer loop** per core that iterates assigned tiles.
3. Materialize **reader/compute/writer** pipelines (L1 circular buffers, prefetch).
4. Tensorize tile ops (`T.gemm`, `tile_load/store`, epilog).
5. Emit `tt.plan.json` (core grid, ranges, worklists, layouts).

**Conceptual persistent shape:**
```python
@T.prim_func
def mm_tt_persistent(...):
    tid = T.get_core_id()                     # backend intrinsic
    worklist = T.tt_load_worklist(tid)        # from tt.plan.json
    for task in worklist:                     # persistent loop
        io, jo = task.io, task.jo
        ko = 0
        while ko < K:
            a_buf.next = T.dram_to_l1(A, io, ko)
            b_buf.next = T.dram_to_l1(B, ko, jo)
            T.barrier()
            T.tt_mma(a_buf.curr, b_buf.curr, accum)
            ko += TK
        T.l1_to_dram(C, io, jo, accum)
```

### Pass Pipeline (high level)
1. Canonicalize grid (`T.Kernel → launch_thread/core`, normalize attrs)
2. **InferTTLayout / PropagateTTLayout** (stamp/normalize buffer layout & shard metadata)
3. **TTTilesToCoreMap** (compute `CoreRange`, build `tt.work_partition`)
4. **LowerTTTileIntrinsics** (tensorize tile ops)
5. **GridToPersistentTT** (inject persistent loops, staging, emit plan)

### `CoreRange` JSON (single source of truth)
```json
{
  "core_grid": [GX, GY],
  "core_ranges": [
    { "start": [sx, sy], "extent": [ex, ey] }
  ],
  "work_partition": {
    "(cx,cy)": [
      { "io": I0, "jo": J0, "len_k": K, "tile_order": "row_major" }
    ]
  },
  "layouts": {
    "A": { "shard": "DRAM", "interleave": true, "stride": [sa0, sa1] },
    "B": { "shard": "DRAM" },
    "C": { "shard": "L1", "tile_id_order": "match_shard" }
  }
}
```

### Guidance: Reductions, Halos, Multicast
- Keep reductions as `T.axis.reduce` until late; decide in‑tile vs cross‑tile accumulation later.
- Represent halos/tails as **irregular tiles** in `work_partition`, avoid scattering conditionals.
- Use attributes for multicast/reader groups; place barriers in the late pass.

### Implementation Checklist (AI Agent)
1. IR sugar: attach attrs (`tt.core_grid`, `tt.core_ranges`, `tt.work_partition`, `tt.layout_desc`).
2. Layout passes: `InferTTLayout`, `PropagateTTLayout` (DRAM/L1 shards, interleave, strides, tile order defaults).
3. Partitioning: `TTTilesToCoreMap` (allow arbitrary ND sharding; tails ok; halo unsupported initially).
4. Tile intrinsics lowering: `LowerTTTileIntrinsics` → TT device intrinsics.
5. Persistent pass: `GridToPersistentTT` (double‑buffering, barriers, per‑core worklists) + emit `tt.plan.json`.
6. Tests: attrs print/round‑trip, pipeline order, example GEMM, shard/layout invariants.
7. Diagnostics: friendly errors for layout mismatches, missing attrs, partition gaps.

---

*End.*