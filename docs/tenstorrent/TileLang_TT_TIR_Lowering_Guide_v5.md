# TileLang → Tenstorrent TIR Lowering Guide (Progressive, Protocol-Late)

**Version:** 5.0  
**Date:** 2025-10-15  
**Audience:** Compiler engineers / backend developers  
**Scope:** TIR surface, attributes, intrinsics, and progressive pass pipeline for the TileLang → Tenstorrent backend with *late* protocol insertion (CB handshakes, DST lifecycle, engine init).

> This guide unifies the prior proposals with a progressive-lowering design that keeps the TIR analyzable and schedulable as long as possible, and inserts protocol details (CB/NOC handshakes, DST lifecycle, compute engine init) only **after** we split into reader/compute/writer kernels.  
> Defaults: **interleaved + bfloat16 + DRAM** for buffers unless annotated otherwise. fileciteturn1file3

---

## 0) Design Goals (TL;DR)

- **Progressive lowering**: early passes attach metadata & restructure loops; late passes insert device protocols. fileciteturn1file1  
- **Protocol-less mid-level TIR**: compute expressed over CB-backed buffers but *without* CB handshakes or DST logic; good for analysis, scheduling, and pattern-matching.
- **Late protocolization**: after **SplitDeviceKernel**, inject:
  1) NOC/CB producer-consumer sequences (reserve/push/wait/pop) **only** into reader/writer,  
  2) **DST** acquire/commit/wait/release **only** into compute,  
  3) **Compute engine init** (`binary_op_init_common`, `add_tiles_init`, `mm_init`, SFPU/FPU configs) **only** into compute. fileciteturn1file5fileciteturn1file8fileciteturn1file10
- **Persistent kernels**: keep `T.launch_core` mid-level; *materialize* the per-core persistent tile loop **during codegen**, not as an actual `for` in TIR.
- **First-class layout/sharding**: attach rich buffer layout + sharding attributes; default to interleaved DRAM; support 2D and ND sharding projection onto compute plane. fileciteturn1file3fileciteturn1file0

---

## 1) Attribute Schema (single source of truth)

All attributes live in `tilelang.tenstorrent.attrs` (Python) and mirrored in C++ visitors.

### PrimFunc-level

| Key | Type | Meaning |
|---|---|---|
| `tt.core_grid` | `[gx, gy]` | Logical compute grid used by `T.launch_core`. |
| `tt.core_ranges` | `CoreRangeSet` | Active core ranges (device topology domain). |
| `tt.partition_mode` | `"global"|"local_shard"` | Work partition policy (global tiles vs shard-local). |
| `tt.grid_tiles` | `[Mt, Nt]` | Global tile counts (M, N). |
| `tt.work_partition` | `Map[str → List[WorkItem]]` | Core → list of tile work items. |
| `tt.runtime_args` | `List[str]` | Ordered kernel args (`start_id`, `count`, etc.). |
| `tt.tile_dfg` | `Map` | Per-kernel tile dataflow graph edges (producer→consumer CB links). |
| `tt.cb_desc` | `Map[name → {page_size, depth, data_format}]` | Conceptual CB geometry (up to 32). |
| `tt.kernel_role` | `"monolithic" \| "reader" \| "compute" \| "writer"` | Assigned after split. |

### Buffer-level

| Key | Type | Meaning |
|---|---|---|
| `tt.layout_desc` | `Dict` | Memory space (`DRAM`/`L1`), memory layout (`interleaved`/`sharded`), dtype, tile shape (default 32×32), sharding (2D/ND) with projection to compute plane. fileciteturn1file3fileciteturn1file0 |
| `tt.tensor_accessor` | `Dict` | Abstract **TensorAccessor** metadata, *not* bound to runtime args yet (compile-time description only). Configured late. fileciteturn1file8 |

> **Why mid-level accessors?** Kernels initially use abstract accessors (capture layout & strides). Only after runtime args are finalized do we assign concrete **TensorAccessorArgs** slots for reader/writer. fileciteturn1file8fileciteturn1file18

---

## 2) TIR Intrinsic Surface

We separate **abstract intrinsics** (protocol-less) from **protocol intrinsics** (inserted late).

### 2.1 Abstract (protocol-less, early)
- `tt.alloc_cb(name, shape, dtype)` – conceptual CB allocation (maps from shared).  
- `tt.read_to_cb(tensor_slice, cb)` – abstract DRAM→CB read, *no* NOC/CB calls yet.  
- `tt.write_from_cb(cb, tensor_slice)` – abstract CB→DRAM write, *no* NOC/CB calls yet.  
- **Compute** (operate on CBs without DST/engine init yet):  
  - `tt.mm.mma(cb_a, cb_b, dst=0, accumulate=bool)` (or `tt.mm.mma_tile(...)`)  
  - `tt.fpu.add(cb_x, cb_y, dst=0)` (binary ops)  
  - `tt.sfpu.unary(op, dst)` (`sin`, `exp`, etc.) – protocol-less form uses `copy_tile` implicitly deferred until late pass. fileciteturn1file10

### 2.2 Protocol intrinsics (inserted late, compute)
- **Engine init (FPU/SFPU/Pack/Unpack)**:  
  - `tt.engine.init_common(cb_in0, cb_in1, cb_out)` (unpack/math/pack)  
  - `tt.fpu.matmul_init(cb_a, cb_b, cb_out)`  
  - `tt.sfpu.init(op, cb_in?, cb_out?)`  
  - `tt.pack.init(cb_out)`  
  These map to Metalium APIs like `binary_op_init_common`, `add_tiles_init`, `mm_init`, etc. fileciteturn1file5

- **DST lifecycle (Math core only):** `tt.dst.acquire()`, `tt.dst.commit()`, `tt.dst.wait()`, `tt.dst.release()` (double-buffered Dst register protocol). fileciteturn1file10

### 2.3 Protocol intrinsics (inserted late, reader/writer)
- **CB handshake & NOC:** `cb_reserve_back`, `get_write_ptr`, `noc_async_read_tile`, `noc_async_read_barrier`, `cb_push_back`, `cb_wait_front`, `get_read_ptr`, `noc_async_write_tile`, `noc_async_write_barrier`, `cb_pop_front`. fileciteturn1file8fileciteturn1file18

> All effect-only calls are emitted as `T.evaluate(tt.*(...))` because they produce **side effects** rather than values; `T.evaluate` preserves them in TIR and gives them sequencing semantics. (TileLang/TIR convention)

---

## 3) Pass Pipeline (Progressive, with *late* split & protocolization)

> The pipeline below merges and clarifies prior plans; notable changes: **SplitDeviceKernel** is late; compute **engine init** is inserted late like DST; persistent loop is **generated in codegen**, not as a TIR `for`. fileciteturn1file4fileciteturn1file16

| # | Pass | Role | Inputs → Outputs | Attributes stamped/updated | TIR effect |
|---:|---|---|---|---|---|
| **A1** | **InferTTLayout** | Normalize buffer defaults and explicit annotations (layout, dtype, tile=32×32; interleaved+bf16+DRAM by default). | Buf decls → `tt.layout_desc` | `tt.layout_desc[buf]` (+ projected shard grid for 2D/ND). | None (metadata only). fileciteturn1file3fileciteturn1file6fileciteturn1file0 |
| **A2** | **PropagateTTLayout** | Derive conceptual CB geometry from layout (page_size from dtype×tile, depth=2 default). | `tt.layout_desc` → `tt.cb_desc` | `tt.cb_desc[name]={page_size,depth,format}` | None (metadata only). |
| **A3** | **AttachTensorAccessorTT** | Attach **abstract** TensorAccessor descriptors to buffers (no runtime binding). | `tt.layout_desc` → `tt.tensor_accessor[buf]` | Accessor spec (compile-time) | None. fileciteturn1file8 |
| **B1** | **LayoutAwareWorkPartitionTT** | Choose `tt.core_grid`, `tt.core_ranges`, and `tt.work_partition`; compute `[Mt,Nt]`. | Shapes+layout → partition | `tt.core_grid`, `tt.core_ranges`, `tt.work_partition`, `tt.grid_tiles`, `tt.partition_mode` | None. |
| **B2** | **GridToCoreGrid** | Convert `with T.Kernel(gx,gy)` to mid-level `T.launch_core`; add axis mapping (`tt.core_map_i/j`). | Grid loops → core launch | (No new attrs) | Replace grid with `cx= T.launch_core("coreIdx.x",gx)`, `cy=...`; keep body intact. fileciteturn1file4 |
| **C1** | **LowerSharedToCB** | Turn `alloc_shared` into `tt.alloc_cb`; turn `T.copy` into abstract `tt.read_to_cb` / `tt.write_from_cb`. | Shared allocs & copies → conceptual CB dataflow | - | Replace shared allocs/copies with CB allocs & abstract data-moves (protocol-less). |
| **C2** | **LowerTTTileIntrinsics** | Tensorize compute to CB-based intrinsics (no DST/engine init). | `T.gemm`/eltwise → `tt.mm.mma`/`tt.fpu.add`/`tt.sfpu.unary` | (May tag `tt.compute.pattern` hints) | Replace high-level ops inside loops with protocol-less compute intrinsics. fileciteturn1file13 |
| **C3** | **BuildTileDFGTT** | Build per-PrimFunc tile dataflow graph to drive CB index assignment & split. | CB allocs + abstract moves | `tt.tile_dfg` (nodes, CB edges, roles) | None (metadata only). |
| **D1 (late)** | **SplitDeviceKernel** | Split **monolithic** kernel into three: `reader`, `compute`, `writer`. Assign roles and CB indexes. | Monolithic → 3 PrimFuncs | `tt.kernel_role`, per-func `tt.runtime_args` seed | Clone & slice TIR by role; assign CB IDs (≤32). fileciteturn1file4 |
| **D2 (late)** | **ConfigureTensorAccessorTT** | Bind **TensorAccessorArgs** ordering/slots, now that runtime args are known and kernels are split (reader/writer only). | `tt.tensor_accessor` + arg order | Update accessors with arg indices, sizes | None (metadata only; consumed in codegen). fileciteturn1file8 |
| **D3 (late)** | **LowerCBIntrinsics** *(reader/writer)* | Lower abstract `tt.read_to_cb` / `tt.write_from_cb` to **NOC/CB protocol** calls. | Abstract moves → protocol | - | Insert `cb_reserve_back`/`get_write_ptr`/`noc_async_*`/`cb_push_back` and the mirror on writer. fileciteturn1file8fileciteturn1file18 |
| **D4 (late)** | **InsertComputeInitTT** *(compute)* | Insert **engine init protocol** for Unpack/Math/Pack (`*_init_common`, `add_tiles_init`, `mm_init`, SFPU configs). | Protocol-less compute → initialized compute | - | Prepend init calls before tile loop(s); no DST yet. fileciteturn1file5 |
| **D5 (late)** | **InsertDSTManagementTT** *(compute)* | Wrap compute with **DST lifecycle** and packing to CB (acquire/commit/wait/release + `pack_tile`). | Protocol-less compute → DST protocolized | - | Enclose K-loop (GEMM) or per-tile (eltwise) with DST protocol; write to `cb_out`. fileciteturn1file10 |
| **E1** | **FinalizePersistentSignatureTT** | Freeze `tt.runtime_args` and constants needed for index recovery (no explicit loops in TIR). | Assemble args | Append `start_id`, `count`, `Mt`, `Nt`, and shard extras for local mode | None. |
| **F** | **VerifyTTIR** | Capacity and conformance checks (≤32 CBs, L1 size, dtype support, metadata presence). | — | — | No structural change. |
| **G** | **Codegen (3 kernels + host)** | Emit `reader.cpp`, `compute.cpp`, `writer.cpp`, `main.cpp`, `tt.plan.json`; materialize persistent **per-core** loop in generated code. | — | — | Codegen-time: emit per-core tile `for (i=0; i<count; ++i)` using `tt.runtime_args` & accessors. fileciteturn1file8fileciteturn1file5 |

> **Why late `SplitDeviceKernel`?** It allows mid-level analyses (scheduling, fusion) to operate on a single TIR, while still yielding a clean dataflow pipeline at the end with precise protocol insertion points. fileciteturn1file1

---

## 4) Pass-by-Pass Examples

Below we show **GEMM** (tile MMA) and **Eltwise Add** (binary) as they evolve.

> Notation: we elide bounds math for brevity; `A_cb/B_cb/C_cb` are conceptual CBs; protocol calls are only shown after late passes.

### 4.1 Initial (user) – shared memory, GPU-like

#### GEMM
```python
@T.prim_func
def gemm(A: T.Buffer((M,K), "bf16"), B: T.Buffer((K,N), "bf16"), C: T.Buffer((M,N), "bf16")):
    with T.Kernel(T.ceildiv(N,128), T.ceildiv(M,128)) as (bx, by):
        A_sh = T.alloc_shared((128, 32), "bf16")
        B_sh = T.alloc_shared((32, 128), "bf16")
        C_frag = T.alloc_fragment((128, 128), "bf16")
        T.clear(C_frag)
        for kt in T.Pipelined(T.ceildiv(K, 32), num_stages=3):
            T.copy(A[by*128, kt*32], A_sh)
            T.copy(B[kt*32, bx*128], B_sh)
            T.gemm(A_sh, B_sh, C_frag)
        T.copy(C_frag, C[by*128, bx*128])
```

#### Eltwise Add
```python
@T.prim_func
def eadd(A: T.Buffer((M,N), "bf16"), B: T.Buffer((M,N), "bf16"), C: T.Buffer((M,N), "bf16")):
    with T.Kernel(T.ceildiv(N,128), T.ceildiv(M,128)) as (bx, by):
        ShA = T.alloc_shared((128,128),"bf16")
        ShB = T.alloc_shared((128,128),"bf16")
        ShC = T.alloc_shared((128,128),"bf16")
        T.copy(A[by*128, bx*128], ShA)
        T.copy(B[by*128, bx*128], ShB)
        for i, j in T.Parallel(128,128):
            ShC[i,j] = ShA[i,j] + ShB[i,j]
        T.copy(ShC, C[by*128, bx*128])
```

### 4.2 After **A1–A3** (layout + CB metadata + abstract accessor)

- Buffers stamped with `tt.layout_desc` (default: interleaved, bf16, DRAM; tile=32×32). fileciteturn1file3  
- `tt.cb_desc` added (page_size=2048 for bf16, depth=2 by default).  
- `tt.tensor_accessor` attached (abstract). fileciteturn1file8

*TIR unchanged (metadata-only).*

### 4.3 **B1–B2** (work partition + core launch)

```python
@T.prim_func
def gemm_core(A, B, C):
    # attrs: tt.core_grid=[gx,gy], tt.grid_tiles=[Mt,Nt], tt.work_partition=...
    cx = T.launch_core("coreIdx.x", gx)
    cy = T.launch_core("coreIdx.y", gy)
    # same body as §4.1 (still shared)
```

(Same for `eadd_core`.)

### 4.4 **C1** LowerSharedToCB (conceptual CBs + abstract moves)

#### GEMM
```python
@T.prim_func
def gemm_cb(A,B,C):
    cx,cy = T.launch_core(...), T.launch_core(...)
    A_cb = tt.alloc_cb("cb_in0", (128,32), "bf16")
    B_cb = tt.alloc_cb("cb_in1", (32,128), "bf16")
    C_cb = tt.alloc_cb("cb_out", (128,128), "bf16")
    for kt in T.Pipelined(T.ceildiv(K,32), num_stages=3):
        T.evaluate(tt.read_to_cb(A[by*128, kt*32], A_cb))      # abstract
        T.evaluate(tt.read_to_cb(B[kt*32, bx*128], B_cb))      # abstract
        T.gemm(A_cb, B_cb, C_cb)                               # still high-level
    T.evaluate(tt.write_from_cb(C_cb, C[by*128, bx*128]))      # abstract
```

#### Eltwise Add
```python
@T.prim_func
def eadd_cb(A,B,C):
    cx,cy = T.launch_core(...), T.launch_core(...)
    A_cb = tt.alloc_cb("cb_in0",(128,128),"bf16")
    B_cb = tt.alloc_cb("cb_in1",(128,128),"bf16")
    C_cb = tt.alloc_cb("cb_out",(128,128),"bf16")
    T.evaluate(tt.read_to_cb(A[by*128, bx*128], A_cb))
    T.evaluate(tt.read_to_cb(B[by*128, bx*128], B_cb))
    for i,j in T.Parallel(128,128):            # left for SFPU pass later if needed
        # temporarily keep Parallel; or rewrite to cb-wise binary op in next pass
        pass
    T.evaluate(tt.write_from_cb(C_cb, C[by*128, bx*128]))
```

### 4.5 **C2** LowerTTTileIntrinsics (protocol-less compute on CBs)

#### GEMM (K-loop MMA on CBs)
```python
@T.prim_func
def gemm_tiles(A,B,C):
    cx,cy = T.launch_core(...), T.launch_core(...)
    A_cb,B_cb,C_cb = ... # as above
    for kt in T.Pipelined(T.ceildiv(K,32)):
        T.evaluate(tt.mm.mma(A_cb, B_cb, dst=0, accumulate=(kt>0)))  # no DST yet
    T.evaluate(tt.write_from_cb(C_cb, C[by*128, bx*128]))
```

#### Eltwise Add (binary tile op on CBs)
```python
@T.prim_func
def eadd_tiles(A,B,C):
    cx,cy = T.launch_core(...), T.launch_core(...)
    A_cb,B_cb,C_cb = ...
    # Reader abstract moves already above; now compute over CBs:
    T.evaluate(tt.fpu.add(A_cb, B_cb, dst=0))   # protocol-less
    T.evaluate(tt.write_from_cb(C_cb, C[by*128, bx*128]))
```

*(Optionally, a **LowerToSFPU** pass could replace `T.Parallel` with `tt.sfpu.unary/binary` protocol-less ops for intra-tile SIMD. DST and pack are still deferred.)*

### 4.6 **C3** BuildTileDFGTT (metadata only)

- Records that `A_cb,B_cb → compute → C_cb`, assigns candidate CB IDs, checks fan-in/out and reuse. Saved to `tt.tile_dfg` for split.

### 4.7 **D1 (late)** SplitDeviceKernel (monolithic → reader/compute/writer)

```python
# KERNEL 1: READER (A,B → CBs)
@T.prim_func
def gemm_reader(A,B):
    # attrs: tt.kernel_role="reader"; tt.runtime_args ~ ["a_addr","b_addr","start_id","count",...]
    cx,cy = T.launch_core(...), T.launch_core(...)
    T.evaluate(tt.read_to_cb(A[...], "cb_in0"))
    T.evaluate(tt.read_to_cb(B[...], "cb_in1"))

# KERNEL 2: COMPUTE (CBs → CB_out)
@T.prim_func
def gemm_compute():
    # attrs: tt.kernel_role="compute"
    cx,cy = T.launch_core(...), T.launch_core(...)
    for kt in T.serial(Kt):
        T.evaluate(tt.mm.mma("cb_in0","cb_in1", dst=0, accumulate=(kt>0)))  # still no DST/pack

# KERNEL 3: WRITER (CB_out → C)
@T.prim_func
def gemm_writer(C):
    # attrs: tt.kernel_role="writer"
    cx,cy = T.launch_core(...), T.launch_core(...)
    T.evaluate(tt.write_from_cb("cb_out", C[...]))
```
(Analogous split for `eadd_*` kernels.) fileciteturn1file13

### 4.8 **D2 (late)** ConfigureTensorAccessorTT (reader/writer only)

- Bind concrete **TensorAccessorArgs** slots & `tile_size_bytes` using finalized runtime arg order; still TIR-metadata only; consumed by codegen. fileciteturn1file8

### 4.9 **D3 (late)** LowerCBIntrinsics (reader/writer protocol)

#### Reader
```python
# gemm_reader after D3
T.evaluate(cb_reserve_back("cb_in0", 1))
T.evaluate(cb_reserve_back("cb_in1", 1))
T.evaluate(noc_async_read_tile(tile_id_expr_A, A_accessor, get_write_ptr("cb_in0")))
T.evaluate(noc_async_read_tile(tile_id_expr_B, B_accessor, get_write_ptr("cb_in1")))
T.evaluate(noc_async_read_barrier())
T.evaluate(cb_push_back("cb_in0", 1))
T.evaluate(cb_push_back("cb_in1", 1))
```
#### Writer
```python
# gemm_writer after D3
T.evaluate(cb_wait_front("cb_out", 1))
T.evaluate(noc_async_write_tile(tile_id_expr_C, C_accessor, get_read_ptr("cb_out")))
T.evaluate(noc_async_write_barrier())
T.evaluate(cb_pop_front("cb_out", 1))
```
(See Metalium examples for the exact API shape.) fileciteturn1file8fileciteturn1file18

### 4.10 **D4 (late)** InsertComputeInitTT (engine init protocol)

```python
# gemm_compute before DST
T.evaluate(tt.engine.init_common("cb_in0","cb_in1","cb_out"))
T.evaluate(tt.fpu.matmul_init("cb_in0","cb_in1","cb_out"))
# …then keep the K-loop with mm.mma(…) as-is
```
For eltwise:
```python
T.evaluate(tt.engine.init_common("cb_in0","cb_in1","cb_out"))
T.evaluate(tt.fpu.binary_init("cb_in0","cb_in1","cb_out", op="add"))
```
Maps to Metalium `binary_op_init_common`, `add_tiles_init`, `mm_init`, etc. fileciteturn1file5

### 4.11 **D5 (late)** InsertDSTManagementTT (DST + pack protocol)

#### GEMM (K-loop accumulation pattern)
```python
T.evaluate(tt.dst.acquire())
for kt in T.serial(Kt):
    T.evaluate(cb_wait_front("cb_in0", 1))
    T.evaluate(cb_wait_front("cb_in1", 1))
    T.evaluate(tt.mm.mma("cb_in0","cb_in1", dst=0, accumulate=(kt>0)))
    T.evaluate(cb_pop_front("cb_in0", 1))
    T.evaluate(cb_pop_front("cb_in1", 1))
T.evaluate(cb_reserve_back("cb_out", 1))
T.evaluate(tt.dst.commit())
T.evaluate(tt.dst.wait())
T.evaluate(pack_tile(dst=0, cb="cb_out", tile_index=0))
T.evaluate(tt.dst.release())
T.evaluate(cb_push_back("cb_out", 1))
```
#### Eltwise (no K-loop; per-tile protocol)
```python
T.evaluate(cb_wait_front("cb_in0", 1))
T.evaluate(cb_wait_front("cb_in1", 1))
T.evaluate(tt.dst.acquire())
T.evaluate(tt.fpu.add("cb_in0","cb_in1", dst=0))
T.evaluate(cb_pop_front("cb_in0", 1))
T.evaluate(cb_pop_front("cb_in1", 1))
T.evaluate(cb_reserve_back("cb_out", 1))
T.evaluate(tt.dst.commit())
T.evaluate(tt.dst.wait())
T.evaluate(pack_tile(dst=0, cb="cb_out", tile_index=0))
T.evaluate(tt.dst.release())
T.evaluate(cb_push_back("cb_out", 1))
```
See Metalium compute examples for the exact ordering across Unpack/Math/Pack. fileciteturn1file5

### 4.12 **E1** FinalizePersistentSignatureTT

- `tt.runtime_args = ["start_id","count","Mt","Nt", ("Sm","Sn","Gy","Gx","sy","sx")?]` based on `tt.partition_mode`.  
- We keep **`T.launch_core`**; the **per-core persistent loop** `for (i=0;i<count;++i)` is generated **during codegen**, not in TIR. (Thin waist design.)

### 4.13 **F, G** Verification & Codegen

- Verify CB count (≤ 32), L1 footprint, dtype support, required metadata.  
- Emit 3 kernels + host; reader/writer execute NOC/CB protocols; compute executes engine init + DST protocol; host uses **TensorAccessor** + run-time args for tiled addressing. fileciteturn1file8fileciteturn1file5

---

## 5) Sharding & Layout Notes (defaults and options)

- Default: **interleaved DRAM** + bf16 + 32×32 tiles. fileciteturn1file3  
- Tiled storage aligns naturally with 32×32 computation; faces (16×16) are an internal detail of the FPU. fileciteturn1file9fileciteturn1file7  
- Sharding strategies: **height**, **width**, **block**; ND sharding supported experimentally; all are projected to a 2D compute plane for core mapping. fileciteturn1file0fileciteturn1file11fileciteturn1file12fileciteturn1file14fileciteturn1file15

---

## 6) FAQ

**Q: Why are many intrinsics wrapped in `T.evaluate`?**  
Because they are **effect-only** (mutate CBs or device registers) and return no SSA value. `T.evaluate` preserves their order and semantics in TIR.

**Q: Where do we allocate CBs?**  
Conceptually in TIR via `tt.alloc_cb` (≤ 32 per kernel). Concrete Metalium CB config is emitted by codegen visitors; TIR keeps the index map & page/depth in `tt.cb_desc`.

**Q: How do TensorAccessors appear in TIR?**  
As buffer-level metadata (`tt.tensor_accessor`) plus codegen binding (`TensorAccessorArgs<slot>()`). Readers/writers consume them when generating `noc_async_*_tile`. fileciteturn1file8

**Q: Persistent loop is not visible in TIR—why?**  
To keep mid-level TIR clean. We expand the per-core tile loop during codegen using `tt.runtime_args` (`start_id`,`count`) and `tt.grid_tiles` for (m,n) recovery.

---

## 7) Checklists

### What must be present *before* SplitDeviceKernel (late D1)?
- `tt.layout_desc`, `tt.cb_desc`, `tt.tensor_accessor` (abstract), `tt.core_grid`, `tt.work_partition`, `tt.grid_tiles`, protocol-less CB compute (`tt.mm.mma` / `tt.fpu.*` / `tt.sfpu.*`).

### What is inserted *after* SplitDeviceKernel?
- Reader/Writer: **NOC/CB protocol** (reserve/push/wait/pop + async read/write).  
- Compute: **engine init** (Unpack/Math/Pack) + **DST lifecycle** + **pack_tile**.  
- Accessors bound to runtime args in reader/writer.  
- `tt.runtime_args` finalized (persistent loop emitted in codegen).

---

## 8) Appendix: Intrinsic Naming Cheatsheet

- **Abstract dataflow**: `tt.read_to_cb`, `tt.write_from_cb`  
- **CB protocol**: `cb_reserve_back`, `cb_push_back`, `cb_wait_front`, `cb_pop_front`, `noc_async_read_tile`, `noc_async_write_tile`, `*_barrier`  
- **Compute (protocol-less)**: `tt.mm.mma`, `tt.fpu.add`, `tt.sfpu.unary(op, dst)`  
- **Engine init (late)**: `tt.engine.init_common`, `tt.fpu.matmul_init`, `tt.fpu.binary_init(op)`, `tt.sfpu.init(op)`  
- **DST**: `tt.dst.acquire`, `tt.dst.commit`, `tt.dst.wait`, `tt.dst.release`, `pack_tile`

---

**References / Source Material**  
Progressive lowering philosophy and examples; dataflow & split details; Metalium CB/NOC and compute APIs; layout & sharding defaults and options: fileciteturn1file1 fileciteturn1file4 fileciteturn1file13 fileciteturn1file16 fileciteturn1file8 fileciteturn1file5 fileciteturn1file10 fileciteturn1file18 fileciteturn1file3 fileciteturn1file6 fileciteturn1file0

