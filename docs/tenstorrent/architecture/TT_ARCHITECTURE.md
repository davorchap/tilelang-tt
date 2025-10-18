# TileLang Tenstorrent Backend Architecture

**Version:** 3.0 (v5 Pipeline)
**Date:** 2025-10-16
**Status:** Production (Python-only backend)

## Overview

The TileLang Tenstorrent backend compiles high-level TileLang DSL kernels to Tenstorrent TT-Metalium API code. This document provides a comprehensive architecture overview of the compilation pipeline, IR transformations, and code generation.

**Key Principle:** Map TileLang's GPU-style grid kernels to Tenstorrent's persistent, tile-based execution model.

---

## Architecture Summary

```
TileLang DSL (Python)
    ↓
TVM IRModule (Frontend Lowering)
    ↓
Apply TT Defaults
    ↓
Transform Pipeline (Metadata + TIR Transforms)
    ↓
Code Generation (3 Kernels + Host)
    ↓
TT-Metalium C++ Code
```

---

## 1. Compilation Pipeline

### Phase 1: Frontend Lowering (Shared with CUDA)

**Entry Point:** `tilelang/engine/phase.py` → `LowerAndLegalize()`

**12 Shared Passes:**
- `BindTarget` - Bind target info to module
- `LetInline` - Inline let expressions
- `AddWrapperForSingleBufStore` - Add buffer store wrapper
- `InjectAssumes` - Inject assumes for TVM prover
- `Simplify` - Simplify IR expressions
- `LayoutReducer` - Set layouts for reducers
- **`LayoutInference`** - Infer fragment/shared memory layouts
- `LowerTileOp` - Lower T.copy, T.fill to loops
- `LowerL2Persistent` - Lower L2 persistent operations
- `LegalizeVectorizedLoop` - Ensure valid vectorization
- `LegalizeSafeMemoryAccess` - Add safety checks
- `LoopVectorizeDynamic` - Vectorize dynamic loops

**Input:** TileLang DSL (T.Kernel, T.copy, T.gemm)
**Output:** Legalized TIR with memory layouts inferred

### Phase 2: Apply TT Defaults

**Entry Point:** `tilelang/tenstorrent/target.py` → `apply_tt_defaults()`

**Annotations Added:**
```python
"tt.schedule": {
  "policy": "contiguous",
  "order": "row_major"
}

"tt.shard": {
  "layout": "tiled",
  "interleaved": true,
  "tile_shape": [32, 32]
}
```

**Purpose:** Ensure GPU-style kernels run on TT with sensible defaults.

### Phase 2.5: Layout-Aware Metadata (v5 Stages A-B)

**Why needed?** Default schedule/shard metadata does not capture user intent for DRAM vs L1 residency, ND sharding, or tile-order annotations. The layout-aware stage introduces explicit buffer- and function-level attributes that downstream passes and codegen can rely on.

**v5 Stage A: Metadata (3 passes)**
- `infer_tt_layout_v5` – Normalize user annotations (`annotate_tt_layout`) and emit `tt.buffer.<name>` dictionaries with memory space, layout, dtype, tile shape, and optional N‑D shard metadata. Validates L1 shards and rejects halo hints.
- `propagate_tt_layout_v5` – Reads buffer metadata and stamps `tt.cb.<name>` attributes describing circular buffer geometry (`page_size`, `depth`, `data_format`) for each DRAM↔L1 copy.
- `attach_tensor_accessor_tt` – Attaches TensorAccessor metadata for buffer addressing.

**v5 Stage B: Partitioning (2 passes)**
- `layout_aware_work_partition_tt_v5` – Chooses per-core work assignments based on buffer residency. Emits `tt.partition_mode` (`global` vs `local_shard`), `tt.core_ranges`, `tt.grid_tiles`, `tt.shard_grid`, `tt.local_shape_tiles`, and canonical `tt.runtime_args`.
- `grid_to_core_grid_v5` – Maps logical grid coordinates to physical core coordinates.

**PrimFunc Attributes after this stage:**
```json
{
  "tt.partition_mode": "global" | "local_shard",
  "tt.grid_tiles": [Mt, Nt],
  "tt.shard_grid": [Gy, Gx],
  "tt.local_shape_tiles": [Sm, Sn],
  "tt.runtime_args": ["start_id","count","Mt","Kt","Nt","Sm","Sn","Gy","Gx","sy","sx"],
  "tt.core_ranges": [[y0,x0],[y1,x1], ...]
}
```

**Buffer Attributes:**
```json
"tt.buffer.A": {
  "memory": "DRAM",
  "layout": "sharded",
  "tile_shape": [32,32],
  "dtype": "bf16",
  "nd_shard": {
    "axes": ["B","H","M","N"],
    "grid": [gB,gH,gM,gN],
    "shard_shape_elems": [sB,sH,sM,sN],
    "order": "row_major",
    "align_tiles": true,
    "projected_grid": [Gy,Gx],
    "projected_shard_tiles": [Sm,Sn]
  }
}
```

This metadata becomes the authoritative source for later stages.

### Phase 3: v5 TT-Specific Transform Pipeline

**Entry Point:** `tilelang/engine/tenstorrent/lower.py` → `OptimizeForTargetTT()`

The v5 pipeline consists of **14 passes organized into 5 stages (A-E)**, all implemented in Python for maintainability and rapid iteration. For complete details including pass-by-pass transformations, dependency graphs, and code examples, see [v5_pipeline.md](v5_pipeline.md).

**Pipeline Summary:**
- **Stage A:** Metadata (3 passes) - Buffer layout inference and CB configuration
- **Stage B:** Partitioning (2 passes) - Work distribution and core mapping
- **Stage C:** Protocol-less Lowering (3 passes) - Abstract tile operations
- **Stage D:** Late Split & Protocol (5 passes) - Kernel splitting and protocol insertion
- **Stage E:** Finalization (1 pass) - Runtime signature and validation

**Common Optimization Passes (11, shared with CUDA):**
- `FlattenBuffer` - Flatten multi-dim buffers to 1D
- `ConfigIndexBitwidth` - Optimize index computation
- `Simplify` - Simplify expressions
- `VectorizeLoop` - Vectorize loops (32-element tiles)
- `StorageRewrite` - Rewrite storage allocations
- `UnrollLoop` - Unroll small loops
- `RenormalizeSplitPattern` - Normalize split patterns
- `RemoveNoOp` - Remove no-op statements
- `RewriteUnsafeSelect` - Rewrite unsafe selects
- `HoistIfThenElse` - Hoist if out of loops
- `VerifyMemory` - Verify memory accesses

**Verification Pass:**
- **`verify_tt_ir`** - Verify TT constraints (grid size, CB counts)

**Total:** 14 v5 TT-specific + 11 shared + 1 verification = 26 passes

**Key Architectural Decision:** All TT backend passes remain in Python for maintainability and rapid iteration. No C++ migration planned.

---

## Attribute Schema

For complete buffer and PrimFunc attribute schemas, see the detailed tables in [v5_pipeline.md](v5_pipeline.md#stage-a-metadata-3-passes). The v5 pipeline maintains comprehensive metadata throughout compilation including:

- **Buffer attributes** (`tt.buffer.<name>`): Memory space, layout, tile shape, dtype, ND sharding
- **CB attributes** (`tt.cb.<name>`): Page size, depth, data format
- **PrimFunc attributes**: Partition mode, grid tiles, core ranges, runtime args
- **TensorAccessor metadata**: Addressing and stride information

**Diagnostics & Guardrails:**
- Halo metadata is rejected (*"halo unsupported"*)
- L1 shards must be tile-aligned and fit within capacity
- Host generator validates TensorAccessor construction

---

## Host & Kernel Responsibilities (Layout-Aware)

### Buffer Creation APIs

| Residency | Layout | Host API | Notes |
|-----------|--------|----------|-------|
| DRAM | interleaved | `CreateBuffer(InterleavedBufferConfig{...}, BufferType::DRAM)` | Default configuration. |
| DRAM | sharded | `CreateBuffer(ShardedBufferConfig{...}, BufferType::DRAM)` | Accepts ND shard metadata. |
| L1 | sharded | `CreateBuffer(ShardedBufferConfig{...}, BufferType::L1)` | Opt-in; enforced alignments. |

### TensorAccessor Policy

- Host `main.cpp` now materialises `TensorAccessorArgs::Create(...)` blobs for every DRAM/L1 buffer and guards against default construction.
- Runtime args expose base tile ranges plus shard geometry so device kernels can compute global indices deterministically.
- Local L1 shards (owned by the executing core) still read/write CBs directly; the guardrail simply validates metadata consistency.

### Runtime Argument Payload

| Mode | Arguments | Host emission |
|------|-----------|---------------|
| Global | `start_id`, `count`, `Mt`, `Kt`, `Nt` | `kRuntimeArgNames = {"tt_start_tile", ...}` |
| Local shard | Above + `Sm`, `Sn`, `Gy`, `Gx`, `sy`, `sx` | Extra entries append shard geometry (`tt_shard_coord_{y,x}`) |

### Tile Order Options

- `row_major` remains the default (global or shard-local).
- `match_shard` iterates shard-local tiles in-row-major order then follows shard assignment.
- `block_linear(k)` reserved for future `RasterizationTT` pass.

---

## Test Matrix (Planned)

| Scenario | Goal | Expected Verification |
|----------|------|-----------------------|
| DRAM interleaved | Default behavior | `tt.partition_mode="global"`, TA compile-args present, default CB geometry. |
| DRAM sharded | Treat sharding as first-class | Host uses `ShardedBufferConfig`, runtime args remain global, tile IDs map via TensorAccessor. |
| L1 sharded | Enforce opt-in residency | `tt.partition_mode="local_shard"`, `tt.core_ranges` == shard grid, runtime args include shard offsets. |
| ND sharding projection | Validate axis → compute mapping | `[Gy,Gx]`, `[Sm,Sn]` derived correctly for mixed axes. |
| Negative halo | Diagnostics | Pass errors with *"halo unsupported"*. |
| L1 overflow | Diagnostics | Fails capacity check. |
| Guardrail | Prevent DRAM TA misuse | Unit test fails on default-constructed TensorAccessorArgs. |

### Phase 4: Code Generation (IR-Driven)

**Entry Point:** `tilelang/tenstorrent/codegen/kernel_generators.py` → Python codegen visitors

**3-Kernel Architecture:**

1. **Reader Kernel** (`reader.cpp`)
   - **Visitor:** `TTReaderCodegenVisitor`
   - **Purpose:** Load data from DRAM to L1 circular buffers via NOC
   - **Generated Code:**
   ```cpp
   void kernel_main() {
       uint32_t dram_addr_a = get_arg_val<uint32_t>(0);
       uint32_t dram_addr_b = get_arg_val<uint32_t>(1);
       uint32_t Mt = get_arg_val<uint32_t>(2);
       uint32_t Kt = get_arg_val<uint32_t>(3);
       uint32_t Nt = get_arg_val<uint32_t>(4);

       for (uint32_t tile = 0; tile < num_tiles; ++tile) {
           cb_reserve_back(cb_in0, 1);
           uint32_t l1_write_addr = get_write_ptr(cb_in0);
           noc_async_read_tile(tile, dram_addr_a, l1_write_addr);
           noc_async_read_barrier();
           cb_push_back(cb_in0, 1);
       }
   }
   ```
   Reader and writer kernels currently mark shard coordinates as intentionally unused (`(void)tt_shard_coord_*`) while the host provides validated TensorAccessor metadata for future tensorisation work.

2. **Compute Kernel** (`compute.cpp`)
   - **Visitor:** `TTComputeCodegenVisitor`
   - **Purpose:** Perform tile computations using Tensix cores
   - **Generated Code:**
   ```cpp
   void MAIN() {
       for (uint32_t tile = 0; tile < num_tiles; ++tile) {
           tile_regs_acquire();

           // K-loop for matmul
           mm_init(cb_in0, cb_in1, cb_out0);
           for (uint32_t k = 0; k < Kt; ++k) {
               cb_wait_front(cb_in0, 1);
               cb_wait_front(cb_in1, 1);

               bool accumulate = (k > 0);
               matmul_tiles(cb_in0, cb_in1, 0, 0, 0, accumulate);

               cb_pop_front(cb_in0, 1);
               cb_pop_front(cb_in1, 1);
           }

           tile_regs_commit();
           tile_regs_wait();
           pack_tile(0, cb_out0);
           tile_regs_release();
       }
   }
   ```

3. **Writer Kernel** (`writer.cpp`)
   - **Visitor:** `TTWriterCodegenVisitor`
   - **Purpose:** Write results from L1 to DRAM via NOC
   - **Generated Code:**
   ```cpp
   void kernel_main() {
       uint32_t dram_addr_c = get_arg_val<uint32_t>(0);

       for (uint32_t tile = 0; tile < num_tiles; ++tile) {
           cb_wait_front(cb_out0, 1);
           uint32_t l1_read_addr = get_read_ptr(cb_out0);
           noc_async_write_tile(tile, l1_read_addr, dram_addr_c);
           noc_async_write_barrier();
           cb_pop_front(cb_out0, 1);
       }
   }
   ```

**Host Program** (`main.cpp`)
- Device initialization
- Circular buffer configuration
- Kernel creation (DataMovement for reader/writer, Compute for compute)
- Runtime argument setup
- Program execution

**Execution Plan** (`tt.plan.json`)
- Grid topology (8x8)
- Per-core tile assignments
- Schedule metadata

---

## 2. Key Design Decisions

### Persistent Loop Model

**GPU Approach:** Launch 64 threadblocks (8x8 grid), each processes one tile
**TT Approach:** Launch 64 cores, each iterates over assigned tiles

**Rationale:** TT cores are persistent - they stay resident and iterate over work. This enables better data reuse and reduces launch overhead.

### Tile-Level vs Intra-Tile Parallelism

TileLang supports two levels of parallelism that map differently on TT hardware:

#### 1. Tile-Level Parallelism (blockIdx → Persistent Cores)

**TileLang Syntax:**
```python
with T.Kernel(8, 8) as (bx, by):  # 8×8 grid of tiles
    C[bx*32:(bx+1)*32, by*32:(by+1)*32] = ...
```

**TT Mapping:**
- `blockIdx.x/y/z` variables → Persistent core loops
- Each core processes multiple tiles sequentially
- Handled by `GridToPersistentTT` pass

**User Perspective:** Users don't care how tile operations are internally implemented - they specify tile-level operations using built-in intrinsics (T.copy, T.gemm, etc.), and the backend handles the details.

#### 2. Intra-Tile Parallelism (threadIdx → SFPU/SIMD)

**TileLang Syntax:**
```python
with T.Kernel(4, 4) as (bx, by):
    for i, j in T.Parallel(32, 32):  # SIMD within each 32×32 tile
        C[bx*32 + i, by*32 + j] = A[bx*32 + i, by*32 + j] + B[bx*32 + i, by*32 + j]
```

**TT Mapping (Planned):**
- `threadIdx.x/y/z` variables → SFPU (SIMD Floating Point Unit) operations
- Element-wise ops within a tile execute as SIMD
- Will be handled by `LowerToSFPU` pass (not yet implemented)

**User Perspective:** Users specify SIMD parallelism at the threadIdx level using T.Parallel(), enabling fine-grained element-wise operations within tiles. The backend will map these to SFPU operations.

**Current Status:**
- ✅ Tile-level parallelism (blockIdx) → Fully implemented
- 🔴 Intra-tile parallelism (threadIdx) → `LowerToSFPU` pass errors out (placeholder)

**Architectural Separation:**
- **GridToPersistentTT**: Handles only blockIdx → persistent loop (tile-level)
- **LowerToSFPU**: Will handle threadIdx → SFPU ops (intra-tile, future)

### 3-Kernel Architecture

**Why 3 Kernels?**
- **Reader:** Specialized for NOC DRAM→L1 transfers (RISC-V processor)
- **Compute:** Specialized for Tensix tile math (FPU/SFPU units)
- **Writer:** Specialized for NOC L1→DRAM transfers (RISC-V processor)

**Benefits:**
- Overlapped execution (reader loads next tile while compute works)
- Hardware specialization (different processor types)
- Circular buffer communication (producer-consumer pattern)

### Circular Buffer Memory Management

**L1 Memory Organization:**
```
Core L1 Memory (1MB)
├── cb_in0 (input A tiles): 2 pages × 2KB
├── cb_in1 (input B tiles): 2 pages × 2KB
└── cb_out0 (output C tiles): 2 pages × 2KB
```

**Double Buffering:** 2 pages per CB enables overlap - reader fills page 1 while compute uses page 0.

### Tile Size: 32×32

**Why 32×32?**
- Matches Tensix hardware tile dimensions
- FP16: 32×32 = 1024 elements × 2 bytes = 2KB per tile
- Efficient for matrix operations (matmul_tiles operates on 32×32 tiles)

### DST Register Double Buffering

**Overview:**
The **Destination (DST) registers** in Tensix cores are shared between the FPU (math engine) and the Packer. To enable pipelining, DST supports double buffering with an explicit handshake protocol:

**DST Lifecycle:**
- `acquire_dst()` - FPU reserves half of DST for computation
- `commit_dst()` - FPU signals computation complete
- `wait_for_tile()` - Packer waits for FPU (called internally by pack_tile)
- `release_dst()` - FPU releases DST back to packer

**Pattern 1: Element-Wise Operations**
```cpp
for (uint32_t i = 0; i < num_tiles; ++i) {
    acquire_dst();              // Acquire per tile
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    add_tiles(cb_a, cb_b, 0, 0, 0);
    cb_reserve_back(cb_c, 1);
    commit_dst();               // Commit per tile
    pack_tile(0, cb_c);
    cb_push_back(cb_c, 1);
    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
    release_dst();              // Release per tile
}
```

**Pattern 2: K-Loop Matrix Operations**
```cpp
for (uint32_t tile_idx = 0; tile_idx < num_output_tiles; ++tile_idx) {
    acquire_dst();              // Acquire BEFORE K-loop
    matmul_tiles_init(cb_c);

    for (uint32_t k = 0; k < Kt; ++k) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        matmul_tiles(cb_a, cb_b, 0, 0, 0, false);
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
    }

    cb_reserve_back(cb_c, 1);
    commit_dst();               // Commit AFTER K-loop
    pack_tile(0, cb_c);
    cb_push_back(cb_c, 1);
    release_dst();              // Release after K-loop
}
```

**Key Difference:**
- Element-wise: acquire/commit/release **per tile** (no accumulation)
- K-loop: acquire/commit/release **per output tile** (accumulation across K dimension)

### IR-Driven Codegen (Not Templates)

**Approach:** Visitor pattern walks TIR AST and emits code based on IR structure
**Benefits:**
- Composable IR transformations
- Pattern detection in transform passes (not codegen)
- Easier to maintain and extend

---

## 3. Transform Pass vs Codegen Responsibilities

### Transform Passes (Smart)

**What They Do:**
- Pattern detection (matmul, element-wise ops)
- IR annotation (AttrStmt with metadata)
- Memory layout inference
- Loop transformations
- Buffer optimizations

**What They DON'T Do:**
- Code generation
- Text emission
- Target-specific intrinsic lowering

### Codegen (Dumb)

**What It Does:**
- Read annotations from IR
- Emit target-specific code (C++ with Metalium APIs)
- Lower intrinsics to final form

**What It SHOULD NOT Do:**
- Pattern detection via heuristics
- IR analysis for optimization decisions
- Guessing what intrinsics to emit

**Key Principle:** Transform passes are smart (detect patterns), codegen is dumb (emit based on annotations).

---

## 4. Current Status & Gaps

### ✅ Complete (v5 Pipeline - 120 tests passing, 21 skipped)

**v5 Pipeline (14 Passes):**
- ✅ Stage A: Metadata (infer_tt_layout_v5, propagate_tt_layout_v5, attach_tensor_accessor_tt)
- ✅ Stage B: Partitioning (layout_aware_work_partition_tt_v5, grid_to_core_grid_v5)
- ✅ Stage C: Protocol-less Lowering (lower_shared_to_cb_v5, lower_tt_tile_intrinsics_v5, build_tile_dfg_tt)
- ✅ Stage D: Late Split & Protocol (split_device_kernel, configure_tensor_accessor_tt, lower_cb_intrinsics, insert_compute_init_tt, insert_dst_management_tt)
- ✅ Stage E: Finalization (finalize_persistent_signature_tt)

**Python-Only Architecture:**
- ✅ All 14 passes implemented in Python
- ✅ Old 5-pass pipeline removed (PR #135)
- ✅ No C++ migration planned - Python provides maintainability and rapid iteration

**IR Pipeline:**
- ✅ Target registration
- ✅ Default annotations (schedule, shard)
- ✅ Layout-aware metadata (buffer residency, ND sharding)
- ✅ Transform pipeline (14 v5 + 11 shared + 1 verification = 26 passes)
- ✅ Verification (TT IR constraints)

**Codegen:**
- ✅ IR-driven visitor infrastructure
- ✅ 3-kernel architecture (reader/compute/writer)
- ✅ Host program generation with TensorAccessor metadata
- ✅ Conditional compilation (real/mock modes)
- ✅ DST lifecycle (acquire→compute→commit→pack→release)
- ✅ Per-core runtime metadata tables

**SDK Integration:**
- ✅ External SDK approach (like CUDA/ROCm)
- ✅ CMake FindMetalium module
- ✅ Real vs Mock build modes
- ✅ CI workflows (mock + SDK validation)

**Test Suite:**
- ✅ 120 tests passing
- ✅ 21 tests skipped (TVM bugs, hardware-specific features)
- ✅ 85.1% pass rate

### 🎯 Next Steps

**SFPU Support (Future):**
- ⚠️ `LowerToSFPU` pass not yet implemented (Python implementation planned)
- ⚠️ T.Parallel (threadIdx) constructs not supported yet
- ✅ Tile-level parallelism (blockIdx) fully working

**SDK Validation:**
- ⚠️ Pending SDK access for hardware testing
- ⚠️ API validation needed (EnqueueWriteBuffer, SetRuntimeArgs)
- ⚠️ Performance tuning and optimization

---

## 5. Comparison: TT vs CUDA

| Aspect | CUDA (GPU) | Tenstorrent (TT) |
|--------|------------|------------------|
| **Execution Model** | Grid of threadblocks (ephemeral) | Grid of persistent cores |
| **Work Distribution** | 1 block = 1 tile | 1 core = N tiles (static assignment) |
| **Memory Hierarchy** | Global → Shared → Registers | DRAM → L1 CB → DST Registers |
| **Synchronization** | `__syncthreads()` | Circular buffer flow control |
| **Kernel Launch** | Host launches grid | Host configures cores, cores iterate |
| **Pattern Detection** | Transform pass (`InferFragment`) | Transform pass (`LowerGemmToTTIntrinsics`) |
| **Codegen Split** | Host/Device (SplitHostDevice) | 3 kernels (Reader/Compute/Writer) |

---

## 6. Example: 256×256 Matmul

### TileLang Source

```python
@T.prim_func
def matmul_256x256(
    A: T.Buffer((256, 256), "float16"),
    B: T.Buffer((256, 256), "float16"),
    C: T.Buffer((256, 256), "float16")
):
    with T.Kernel(8, 8) as (bx, by):  # 8×8 = 64 output tiles
        for kt in T.serial(8):  # K-loop (8 tiles)
            for i, j in T.Parallel(32, 32):
                C[bx*32+i, by*32+j] += A[bx*32+i, kt*32+j] * B[kt*32+i, by*32+j]
```

### After Transforms

**Persistent Loop:**
```python
core_id = get_core_id()
start_tile, count = get_tile_assignment(core_id)  # e.g., core 0: tiles 0-0

for tile_id in range(start_tile, start_tile + count):
    bx = tile_id // 8
    by = tile_id % 8

    for kt in T.serial(8):
        # Annotated with tt.matmul_k_loop
        C[bx*32:(bx+1)*32, by*32:(by+1)*32] += ...
```

### Generated Code

**Reader Kernel:**
```cpp
for (uint32_t out_tile = 0; out_tile < num_out_tiles; ++out_tile) {
    for (uint32_t kt = 0; kt < Kt; ++kt) {
        // Read A[out_m, kt]
        cb_reserve_back(cb_in0, 1);
        noc_async_read_tile(...);
        cb_push_back(cb_in0, 1);

        // Read B[kt, out_n]
        cb_reserve_back(cb_in1, 1);
        noc_async_read_tile(...);
        cb_push_back(cb_in1, 1);
    }
}
```

**Compute Kernel:**
```cpp
for (uint32_t i = 0; i < tt_count; ++i) {
    tile_regs_acquire();
    mm_init(cb_in0, cb_in1, cb_out0);

    for (uint32_t kt = 0; kt < Kt; ++kt) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        matmul_tiles(cb_in0, cb_in1, 0, 0, 0, kt > 0);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }

    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out0);
    tile_regs_release();
}
```

**Writer Kernel:**
```cpp
for (uint32_t out_tile = 0; out_tile < num_out_tiles; ++out_tile) {
    cb_wait_front(cb_out0, 1);
    noc_async_write_tile(...);
    cb_pop_front(cb_out0, 1);
}
```

---

## 7. File Organization

```
tilelang-tt/
├── tilelang/
│   ├── engine/tenstorrent/
│   │   └── lower.py                           # TT lowering pipeline (v5)
│   └── tenstorrent/
│       ├── __init__.py
│       ├── annotations.py                     # annotate_tt_layout / annotate_tt_schedule
│       ├── target.py                          # apply_tt_defaults()
│       ├── codegen/
│       │   ├── kernel_generators.py           # Reader/compute/writer codegen
│       │   ├── host_generator.py              # Host program generation
│       │   ├── runtime_plan.py                # tt.plan.json generation
│       │   ├── intrinsics.py                  # Intrinsic registry
│       │   └── visitors.py                    # C++ emission visitors
│       └── passes/
│           ├── __init__.py
│           ├── _common.py
│           ├── pipeline.py                    # v5 14-pass pipeline
│           ├── infer_tt_layout_v5.py          # Stage A1
│           ├── propagate_tt_layout_v5.py      # Stage A2
│           ├── attach_tensor_accessor_tt.py   # Stage A3
│           ├── layout_aware_work_partition_tt_v5.py  # Stage B1
│           ├── grid_to_core_grid_v5.py        # Stage B2
│           ├── lower_shared_to_cb_v5.py       # Stage C1
│           ├── lower_tt_tile_intrinsics_v5.py # Stage C2
│           ├── build_tile_dfg_tt.py           # Stage C3
│           ├── split_device_kernel.py         # Stage D1
│           ├── configure_tensor_accessor_tt.py # Stage D2
│           ├── lower_cb_intrinsics.py         # Stage D3
│           ├── insert_compute_init_tt.py      # Stage D4
│           ├── insert_dst_management_tt.py    # Stage D5
│           └── finalize_persistent_signature_tt.py  # Stage E1
├── src/
│   └── transform/tenstorrent/
│       └── verify_tt_ir.cc                    # TT constraint verification (C++)
└── testing/python/tenstorrent/
    ├── test_target_registration.py            # Target registration
    ├── test_v5_passes_integration.py          # v5 pipeline integration
    ├── test_codegen_pipeline.py               # Codegen integration
    ├── test_jit_decorator.py                  # JIT decorator
    └── test_no_templates.py                   # IR-driven codegen validation
```

---

## 8. Build Modes

### Mock Mode (Default)

**Purpose:** Development without hardware

```bash
cmake -B build -DUSE_LLVM=true
cmake --build build -j$(nproc)
```

**Features:**
- ✅ No hardware required
- ✅ Fast iteration
- ✅ Complete code generation
- ✅ 120 tests passing (21 skipped)

**Limitations:**
- ❌ Cannot execute on hardware
- ❌ Mock APIs (void functions)

### Real Mode (With SDK)

**Purpose:** Hardware execution

```bash
export TT_METAL_HOME=/path/to/tt-metal
cmake -B build -DUSE_LLVM=true -DUSE_REAL_METALIUM=ON
cmake --build build -j$(nproc)
```

**Features:**
- ✅ Real Metalium APIs
- ✅ Hardware execution ready
- ✅ Performance profiling

**Requirements:**
- ✅ TT-Metalium SDK installed
- ✅ Tenstorrent device (for execution)

See [METALIUM_SETUP_GUIDE.md](./METALIUM_SETUP_GUIDE.md) for SDK setup.

---

## 9. References

**Implementation Details:**
- [GPU_vs_Tenstorrent_Analysis.md](./GPU_vs_Tenstorrent_Analysis.md) - GPU vs TT architecture & compiler comparison
- [PASS_TABLE_SHARED.md](./PASS_TABLE_SHARED.md) - Shared lowering/optimization passes
- [PASS_TABLE_GPU.md](./PASS_TABLE_GPU.md) - CUDA/ROCm-specific passes
- [passes/README.md](../passes/README.md) - Tenstorrent pass index and reference
- [architecture/v5_pipeline.md](./v5_pipeline.md) - Authoritative v5 pipeline reference
- [planning/future-enhancements.md](../planning/future-enhancements.md) - Future enhancements and roadmap

**Setup & Usage:**
- [METALIUM_SETUP_GUIDE.md](./METALIUM_SETUP_GUIDE.md) - SDK installation
- [local_build_guide.md](./local_build_guide.md) - Local build instructions
- [CI.md](./CI.md) - Continuous integration

**Validation:**
- [planning/sdk-validation-plan.md](../planning/sdk-validation-plan.md) - SDK validation phases

---

## 10. Contributing

### Development Workflow

1. **Feature Development** (Mock Mode):
   ```bash
   bash maint/scripts/local_build_and_test_tt.sh
   ```

2. **Create Pull Request:**
   ```bash
   git checkout -b feature-name
   git commit -m "Description"
   git push -u origin feature-name
   gh pr create --repo davorchap/tilelang-tt --base main
   ```

3. **SDK Validation** (When Available):
   ```bash
   export TT_METAL_HOME=/path/to/tt-metal
   cmake -B build -DUSE_REAL_METALIUM=ON
   ```

### Testing

```bash
# All TT backend tests (120 passing, 21 skipped)
pytest testing/python/tenstorrent/ -v

# Quick summary
pytest testing/python/tenstorrent/ --tb=no -q

# Specific categories
pytest testing/python/tenstorrent/test_target_registration.py -v     # Target registration
pytest testing/python/tenstorrent/test_v5_passes_integration.py -v   # v5 pipeline integration
pytest testing/python/tenstorrent/test_codegen_pipeline.py -v        # Code generation
pytest testing/python/tenstorrent/test_jit_decorator.py -v           # JIT decorator
```

---

## Appendix A: Runtime Plan Specification

The `tt.plan.json` file is the single source of truth for coordinating host and device execution in the Tenstorrent backend. It contains all necessary information for launching kernels, configuring data movement, and managing core resources.

### File Format

The runtime plan is a JSON file with the following structure:

```json
{
  "core_grid": [gx, gy],
  "core_ranges": [...],
  "work_partition": {...},
  "layouts": {...}
}
```

### Field Descriptions

#### core_grid
**Type:** `[int, int]`

Specifies the dimensions of the core grid.

```json
"core_grid": [4, 4]  // 4×4 grid of cores (16 cores total)
```

#### core_ranges
**Type:** `Array<CoreRange>`

Defines the active rectangular regions of cores. Multiple disjoint ranges are supported.

```json
// Single range covering entire grid:
"core_ranges": [
  {"start": [0, 0], "extent": [4, 4]}
]

// Multiple disjoint ranges:
"core_ranges": [
  {"start": [0, 0], "extent": [2, 2]},
  {"start": [2, 2], "extent": [2, 2]}
]
```

#### work_partition
**Type:** `Map<string, Array<WorkItem>>`

Maps core coordinates to lists of work items. Keys are stringified coordinates `"(cx,cy)"`.

**WorkItem Structure:**
- `io`: M-dimension tile index
- `jo`: N-dimension tile index
- `len_k`: Optional K-dimension extent
- `tile_order`: Optional traversal order (`"row_major"`, `"column_major"`, `"match_shard"`, `"z_order"`)

```json
"work_partition": {
  "(0,0)": [
    {"io": 0, "jo": 0, "len_k": 128, "tile_order": "row_major"},
    {"io": 0, "jo": 1, "len_k": 128, "tile_order": "row_major"}
  ],
  "(0,1)": [
    {"io": 1, "jo": 0, "len_k": 128, "tile_order": "row_major"}
  ]
}
```

#### layouts
**Type:** `Map<string, LayoutDescriptor>`

Describes memory layout and sharding for each buffer.

```json
"layouts": {
  "A": {
    "shard": "DRAM",
    "interleave": true,
    "stride": [1024, 32]
  },
  "B": {
    "shard": "DRAM",
    "interleave": false
  },
  "C": {
    "shard": "L1",
    "tile_id_order": "row_major"
  }
}
```

### Complete Example

Runtime plan for a 128×128 matrix multiplication on a 2×2 core grid:

```json
{
  "core_grid": [2, 2],
  "core_ranges": [
    {"start": [0, 0], "extent": [2, 2]}
  ],
  "work_partition": {
    "(0,0)": [
      {"io": 0, "jo": 0, "len_k": 128, "tile_order": "row_major"},
      {"io": 0, "jo": 1, "len_k": 128, "tile_order": "row_major"}
    ],
    "(0,1)": [
      {"io": 0, "jo": 2, "len_k": 128, "tile_order": "row_major"},
      {"io": 0, "jo": 3, "len_k": 128, "tile_order": "row_major"}
    ],
    "(1,0)": [
      {"io": 1, "jo": 0, "len_k": 128, "tile_order": "row_major"},
      {"io": 1, "jo": 1, "len_k": 128, "tile_order": "row_major"}
    ],
    "(1,1)": [
      {"io": 1, "jo": 2, "len_k": 128, "tile_order": "row_major"},
      {"io": 1, "jo": 3, "len_k": 128, "tile_order": "row_major"}
    ]
  },
  "layouts": {
    "A": {"shard": "DRAM", "interleave": true},
    "B": {"shard": "DRAM", "interleave": true},
    "C": {"shard": "L1", "tile_id_order": "row_major"}
  }
}
```

### Python API

#### Creating a Plan
```python
from tilelang.tenstorrent import (
    CoreRange, WorkItem, plan_dict
)

# Define components
core_grid = (4, 4)
core_ranges = [CoreRange((0, 0), (4, 4))]
work_partition = {
    "(0,0)": [WorkItem(io=0, jo=0, len_k=128)]
}
layouts = {
    "A": {"shard": "DRAM"},
    "B": {"shard": "DRAM"},
    "C": {"shard": "L1"}
}

# Create plan dictionary
plan = plan_dict(core_grid, core_ranges, work_partition, layouts)
```

#### Emitting from IR
```python
from tilelang.tenstorrent import emit_tt_plan

# Emit plan from a PrimFunc with metadata
emit_tt_plan(func, out_path="my_plan.json")
```

#### Loading and Validation
```python
from tilelang.tenstorrent import load_tt_plan, validate_plan

# Load plan from file
plan = load_tt_plan("my_plan.json")

# Validate plan structure
errors = validate_plan(plan)
if errors:
    print("Plan validation failed:")
    for error in errors:
        print(f"  - {error}")
```

### Host Runtime Usage

The host runtime reads the plan to configure core activation, memory allocations, DMA engines, and kernel launches:

```cpp
// Pseudo-code for host runtime
TTRuntimePlan plan = LoadPlan("tt.plan.json");

// Activate cores based on core_ranges
for (auto& range : plan.core_ranges) {
    ActivateCores(range.start, range.extent);
}

// Allocate buffers based on layouts
for (auto& [name, layout] : plan.layouts) {
    if (layout.shard == "DRAM") {
        AllocateDRAM(name, layout);
    } else {
        AllocateL1(name, layout);
    }
}

// Launch kernels with work assignments
for (auto& [core, work_items] : plan.work_partition) {
    LaunchKernel(core, work_items);
}
```

### Device Kernel Usage

Device kernels use the plan to determine their work assignment:

```cpp
// Pseudo-code for device kernel
uint32_t core_id = GetCoreID();
WorkList my_work = GetWorkForCore(core_id);

for (auto& work_item : my_work) {
    ProcessTile(work_item.io, work_item.jo, work_item.len_k);
}
```

### Validation Rules

The runtime plan must satisfy:
1. **Core ranges within grid bounds**: All core ranges must fit within grid dimensions
2. **Non-overlapping ranges**: Core ranges should not overlap (unless intentionally replicated)
3. **Complete tile coverage**: All required output tiles must be assigned to cores
4. **Valid layout shards**: Shard values must be "DRAM" or "L1"
5. **Consistent buffer references**: All buffers in layouts must correspond to function parameters

### Performance Considerations

- **Work balance**: Distribute work items evenly across cores
- **Memory locality**: Place frequently accessed buffers in L1
- **Interleaving**: Enable for better bank utilization
- **Tile order**: Match computation pattern for cache efficiency

---

**Last Updated:** 2025-10-17
**Maintainer:** TileLang Tenstorrent Team
**Repository:** https://github.com/davorchap/tilelang-tt
