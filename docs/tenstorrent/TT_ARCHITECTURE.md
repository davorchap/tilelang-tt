# TileLang Tenstorrent Backend Architecture

**Version:** 2.0
**Date:** 2025-10-08
**Status:** Production

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

**Entry Point:** `tilelang/tt/target.py` → `apply_tt_defaults()`

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

**Purpose:** Ensure backward compatibility - GPU-style kernels run on TT with sensible defaults.

### Phase 2.5: Layout-Aware Metadata (Planned)

**Why add a new stage?** Default schedule/shard metadata does not capture user intent for DRAM vs L1 residency, ND sharding, or tile-order annotations. The layout-aware stage introduces explicit buffer- and function-level attributes that downstream passes and codegen can rely on.

**Passes (new):**
- `InferTTLayout` – Normalize user annotations (`annotate_tt_layout`) and emit `tt.buffer.<name>` dictionaries with memory space, layout, dtype, tile shape, and optional N‑D shard metadata. Validates L1 shards and rejects halo hints.
- `PropagateTTLayout` – Reads buffer metadata and stamps `tt.cb.<name>` attributes describing circular buffer geometry (`page_size`, `depth`, `data_format`) for each DRAM↔L1 copy.
- `LayoutAwareWorkPartitionTT` – Chooses per-core work assignments based on buffer residency. Emits `tt.partition_mode` (`global` vs `local_shard`), `tt.core_ranges`, `tt.grid_tiles`, `tt.shard_grid`, `tt.local_shape_tiles`, and canonical `tt.runtime_args`.

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

This metadata becomes the authoritative source for later stages, allowing legacy heuristics to be removed.

### Phase 3: TT-Specific Optimization

**Entry Point:** `tilelang/engine/tt/lower.py` → `OptimizeForTargetTT()`

**TT-Specific Transform Passes (updated set):**

1. **`InferTTLayout`** *(new)* – Canonicalize buffer layout schema, validate N‑D sharding, enforce L1 constraints.
2. **`PropagateTTLayout`** *(new)* – Derive circular buffer (`tt.cb.*`) metadata from layout information.
3. **`LayoutAwareWorkPartitionTT`** *(new)* – Select cores and runtime ranges based on residency (`global` vs `local_shard`).
4. **`grid_to_persistent_tt`** *(updated)* – Transform GPU grid to persistent loop using shard-aware `(m, n)` recovery.
   - Input: GPU-style grid kernel (`with T.Kernel(8, 8) as (bx, by)`)
   - Output: Persistent kernel with tile iteration
   ```python
   # Before
   with T.Kernel(8, 8) as (bx, by):
       C[bx*32:(bx+1)*32, by*32:(by+1)*32] = ...

   # After
   core_id = get_core_id()
   start_tile, count = get_tile_assignment(core_id)
   for tile_id in range(start_tile, start_tile + count):
       bx = tile_id // 8
       by = tile_id % 8
       C[bx*32:(bx+1)*32, by*32:(by+1)*32] = ...
   ```

5. **`memory_space_lower_tt`** - Lower DRAM to L1 circular buffers
   - Input: DRAM buffer allocations
   - Output: L1 circular buffer allocations
   ```python
   # Before
   A = T.Buffer((256, 256), "float16")  # DRAM

   # After
   cb_in0 = CircularBuffer(cb_id=0, num_pages=2, page_size=2048)  # L1
   ```

6. **`tile_pad_tt`** - Pad buffers to 32×32 tile boundaries
   - Input: Arbitrary buffer shapes
   - Output: Tile-aligned shapes (multiples of 32)

Legacy passes `infer_default_tt_schedule` and `tt_tiles_to_core_map` remain available for backward compatibility but will be phased out once the layout-aware stack ships.

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

**Total:** 6 TT-specific + 11 shared + 1 verification = 18 passes

---

## Layout Attribute Schema (Details)

### Buffer Attributes (`tt.buffer.<name>`)

| Field | Meaning | Notes |
|-------|---------|-------|
| `memory` | `"DRAM"` or `"L1"` | Drives buffer residency and host API choice. |
| `layout` | `"interleaved"` or `"sharded"` | Mirrors TT-Metalium layout enums. |
| `tile_shape` | `[height, width]` | Defaults to `[32, 32]`; configurable in future. |
| `dtype` | Element type | `bf16`, `fp16`, `fp32`, etc. |
| `nd_shard.axes` | Logical axis labels | User-defined names to keep intent clear. |
| `nd_shard.grid` | Cores per axis | Product equals total shards. |
| `nd_shard.shard_shape_elems` | Elements per shard per axis | Prior to tilization. |
| `nd_shard.order` | Traversal hint | `row_major`, `match_shard`, `block_linear(k)`. |
| `nd_shard.align_tiles` | Bool | Must be true for L1 shards in v1. |
| `nd_shard.projected_grid` | `[Gy, Gx]` | 2-D projection on compute plane (derived). |
| `nd_shard.projected_shard_tiles` | `[Sm, Sn]` | Tiles per shard over compute plane (derived). |

### PrimFunc Attributes

| Attribute | Source Pass | Consumer | Description |
|-----------|-------------|----------|-------------|
| `tt.partition_mode` | LayoutAwareWorkPartitionTT | GridToPersistentTT, host | `"global"` vs `"local_shard"`. |
| `tt.grid_tiles` | LayoutAwareWorkPartitionTT | GridToPersistentTT, kernels | `[Mt, Nt]` global tile counts. |
| `tt.shard_grid` | LayoutAwareWorkPartitionTT | Host + kernels | `[Gy, Gx]` shard projection dims. |
| `tt.local_shape_tiles` | LayoutAwareWorkPartitionTT | Kernels | `[Sm, Sn]` tiles within one shard. |
| `tt.core_ranges` | LayoutAwareWorkPartitionTT | Host | CoreRangeSet for launches. |
| `tt.runtime_args` | LayoutAwareWorkPartitionTT | Host + kernels | Ordered runtime arg names. |
| `tt.cb.<buffer>` | PropagateTTLayout | MemorySpaceLowerTT, kernels | Circular buffer geometry. |

### Diagnostics & Guardrails

- Halo metadata is rejected (*"halo unsupported"*).
- L1 shards must be tile-aligned and fit within capacity (*"L1 shard exceeds capacity"*).
- Default-constructed `TensorAccessorArgs()` for DRAM buffers triggers a guardrail failure.

---

## Host & Kernel Responsibilities (Layout-Aware)

### Buffer Creation APIs

| Residency | Layout | Host API | Notes |
|-----------|--------|----------|-------|
| DRAM | interleaved | `CreateBuffer(InterleavedBufferConfig{...}, BufferType::DRAM)` | Default configuration. |
| DRAM | sharded | `CreateBuffer(ShardedBufferConfig{...}, BufferType::DRAM)` | Accepts ND shard metadata. |
| L1 | sharded | `CreateBuffer(ShardedBufferConfig{...}, BufferType::L1)` | Opt-in; enforced alignments. |

### TensorAccessor Policy

- Compile-time blobs must be built via `TensorAccessorArgs(*buffer)` so layout metadata is captured.
- Runtime args supply base addresses plus tile geometry fields (global + shard).
- Local L1 shards (owned by the executing core) use CB pointers directly; no TA required.

### Runtime Argument Payload

| Mode | Arguments |
|------|-----------|
| Global | `start_id`, `count`, `Mt`, `Kt`, `Nt` |
| Local shard | Above + `Sm`, `Sn`, `Gy`, `Gx`, `sy`, `sx` |

### Tile Order Options

- `row_major` remains the default (global or shard-local).
- `match_shard` iterates shard-local tiles in-row-major order then follows shard assignment.
- `block_linear(k)` reserved for future `RasterizationTT` pass.

---

## Test Matrix (Planned)

| Scenario | Goal | Expected Verification |
|----------|------|-----------------------|
| DRAM interleaved | Preserve legacy behavior | `tt.partition_mode="global"`, TA compile-args present, default CB geometry. |
| DRAM sharded | Treat sharding as first-class | Host uses `ShardedBufferConfig`, runtime args remain global, tile IDs map via TensorAccessor. |
| L1 sharded | Enforce opt-in residency | `tt.partition_mode="local_shard"`, `tt.core_ranges` == shard grid, runtime args include shard offsets. |
| ND sharding projection | Validate axis → compute mapping | `[Gy,Gx]`, `[Sm,Sn]` derived correctly for mixed axes. |
| Negative halo | Diagnostics | Pass errors with *"halo unsupported"*. |
| L1 overflow | Diagnostics | Fails capacity check. |
| Guardrail | Prevent DRAM TA misuse | Unit test fails on default-constructed TensorAccessorArgs. |

### Phase 4: Code Generation (IR-Driven)

**Entry Point:** `src/target/tt/codegen_tt.cc` → `CodegenTT::Build()`

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
   > **Upcoming change:** Reader/Writer kernels will instantiate `TensorAccessor` objects using compile-time blobs provided via `TensorAccessorArgs(*buffer)` to handle DRAM interleaved vs sharded layouts transparently.

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

### ✅ Complete (95 tests passing)

**IR Pipeline:**
- ✅ Target registration
- ✅ Default annotations (schedule, shard)
- ✅ Metadata inference (tile assignments, DRAM layout)
- ✅ Transform pipeline (6 TT-specific + 11 shared passes)
- ✅ Verification (TT IR constraints)

**Codegen:**
- ✅ IR-driven visitor infrastructure
- ✅ 3-kernel architecture (reader/compute/writer)
- ✅ Host program generation
- ✅ Conditional compilation (real/mock modes)
- ✅ DST lifecycle (acquire→compute→commit→pack→release)

**SDK Integration:**
- ✅ External SDK approach (like CUDA/ROCm)
- ✅ CMake FindMetalium module
- ✅ Real vs Mock build modes
- ✅ CI workflows (mock + SDK validation)

### 🚧 Incomplete (Next Steps)

**Pattern Detection:**
- ❌ `tensorize_tt` pass only handles T.gemm() intrinsic calls
- ❌ Manual matmul loops not detected
- ❌ Element-wise operations not annotated
- ⚠️ Codegen uses heuristics (variable name "kt" → K-loop) instead of annotations

**Issue:** Generated K-loop has scaffolding but body still has raw array operations instead of Metalium intrinsics.

**Solution:** Extend `tensorize_tt.cc` to detect manual loop patterns and annotate IR. See [IR_LOWERING_TASKS.md](./IR_LOWERING_TASKS.md) for implementation plan.

**SDK Validation:**
- ⚠️ Pending SDK access for hardware testing
- ⚠️ API gaps may exist (EnqueueWriteBuffer, SetRuntimeArgs)
- ⚠️ Performance tuning needed

---

## 5. Comparison: TT vs CUDA

| Aspect | CUDA (GPU) | Tenstorrent (TT) |
|--------|------------|------------------|
| **Execution Model** | Grid of threadblocks (ephemeral) | Grid of persistent cores |
| **Work Distribution** | 1 block = 1 tile | 1 core = N tiles (static assignment) |
| **Memory Hierarchy** | Global → Shared → Registers | DRAM → L1 CB → DST Registers |
| **Synchronization** | `__syncthreads()` | Circular buffer flow control |
| **Kernel Launch** | Host launches grid | Host configures cores, cores iterate |
| **Pattern Detection** | Transform pass (`InferFragment`) | Transform pass (`tensorize_tt`) |
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
│   ├── engine/tt/
│   │   ├── __init__.py
│   │   ├── adapter.py          # Engine adapter (entry point)
│   │   └── lower.py            # TT lowering pipeline
│   └── tt/
│       ├── __init__.py
│       ├── target.py           # apply_tt_defaults()
│       └── passes.py           # Python bindings for TT passes
├── src/
│   ├── transform/tt/
│   │   ├── infer_tt_schedule.cc        # Compute tile assignments
│   │   ├── infer_tt_shard.cc           # DRAM layout descriptors
│   │   ├── grid_to_persistent_tt.cc    # Grid → persistent loop
│   │   ├── tt_tiles_to_core_map.cc     # Tile assignments → NOC coords
│   │   ├── memory_space_lower_tt.cc    # DRAM → L1 CB
│   │   ├── tile_pad_tt.cc              # Pad to 32×32
│   │   ├── tensorize_tt.cc             # Pattern detection (INCOMPLETE)
│   │   └── verify_tt_ir.cc             # TT constraint verification
│   └── target/tt/
│       ├── codegen_tt.cc                      # Main codegen entry
│       ├── codegen_tt_visitor_base.cc         # Base visitor
│       ├── codegen_tt_compute_visitor.cc      # Compute kernel
│       ├── codegen_tt_reader_visitor.cc       # Reader kernel
│       └── codegen_tt_writer_visitor.cc       # Writer kernel
└── testing/python/tt/
    ├── test_target_registration.py      # Target registration (8 tests)
    ├── test_passes.py                   # Metadata inference (7 tests)
    ├── test_grid_to_persistent_tt.py    # Persistent loop (12 tests)
    ├── test_tt_tiles_to_core_map.py     # NOC mapping (5 tests)
    ├── test_memory_space_lower_tt.py    # CB lowering (8 tests)
    ├── test_tile_pad_tt.py              # Tile padding (6 tests)
    ├── test_tensorize_tt.py             # Tensorization (8 tests)
    ├── test_verify_tt_ir.py             # Verification (8 tests)
    └── test_codegen_tt.py               # Code generation (33 tests)
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
- ✅ All 95 tests pass

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
- [IR_LOWERING_ANALYSIS.md](./IR_LOWERING_ANALYSIS.md) - GPU vs TT pipeline comparison
- [PASS_TABLE.md](./PASS_TABLE.md) - Comprehensive pass reference (60+ passes)
- [IR_LOWERING_TASKS.md](./IR_LOWERING_TASKS.md) - Pattern detection implementation tasks

**Setup & Usage:**
- [METALIUM_SETUP_GUIDE.md](./METALIUM_SETUP_GUIDE.md) - SDK installation
- [local_build_guide.md](./local_build_guide.md) - Local build instructions
- [CI.md](./CI.md) - Continuous integration

**Validation:**
- [METALIUM_SDK_VALIDATION_PLAN.md](./METALIUM_SDK_VALIDATION_PLAN.md) - SDK validation phases

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
# All TT backend tests
pytest testing/python/tt/ -v

# Specific categories
pytest testing/python/tt/test_passes.py -v           # Metadata inference
pytest testing/python/tt/test_grid_to_persistent_tt.py -v  # Persistent loop
pytest testing/python/tt/test_codegen_tt.py -v       # Code generation
```

---

**Last Updated:** 2025-10-08
**Maintainer:** TileLang Tenstorrent Team
**Repository:** https://github.com/davorchap/tilelang-tt
