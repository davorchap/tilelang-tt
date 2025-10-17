# v5 Pipeline Reference

**Version:** 1.0
**Date:** 2025-10-16
**Status:** Production

## Overview

The v5 pipeline is the authoritative TileLang Tenstorrent backend compilation pipeline. It consists of **14 passes** organized into **5 stages (A-E)**, all implemented in Python for maintainability and rapid iteration.

**Key Design Principles:**
1. **Progressive Lowering**: Early metadata → Late protocol
2. **Protocol-less Mid-level**: No NOC/CB/DST until Stage D
3. **No Heuristics**: Pattern matching based on IR structure, not names
4. **Standard Metadata**: Consistent attribute schema throughout
5. **Python Implementation**: All passes in Python (no C++ migration planned)

---

## Pipeline Stages

```
Stage A: Metadata (3 passes)
  ↓ Infer and propagate buffer metadata, attach tensor accessors
Stage B: Partitioning (2 passes)
  ↓ Determine per-core work assignments, map to physical cores
Stage C: Protocol-less Lowering (3 passes)
  ↓ Lower to abstract tile operations (no protocol yet)
Stage D: Late Split & Protocol (5 passes)
  ↓ Split kernels, insert NOC/CB/DST protocol
Stage E: Finalization (1 pass)
  ↓ Finalize runtime signature and metadata
```

---

## Stage A: Metadata (3 passes)

### A1: `infer_tt_layout_v5`

**Purpose:** Canonicalize buffer layout schema and validate constraints

**Input:** TIR with user annotations (`annotate_tt_layout`)

**Output:** PrimFunc with `tt.buffer.<name>` attributes

**Key Operations:**
- Normalize user layout annotations
- Validate L1 shard constraints (tile-aligned, capacity checks)
- Reject unsupported features (halo hints)
- Emit buffer metadata: memory space, layout, dtype, tile shape, ND shard

**Buffer Attribute Schema:**
```json
"tt.buffer.A": {
  "memory": "DRAM" | "L1",
  "layout": "interleaved" | "sharded",
  "tile_shape": [32, 32],
  "dtype": "bf16" | "fp16" | "fp32",
  "nd_shard": {
    "axes": ["B", "H", "M", "N"],
    "grid": [gB, gH, gM, gN],
    "shard_shape_elems": [sB, sH, sM, sN],
    "order": "row_major" | "match_shard" | "block_linear",
    "align_tiles": true,
    "projected_grid": [Gy, Gx],
    "projected_shard_tiles": [Sm, Sn]
  }
}
```

**File:** `tilelang/tenstorrent/passes/infer_tt_layout_v5.py`

---

### A2: `propagate_tt_layout_v5`

**Purpose:** Derive circular buffer geometry from buffer metadata

**Input:** PrimFunc with `tt.buffer.*` attributes

**Output:** PrimFunc with `tt.cb.<name>` attributes

**Key Operations:**
- Read buffer metadata (memory, layout, tile_shape, dtype)
- Calculate circular buffer parameters:
  - `page_size` = tile_height × tile_width × dtype_size (e.g., 32 × 32 × 2 = 2048 for bf16)
  - `depth` = 2 (double buffering)
  - `data_format` = derived from dtype
- Stamp `tt.cb.*` attributes for each DRAM↔L1 transfer

**CB Attribute Schema:**
```json
"tt.cb.A": {
  "cb_id": 0,
  "page_size": 2048,
  "depth": 2,
  "data_format": "bfloat16"
}
```

**File:** `tilelang/tenstorrent/passes/propagate_tt_layout_v5.py`

---

### A3: `attach_tensor_accessor_tt`

**Purpose:** Attach TensorAccessor metadata for buffer addressing

**Input:** PrimFunc with buffer and CB metadata

**Output:** PrimFunc with `tt.tensor_accessor.<name>` attributes

**Key Operations:**
- Create TensorAccessor args for each buffer
- Enable deterministic global index computation in device kernels
- Support both DRAM and L1 buffer addressing
- Guard against default-constructed accessors

**TensorAccessor Metadata:**
- Base tile ranges
- Shard geometry (for sharded buffers)
- Stride information
- Address computation parameters

**File:** `tilelang/tenstorrent/passes/attach_tensor_accessor_tt.py`

---

## Stage B: Partitioning (2 passes)

### B1: `layout_aware_work_partition_tt_v5`

**Purpose:** Determine per-core work assignments based on buffer residency

**Input:** PrimFunc with buffer metadata

**Output:** PrimFunc with partition and runtime argument metadata

**Key Operations:**
- Analyze buffer residency (DRAM vs L1)
- Choose partition mode:
  - `global`: DRAM interleaved/sharded buffers
  - `local_shard`: L1 sharded buffers (opt-in)
- Calculate grid dimensions and tile assignments
- Emit canonical runtime arguments
- Generate CoreRangeSet for kernel launches

**PrimFunc Attributes Emitted:**
```json
{
  "tt.partition_mode": "global" | "local_shard",
  "tt.grid_tiles": [Mt, Nt],
  "tt.shard_grid": [Gy, Gx],
  "tt.local_shape_tiles": [Sm, Sn],
  "tt.core_ranges": [[y0, x0], [y1, x1], ...],
  "tt.runtime_args": ["start_id", "count", "Mt", "Kt", "Nt", "Sm", "Sn", "Gy", "Gx", "sy", "sx"]
}
```

**Runtime Argument Schema:**

| Mode | Arguments | Description |
|------|-----------|-------------|
| Global | `start_id`, `count`, `Mt`, `Kt`, `Nt` | Tile range and global dimensions |
| Local Shard | Above + `Sm`, `Sn`, `Gy`, `Gx`, `sy`, `sx` | Plus shard geometry and coordinates |

**File:** `tilelang/tenstorrent/passes/layout_aware_work_partition_tt_v5.py`

---

### B2: `grid_to_core_grid_v5`

**Purpose:** Map logical grid coordinates to physical core coordinates

**Input:** PrimFunc with partition metadata

**Output:** PrimFunc with physical core mapping

**Key Operations:**
- Transform GPU-style grid kernel to persistent loop
- Map `blockIdx.x/y/z` to persistent core iteration
- Calculate per-core tile assignments
- Generate core ID → tile range mapping

**Transformation Example:**
```python
# Before (GPU-style grid)
with T.Kernel(8, 8) as (bx, by):
    C[bx*32:(bx+1)*32, by*32:(by+1)*32] = ...

# After (persistent loop)
core_id = get_core_id()
start_tile, count = get_tile_assignment(core_id)
for tile_id in range(start_tile, start_tile + count):
    bx = tile_id // 8
    by = tile_id % 8
    C[bx*32:(bx+1)*32, by*32:(by+1)*32] = ...
```

**File:** `tilelang/tenstorrent/passes/grid_to_core_grid_v5.py`

---

## Stage C: Protocol-less Lowering (3 passes)

### C1: `lower_shared_to_cb_v5`

**Purpose:** Lower shared memory allocations to circular buffers (abstract, no protocol)

**Input:** TIR with shared memory allocations

**Output:** TIR with CB allocations (protocol-free)

**Key Operations:**
- Replace shared memory allocations with CB references
- Use `tt.cb.*` metadata from Stage A
- No NOC/CB protocol insertion yet (comes in Stage D)
- Maintain abstract tile-level operations

**Transformation:**
```python
# Before
shared_mem = T.alloc_buffer((32, 32), "float16", scope="shared")

# After
cb_in0 = T.alloc_buffer((2, 2048), "uint8", scope="cb")  # Abstract CB
```

**File:** `tilelang/tenstorrent/passes/lower_shared_to_cb_v5.py`

---

### C2: `lower_tt_tile_intrinsics_v5`

**Purpose:** Lower TT tile operations (matmul, elementwise) to intrinsics

**Input:** TIR with high-level tile operations

**Output:** TIR with TT tile intrinsics

**Key Operations:**
- Detect tile operation patterns (matmul, add, mul, etc.)
- Lower to TT tile intrinsics:
  - `matmul_tiles()` for matrix multiplication
  - `add_tiles()`, `mul_tiles()` for elementwise ops
- Preserve K-loop structure for accumulation
- Annotate operations for codegen

**Intrinsic Lowering:**
```python
# Before
C[bx*32:(bx+1)*32, by*32:(by+1)*32] += A[...] * B[...]

# After
T.call_extern("matmul_tiles", cb_a, cb_b, 0, 0, 0, accumulate=True)
```

**Supported Operations:**
- `matmul_tiles` - Matrix multiplication
- `add_tiles` - Element-wise addition
- `mul_tiles` - Element-wise multiplication
- `sub_tiles` - Element-wise subtraction
- More ops as needed

**File:** `tilelang/tenstorrent/passes/lower_tt_tile_intrinsics_v5.py`

---

### C3: `build_tile_dfg_tt`

**Purpose:** Build tile dataflow graph for optimization

**Input:** TIR with tile intrinsics

**Output:** TIR with dataflow metadata

**Key Operations:**
- Analyze tile data dependencies
- Build dataflow graph (producer-consumer relationships)
- Identify opportunities for optimization:
  - Tile reuse
  - CB allocation sharing
  - Overlapped execution
- Emit dataflow metadata for later passes

**Dataflow Analysis:**
- Track tile lifetimes
- Identify CB reuse opportunities
- Detect parallel execution opportunities
- Prepare for kernel splitting (Stage D)

**File:** `tilelang/tenstorrent/passes/build_tile_dfg_tt.py`

---

## Stage D: Late Split & Protocol (5 passes)

### D1: `split_device_kernel`

**Purpose:** Split single kernel into reader/compute/writer kernels

**Input:** Single monolithic device kernel

**Output:** Three separate kernels (reader, compute, writer)

**Key Operations:**
- Analyze kernel structure (loads, computes, stores)
- Split into three kernels:
  - **Reader**: DRAM → L1 transfers (NOC)
  - **Compute**: Tile computations (Tensix)
  - **Writer**: L1 → DRAM transfers (NOC)
- Maintain CB-based communication between kernels
- Preserve operation ordering and dependencies

**Kernel Roles:**

| Kernel | Processor | Operations | CBs |
|--------|-----------|------------|-----|
| Reader | RISC-V (Data Movement) | `noc_async_read_tile` | Produces `cb_in0`, `cb_in1` |
| Compute | Tensix (Compute) | `matmul_tiles`, `add_tiles` | Consumes `cb_in*`, produces `cb_out0` |
| Writer | RISC-V (Data Movement) | `noc_async_write_tile` | Consumes `cb_out0` |

**File:** `tilelang/tenstorrent/passes/split_device_kernel.py`

---

### D2: `configure_tensor_accessor_tt`

**Purpose:** Configure TensorAccessor for each kernel

**Input:** Three kernels (reader, compute, writer)

**Output:** Kernels with TensorAccessor configuration

**Key Operations:**
- Attach TensorAccessor metadata to each kernel
- Configure addressing for each buffer:
  - Reader: DRAM source addresses
  - Writer: DRAM destination addresses
  - Compute: Uses runtime args for index computation
- Enable deterministic tile addressing
- Support both interleaved and sharded buffers

**TensorAccessor Configuration:**
- Reader: Source buffer TensorAccessors
- Compute: Runtime args for (m, n) → tile_id mapping
- Writer: Destination buffer TensorAccessors

**File:** `tilelang/tenstorrent/passes/configure_tensor_accessor_tt.py`

---

### D3: `lower_cb_intrinsics`

**Purpose:** Lower abstract CB operations to NOC/CB API calls

**Input:** Kernels with abstract CB operations

**Output:** Kernels with concrete NOC/CB API calls

**Key Operations:**
- **Reader Kernel:**
  - Insert `cb_reserve_back()` before write
  - Insert `noc_async_read_tile()` for DRAM → L1
  - Insert `noc_async_read_barrier()` for completion
  - Insert `cb_push_back()` after write

- **Compute Kernel:**
  - Insert `cb_wait_front()` before read
  - Keep tile intrinsics (`matmul_tiles`, etc.)
  - Insert `cb_pop_front()` after read
  - Insert `cb_reserve_back()` before output write
  - Insert `cb_push_back()` after output write

- **Writer Kernel:**
  - Insert `cb_wait_front()` before read
  - Insert `noc_async_write_tile()` for L1 → DRAM
  - Insert `noc_async_write_barrier()` for completion
  - Insert `cb_pop_front()` after read

**API Calls Inserted:**

| API Call | Purpose | Kernel |
|----------|---------|--------|
| `cb_reserve_back(cb, n)` | Reserve output pages | Reader, Compute |
| `cb_push_back(cb, n)` | Publish output pages | Reader, Compute |
| `cb_wait_front(cb, n)` | Wait for input pages | Compute, Writer |
| `cb_pop_front(cb, n)` | Release input pages | Compute, Writer |
| `noc_async_read_tile(...)` | DRAM → L1 transfer | Reader |
| `noc_async_write_tile(...)` | L1 → DRAM transfer | Writer |
| `noc_async_read_barrier()` | Wait for read completion | Reader |
| `noc_async_write_barrier()` | Wait for write completion | Writer |

**File:** `tilelang/tenstorrent/passes/lower_cb_intrinsics.py`

---

### D4: `insert_compute_init_tt`

**Purpose:** Insert compute initialization (DST acquire, mm_init)

**Input:** Compute kernel with tile intrinsics

**Output:** Compute kernel with initialization code

**Key Operations:**
- Determine operation type (matmul vs elementwise)
- Insert appropriate initialization:
  - **Matmul**: `acquire_dst()` + `mm_init(cb_in0, cb_in1, cb_out0)`
  - **Elementwise**: `acquire_dst()` per tile
- Position initialization relative to K-loop structure
- Prepare for DST lifecycle management (D5)

**Initialization Patterns:**

**Pattern 1: Matmul (K-loop)**
```cpp
for (uint32_t tile_idx = 0; tile_idx < num_output_tiles; ++tile_idx) {
    acquire_dst();                          // ← Inserted by D4
    mm_init(cb_in0, cb_in1, cb_out0);     // ← Inserted by D4

    for (uint32_t k = 0; k < Kt; ++k) {
        matmul_tiles(cb_in0, cb_in1, 0, 0, 0, k > 0);
    }
    // D5 will insert commit/pack/release here
}
```

**Pattern 2: Elementwise (per-tile)**
```cpp
for (uint32_t i = 0; i < num_tiles; ++i) {
    acquire_dst();                          // ← Inserted by D4
    add_tiles(cb_a, cb_b, 0, 0, 0);
    // D5 will insert commit/pack/release here
}
```

**File:** `tilelang/tenstorrent/passes/insert_compute_init_tt.py`

---

### D5: `insert_dst_management_tt`

**Purpose:** Insert DST lifecycle (acquire→commit→pack→release)

**Input:** Compute kernel with initialization

**Output:** Compute kernel with complete DST lifecycle

**Key Operations:**
- Insert DST commit after computation
- Insert pack_tile for CB output
- Insert DST release after packing
- Handle different operation patterns:
  - **K-loop**: acquire before, commit/pack/release after loop
  - **Per-tile**: acquire/commit/pack/release per tile

**Complete DST Lifecycle:**

**Matmul (K-loop accumulation):**
```cpp
for (uint32_t tile_idx = 0; tile_idx < num_output_tiles; ++tile_idx) {
    acquire_dst();                          // D4: Acquire
    mm_init(cb_in0, cb_in1, cb_out0);     // D4: Init

    for (uint32_t k = 0; k < Kt; ++k) {
        cb_wait_front(cb_in0, 1);          // D3: CB protocol
        cb_wait_front(cb_in1, 1);          // D3: CB protocol
        matmul_tiles(cb_in0, cb_in1, 0, 0, 0, k > 0);  // C2: Intrinsic
        cb_pop_front(cb_in0, 1);           // D3: CB protocol
        cb_pop_front(cb_in1, 1);           // D3: CB protocol
    }

    cb_reserve_back(cb_out0, 1);           // D3: CB protocol
    commit_dst();                           // D5: Commit ←
    pack_tile(0, cb_out0);                 // D5: Pack ←
    cb_push_back(cb_out0, 1);              // D3: CB protocol
    release_dst();                          // D5: Release ←
}
```

**Elementwise (per-tile):**
```cpp
for (uint32_t i = 0; i < num_tiles; ++i) {
    acquire_dst();                          // D4: Acquire
    cb_wait_front(cb_a, 1);                // D3: CB protocol
    cb_wait_front(cb_b, 1);                // D3: CB protocol
    add_tiles(cb_a, cb_b, 0, 0, 0);        // C2: Intrinsic
    cb_reserve_back(cb_c, 1);              // D3: CB protocol
    commit_dst();                           // D5: Commit ←
    pack_tile(0, cb_c);                    // D5: Pack ←
    cb_push_back(cb_c, 1);                 // D3: CB protocol
    cb_pop_front(cb_a, 1);                 // D3: CB protocol
    cb_pop_front(cb_b, 1);                 // D3: CB protocol
    release_dst();                          // D5: Release ←
}
```

**DST Lifecycle APIs:**
- `acquire_dst()` - FPU reserves DST half for computation
- `commit_dst()` - FPU signals computation complete
- `pack_tile(dst_index, cb)` - Packer writes tile to CB (internally calls `wait_for_tile()`)
- `release_dst()` - FPU releases DST back to packer

**File:** `tilelang/tenstorrent/passes/insert_dst_management_tt.py`

---

## Stage E: Finalization (1 pass)

### E1: `finalize_persistent_signature_tt`

**Purpose:** Finalize runtime signature and metadata

**Input:** Complete kernel IR with all protocol

**Output:** Final IR ready for codegen

**Key Operations:**
- Finalize runtime argument signature
- Validate metadata completeness:
  - All buffers have TensorAccessor metadata
  - All CBs properly configured
  - Core ranges defined
  - Runtime args schema complete
- Emit final metadata for host codegen:
  - Per-core runtime argument tables
  - CB configuration tables
  - Kernel launch parameters
- Verify guardrails:
  - No default-constructed TensorAccessors
  - All shard metadata present for local_shard mode
  - CB IDs unique and valid

**Final Metadata Validation:**
```python
# Required for each kernel
assert "tt.partition_mode" in func.attrs
assert "tt.grid_tiles" in func.attrs
assert "tt.core_ranges" in func.attrs
assert "tt.runtime_args" in func.attrs

# Per-buffer validation
for buf in buffers:
    assert f"tt.buffer.{buf.name}" in func.attrs
    assert f"tt.cb.{buf.name}" in func.attrs
    assert f"tt.tensor_accessor.{buf.name}" in func.attrs
```

**Output for Codegen:**
- Three complete kernel IRs (reader, compute, writer)
- Host metadata (runtime args, CB configs, core ranges)
- Execution plan metadata (grid topology, tile assignments)

**File:** `tilelang/tenstorrent/passes/finalize_persistent_signature_tt.py`

---

## Pass Dependencies

```
Stage A: Metadata
├─ A1: infer_tt_layout_v5
│  └─ Provides: tt.buffer.*
├─ A2: propagate_tt_layout_v5
│  ├─ Requires: tt.buffer.*
│  └─ Provides: tt.cb.*
└─ A3: attach_tensor_accessor_tt
   ├─ Requires: tt.buffer.*, tt.cb.*
   └─ Provides: tt.tensor_accessor.*

Stage B: Partitioning
├─ B1: layout_aware_work_partition_tt_v5
│  ├─ Requires: tt.buffer.*
│  └─ Provides: tt.partition_mode, tt.grid_tiles, tt.core_ranges, tt.runtime_args
└─ B2: grid_to_core_grid_v5
   ├─ Requires: tt.partition_mode, tt.grid_tiles
   └─ Transforms: GPU grid → persistent loop

Stage C: Protocol-less Lowering
├─ C1: lower_shared_to_cb_v5
│  ├─ Requires: tt.cb.*
│  └─ Transforms: shared mem → CB allocations
├─ C2: lower_tt_tile_intrinsics_v5
│  └─ Transforms: tile ops → TT intrinsics
└─ C3: build_tile_dfg_tt
   ├─ Requires: tile intrinsics
   └─ Provides: dataflow metadata

Stage D: Late Split & Protocol
├─ D1: split_device_kernel
│  ├─ Requires: dataflow metadata
│  └─ Splits: 1 kernel → 3 kernels
├─ D2: configure_tensor_accessor_tt
│  ├─ Requires: tt.tensor_accessor.*, split kernels
│  └─ Configures: per-kernel TensorAccessors
├─ D3: lower_cb_intrinsics
│  ├─ Requires: split kernels, tt.cb.*
│  └─ Inserts: NOC/CB API calls
├─ D4: insert_compute_init_tt
│  ├─ Requires: compute kernel with intrinsics
│  └─ Inserts: acquire_dst, mm_init
└─ D5: insert_dst_management_tt
   ├─ Requires: compute kernel with init
   └─ Inserts: commit_dst, pack_tile, release_dst

Stage E: Finalization
└─ E1: finalize_persistent_signature_tt
   ├─ Requires: all previous metadata
   └─ Validates: metadata completeness, emits final IR
```

---

## Metadata Flow

### Buffer Metadata (Stage A)

```
User Annotations (annotate_tt_layout)
    ↓ A1: infer_tt_layout_v5
tt.buffer.* attributes (memory, layout, tile_shape, nd_shard)
    ↓ A2: propagate_tt_layout_v5
tt.cb.* attributes (cb_id, page_size, depth, data_format)
    ↓ A3: attach_tensor_accessor_tt
tt.tensor_accessor.* attributes (addressing metadata)
```

### Partition Metadata (Stage B)

```
tt.buffer.* (memory residency)
    ↓ B1: layout_aware_work_partition_tt_v5
tt.partition_mode, tt.grid_tiles, tt.core_ranges, tt.runtime_args
    ↓ B2: grid_to_core_grid_v5
Persistent loop with per-core tile assignments
```

### Protocol Insertion (Stages C-D)

```
Abstract tile operations (Stage C)
    ↓ C1-C3: Lower to intrinsics, build DFG
Intrinsics with dataflow metadata
    ↓ D1: split_device_kernel
Three kernels (reader, compute, writer)
    ↓ D2-D5: Configure and insert protocol
Complete kernels with NOC/CB/DST protocol
```

### Finalization (Stage E)

```
Complete kernel IR + metadata
    ↓ E1: finalize_persistent_signature_tt
Validated IR ready for codegen
    ↓ Codegen
reader.cpp, compute.cpp, writer.cpp, main.cpp, tt.plan.json
```

---

## Code Generation

After Stage E, the IR is passed to the codegen visitors:

### Codegen Visitors

1. **`TTReaderCodegenVisitor`** (`src/target/tenstorrent/codegen_tt_reader_visitor.cc`)
   - Generates `reader.cpp`
   - Emits NOC read operations
   - Produces circular buffers (`cb_in0`, `cb_in1`)

2. **`TTComputeCodegenVisitor`** (`src/target/tenstorrent/codegen_tt_compute_visitor.cc`)
   - Generates `compute.cpp`
   - Emits tile intrinsics (`matmul_tiles`, `add_tiles`, etc.)
   - Includes DST lifecycle (acquire, commit, pack, release)

3. **`TTWriterCodegenVisitor`** (`src/target/tenstorrent/codegen_tt_writer_visitor.cc`)
   - Generates `writer.cpp`
   - Emits NOC write operations
   - Consumes circular buffer (`cb_out0`)

4. **`TTHostCodegenVisitor`** (`src/target/tenstorrent/codegen_tt_host_visitor.cc`)
   - Generates `main.cpp`
   - Device initialization
   - CB configuration
   - Kernel creation and launch
   - Runtime argument setup
   - TensorAccessor materialization

5. **`TTPlanCodegenVisitor`** (`src/target/tenstorrent/codegen_tt_plan_visitor.cc`)
   - Generates `tt.plan.json`
   - Grid topology
   - Per-core tile assignments
   - Schedule metadata

### Generated Files

```
build/
└── kernels/
    ├── reader.cpp          # DRAM → L1 transfers
    ├── compute.cpp         # Tile computations
    ├── writer.cpp          # L1 → DRAM transfers
    ├── main.cpp            # Host program
    └── tt.plan.json        # Execution plan
```

---

## Testing

### Integration Tests

**File:** `testing/python/tenstorrent/test_v5_passes_integration.py`

Tests for each stage:
- `test_stage_a_metadata` - A1-A3 passes
- `test_stage_b_partitioning` - B1-B2 passes
- `test_stage_c_lowering` - C1-C3 passes
- `test_stage_d_protocol` - D1-D5 passes
- `test_stage_e_finalization` - E1 pass

### End-to-End Tests

**Files:**
- `testing/python/tenstorrent/test_codegen_pipeline.py` - Full pipeline
- `testing/python/tenstorrent/test_jit_decorator.py` - JIT compilation
- `testing/python/tenstorrent/test_examples_run.py` - Example workloads

### Test Coverage

- **120 tests passing** (85.1% pass rate)
- **21 tests skipped** (TVM bugs, hardware-specific features)

---

## Future Work

### Planned Enhancements

1. **LowerToSFPU Pass** (Python implementation)
   - Support for T.Parallel (threadIdx) constructs
   - SFPU (SIMD Floating Point Unit) operations
   - Element-wise operations within tiles
   - See: `docs/tenstorrent/passes/lower_to_sfpu.md`

2. **Optimization Passes**
   - CB allocation sharing
   - Tile reuse optimization
   - Overlapped execution optimization

3. **Advanced Sharding**
   - Block linear order support
   - Halo exchange support (currently rejected)
   - Dynamic sharding configurations

### Hardware Validation

- SDK-backed hardware testing (pending device access)
- Performance profiling and tuning
- API validation with real Metalium SDK

---

## References

### Architecture Documents
- [TT_ARCHITECTURE.md](TT_ARCHITECTURE.md) - Complete backend architecture
- [GPU_vs_Tenstorrent_Analysis.md](GPU_vs_Tenstorrent_Analysis.md) - GPU vs TT comparison
- [TileLang_TT_TIR_Lowering_Guide_v5.md](TileLang_TT_TIR_Lowering_Guide_v5.md) - v5 lowering guide

### Pass Documentation
- [passes/README.md](../passes/README.md) - Pass documentation index
- Individual pass docs in `docs/tenstorrent/passes/`

### Planning and Status
- [planning/TT_Pass_Status.md](../planning/TT_Pass_Status.md) - v5 implementation status
- [planning/TT_BACKEND_TASKS.md](../planning/TT_BACKEND_TASKS.md) - Backend tasks

### Setup and Build
- [setup/local_build_guide.md](../setup/local_build_guide.md) - Local build instructions
- [setup/CI.md](../setup/CI.md) - CI/CD workflows

---

## Changelog

### Version 1.0 (2025-10-16)
- Initial v5 pipeline reference document
- Documented all 14 passes in stages A-E
- Added metadata flow diagrams
- Included codegen visitor reference
- Complete dependency graph

---

**Maintainer:** TileLang Tenstorrent Team
**Repository:** https://github.com/davorchap/tilelang-tt
**Last Updated:** 2025-10-16
