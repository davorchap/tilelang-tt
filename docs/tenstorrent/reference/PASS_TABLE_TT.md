# TileLang Tenstorrent Pass Reference (v5 Pipeline)

**Document Version:** 3.0
**Date:** 2025-10-16
**Status:** Active

## Overview

This document describes the v5 Tenstorrent-specific transformation pipeline. The v5 pipeline consists of **14 passes** organized into **5 stages (A-E)**, all implemented in Python.

Shared passes are documented in `PASS_TABLE_SHARED.md`, and GPU-only passes in `PASS_TABLE_GPU.md`.

```
             ┌──────────────────────────────┐
             │   Lower & Legalize (shared) │
             │   PASS_TABLE_SHARED.md      │
             └─────────────┬────────────────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
┌──────────────────────┐      ┌────────────────────────────┐
│ GPU / HIP Pipeline   │      │ Tenstorrent v5 Pipeline    │
│ (PASS_TABLE_GPU.md)  │      │ (PASS_TABLE_TT.md)         │
├──────────────────────┤      ├────────────────────────────┤
│ • InferFragment      │      │ 14 passes in stages A-E:   │
│ • Warp/thread passes │      │ • Metadata (A1-A3)         │
│ • SplitHostDevice    │      │ • Partitioning (B1-B2)     │
│ • CUDA/HIP codegen   │      │ • Protocol-less (C1-C3)    │
└──────────────────────┘      │ • Late Split (D1-D5)       │
                               │ • Finalization (E1)        │
                               └────────────────────────────┘
```

## v5 Design Principles

1. **Progressive Lowering**: Early metadata → Late protocol
2. **Protocol-less Mid-level**: No NOC/CB/DST until Stage D
3. **No Heuristics**: Pattern matching based on IR structure, not names
4. **Standard Metadata**: Consistent attribute schema throughout
5. **Python Implementation**: All passes in Python (no C++ migration planned)

---

## Stage A: Metadata (3 passes)

**Purpose:** Infer and propagate buffer metadata, attach tensor accessors

| Pass | Location | Purpose | Metadata Added |
|------|----------|---------|----------------|
| **A1: infer_tt_layout_v5** | `tilelang/tenstorrent/passes/infer_tt_layout_v5.py` | Canonicalize buffer layout schema, validate N-D sharding | `tt.buffer.<name>` (memory, layout, tile_shape, nd_shard) |
| **A2: propagate_tt_layout_v5** | `tilelang/tenstorrent/passes/propagate_tt_layout_v5.py` | Derive circular buffer metadata from layout | `tt.cb.<name>` (cb_id, page_size, depth, data_format) |
| **A3: attach_tensor_accessor_tt** | `tilelang/tenstorrent/passes/attach_tensor_accessor_tt.py` | Attach TensorAccessor metadata for buffer addressing | `tt.tensor_accessor.<name>` (addressing metadata) |

### A1: infer_tt_layout_v5

**Input:** TIR with user annotations (`annotate_tt_layout`)
**Output:** PrimFunc with `tt.buffer.<name>` attributes

**Key Operations:**
- Normalize user layout annotations
- Validate L1 shard constraints (tile-aligned, capacity checks)
- Reject unsupported features (halo hints)
- Emit buffer metadata: memory space, layout, dtype, tile shape, ND shard

**Example Transform:**
```python
# Before (user annotation)
A = T.Buffer((256, 256), "bf16")
annotate_tt_layout(A, memory="DRAM", layout="interleaved")

# After (metadata added)
# PrimFunc attrs contain:
"tt.buffer.A": {
  "memory": "DRAM",
  "layout": "interleaved",
  "tile_shape": [32, 32],
  "dtype": "bf16"
}
```

### A2: propagate_tt_layout_v5

**Input:** PrimFunc with `tt.buffer.*` attributes
**Output:** PrimFunc with `tt.cb.<name>` attributes

**Key Operations:**
- Read buffer metadata (memory, layout, tile_shape, dtype)
- Calculate CB parameters: page_size, depth, data_format
- Stamp `tt.cb.*` attributes for each DRAM↔L1 transfer

**Example Transform:**
```python
# Input metadata
"tt.buffer.A": {"tile_shape": [32, 32], "dtype": "bf16"}

# Output metadata
"tt.cb.A": {
  "cb_id": 0,
  "page_size": 2048,  # 32 × 32 × 2 bytes
  "depth": 2,          # double buffering
  "data_format": "bfloat16"
}
```

### A3: attach_tensor_accessor_tt

**Input:** PrimFunc with buffer and CB metadata
**Output:** PrimFunc with `tt.tensor_accessor.<name>` attributes

**Key Operations:**
- Create TensorAccessor args for each buffer
- Enable deterministic global index computation
- Support both DRAM and L1 buffer addressing
- Guard against default-constructed accessors

---

## Stage B: Partitioning (2 passes)

**Purpose:** Determine per-core work assignments, map to physical cores

| Pass | Location | Purpose | Metadata Added |
|------|----------|---------|----------------|
| **B1: layout_aware_work_partition_tt_v5** | `tilelang/tenstorrent/passes/layout_aware_work_partition_tt_v5.py` | Choose per-core work assignments based on buffer residency | `tt.partition_mode`, `tt.grid_tiles`, `tt.core_ranges`, `tt.runtime_args` |
| **B2: grid_to_core_grid_v5** | `tilelang/tenstorrent/passes/grid_to_core_grid_v5.py` | Map logical grid to physical core coordinates | Persistent loop transformation |

### B1: layout_aware_work_partition_tt_v5

**Input:** PrimFunc with buffer metadata
**Output:** PrimFunc with partition and runtime argument metadata

**Key Operations:**
- Analyze buffer residency (DRAM vs L1)
- Choose partition mode (`global` or `local_shard`)
- Calculate grid dimensions and tile assignments
- Emit canonical runtime arguments
- Generate CoreRangeSet for kernel launches

**Example Transform:**
```python
# Input: Buffer metadata shows DRAM interleaved

# Output: Partition metadata
"tt.partition_mode": "global"
"tt.grid_tiles": [8, 8]  # Mt, Nt
"tt.core_ranges": [[0,0], [7,7]]  # 8x8 grid
"tt.runtime_args": ["start_id", "count", "Mt", "Kt", "Nt"]
```

### B2: grid_to_core_grid_v5

**Input:** PrimFunc with partition metadata
**Output:** PrimFunc with persistent loop

**Key Operations:**
- Transform GPU-style grid kernel to persistent loop
- Map `blockIdx.x/y/z` to persistent core iteration
- Calculate per-core tile assignments

**Example Transform:**
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

---

## Stage C: Protocol-less Lowering (3 passes)

**Purpose:** Lower to abstract tile operations (no protocol yet)

| Pass | Location | Purpose | Transformation |
|------|----------|---------|----------------|
| **C1: lower_shared_to_cb_v5** | `tilelang/tenstorrent/passes/lower_shared_to_cb_v5.py` | Lower shared memory to circular buffers (abstract) | shared mem → CB allocations |
| **C2: lower_tt_tile_intrinsics_v5** | `tilelang/tenstorrent/passes/lower_tt_tile_intrinsics_v5.py` | Lower TT tile operations to intrinsics | tile ops → TT intrinsics |
| **C3: build_tile_dfg_tt** | `tilelang/tenstorrent/passes/build_tile_dfg_tt.py` | Build tile dataflow graph | dataflow metadata |

### C1: lower_shared_to_cb_v5

**Input:** TIR with shared memory allocations
**Output:** TIR with CB allocations (protocol-free)

**Key Operations:**
- Replace shared memory allocations with CB references
- Use `tt.cb.*` metadata from Stage A
- No NOC/CB protocol insertion yet (comes in Stage D)

**Example Transform:**
```python
# Before
shared_mem = T.alloc_buffer((32, 32), "float16", scope="shared")

# After
cb_in0 = T.alloc_buffer((2, 2048), "uint8", scope="cb")  # Abstract CB
```

### C2: lower_tt_tile_intrinsics_v5

**Input:** TIR with high-level tile operations
**Output:** TIR with TT tile intrinsics

**Key Operations:**
- Detect tile operation patterns (matmul, add, mul, etc.)
- Lower to TT tile intrinsics
- Preserve K-loop structure for accumulation
- Annotate operations for codegen

**Example Transform:**
```python
# Before
C[bx*32:(bx+1)*32, by*32:(by+1)*32] += A[...] * B[...]

# After
T.call_extern("matmul_tiles", cb_a, cb_b, 0, 0, 0, accumulate=True)
```

### C3: build_tile_dfg_tt

**Input:** TIR with tile intrinsics
**Output:** TIR with dataflow metadata

**Key Operations:**
- Analyze tile data dependencies
- Build dataflow graph (producer-consumer relationships)
- Identify optimization opportunities
- Emit dataflow metadata for Stage D

---

## Stage D: Late Split & Protocol (5 passes)

**Purpose:** Split kernels and insert NOC/CB/DST protocol

| Pass | Location | Purpose | Transformation |
|------|----------|---------|----------------|
| **D1: split_device_kernel** | `tilelang/tenstorrent/passes/split_device_kernel.py` | Split single kernel into 3 kernels | 1 kernel → reader/compute/writer |
| **D2: configure_tensor_accessor_tt** | `tilelang/tenstorrent/passes/configure_tensor_accessor_tt.py` | Configure TensorAccessor per kernel | per-kernel addressing |
| **D3: lower_cb_intrinsics** | `tilelang/tenstorrent/passes/lower_cb_intrinsics.py` | Insert NOC/CB API calls | abstract CB → NOC/CB protocol |
| **D4: insert_compute_init_tt** | `tilelang/tenstorrent/passes/insert_compute_init_tt.py` | Insert compute initialization | acquire_dst, mm_init |
| **D5: insert_dst_management_tt** | `tilelang/tenstorrent/passes/insert_dst_management_tt.py` | Insert DST lifecycle | commit/pack/release |

### D1: split_device_kernel

**Input:** Single monolithic device kernel
**Output:** Three separate kernels (reader, compute, writer)

**Key Operations:**
- Analyze kernel structure (loads, computes, stores)
- Split into reader/compute/writer
- Maintain CB-based communication
- Preserve operation ordering

**Example Transform:**
```python
# Before (monolithic kernel)
def kernel():
    # Load data
    tile_a = load_from_dram(A)
    tile_b = load_from_dram(B)
    # Compute
    tile_c = matmul(tile_a, tile_b)
    # Store result
    store_to_dram(C, tile_c)

# After (3 kernels)
def reader():
    tile_a = load_from_dram(A)
    tile_b = load_from_dram(B)
    push_to_cb(cb_in0, tile_a)
    push_to_cb(cb_in1, tile_b)

def compute():
    tile_a = pop_from_cb(cb_in0)
    tile_b = pop_from_cb(cb_in1)
    tile_c = matmul(tile_a, tile_b)
    push_to_cb(cb_out0, tile_c)

def writer():
    tile_c = pop_from_cb(cb_out0)
    store_to_dram(C, tile_c)
```

### D3: lower_cb_intrinsics

**Input:** Kernels with abstract CB operations
**Output:** Kernels with concrete NOC/CB API calls

**Key Operations:**
- **Reader**: Insert `cb_reserve_back`, `noc_async_read_tile`, `noc_async_read_barrier`, `cb_push_back`
- **Compute**: Insert `cb_wait_front`, `cb_pop_front`, `cb_reserve_back`, `cb_push_back`
- **Writer**: Insert `cb_wait_front`, `noc_async_write_tile`, `noc_async_write_barrier`, `cb_pop_front`

**Example Transform (Reader):**
```python
# Before
push_to_cb(cb_in0, tile_a)

# After
cb_reserve_back(cb_in0, 1)
uint32_t l1_write_addr = get_write_ptr(cb_in0)
noc_async_read_tile(tile, dram_addr_a, l1_write_addr)
noc_async_read_barrier()
cb_push_back(cb_in0, 1)
```

### D4-D5: insert_compute_init_tt + insert_dst_management_tt

**Input:** Compute kernel with tile intrinsics
**Output:** Compute kernel with complete DST lifecycle

**Example Transform (Matmul K-loop):**
```python
# After D4 (insert_compute_init_tt)
for tile_idx in range(num_output_tiles):
    acquire_dst()                          # ← D4
    mm_init(cb_in0, cb_in1, cb_out0)     # ← D4

    for k in range(Kt):
        matmul_tiles(cb_in0, cb_in1, 0, 0, 0, k > 0)

# After D5 (insert_dst_management_tt)
for tile_idx in range(num_output_tiles):
    acquire_dst()
    mm_init(cb_in0, cb_in1, cb_out0)

    for k in range(Kt):
        cb_wait_front(cb_in0, 1)           # ← D3
        cb_wait_front(cb_in1, 1)           # ← D3
        matmul_tiles(cb_in0, cb_in1, 0, 0, 0, k > 0)
        cb_pop_front(cb_in0, 1)            # ← D3
        cb_pop_front(cb_in1, 1)            # ← D3

    cb_reserve_back(cb_out0, 1)            # ← D3
    commit_dst()                            # ← D5
    pack_tile(0, cb_out0)                  # ← D5
    cb_push_back(cb_out0, 1)               # ← D3
    release_dst()                           # ← D5
```

---

## Stage E: Finalization (1 pass)

**Purpose:** Finalize runtime signature and metadata

| Pass | Location | Purpose | Validation |
|------|----------|---------|------------|
| **E1: finalize_persistent_signature_tt** | `tilelang/tenstorrent/passes/finalize_persistent_signature_tt.py` | Finalize runtime args and validate metadata | Completeness checks, guardrails |

### E1: finalize_persistent_signature_tt

**Input:** Complete kernel IR with all protocol
**Output:** Final IR ready for codegen

**Key Operations:**
- Finalize runtime argument signature
- Validate metadata completeness:
  - All buffers have TensorAccessor metadata
  - All CBs properly configured
  - Core ranges defined
  - Runtime args schema complete
- Emit final metadata for host codegen
- Verify guardrails

---

## Shared Optimization Passes (11 passes)

Applied after Stage E, shared with GPU target:

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

See [PASS_TABLE_SHARED.md](./PASS_TABLE_SHARED.md) for details.

---

## Verification Pass

| Pass | Location | Purpose |
|------|----------|---------|
| **verify_tt_ir** | `tilelang/tenstorrent/passes/verify_tt_ir.py` | Verify TT constraints (grid size, CB counts) |

---

## Pipeline Dependencies

```
Apply TT Defaults
  ↓
Lower & Legalize (shared passes)
  ↓
┌─────────────────────────────────────────┐
│ Stage A: Metadata (3 passes)            │
│  A1: infer_tt_layout_v5                 │
│  A2: propagate_tt_layout_v5             │
│  A3: attach_tensor_accessor_tt          │
└───────────────────┬─────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ Stage B: Partitioning (2 passes)        │
│  B1: layout_aware_work_partition_tt_v5  │
│  B2: grid_to_core_grid_v5               │
└───────────────────┬─────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ Stage C: Protocol-less (3 passes)       │
│  C1: lower_shared_to_cb_v5              │
│  C2: lower_tt_tile_intrinsics_v5        │
│  C3: build_tile_dfg_tt                  │
└───────────────────┬─────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ Stage D: Late Split & Protocol (5)      │
│  D1: split_device_kernel                │
│  D2: configure_tensor_accessor_tt       │
│  D3: lower_cb_intrinsics                │
│  D4: insert_compute_init_tt             │
│  D5: insert_dst_management_tt           │
└───────────────────┬─────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ Stage E: Finalization (1 pass)          │
│  E1: finalize_persistent_signature_tt   │
└───────────────────┬─────────────────────┘
                    ↓
Common optimizations (11 shared passes)
  ↓
Verification (verify_tt_ir)
  ↓
Codegen (5 artifacts)
```

---

## Codegen (IR-Driven)

| Component | Location | Output | Purpose |
|-----------|----------|--------|---------|
| **TTReaderCodegenVisitor** | `src/target/tenstorrent/codegen_tt_reader_visitor.cc` | `reader.cpp` | DRAM → L1 transfers |
| **TTComputeCodegenVisitor** | `src/target/tenstorrent/codegen_tt_compute_visitor.cc` | `compute.cpp` | Tile computations |
| **TTWriterCodegenVisitor** | `src/target/tenstorrent/codegen_tt_writer_visitor.cc` | `writer.cpp` | L1 → DRAM transfers |
| **TTHostCodegenVisitor** | `src/target/tenstorrent/codegen_tt_host_visitor.cc` | `main.cpp` | Host program |
| **TTPlanCodegenVisitor** | `src/target/tenstorrent/codegen_tt_plan_visitor.cc` | `tt.plan.json` | Execution plan |

**Output:** 5 TT artifacts (reader, compute, writer, host, plan)

**Note:** TT codegen is **IR-driven** - visitors walk the annotated TIR and emit code based on metadata attributes.

---

## Key Files

**Python Pipeline:**
- `tilelang/engine/tenstorrent/lower.py` - Orchestration (`OptimizeForTargetTT`)
- `tilelang/tenstorrent/passes/` - All 14 pass implementations
- `tilelang/tenstorrent/attrs.py` - Metadata dataclasses

**Codegen:**
- `src/target/tenstorrent/codegen_tt*.cc` - IR-driven visitors

**Tests:**
- `testing/python/tenstorrent/test_v5_passes_integration.py` - Integration tests
- `testing/python/tenstorrent/test_codegen_pipeline.py` - Full pipeline
- **120 tests passing, 21 skipped** (85.1% pass rate)

---

## References

- [v5_pipeline.md](../architecture/v5_pipeline.md) - Authoritative v5 pipeline reference (800+ lines)
- [TT_ARCHITECTURE.md](../architecture/TT_ARCHITECTURE.md) - Complete backend architecture
- [passes/README.md](../passes/README.md) - Pass documentation index
- [PASS_TABLE_SHARED.md](./PASS_TABLE_SHARED.md) - Shared optimization passes
- [PASS_TABLE_GPU.md](./PASS_TABLE_GPU.md) - GPU-specific passes

---

## Changelog

### Version 3.0 (2025-10-16)
- Complete rewrite for v5 pipeline
- 14 passes organized in stages A-E
- All Python implementation
- Removed old 5-pass pipeline references
- Added comprehensive examples for each stage

### Version 2.0 (2025-10-14)
- Original 5-pass pipeline documentation

---

**Maintainer:** TileLang Tenstorrent Team
**Last Updated:** 2025-10-16
