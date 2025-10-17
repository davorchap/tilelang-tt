# GPU vs Tenstorrent Architecture Analysis

**Document Version:** 2.0
**Date:** 2025-10-17
**Status:** Production

## Overview

This document provides a comprehensive analysis comparing GPU (CUDA/ROCm) and Tenstorrent architectures, covering both high-level architectural differences and detailed compiler pipeline transformations. It combines the best content from historical documentation with current implementation details.

**Scope:**
- Execution model differences (transient vs persistent)
- Memory hierarchy and data movement patterns
- Compiler pipeline analysis (IR lowering and tensorization)
- Code generation approaches

**Related Documents:**
- [v5_pipeline.md](v5_pipeline.md) - Current TT backend pipeline reference
- [TT_ARCHITECTURE.md](TT_ARCHITECTURE.md) - Complete TT backend architecture

---

## Table of Contents

1. [Execution Models](#execution-models)
2. [Memory Hierarchy](#memory-hierarchy)
3. [Data Movement Patterns](#data-movement-patterns)
4. [Compute Operations](#compute-operations)
5. [Compiler Pipeline Analysis](#compiler-pipeline-analysis)
6. [Code Generation Strategy](#code-generation-strategy)
7. [Summary and Comparison Tables](#summary-and-comparison-tables)
8. [Implications for TileLang](#implications-for-tilelang)

---

## Execution Models

### GPU (CUDA/HIP): Massive Parallelism with Transient Kernels

**Grid-based model:**
```python
# GPU kernel launch
grid_dim = (Mt, Nt)  # Grid of thread blocks
block_dim = (32, 32)  # Threads per block

# Each block processes one output tile independently
@cuda.kernel
def matmul_gpu(A, B, C):
    bx, by = blockIdx.x, blockIdx.y  # Block coordinates
    tx, ty = threadIdx.x, threadIdx.y  # Thread coordinates

    # Each block computes C[by, bx] = A[by, :] @ B[:, bx]
    # Block exits after computing its tile
```

**Key characteristics:**
- **Transient execution**: Each block processes its assigned tile and exits
- **Hardware scheduling**: GPU hardware schedules blocks to SMs dynamically
- **No persistent state**: Blocks don't communicate or persist across kernel launches
- **Occupancy-driven**: Performance depends on maximizing SM occupancy

### Tenstorrent: Persistent Per-Core Loops

**Persistent kernel model:**
```cpp
// Tenstorrent persistent kernel
void MAIN() {
    // Runtime args provided by host
    uint32_t start_id = get_arg_val<uint32_t>(0);
    uint32_t count    = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Nt = get_arg_val<uint32_t>(3);

    // Core runs a persistent loop over assigned tiles
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t tile_id = start_id + i;
        uint32_t by = tile_id / Nt;  // Row index (row-major)
        uint32_t bx = tile_id % Nt;  // Column index

        // Process tile (by, bx)
        // Core remains active, processing multiple tiles
    }
}
```

**Key characteristics:**
- **Persistent execution**: Each Tensix core runs a long-lived kernel loop
- **Static partitioning**: Host assigns tile ranges to cores before launch
- **Per-core state**: Cores can maintain state across tile iterations
- **Explicit scheduling**: Developer controls tile-to-core mapping

---

## Memory Hierarchy

### GPU Memory Hierarchy

```
Global Memory (DRAM)   →  Gigabytes, ~500 GB/s, high latency
    ↕
L2 Cache               →  Megabytes, shared across SMs
    ↕
L1 Cache/Shared Memory →  Kilobytes per SM, low latency
    ↕
Registers              →  Thousands per thread, ultra-low latency
```

**Access patterns:**
- **Coalescing**: Threads in a warp access consecutive addresses
- **Shared memory**: Explicitly managed scratch space for block collaboration
- **Caching**: L1/L2 caches handle some locality automatically

### Tenstorrent Memory Hierarchy

```
DRAM                   →  External DDR, accessed via NOC
    ↕ (NOC async read/write)
L1 Circular Buffers    →  1 MB per Tensix core, explicit staging
    ↕ (cb_wait_front/cb_pop_front)
Destination Registers  →  Tile-sized (32×32 elements), compute accumulation
```

**Access patterns:**
- **Circular buffers (CBs)**: Explicit producer-consumer staging areas in L1
- **NOC operations**: Async reads/writes move tiles between DRAM and L1
- **Tile-oriented**: All operations work on 32×32 tiles (2KB for FP16)
- **No implicit caching**: Developer must explicitly manage L1 staging

**Example CB usage:**
```cpp
// Reader kernel: DRAM → L1
cb_reserve_back(cb_in0, 1);  // Reserve space for 1 tile
noc_async_read(dram_addr, get_write_ptr(cb_in0), TILE_SIZE);
noc_async_read_barrier();
cb_push_back(cb_in0, 1);  // Signal tile is ready

// Compute kernel: Consume from CB
cb_wait_front(cb_in0, 1);  // Wait for tile availability
// ... process tile ...
cb_pop_front(cb_in0, 1);  // Release tile
```

---

## Data Movement Patterns

### GPU: Implicit + Explicit Transfers

**Global → Shared:**
```cuda
__shared__ float tile_A[32][32];
tile_A[ty][tx] = A[by * 32 + ty, k * 32 + tx];  // Coalesced load
__syncthreads();  // Synchronize before use
```

**Characteristics:**
- **Implicit caching**: L1/L2 caches reduce DRAM traffic
- **Explicit shared memory**: Developer copies global→shared for reuse
- **Thread-level granularity**: Each thread loads individual elements

### Tenstorrent: Explicit NOC Operations

**DRAM → L1 (Reader kernel):**
```cpp
// Compute tile address in DRAM (row-major tiled layout)
uint32_t tile_idx = by * Nt + bx;
uint64_t dram_addr = base_addr + tile_idx * TILE_SIZE;

// Asynchronous NOC read
cb_reserve_back(cb_in, 1);
noc_async_read_tile(dram_addr, get_write_ptr(cb_in), TILE_SIZE);
noc_async_read_barrier();  // Ensure transfer completes
cb_push_back(cb_in, 1);  // Signal compute kernel
```

**L1 → DRAM (Writer kernel):**
```cpp
cb_wait_front(cb_out, 1);  // Wait for compute result
uint64_t dram_addr = base_addr + tile_idx * TILE_SIZE;
noc_async_write_tile(get_read_ptr(cb_out), dram_addr, TILE_SIZE);
noc_async_write_barrier();  // Ensure write completes
cb_pop_front(cb_out, 1);  // Release CB space
```

**Characteristics:**
- **Three-kernel pattern**: Separate reader, compute, writer kernels
- **Tile granularity**: Move entire 32×32 tiles (2KB for FP16)
- **Asynchronous NOC**: Network-on-chip handles inter-core/DRAM transfers
- **CB synchronization**: Producer-consumer protocol via circular buffers

---

## Compute Operations

### GPU: Tensor Cores + CUDA Cores

**Matrix multiplication (Tensor Cores):**
```cuda
// WMMA API (Warp Matrix Multiply-Accumulate)
wmma::fragment<...> a_frag, b_frag, c_frag;
wmma::load_matrix_sync(a_frag, tile_A, 32);
wmma::load_matrix_sync(b_frag, tile_B, 32);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // C += A @ B
wmma::store_matrix_sync(tile_C, c_frag, 32);
```

**Characteristics:**
- **Warp-level operations**: 32 threads cooperate on matrix fragments
- **Mixed precision**: FP16/BF16 inputs, FP32 accumulation
- **Implicit accumulation**: Tensor cores accumulate in registers

### Tenstorrent: Tile-Based Compute Intrinsics

**Matrix multiplication (matmul_tiles):**
```cpp
// Initialize matmul for CB indices
mm_init(cb_in0, cb_in1, cb_out);

// Acquire destination tile registers
tile_regs_acquire();

// K-loop: Accumulate partial products
for (uint32_t k = 0; k < Kt; ++k) {
    cb_wait_front(cb_in0, 1);  // Wait for A tile
    cb_wait_front(cb_in1, 1);  // Wait for B tile

    matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);  // C[0] += A[0] @ B[0]

    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);
}

// Pack result from DST to CB
tile_regs_commit();
tile_regs_wait();
cb_reserve_back(cb_out, 1);
pack_tile(0, cb_out);  // DST[0] → CB[out]
cb_push_back(cb_out, 1);
tile_regs_release();
```

**Characteristics:**
- **Tile-level operations**: `matmul_tiles`, `add_tiles`, `mul_tiles`, etc.
- **Destination registers (DST)**: Tile-sized accumulators for matmul
- **Explicit lifecycle**: Acquire→Compute→Commit→Pack→Release
- **Tilize/Untilize**: Convert between row-major and tile layout

**Other tile operations:**
```cpp
tilize(cb_in, cb_out, tile_idx);        // Row-major → Tile layout
untilize(cb_in, cb_out, tile_idx);      // Tile → Row-major layout
add_tiles(cb_a, cb_b, tile_a, tile_b, dst_idx);
mul_tiles(cb_a, cb_b, tile_a, tile_b, dst_idx);
exp_tile(cb_in, tile_in, dst_idx);      // Elementwise exp
```

---

## Compiler Pipeline Analysis

This section analyzes how TileLang lowers IR for GPU vs Tenstorrent targets, with focus on **where pattern detection and tensorization occur**.

### Key Architectural Decision

**Question:** Should matmul pattern detection and intrinsic emission happen in transform passes or codegen?

**Answer:** Transform passes ✅

**Rationale:** GPU backends (CUDA) use transform passes for pattern detection and annotation, keeping codegen "dumb" (just emit based on annotations). This separation of concerns makes the compiler more maintainable and allows IR optimizations to work on annotated patterns.

### GPU (CUDA/ROCm) Lowering Pipeline

#### Phase 1: Frontend Lowering (Shared)

**Location:** `tilelang/engine/phase.py` → `LowerAndLegalize()`

This phase is **backend-agnostic** and shared between CUDA and TT. Key passes include:
- **LayoutInference**: Infers memory layouts for fragments (TensorCore tiles) and shared memory
- **LowerTileOp**: Lowers high-level tile operations to loops and buffer operations
- **LayoutReducer**: Configures layouts for reduction operations

#### Phase 2: Target-Specific Optimization

**Location:** `tilelang/engine/phase.py` → `OptimizeForTarget()`

CUDA-specific passes:
- **InferFragment**: Detects TensorCore operations (wmma, mma) and annotates fragment metadata
- **WarpSpecialized**: Enables async warp specialization for Hopper
- **RewriteWgmmaSync**: Rewrites wgmma operations for Hopper architecture
- **SplitHostDevice**: Splits IR into host and device functions

#### Phase 3: TensorCore Pattern Detection

**How GPU Detects TensorCore Operations:**

```python
# GPU approach: Look for intrinsic calls in the IR
class FragmentGetter(tvm.tir.stmt_functor.StmtExprVisitor):
    def __init__(self):
        super().__init__()
        self.fragments = {}

    def visit_call(self, op):
        # Detect TensorCore load/store intrinsics
        if op.op.name in ["tvm_load_matrix_sync", "tvm_store_matrix_sync"]:
            # Extract shape: m, n, k from call args
            m, n, k = self._extract_shape(op.args)
            # Get memory scope (wmma.matrix_a, wmma.matrix_b, wmma.accumulator)
            scope = self._get_memory_scope(op.args)
            layout = self._get_layout(op.args)
            # Store fragment metadata
            buffer_var = op.args[0]
            self.fragments[buffer_var] = FragmentInfo(m, n, k, layout, scope)
```

The transform pass annotates these with metadata (`fragment_shape`, `fragment_layout`) that codegen later uses to emit PTX assembly.

#### Phase 4: Device Codegen

**Location:** `tilelang/engine/lower.py` → `device_codegen()`

- **LowerIntrin**: Lowers TVM intrinsics to target-specific code (PTX asm for CUDA)
- Codegen reads fragment annotations and emits PTX assembly

### Tenstorrent Lowering Pipeline (v5)

For detailed information about the current v5 pipeline, see [v5_pipeline.md](v5_pipeline.md). This section provides a high-level comparison.

#### Phase 1: Apply TT Defaults

**Location:** `tilelang/engine/tenstorrent/lower.py` → `apply_tt_defaults()`

Apply default TT annotations for schedule and layout.

#### Phase 2: Frontend Lowering

Calls the same `LowerAndLegalize()` as CUDA (shared passes).

#### Phase 3: TT-Specific Optimizations

**Location:** `tilelang/engine/tenstorrent/lower.py` → `OptimizeForTargetTT()`

TT-specific passes (see [v5_pipeline.md](v5_pipeline.md) for complete details):
- **Stage A**: Metadata passes (InferTTLayout, PropagateTTLayout, AttachTensorAccessor)
- **Stage B**: Partitioning (LayoutAwareWorkPartition, GridToCoreGrid)
- **Stage C**: Protocol-less lowering (LowerSharedToCB, LowerTTTileIntrinsics, BuildTileDFG)
- **Stage D**: Late split & protocol (SplitDeviceKernel, ConfigureTensorAccessor, LowerCBIntrinsics, InsertComputeInit, InsertDSTManagement)
- **Stage E**: Finalization (FinalizePersistentSignature)

#### Phase 4: LowerGemmToTTIntrinsics Pass

**Current Implementation:**
- Consumes frontend-issued `tl.gemm` intrinsics (mirroring CUDA pipeline)
- Resolves CB IDs from metadata
- Expands each `tl.gemm` into TT intrinsic sequence
- No pattern matching heuristics - purely IR-driven

#### Phase 5: Device Splitting

**Key Difference:** Unlike CUDA which splits IR into host/device functions, TT keeps IR intact and splits during codegen into three kernels (reader/compute/writer).

#### Phase 6: Codegen

IR-driven visitors generate three separate kernels:
- **Reader**: DRAM → L1 transfers
- **Compute**: Tile computations
- **Writer**: L1 → DRAM transfers

---

## Code Generation Strategy

### GPU Codegen Approach

```
Annotated IR with fragment metadata
    ↓
Read annotations
    ↓
Emit PTX/AMDGPU assembly based on annotations
```

**Example:**
```cpp
if (call->op == "tvm_mma_sync") {
  // Read fragment_shape attribute from annotated buffers
  auto shape_attr = GetAttr(buffer_d, "fragment_shape");
  auto [m, n, k] = ParseShape(shape_attr);

  // Emit PTX assembly for TensorCore
  stream << "mma.sync.aligned.m" << m << "n" << n << "k" << k
         << ".row.col.f16.f16.f16.f16 "
         << "{" << d_regs << "}, "
         << "{" << a_regs << "}, "
         << "{" << b_regs << "}, "
         << "{" << c_regs << "};\n";
}
```

### Tenstorrent Codegen Approach

```
IR with tt.* intrinsics and metadata
    ↓
Three-visitor split during codegen
    ↓
Generate reader.cpp, compute.cpp, writer.cpp
```

**Example:**
```cpp
void TTComputeCodegenVisitor::VisitStmt_(const EvaluateNode* op) {
  // Directly emit TT intrinsics from IR
  if (op->value.as<CallNode>()->op == "tt.matmul_tiles") {
    EmitLine("matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);");
  }
}
```

---

## Summary and Comparison Tables

### Execution Model Comparison

| Aspect | GPU (CUDA/HIP) | Tenstorrent |
|--------|----------------|-------------|
| **Model** | Transient blocks | Persistent per-core loops |
| **Scheduling** | Hardware-scheduled dynamically | Static partitioning by host |
| **State** | No persistent state between launches | Per-core state across iterations |
| **Parallelism** | Massive (thousands of blocks) | Moderate (64-120 Tensix cores) |

### Memory Hierarchy Comparison

| Aspect | GPU (CUDA/HIP) | Tenstorrent |
|--------|----------------|-------------|
| **Caching** | L1/L2 caches (implicit) | No implicit caching |
| **Scratch** | Shared memory (explicit) | L1 circular buffers (explicit) |
| **Staging** | Thread-level loads | Tile-level NOC operations |
| **Granularity** | Individual elements | 32×32 tiles |

### Compiler Pipeline Comparison

| Aspect | GPU (CUDA) | Tenstorrent |
|--------|------------|-------------|
| **Pattern Detection** | Transform pass (`InferFragment`) | Transform pass (`LowerTTTileIntrinsics`) |
| **Annotation Method** | AttrStmt metadata on buffers | Evaluate nodes with `tt.*` intrinsics |
| **Intrinsic Insertion** | Frontend lowering | Transform passes (Stage C/D) |
| **Codegen Role** | Emit from annotations | Emit from intrinsic-bearing IR |
| **Device Splitting** | `SplitHostDevice` in IR | 3-visitor split during codegen |
| **Pipeline Stages** | 3 phases | 5 stages (A-E) with 14 passes |

### Programming Model Comparison

| Aspect | GPU (CUDA/HIP) | Tenstorrent |
|--------|----------------|-------------|
| **Model** | SIMT (Single Instruction, Multiple Threads) | SPMD + Dataflow |
| **Tile Size** | Flexible (warp-level fragments) | Fixed 32×32 tiles |
| **Compute** | Tensor Cores (WMMA) | Tile intrinsics (`matmul_tiles`) |
| **Synchronization** | `__syncthreads()`, barriers | CB producer-consumer protocol |
| **Data Movement** | Implicit caching + `__shared__` | Explicit NOC + CBs |

---

## Implications for TileLang

### What Works Naturally
- Grid-style kernels with block indices (`bx`, `by`)
- Matrix multiplication and convolution patterns
- Tile-aligned shapes (multiples of 32)

### What Requires Adaptation
- **Persistent loop wrapper**: Grid blocks → per-core tile iteration
- **Circular buffer management**: Allocate CBs for `T.copy` operations
- **Tile padding**: Non-tile-aligned shapes need padding + masking
- **Explicit scheduling**: Specify tile-to-core mapping (vs GPU's dynamic scheduling)
- **Three-kernel pattern**: Split reader/compute/writer for pipelined execution

### Developer Experience

TileLang provides a GPU-like surface that lowers to Tenstorrent's model:

```python
import tilelang.language as T
import tilelang.tenstorrent as tt

@T.prim_func
def matmul(A: T.Buffer, B: T.Buffer, C: T.Buffer):
    with T.Kernel(T.ceildiv(M, 32), T.ceildiv(N, 32)) as (bx, by):
        # GPU-style grid kernel (user code stays familiar)
        # ...

# Apply TT backend
mod = tvm.IRModule.from_expr(matmul)
mod = tt.apply_tt_defaults(mod)         # Inject TT annotations
mod = tt.apply_tt_metadata_passes(mod)  # Lower to persistent + CBs
artifacts = tt.emit_tt_artifacts(mod)   # Generate reader/compute/writer
```

The backend handles the transformation to Tenstorrent's execution model transparently.

---

## References

### Tenstorrent Documentation
- [v5_pipeline.md](v5_pipeline.md) - Current v5 pipeline reference (14 passes, 5 stages)
- [TT_ARCHITECTURE.md](TT_ARCHITECTURE.md) - Complete backend architecture with runtime plan spec
- [passes/README.md](../passes/README.md) - Pass documentation index

### External Resources
- [TT-Metalium Programming Guide](https://docs.tenstorrent.com/) - Tenstorrent SDK docs
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - CUDA reference

---

## Changelog

### Version 2.0 (2025-10-17)
- Combined content from historical GPU_vs_Tenstorrent.md and IR_LOWERING_ANALYSIS.md
- Updated all references to v5 pipeline
- Removed C++ implementation details (Python-only)
- Added clear cross-references to v5_pipeline.md
- Restructured for better flow and eliminated redundancy

### Version 1.0 (Historical)
- Original GPU_vs_Tenstorrent.md (5KB) - High-level comparison
- Original IR_LOWERING_ANALYSIS.md (21KB) - Detailed pipeline analysis

---

**Maintainer:** TileLang Tenstorrent Team
**Repository:** https://github.com/davorchap/tilelang-tt
**Last Updated:** 2025-10-17