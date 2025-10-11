# GPU vs Tenstorrent Programming Model Comparison

This document contrasts the GPU and Tenstorrent execution models to explain how TileLang kernels are adapted for Tenstorrent hardware.

## Table of Contents
- [Execution Model](#execution-model)
- [Memory Hierarchy](#memory-hierarchy)
- [Data Movement](#data-movement)
- [Compute Operations](#compute-operations)
- [Compilation Flow](#compilation-flow)

---

## Execution Model

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

## Data Movement

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

## Compilation Flow

### GPU: NVCC/HIPCC → PTX/AMDGPU → Binary

```
TileLang DSL (Python)
    ↓
TVM TIR (IR)
    ↓
GPU Codegen (PTX/HIP)
    ↓
NVCC/HIPCC
    ↓
Executable (cubin/hsaco)
```

**Key transformations:**
- **Thread hierarchy**: Map loops to `threadIdx`/`blockIdx`
- **Memory promotion**: Allocate shared memory for reuse
- **Intrinsic lowering**: Replace high-level ops with PTX/AMDGPU intrinsics

### Tenstorrent: TileLang → TIR → TT Kernels → Metalium

```
TileLang DSL (Python)
    ↓
TVM TIR (IR)
    ↓
TT Transform Passes (see [TT_ARCHITECTURE.md](TT_ARCHITECTURE.md#layout-aware-metadata) for canonical ordering):
  - infer_default_tt_schedule / infer_default_tt_shard *(legacy defaults)*
  - apply_layout_aware_metadata_passes *(InferTTLayout → PropagateTTLayout → LayoutAwareWorkPartitionTT)*
  - grid_to_persistent_tt *(GPU grid → persistent loop, shard-aware)*
  - tt_tiles_to_core_map *(legacy fallback when layout metadata is absent)*
  - memory_space_lower_tt *(Lower T.copy → CB/NOC ops using `tt.cb.*`)*
  - tile_pad_tt *(Insert padding for non-tile-aligned shapes)*
  - lower_gemm_to_tt_intrinsics *(Replace patterns with `matmul_tiles`, etc.; matcher upgrades pending)*
  - verify_tt_ir *(Validate constraints and runtime args)*
    ↓
TT Codegen (detailed in [TT_ARCHITECTURE.md](TT_ARCHITECTURE.md#code-generation)): 
  - Reader kernel (`reader.cpp`)
  - Compute kernel (`compute.cpp`)
  - Writer kernel (`writer.cpp`)
  - Host metadata summary (`main.cpp`) with TensorAccessor payloads
  - Execution plan (`tt.plan.json`)
    ↓
TT-Metalium SDK (Compile & Link)
    ↓
Executable (runs on Grayskull/Wormhole)
```

**Key transformations:**
- **Persistent scheduler**: Wrap body in `for (i = 0; i < count; ++i)`
- **CB allocation**: Create circular buffers for staging (depth from schedule)
- **NOC lowering**: Replace `T.copy` with `noc_async_read/write`
- **Tensorization**: Match patterns → `matmul_tiles`, `tilize`, etc.
- **Three-kernel split**: Separate data movement from compute

---

## Summary Table

| Aspect | GPU (CUDA/HIP) | Tenstorrent (TT-Metalium) |
|--------|----------------|---------------------------|
| **Execution** | Transient blocks, hardware-scheduled | Persistent per-core loops, static partitioning |
| **Parallelism** | Massive (thousands of blocks) | Moderate (64-120 Tensix cores) |
| **Memory** | L1/L2 caches + explicit shared | Explicit L1 circular buffers |
| **Data Movement** | Implicit caching + `__shared__` | Explicit NOC operations + CBs |
| **Tile Size** | Flexible (warp-level fragments) | Fixed 32×32 tiles |
| **Compute** | Tensor Cores (WMMA) | Tile intrinsics (`matmul_tiles`) |
| **Synchronization** | `__syncthreads()`, barriers | CB producer-consumer protocol |
| **Programming Model** | SIMT (Single Instruction, Multiple Threads) | SPMD (Single Program, Multiple Data) + Dataflow |

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
import tilelang.tt as tt

@T.prim_func
def matmul(A: T.Buffer, B: T.Buffer, C: T.Buffer):
    with T.Kernel(T.ceildiv(M, 32), T.ceildiv(N, 32)) as (bx, by):
        # GPU-style grid kernel (user code stays familiar)
        # ...

# Apply TT backend
mod = tvm.IRModule.from_expr(matmul)
mod = tt.apply_tt_defaults(mod)         # Inject TT annotations
mod = tt.apply_tt_metadata_passes(mod)          # Lower to persistent + CBs
artifacts = tt.emit_tt_artifacts(mod)   # Generate reader/compute/writer
```

The backend handles the transformation to Tenstorrent's execution model transparently.

---

## References

- **Tenstorrent Metalium Docs**: [TT-Metalium Programming Guide](https://docs.tenstorrent.com/)
- **TileLang TT Architecture**: [TT_ARCHITECTURE.md](TT_ARCHITECTURE.md)
- **CUDA Programming Guide**: [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
