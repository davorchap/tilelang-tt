# Kernel Authoring Comparison: GPU vs Tenstorrent

This document provides side-by-side comparisons of writing kernels for GPU (CUDA/HIP) vs Tenstorrent, using TileLang as the common frontend.

## Table of Contents
- [Matrix Multiplication (GEMM)](#matrix-multiplication-gemm)
- [Elementwise Operations](#elementwise-operations)
- [Reduction Operations](#reduction-operations)
- [Memory Management](#memory-management)
- [Annotations and Metadata](#annotations-and-metadata)

---

## Matrix Multiplication (GEMM)

### TileLang Frontend (Common for Both Targets)

```python
import tilelang.language as T

@T.prim_func
def matmul(A: T.Buffer[(M, K), "float16"],
           B: T.Buffer[(K, N), "float16"],
           C: T.Buffer[(M, N), "float16"]):
    with T.Kernel(T.ceildiv(M, 32), T.ceildiv(N, 32)) as (bx, by):
        # Allocate shared memory tiles
        A_tile = T.alloc_shared((32, 32), "float16")
        B_tile = T.alloc_shared((32, 32), "float16")
        C_tile = T.alloc_local((32, 32), "float16")

        # Initialize accumulator
        for i, j in T.Parallel(32, 32):
            C_tile[i, j] = 0.0

        # K-loop: Accumulate partial products
        for k in T.Pipelined(T.ceildiv(K, 32)):
            # Load tiles from global memory
            for i, j in T.Parallel(32, 32):
                A_tile[i, j] = A[by * 32 + i, k * 32 + j]
                B_tile[i, j] = B[k * 32 + i, bx * 32 + j]

            # Compute: C_tile += A_tile @ B_tile
            for i, j, kk in T.Parallel(32, 32, 32):
                C_tile[i, j] += A_tile[i, kk] * B_tile[kk, j]

        # Write result back to global memory
        for i, j in T.Parallel(32, 32):
            C[by * 32 + i, bx * 32 + j] = C_tile[i, j]
```

### GPU Backend (Automatic)

**What happens automatically:**
```python
# Compile for CUDA
mod = tvm.IRModule.from_expr(matmul)
mod = tl.compilation.apply_default_schedule(mod, target="cuda")
cuda_code, runtime = tl.compilation.lower_and_build(mod, target="cuda")
```

**Generated CUDA kernel (conceptual):**
```cuda
__global__ void matmul_kernel(float16* A, float16* B, float16* C, int M, int K, int N) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float16 A_tile[32][32];
    __shared__ float16 B_tile[32][32];
    float C_acc = 0.0f;

    for (int k = 0; k < K / 32; ++k) {
        // Load tiles cooperatively (all threads in block)
        A_tile[ty][tx] = A[(by * 32 + ty) * K + (k * 32 + tx)];
        B_tile[ty][tx] = B[(k * 32 + ty) * N + (bx * 32 + tx)];
        __syncthreads();

        // Compute partial dot product
        for (int kk = 0; kk < 32; ++kk) {
            C_acc += A_tile[ty][kk] * B_tile[kk][tx];
        }
        __syncthreads();
    }

    C[(by * 32 + ty) * N + (bx * 32 + tx)] = C_acc;
}
```

### Tenstorrent Backend (With TileLang)

**Apply TT-specific annotations:**
```python
import tilelang.tt as tt

# Compile for Tenstorrent
mod = tvm.IRModule.from_expr(matmul)
mod = tt.apply_tt_defaults(mod)      # Add default schedule/sharding
mod = tt.apply_tt_metadata_passes(mod)       # Lower to persistent + CBs + tensorization
artifacts = tt.emit_tt_artifacts(mod)  # Generate reader/compute/writer kernels

# Artifacts:
#   - reader.cc: Load A/B tiles from DRAM → L1 CBs
#   - compute.cc: matmul_tiles(cb_in0, cb_in1, cb_out)
#   - writer.cc: Write C tiles from L1 CB → DRAM
#   - host_stub.cc: Program, CBs, SetRuntimeArgs
```

**Generated TT Compute Kernel:**
```cpp
#include <tt_metal/compute_kernel_api.h>

void MAIN() {
    // Runtime arguments (set by host)
    uint32_t start_id = get_arg_val<uint32_t>(0);
    uint32_t count    = get_arg_val<uint32_t>(1);
    uint32_t Kt       = get_arg_val<uint32_t>(2);  // K tiles

    mm_init(cb::c_in0, cb::c_in1, cb::c_out0);

    // Persistent loop over assigned output tiles
    for (uint32_t i = 0; i < count; ++i) {
        tile_regs_acquire();

        // K-loop: Accumulate A @ B
        for (uint32_t k = 0; k < Kt; ++k) {
            cb_wait_front(cb::c_in0, 1);  // Wait for A tile
            cb_wait_front(cb::c_in1, 1);  // Wait for B tile

            matmul_tiles(cb::c_in0, cb::c_in1, 0, 0, 0, false);

            cb_pop_front(cb::c_in0, 1);
            cb_pop_front(cb::c_in1, 1);
        }

        // Pack result to output CB
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb::c_out0, 1);
        pack_tile(0, cb::c_out0);
        cb_push_back(cb::c_out0, 1);
        tile_regs_release();
    }
}
```

**Generated TT Reader Kernel:**
```cpp
#include <tt_metal/data_movement_kernel_api.h>

void kernel_main() {
    uint32_t dram_addr_a = get_arg_val<uint32_t>(0);
    uint32_t dram_addr_b = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t count = get_arg_val<uint32_t>(3);
    uint32_t Kt = get_arg_val<uint32_t>(4);
    uint32_t Nt = get_arg_val<uint32_t>(5);

    for (uint32_t i = 0; i < count; ++i) {
        uint32_t out_tile_id = start_id + i;
        uint32_t by = out_tile_id / Nt;
        uint32_t bx = out_tile_id % Nt;

        for (uint32_t k = 0; k < Kt; ++k) {
            // Read A[by, k]
            uint32_t tile_a_idx = by * Kt + k;
            cb_reserve_back(cb::c_in0, 1);
            noc_async_read_tile(tile_a_idx, dram_addr_a, get_write_ptr(cb::c_in0));
            noc_async_read_barrier();
            cb_push_back(cb::c_in0, 1);

            // Read B[k, bx]
            uint32_t tile_b_idx = k * Nt + bx;
            cb_reserve_back(cb::c_in1, 1);
            noc_async_read_tile(tile_b_idx, dram_addr_b, get_write_ptr(cb::c_in1));
            noc_async_read_barrier();
            cb_push_back(cb::c_in1, 1);
        }
    }
}
```

---

## Elementwise Operations

### TileLang Frontend

```python
@T.prim_func
def elementwise_add(A: T.Buffer[(M, N), "float16"],
                    B: T.Buffer[(M, N), "float16"],
                    C: T.Buffer[(M, N), "float16"]):
    with T.Kernel(T.ceildiv(M, 32), T.ceildiv(N, 32)) as (bx, by):
        for i, j in T.Parallel(32, 32):
            C[by * 32 + i, bx * 32 + j] = \
                A[by * 32 + i, bx * 32 + j] + B[by * 32 + i, bx * 32 + j]
```

### GPU Backend Output

```cuda
__global__ void elementwise_add_kernel(float16* A, float16* B, float16* C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    C[idy * N + idx] = A[idy * N + idx] + B[idy * N + idx];
}
```

### Tenstorrent Backend Output (Compute Kernel)

```cpp
void MAIN() {
    uint32_t start_id = get_arg_val<uint32_t>(0);
    uint32_t count = get_arg_val<uint32_t>(1);

    add_tiles_init();

    for (uint32_t i = 0; i < count; ++i) {
        tile_regs_acquire();

        cb_wait_front(cb::c_in0, 1);
        cb_wait_front(cb::c_in1, 1);

        add_tiles(cb::c_in0, cb::c_in1, 0, 0, 0);  // Tile-wise add

        cb_pop_front(cb::c_in0, 1);
        cb_pop_front(cb::c_in1, 1);

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb::c_out0, 1);
        pack_tile(0, cb::c_out0);
        cb_push_back(cb::c_out0, 1);
        tile_regs_release();
    }
}
```

---

## Reduction Operations

### TileLang Frontend (Sum Reduction)

```python
@T.prim_func
def reduce_sum(A: T.Buffer[(M, N), "float16"],
               B: T.Buffer[(M,), "float16"]):
    with T.Kernel(T.ceildiv(M, 32)) as bx:
        partial_sum = T.alloc_local((32,), "float32")

        # Initialize
        for i in T.Parallel(32):
            partial_sum[i] = 0.0

        # Reduce across columns
        for j in range(N):
            for i in T.Parallel(32):
                partial_sum[i] += A[bx * 32 + i, j]

        # Write result
        for i in T.Parallel(32):
            B[bx * 32 + i] = partial_sum[i]
```

### GPU Backend

```cuda
__global__ void reduce_sum_kernel(float16* A, float16* B, int M, int N) {
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    float acc = 0.0f;
    for (int j = 0; j < N; ++j) {
        acc += A[(bx * 32 + tx) * N + j];
    }

    B[bx * 32 + tx] = acc;
}
```

### Tenstorrent Backend

**Compute kernel (reduction requires custom implementation):**
```cpp
void MAIN() {
    uint32_t start_id = get_arg_val<uint32_t>(0);
    uint32_t count = get_arg_val<uint32_t>(1);
    uint32_t N_tiles = get_arg_val<uint32_t>(2);

    for (uint32_t i = 0; i < count; ++i) {
        tile_regs_acquire();

        // Load first tile
        cb_wait_front(cb::c_in0, 1);
        copy_tile(cb::c_in0, 0, 0);  // Copy to DST[0]
        cb_pop_front(cb::c_in0, 1);

        // Accumulate remaining tiles
        for (uint32_t j = 1; j < N_tiles; ++j) {
            cb_wait_front(cb::c_in0, 1);
            add_tiles(cb::c_in0, cb::c_in0, 0, 0, 0);  // DST[0] += tile
            cb_pop_front(cb::c_in0, 1);
        }

        // Pack result
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb::c_out0, 1);
        pack_tile(0, cb::c_out0);
        cb_push_back(cb::c_out0, 1);
        tile_regs_release();
    }
}
```

---

## Memory Management

### GPU: Shared Memory

```python
@T.prim_func
def use_shared_memory(...):
    with T.Kernel(...) as (bx, by):
        shared_tile = T.alloc_shared((32, 32), "float16")  # __shared__

        # Cooperative load
        for i, j in T.Parallel(32, 32):
            shared_tile[i, j] = global_mem[...]

        # Synchronize before use
        T.syncthreads()

        # Use shared_tile...
```

**GPU output:**
```cuda
__shared__ float16 shared_tile[32][32];
shared_tile[ty][tx] = global_mem[...];
__syncthreads();
```

### Tenstorrent: Circular Buffers

```python
@T.prim_func
def use_circular_buffers(...):
    with T.Kernel(...) as (bx, by):
        # T.copy automatically lowers to CBs
        for i, j in T.Parallel(32, 32):
            local_tile[i, j] = T.copy(global_mem[...])  # DRAM → CB → local

        # Compute uses tiles from CB
        ...
```

**TT output (lower_copy_to_tt pass):**
```cpp
// Reader kernel
cb_reserve_back(cb::c_in0, 1);
noc_async_read(..., get_write_ptr(cb::c_in0), ...);
noc_async_read_barrier();
cb_push_back(cb::c_in0, 1);

// Compute kernel
cb_wait_front(cb::c_in0, 1);
// ... use tile from cb::c_in0 ...
cb_pop_front(cb::c_in0, 1);
```

---

## Annotations and Metadata

### GPU Annotations (Minimal)

```python
# GPU relies heavily on automatic scheduling
@T.prim_func
def kernel(...):
    with T.Kernel(grid_x, grid_y) as (bx, by):
        # TileLang/TVM handles thread mapping, shared memory allocation, etc.
        ...
```

### Tenstorrent Annotations (More Explicit)

```python
import tilelang.tt as tt

@T.prim_func
def kernel(...):
    with T.Kernel(grid_x, grid_y) as (bx, by):
        ...

# Apply TT-specific annotations
mod = tvm.IRModule.from_expr(kernel)

# Option 1: Use defaults (contiguous, row-major)
mod = tt.apply_tt_defaults(mod)

# Option 2: Explicit scheduling
from tilelang.tt import annotate_tt_schedule, annotate_tt_sharding

@annotate_tt_schedule(policy="contiguous", order="row_major", chunk_k_tiles=4)
@annotate_tt_sharding({
    "A": {"layout": "row_major"},
    "B": {"layout": "column_major"}
})
@T.prim_func
def kernel_with_annotations(...):
    ...
```

**What these control:**
- `policy`: How tiles are assigned to cores ("contiguous", "block_linear", etc.)
- `order`: Tile iteration order ("row_major", "column_major", "z_order")
- `chunk_k_tiles`: K-dimension chunking for pipelined execution
- `sharding`: DRAM layout and replication strategy per buffer

---

## Summary Table

| Feature | GPU (CUDA/HIP) | Tenstorrent (TT-Metalium) |
|---------|----------------|---------------------------|
| **Frontend** | Same TileLang DSL | Same TileLang DSL |
| **Grid Model** | Automatic block scheduling | Persistent per-core loops |
| **Shared Memory** | `T.alloc_shared` → `__shared__` | `T.copy` → L1 Circular Buffers |
| **Synchronization** | `T.syncthreads()` → `__syncthreads()` | CB producer-consumer protocol |
| **Data Movement** | Implicit via L1/L2 cache | Explicit NOC operations |
| **Tile Size** | Flexible (warp-level) | Fixed 32×32 tiles |
| **Matrix Ops** | Automatic → WMMA/Tensor Cores | Automatic → `matmul_tiles` |
| **Annotations** | Minimal (auto-scheduling) | Explicit (schedule/sharding) |
| **Output** | Single `.cu`/`.hip` file | 3 kernels + host program |

---

## Best Practices

### For GPU Targets
1. **Use automatic scheduling**: TileLang's default GPU schedules are highly optimized
2. **Leverage shared memory**: Mark reused data as `T.alloc_shared`
3. **Maximize occupancy**: Use block sizes that saturate SMs (e.g., 256 threads/block)
4. **Coalesce memory access**: Access consecutive elements across threads

### For Tenstorrent Targets
1. **Think in tiles**: All shapes should be multiples of 32 (or use padding)
2. **Annotate schedules**: Specify tile-to-core mapping for performance
3. **Pipeline K-loops**: Use `chunk_k_tiles` to overlap data movement and compute
4. **Minimize DRAM traffic**: Reuse tiles across cores via multicast (when supported)
5. **Test with dry-run**: Use `emit_tt_artifacts()` to inspect generated kernels

---

## Examples

See `examples/tenstorrent/` for complete examples:
- `example_gemm_cb_pipeline_tt.py`: Pipelined GEMM with circular buffers
- `example_flash_attention_tt.py`: Flash Attention adapted for TT
- `example_reduction_sum_tt.py`: Row-wise sum reduction
- `example_elementwise_multi_tt.py`: Multi-operand elementwise operations

---

## References
- [GPU_vs_Tenstorrent.md](GPU_vs_Tenstorrent.md) - Programming model comparison
- [TT_ARCHITECTURE.md](TT_ARCHITECTURE.md) - Complete TT backend architecture
- [TileLang Documentation](../../README.md) - TileLang language reference
