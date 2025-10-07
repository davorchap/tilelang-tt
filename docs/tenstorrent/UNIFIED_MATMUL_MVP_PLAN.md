# Unified TileLang Tenstorrent Backend Plan: Matmul MVP

**Status:** UNIFIED - Consolidates project_1, MVP_PROGRESS_SUMMARY, and MVP_PHASE2_PLAN
**Last Updated:** 2025-10-07
**Objective:** Complete dry-run matmul code generation with correct Metalium APIs

---

## Executive Summary

This document unifies three previously divergent planning documents:
- **project_1.md** - Original comprehensive MVP plan (authoritative baseline)
- **MVP_PROGRESS_SUMMARY.md** - Progress tracker showing WS1-3 complete (60%)
- **MVP_PHASE2_PLAN.md** - Corrective action plan for WS4-6 API issues

**Current Status:**
- ✅ **WS1-3 Complete (60%)** - Foundation, metadata inference, basic transforms
- 🚧 **WS4-6 In Progress (35%)** - Codegen exists but needs API correctness fixes
- ⏳ **WS7 = Fixing WS4-6** - Not a new workstream, but corrections to codegen

**Key Insight:** MVP_PHASE2_PLAN and WS7 are NOT new workstreams—they're fixing incomplete WS4-6 implementation. This unified plan consolidates them back into WS4-6.

---

## Divergence Analysis

### What Happened

1. **project_1.md** (original) defined comprehensive MVP with 6 workstreams
2. **Implementation proceeded** through WS1-3 successfully
3. **WS4-6 implementation** had gaps:
   - Compute kernel missing K-loop (not real matmul)
   - Reader/writer using wrong tile indexing
   - Mock APIs instead of real Metalium structure
   - Hardcoded dimensions instead of reading from IR
4. **MVP_PROGRESS_SUMMARY.md** created to track WS1-3 completion
5. **MVP_PHASE2_PLAN.md** created as reactive fix plan
6. **WS7** created as a "new workstream" but actually fixing WS4-6

### Resolution

This unified plan:
- Uses **project_1.md as authoritative baseline** (most comprehensive)
- Incorporates **MVP_PROGRESS_SUMMARY learnings** (what worked in WS1-3)
- **Folds MVP_PHASE2_PLAN and WS7 back into WS4-6** (they're fixes, not new work)
- Provides **single source of truth** going forward

---

## Architecture: TT vs CUDA Stack

### IR Flow Comparison

**CUDA Stack:**
```
TileLang Frontend → TIR
  ↓
Lower to CUDA primitives
  ↓
Thread block transforms (persist_threadblock, etc.)
  ↓
CUDA CodeGen (codegen_cuda.cc)
  ↓
NVRTC compile → PTX → cubin
```

**TT Stack (Unified):**
```
TileLang Frontend → TIR
  ↓
WS1: Target Registration & Default Annotations
  ├─ Recognize target="tenstorrent"
  ├─ Apply defaults: contiguous schedule, DRAM interleaved
  └─ Output: TIR with tt_* attributes
  ↓
WS2: Schedule & Sharding Metadata Inference
  ├─ InferDefaultTTSchedule: compute per-core tile ranges
  ├─ InferDefaultTTShard: generate DRAM layout descriptors
  └─ Output: TIR with tt_grid_*, tt_buffer_* metadata
  ↓
WS3: TIR Transform Pipeline
  ├─ GridToPersistentTT: wrap body in per-core loop
  ├─ (Deferred) TTShardToCoreMap: CoreRangeSet topology
  ├─ (Deferred) MemorySpaceLowerTT: circular buffers
  ├─ (Deferred) TilePadTT: padding insertion
  ├─ (Deferred) TensorizeTT: matmul intrinsics
  └─ Output: Persistent TIR ready for codegen
  ↓
WS4-6: Code Generation (NEEDS COMPLETION)
  ├─ EmitTTComputeKernel: K-loop matmul with CB operations
  ├─ EmitTTReaderWriter: DRAM→L1→DRAM with tile indexing
  ├─ EmitTTHostProgram: Real Metalium APIs
  └─ Output: compute.cpp, reader.cpp, writer.cpp, host.cpp, tt.plan.json
```

### Key Architectural Differences

| Aspect | CUDA | Tenstorrent |
|--------|------|-------------|
| **Work Assignment** | Dynamic hardware scheduler | Static per-core tile assignment |
| **Kernel Lifetime** | Short-lived blocks | Persistent per-core loops |
| **Memory Pipeline** | Single kernel with shared mem | 3 kernels (reader/compute/writer) |
| **Granularity** | Software-chosen tiles | Native 32×32 tiles (tilized DRAM) |
| **Scratchpad** | Shared memory (per-SM) | L1 Circular Buffers (per-core) |
| **Indexing** | blockIdx.x/y/z from hardware | Computed from tile_id: `bx = tile_id % grid_x` |
| **Data Movement** | Global↔shared in-kernel | Separate reader kernel (DRAM→L1) |
| **Pipelining** | cp.async + multi-buffering | CB depth + split reader/compute/writer |

**Critical Difference for Matmul:**
- **CUDA:** Single kernel with K-loop over shared memory tiles
- **TT:** Reader loads A[m,k] and B[k,n] sequentially, compute has K-loop consuming from CBs

---

## Matmul IR with TT Extensions

### Input: TileLang Matmul (256×256, bf16)

```python
@T.prim_func
def matmul(A: T.Buffer[(256, 256), "float16"],
           B: T.Buffer[(256, 256), "float16"],
           C: T.Buffer[(256, 256), "float16"]):
    with T.Kernel(T.ceildiv(256, 32), T.ceildiv(256, 32)) as (bx, by):
        # Grid: 8×8 = 64 tiles
        # Each tile computes C[by*32:(by+1)*32, bx*32:(bx+1)*32]

        A_shared = T.alloc_shared((32, 32), "float16")
        B_shared = T.alloc_shared((32, 32), "float16")
        C_local = T.alloc_fragment((32, 32), "float16")

        # K-loop (omitted for brevity, full matmul has this)
        for k in range(0, 256, 32):
            # Load A[by*32:(by+1)*32, k:k+32]
            # Load B[k:k+32, bx*32:(bx+1)*32]
            # Accumulate C_local += A_shared @ B_shared

        # Store C_local to C[by*32:(by+1)*32, bx*32:(bx+1)*32]
```

### After WS1: Default TT Annotations

```python
func.attrs = {
    "tt_schedule_policy": "contiguous",
    "tt_schedule_order": "row_major",
    "tt_layout_type": "dram_interleaved",
    "tt_tile_height": 32,
    "tt_tile_width": 32,
}
```

### After WS2: Schedule & Sharding Metadata

```python
func.attrs = {
    # WS1 attrs preserved...

    # Schedule metadata
    "tt_num_tiles": 64,           # 8×8 grid
    "tt_grid_x": 8,
    "tt_grid_y": 8,
    "tt_grid_z": 1,
    "tt_num_cores": 64,           # Tensix cores
    "tt_tiles_per_core": [        # Contiguous assignment
        [0, 1],   # Core 0: tile 0
        [1, 1],   # Core 1: tile 1
        ...
        [63, 1],  # Core 63: tile 63
    ],

    # Sharding metadata (per buffer)
    "tt_buffer_A_layout": "dram_interleaved",
    "tt_buffer_A_tile_shape": [32, 32],
    "tt_buffer_A_num_tiles_height": 8,
    "tt_buffer_A_num_tiles_width": 8,
    "tt_buffer_A_needs_padding": 0,

    "tt_buffer_B_layout": "dram_interleaved",
    "tt_buffer_B_tile_shape": [32, 32],
    "tt_buffer_B_num_tiles_height": 8,
    "tt_buffer_B_num_tiles_width": 8,
    "tt_buffer_B_needs_padding": 0,

    "tt_buffer_C_layout": "dram_interleaved",
    "tt_buffer_C_tile_shape": [32, 32],
    "tt_buffer_C_num_tiles_height": 8,
    "tt_buffer_C_num_tiles_width": 8,
    "tt_buffer_C_needs_padding": 0,
}
```

### After WS3: Persistent Loop Structure

```cpp
// Conceptual C++ output (actual IR is TIR)
void kernel(int32_t tt_start_id, int32_t tt_count,
            int32_t grid_x, int32_t grid_y) {
    // Persistent outer loop
    for (int i = 0; i < tt_count; ++i) {
        int tile_id = tt_start_id + i;

        // Recover block indices
        int bx = tile_id % grid_x;
        int by = tile_id / grid_x;

        // Original kernel body (still has T.alloc_shared, etc.)
        // These will be lowered in future transforms
        ...
    }
}
```

### Target After WS4-6 (Needs Completion): Generated Kernels

**Compute Kernel (compute.cpp):**
```cpp
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/matmul.h"

void MAIN {
    uint32_t out_tile_start_id = get_arg_val<uint32_t>(0);
    uint32_t num_output_tiles = get_arg_val<uint32_t>(1);
    uint32_t Kt = get_arg_val<uint32_t>(2);  // K dimension in tiles

    constexpr uint32_t CB_A = 0;
    constexpr uint32_t CB_B = 1;
    constexpr uint32_t CB_C = 2;

    matmul_tiles_init(CB_A, CB_B, CB_C);

    // Per-output-tile loop
    for (uint32_t out_tile = 0; out_tile < num_output_tiles; ++out_tile) {
        // K-loop for matmul accumulation
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            cb_wait_front(CB_A, 1);  // Wait for A[m, kt]
            cb_wait_front(CB_B, 1);  // Wait for B[kt, n]

            matmul_tiles(CB_A, CB_B, CB_C, /* accumulate */ kt > 0);

            cb_pop_front(CB_A, 1);
            cb_pop_front(CB_B, 1);
        }

        cb_push_back(CB_C, 1);  // Output tile ready
    }
}
```

**Reader Kernel (reader.cpp):**
```cpp
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dram_addr_a = get_arg_val<uint32_t>(0);
    uint32_t dram_addr_b = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);  // M in tiles
    uint32_t Kt = get_arg_val<uint32_t>(3);  // K in tiles
    uint32_t Nt = get_arg_val<uint32_t>(4);  // N in tiles
    uint32_t out_tile_start_id = get_arg_val<uint32_t>(5);
    uint32_t num_out_tiles = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in0 = 0;  // CB for A
    constexpr uint32_t cb_id_in1 = 1;  // CB for B

    // For each output tile this core computes
    for (uint32_t out_tile = 0; out_tile < num_out_tiles; ++out_tile) {
        uint32_t current_tile_id = out_tile_start_id + out_tile;
        uint32_t out_m = current_tile_id / Nt;
        uint32_t out_n = current_tile_id % Nt;

        // Load row of A and column of B
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            // Read A[out_m, kt]
            uint32_t tile_a_idx = out_m * Kt + kt;
            cb_reserve_back(cb_id_in0, 1);
            uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            noc_async_read_tile(tile_a_idx, dram_addr_a, l1_write_addr_in0);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, 1);

            // Read B[kt, out_n]
            uint32_t tile_b_idx = kt * Nt + out_n;
            cb_reserve_back(cb_id_in1, 1);
            uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
            noc_async_read_tile(tile_b_idx, dram_addr_b, l1_write_addr_in1);
            noc_async_read_barrier();
            cb_push_back(cb_id_in1, 1);
        }
    }
}
```

**Writer Kernel (writer.cpp):**
```cpp
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dram_addr_c = get_arg_val<uint32_t>(0);
    uint32_t out_tile_start_id = get_arg_val<uint32_t>(1);
    uint32_t num_out_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = 2;  // CB for C

    for (uint32_t out_tile = 0; out_tile < num_out_tiles; ++out_tile) {
        uint32_t tile_idx = out_tile_start_id + out_tile;

        cb_wait_front(cb_id_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        noc_async_write_tile(tile_idx, l1_read_addr, dram_addr_c);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, 1);
    }
}
```

**Host Program (host.cpp):**
```cpp
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt::tt_metal;

int main() {
    // Device setup
    Device* device = CreateDevice(0);
    Program program = CreateProgram();

    // Buffer creation (interleaved DRAM)
    uint32_t M = 256, K = 256, N = 256;
    uint32_t Mt = 8, Kt = 8, Nt = 8;
    uint32_t single_tile_size = 2 * 1024;  // 32×32 fp16

    InterleavedBufferConfig a_config{
        .device = device,
        .size = M * K * sizeof(uint16_t),
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM
    };
    auto src0_buffer = CreateBuffer(a_config);
    auto src1_buffer = CreateBuffer(a_config);
    auto dst_buffer = CreateBuffer(a_config);

    // CB configuration (per core)
    CoreRange core = {CoreCoord{0, 0}};  // Example: single core

    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(2 * single_tile_size, {{0, DataFormat::Float16_b}})
            .set_page_size(0, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(2 * single_tile_size, {{1, DataFormat::Float16_b}})
            .set_page_size(1, single_tile_size);
    CreateCircularBuffer(program, core, cb_src1_config);

    CircularBufferConfig cb_output_config =
        CircularBufferConfig(1 * single_tile_size, {{2, DataFormat::Float16_b}})
            .set_page_size(2, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    // Kernel creation
    auto reader_kernel = CreateKernel(
        program,
        "reader.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default
        }
    );

    auto compute_kernel = CreateKernel(
        program,
        "compute.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .dst_full_sync_en = true
        }
    );

    auto writer_kernel = CreateKernel(
        program,
        "writer.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default
        }
    );

    // Runtime args (example for core 0)
    uint32_t start_tile_id = 0;
    uint32_t num_tiles_per_core = 1;

    SetRuntimeArgs(program, reader_kernel, core, {
        src0_buffer->address(),
        src1_buffer->address(),
        Mt, Kt, Nt,
        start_tile_id,
        num_tiles_per_core
    });

    SetRuntimeArgs(program, compute_kernel, core, {
        start_tile_id,
        num_tiles_per_core,
        Kt
    });

    SetRuntimeArgs(program, writer_kernel, core, {
        dst_buffer->address(),
        start_tile_id,
        num_tiles_per_core
    });

    // Enqueue (dry-run: commented out)
    // CommandQueue& cq = device->command_queue();
    // EnqueueWriteBuffer(cq, src0_buffer, a_data, false);
    // EnqueueProgram(cq, program, false);
    // EnqueueReadBuffer(cq, dst_buffer, output_data, true);
    // Finish(cq);

    CloseDevice(device);
    return 0;
}
```

---

## Unified Workstream Plan

### ✅ Workstream 1: Frontend Integration & Target Selection (COMPLETE)

**Status:** 100% Complete (8 tests passing)

**Delivered:**
- Target registration in `tilelang/utils/target.py`
- Engine adapter in `tilelang/engine/tt/`
- Default annotation helper in `tilelang/tt/target.py`

**Tests:** `testing/python/tt/test_target_registration.py`

---

### ✅ Workstream 2: Schedule & Sharding Metadata (COMPLETE)

**Status:** 100% Complete (7 tests passing)

**Delivered:**
- `src/transform/tt/infer_tt_schedule.cc` - Schedule inference pass
- `src/transform/tt/infer_tt_shard.cc` - Sharding inference pass
- `tilelang/tt/passes.py` - Python bindings

**Tests:** `testing/python/tt/test_ws2_passes.py`

---

### ✅ Workstream 3: TIR Transform Pipeline (FOUNDATION COMPLETE)

**Status:** 50% Complete (3 tests passing, foundation only)

**Delivered:**
- `src/transform/tt/grid_to_persistent_tt.cc` - GridToPersistentTT transform

**Deferred to Post-MVP:**
- TTShardToCoreMap
- MemorySpaceLowerTT
- TilePadTT
- TensorizeTT
- VerifyTTIR

**Rationale:** Unblock WS4 codegen; remaining transforms add incrementally

**Tests:** `testing/python/tt/test_ws3_grid_to_persistent.py`

---

### 🚧 Workstream 4-6: Code Generation (NEEDS COMPLETION)

**Current Status:** 40% Complete (basic structure exists, API correctness needed)

**Problem:** Initial implementation (WS4-6) generated code but with critical gaps:
1. Compute kernel missing K-loop (not real matmul)
2. Reader/writer using sequential tile loading (wrong for matmul)
3. Mock APIs instead of real Metalium structure
4. Hardcoded dimensions instead of reading from IR
5. Incorrect runtime args
6. Wrong CB sizes

**Resolution:** Complete WS4-6 properly by fixing all API correctness issues

#### WS4: Compute Kernel Code Generation

**File:** `src/target/tt/codegen_tt.cc` - `EmitTTComputeKernel()`

**Tasks:**
1. ✅ Extract buffer dimensions from PrimFunc (Mt, Kt, Nt)
2. ❌ **Add K-loop for matmul accumulation** (CRITICAL)
3. ❌ **Add matmul_tiles_init() and matmul_tiles()** (CRITICAL)
4. ❌ Fix CB patterns (wait/pop inside K-loop)
5. ❌ Update runtime args schema

**Expected Output:**
```cpp
void MAIN {
    uint32_t out_tile_start_id = get_arg_val<uint32_t>(0);
    uint32_t num_output_tiles = get_arg_val<uint32_t>(1);
    uint32_t Kt = get_arg_val<uint32_t>(2);

    matmul_tiles_init(CB_A, CB_B, CB_C);

    for (uint32_t out_tile = 0; out_tile < num_output_tiles; ++out_tile) {
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            cb_wait_front(CB_A, 1);
            cb_wait_front(CB_B, 1);
            matmul_tiles(CB_A, CB_B, CB_C, kt > 0);
            cb_pop_front(CB_A, 1);
            cb_pop_front(CB_B, 1);
        }
        cb_push_back(CB_C, 1);
    }
}
```

**Tests:** `testing/python/tt/test_ws4_codegen.py`

---

#### WS5: Reader/Writer Kernel Code Generation

**File:** `src/target/tt/codegen_tt.cc` - `EmitTTReaderKernel()`, `EmitTTWriterKernel()`

**Tasks:**
1. ❌ **Fix reader tile indexing for matmul** (A[m,k] and B[k,n])
2. ❌ Add proper DRAM address calculations
3. ❌ Update runtime args (add Mt, Kt, Nt, DRAM addrs)
4. ❌ Fix writer tile output indexing

**Expected Reader Output:** (see full code above)

**Tests:** `testing/python/tt/test_ws5_reader_writer.py`

---

#### WS6: Host Program Code Generation

**File:** `src/target/tt/codegen_tt.cc` - `EmitTTHostProgram()`

**Tasks:**
1. ❌ **Replace mock APIs with real Metalium structure** (CRITICAL)
2. ❌ Add proper buffer creation (InterleavedBufferConfig)
3. ❌ Add CreateKernel with DataMovementConfig/ComputeConfig
4. ❌ Add SetRuntimeArgs for each kernel
5. ❌ Fix CB configuration (correct sizes)
6. ❌ Add multi-core work distribution

**Expected Output:** (see full code above)

**Tests:** `testing/python/tt/test_ws6_host_program.py`

---

### ⏳ Workstream Testing & Validation

**File:** `testing/python/tt/test_mvp_acceptance.py`

**MVP GEMM Acceptance Test:**
```python
def test_mvp_gemm_256x256_full_pipeline():
    # Create 256×256 matmul
    mod = create_gemm_module(256, 256, 256)

    # Apply full pipeline
    mod = tt.apply_tt_defaults(mod)           # WS1
    mod = tt.apply_ws2_passes(mod)            # WS2
    mod = tt.apply_ws3_passes(mod)            # WS3
    artifacts = tt.emit_tt_artifacts(mod)     # WS4-6

    # Verify artifacts
    assert "compute.cpp" in artifacts
    assert "reader.cpp" in artifacts
    assert "writer.cpp" in artifacts
    assert "host.cpp" in artifacts
    assert "tt.plan.json" in artifacts

    # Validate compute kernel has K-loop
    compute_cpp = artifacts["compute.cpp"]
    assert "matmul_tiles_init" in compute_cpp
    assert "for (uint32_t kt = 0; kt < Kt; ++kt)" in compute_cpp

    # Validate reader has correct tile indexing
    reader_cpp = artifacts["reader.cpp"]
    assert "uint32_t tile_a_idx = out_m * Kt + kt" in reader_cpp
    assert "uint32_t tile_b_idx = kt * Nt + out_n" in reader_cpp

    # Validate host uses real Metalium APIs
    host_cpp = artifacts["host.cpp"]
    assert "using namespace tt::tt_metal" in host_cpp
    assert "CreateDevice" in host_cpp
    assert "CreateProgram" in host_cpp
    assert "CreateKernel" in host_cpp
    assert "SetRuntimeArgs" in host_cpp
```

---

## Implementation Timeline

**Total Estimated Effort:** 10-14 hours

### Phase 1: Fix Compute Kernel (2-3 hours)
- Add K-loop structure
- Add matmul tile operations
- Update runtime args

### Phase 2: Fix Reader/Writer (2-3 hours)
- Implement matmul tile indexing
- Add DRAM address calculations
- Update runtime args

### Phase 3: Fix Host Program (3-4 hours)
- Replace all mock APIs
- Add real Metalium structure
- Fix CB configuration
- Add multi-core support

### Phase 4: Extract Buffer Dimensions (1-2 hours)
- Read from PrimFunc params
- Calculate Mt, Kt, Nt
- Pass to all codegen functions

### Phase 5: Testing & Validation (2 hours)
- Update existing tests
- Add MVP acceptance test
- Verify dry-run compilation

---

## Success Criteria

**MVP Complete When:**
- [ ] All 6 workstreams at 100%
- [ ] Compute kernel has correct K-loop matmul
- [ ] Reader loads A[m,k] and B[k,n] correctly
- [ ] Writer outputs C tiles correctly
- [ ] Host uses real Metalium APIs (not mocks)
- [ ] Runtime args match Metalium examples
- [ ] Buffer dimensions from IR (not hardcoded)
- [ ] CB sizes correct (double-buffered inputs, single output)
- [ ] 40+ tests passing (WS1-6 + acceptance)
- [ ] Generated code compiles (dry-run)
- [ ] tt.plan.json validates correctly

---

## Post-MVP Roadmap

After MVP completion, address deferred items:

**Phase 1: Complete WS3 Transforms**
- TTShardToCoreMap (CoreRangeSet topology)
- MemorySpaceLowerTT (circular buffers, L1 allocation)
- TilePadTT (padding for non-tile-aligned)
- TensorizeTT (matmul intrinsics)
- VerifyTTIR (IR validation)

**Phase 2: Advanced Features**
- Multi-device support
- Advanced scheduling (strided, rectangular)
- Custom sharding annotations
- Performance optimization

**Phase 3: Hardware Integration**
- Hardware execution (beyond dry-run)
- Performance profiling
- Autotuning integration

---

## Related Documentation

- **Architecture:** `GPU_vs_Tenstorrent.md`
- **Build Guide:** `local_build_guide.md`
- **CI Documentation:** `CI.md`
- **Workstream Status:**
  - `workstream1/WS1_STATUS.md`
  - `workstream2/WS2_STATUS.md`
  - `workstream3/WS3_STATUS.md`
  - `workstream4/WS4_STATUS.md`
  - `workstream5/WS5_STATUS.md`
  - `workstream6/WS6_STATUS.md`

---

## Appendix: Deprecated Documents

The following documents are **superseded** by this unified plan:

1. **project_1_prompt.md** - Original prompt (historical reference)
2. **MVP_PROGRESS_SUMMARY.md** - Interim progress tracker (content integrated here)
3. **mvp/MVP_PHASE2_PLAN.md** - Reactive fix plan (content integrated into WS4-6)
4. **workstream7/WS7_STATUS.md** - Mislabeled fixes (actually WS4-6 corrections)

**Action:** Mark these as deprecated and reference this unified plan going forward.

---

**Document Version:** 1.0 (UNIFIED)
**Author:** Claude Code
**Date:** 2025-10-07
