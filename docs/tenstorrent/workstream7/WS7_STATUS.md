# Workstream 7 Status - API Correctness & Matmul Implementation

**Last Updated:** 2025-10-07

## Overview

Workstream 7 focuses on achieving **API correctness** by aligning generated code with real TT-Metalium structure and implementing proper matmul computation.

**Goal:** Generate correct, compilable matmul code matching Metalium examples.

**Status:** 🚧 **IN PROGRESS** - Phase 2 of MVP

## Progress Summary

| Task | Status | Priority | Blocker |
|------|--------|----------|---------|
| **WS7 Documentation** | ✅ **COMPLETE** | High | None |
| **Fix Compute Kernel K-Loop** | ❌ TODO | **Critical** | None |
| **Fix Reader Tile Indexing** | ❌ TODO | **Critical** | Compute |
| **Fix Writer Tile Output** | ❌ TODO | **Critical** | Compute |
| **Fix Runtime Args** | ❌ TODO | **Critical** | None |
| **Replace Mock Host APIs** | ❌ TODO | **Critical** | None |
| **Extract Buffer Dimensions** | ❌ TODO | High | None |
| **Fix CB Sizes** | ❌ TODO | High | None |
| **Multi-Core Work Distribution** | ❌ TODO | High | Runtime Args |
| **Integration Tests** | ❌ TODO | High | Implementation |

**Overall WS7 Progress:** 5% (Planning only)

---

## Key Issues to Fix

### 1. Compute Kernel - Missing K Loop

**Current (WRONG):**
```cpp
// Only processes 1 tile each
cb_wait_front(CB_A, 1);
cb_wait_front(CB_B, 1);
// TODO: matmul
cb_pop_front(CB_A, 1);
cb_pop_front(CB_B, 1);
```

**Target (CORRECT):**
```cpp
// Matmul: C[m,n] += sum(A[m,k] * B[k,n])
matmul_tiles_init(CB_A, CB_B, CB_C);

for (uint32_t out_tile = 0; out_tile < num_output_tiles; ++out_tile) {
    for (uint32_t kt = 0; kt < Kt; ++kt) {
        cb_wait_front(CB_A, 1);
        cb_wait_front(CB_B, 1);

        matmul_tiles(CB_A, CB_B, CB_C, /* accumulate */ kt > 0);

        cb_pop_front(CB_A, 1);
        cb_pop_front(CB_B, 1);
    }
    cb_push_back(CB_C, 1);
}
```

### 2. Reader Kernel - Wrong Tile Loading

**Current (WRONG):**
```cpp
// Loads tiles sequentially
for (uint32_t i = 0; i < num_tiles; ++i) {
    noc_async_read(...);
}
```

**Target (CORRECT):**
```cpp
// Load specific tiles for matmul
for (uint32_t out_tile = start_tile; out_tile < end_tile; ++out_tile) {
    uint32_t out_m = out_tile / Nt;
    uint32_t out_n = out_tile % Nt;

    for (uint32_t kt = 0; kt < Kt; ++kt) {
        // A[out_m, kt]
        uint32_t tile_a_idx = out_m * Kt + kt;
        read_tile(tile_a_idx, a_addr, CB_A);

        // B[kt, out_n]
        uint32_t tile_b_idx = kt * Nt + out_n;
        read_tile(tile_b_idx, b_addr, CB_B);
    }
}
```

### 3. Runtime Args - Missing Parameters

**Current (WRONG):**
```cpp
uint32_t start_id = get_arg_val<uint32_t>(0);
uint32_t count    = get_arg_val<uint32_t>(1);
uint32_t grid_x   = get_arg_val<uint32_t>(2);
uint32_t grid_y   = get_arg_val<uint32_t>(3);
```

**Target (CORRECT):**
```cpp
// Reader kernel args
uint32_t dram_addr_a       = get_arg_val<uint32_t>(0);
uint32_t dram_addr_b       = get_arg_val<uint32_t>(1);
uint32_t Mt                = get_arg_val<uint32_t>(2);
uint32_t Kt                = get_arg_val<uint32_t>(3);
uint32_t Nt                = get_arg_val<uint32_t>(4);
uint32_t out_tile_start_id = get_arg_val<uint32_t>(5);
uint32_t num_out_tiles     = get_arg_val<uint32_t>(6);

// Compute kernel args
uint32_t out_tile_start_id = get_arg_val<uint32_t>(0);
uint32_t num_output_tiles  = get_arg_val<uint32_t>(1);
uint32_t Kt                = get_arg_val<uint32_t>(2);

// Writer kernel args
uint32_t dram_addr_c       = get_arg_val<uint32_t>(0);
uint32_t out_tile_start_id = get_arg_val<uint32_t>(1);
uint32_t num_out_tiles     = get_arg_val<uint32_t>(2);
uint32_t Nt                = get_arg_val<uint32_t>(3);
```

### 4. Host APIs - Mock vs Real Metalium

**Current (WRONG):**
```cpp
class Device { ... };  // Mock
Program program;
program.AddKernel(...);
```

**Target (CORRECT):**
```cpp
#include "tt_metal/host_api.hpp"

using namespace tt::tt_metal;

Device* device = CreateDevice(0);
Program program = CreateProgram();

auto reader_kernel = CreateKernel(
    program,
    "reader_bmm_single_tile.cpp",
    core,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default
    }
);

SetRuntimeArgs(program, reader_kernel, core, {
    src0_buffer->address(),
    src1_buffer->address(),
    Mt, Kt, Nt,
    start_tile, num_tiles
});
```

---

## Implementation Plan

### Phase 1: Fix Compute Kernel (2-3 hours)

**File:** `src/target/tt/codegen_tt.cc`

**Tasks:**
1. Add K-loop for matmul accumulation
2. Add `matmul_tiles_init()` call
3. Add `matmul_tiles()` with accumulate flag
4. Fix CB patterns (wait/pop inside K-loop, push after)
5. Update runtime args (out_tile_start_id, num_output_tiles, Kt)

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

### Phase 2: Fix Reader Kernel (2-3 hours)

**File:** `src/target/tt/codegen_tt.cc`

**Tasks:**
1. Add proper tile indexing for matmul
2. Calculate out_m, out_n from output tile ID
3. Loop over K tiles for each output
4. Calculate A and B tile indices
5. Update runtime args

**Expected Output:**
```cpp
void kernel_main() {
    uint32_t dram_addr_a = get_arg_val<uint32_t>(0);
    uint32_t dram_addr_b = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t out_tile_start_id = get_arg_val<uint32_t>(5);
    uint32_t num_out_tiles = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    for (uint32_t out_tile = 0; out_tile < num_out_tiles; ++out_tile) {
        uint32_t current_tile_id = out_tile_start_id + out_tile;
        uint32_t out_m = current_tile_id / Nt;
        uint32_t out_n = current_tile_id % Nt;

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

### Phase 3: Fix Writer Kernel (1 hour)

**File:** `src/target/tt/codegen_tt.cc`

**Tasks:**
1. Update tile indexing for output
2. Update runtime args

**Expected Output:**
```cpp
void kernel_main() {
    uint32_t dram_addr_c = get_arg_val<uint32_t>(0);
    uint32_t out_tile_start_id = get_arg_val<uint32_t>(1);
    uint32_t num_out_tiles = get_arg_val<uint32_t>(2);
    uint32_t Nt = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_out = 2;

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

### Phase 4: Fix Host Code (3-4 hours)

**File:** `src/target/tt/codegen_tt.cc`

**Tasks:**
1. Replace mock Device/Program/etc. with real Metalium structure
2. Add proper buffer creation (InterleavedBufferConfig)
3. Add CreateKernel with DataMovementConfig
4. Add SetRuntimeArgs for each kernel per core
5. Add multi-core work distribution
6. Fix CB configuration

**Expected Output:**
```cpp
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt::tt_metal;

int main() {
    // Device setup
    Device* device = CreateDevice(0);
    Program program = CreateProgram();

    // Buffer creation
    InterleavedBufferConfig dram_config{
        .device = device,
        .size = M * K * sizeof(uint16_t),
        .page_size = M * K * sizeof(uint16_t),
        .buffer_type = BufferType::DRAM
    };
    auto src0_buffer = CreateBuffer(dram_config);

    // CB configuration
    uint32_t single_tile_size = 2 * 1024;  // 32x32 fp16
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(2 * single_tile_size, {{CB_A, DataFormat::Float16_b}})
            .set_page_size(CB_A, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    // Kernel creation
    auto reader_kernel = CreateKernel(
        program,
        "reader_bmm_single_tile.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default
        }
    );

    // Runtime args
    SetRuntimeArgs(program, reader_kernel, core, {
        src0_buffer->address(),
        src1_buffer->address(),
        Mt, Kt, Nt,
        start_tile_id,
        num_tiles_per_core
    });

    // Enqueue
    CommandQueue& cq = device->command_queue();
    EnqueueWriteBuffer(cq, src0_buffer, a_data, false);
    EnqueueProgram(cq, program, false);
    EnqueueReadBuffer(cq, dst_buffer, output_data, true);
    Finish(cq);

    CloseDevice(device);
}
```

### Phase 5: Extract Buffer Dimensions (1-2 hours)

**File:** `src/target/tt/codegen_tt.cc`

**Tasks:**
1. Read buffer shapes from PrimFunc params
2. Calculate Mt, Kt, Nt
3. Pass to all codegen functions
4. Remove hardcoded dimensions

**Code:**
```cpp
// Extract dimensions from PrimFunc
auto params = func->params;
auto buffer_A = Downcast<Buffer>(params[0]);
auto buffer_B = Downcast<Buffer>(params[1]);
auto buffer_C = Downcast<Buffer>(params[2]);

int M = buffer_A->shape[0].as<IntImmNode>()->value;
int K = buffer_A->shape[1].as<IntImmNode>()->value;
int N = buffer_B->shape[1].as<IntImmNode>()->value;

int Mt = (M + 31) / 32;  // Tiles in M dimension
int Kt = (K + 31) / 32;  // Tiles in K dimension
int Nt = (N + 31) / 32;  // Tiles in N dimension
```

### Phase 6: Testing & Validation (2 hours)

**File:** `testing/python/tt/test_ws7_api_correctness.py`

**Test Cases:**
1. `test_compute_kernel_k_loop()` - Verify K-loop structure
2. `test_reader_tile_indexing()` - Verify A[m,k] and B[k,n] indexing
3. `test_runtime_args_complete()` - Verify all args present
4. `test_host_apis_structure()` - Verify Metalium API usage
5. `test_buffer_dimensions_extraction()` - Verify dimensions from TileLang
6. `test_cb_sizes_correct()` - Verify CB configurations
7. `test_generated_code_compiles()` - Dry-run compilation test

---

## Build & Test Instructions

```bash
# Rebuild with WS7 changes
bash maint/scripts/local_build_and_test_tt.sh --skip-deps --jobs 4

# Run WS7 tests
pytest testing/python/tt/test_ws7_api_correctness.py -v
```

---

## Dependencies

- **WS1-6 complete** ✓
- **TT-Metalium documentation** (reference)
- **C++ compiler** (for dry-run compilation)

---

## Success Criteria

WS7 is complete when:
- [ ] Compute kernel has proper K-loop
- [ ] Reader loads correct A[m,k] and B[k,n] tiles
- [ ] Runtime args match Metalium examples
- [ ] Host code uses real Metalium API structure
- [ ] Buffer dimensions extracted from TileLang
- [ ] CB sizes match requirements
- [ ] Generated code compiles (dry-run)
- [ ] 10+ integration tests passing
- [ ] No regressions in existing 36 tests

---

## Timeline

**Estimated Effort:** 10-14 hours

- **Phase 1:** 2-3 hours
- **Phase 2:** 2-3 hours
- **Phase 3:** 1 hour
- **Phase 4:** 3-4 hours
- **Phase 5:** 1-2 hours
- **Phase 6:** 2 hours

---

## Related Documentation

- [MVP Phase 2 Plan](../../mvp/MVP_PHASE2_PLAN.md)
- [WS4 Status](../workstream4/WS4_STATUS.md) - Compute kernel foundation
- [WS5 Status](../workstream5/WS5_STATUS.md) - Reader/writer kernels
- [WS6 Status](../workstream6/WS6_STATUS.md) - Host program
- [Metalium Matmul Guide](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_single_core.html)

---
