# MVP Phase 2: Correct APIs + Compilable Matmul

> **⚠️ DEPRECATED - 2025-10-07**
>
> This "Phase 2" plan has been integrated into **[UNIFIED_MATMUL_MVP_PLAN.md](../UNIFIED_MATMUL_MVP_PLAN.md)**.
>
> The issues identified here are now part of WS4-6 completion tasks in the unified plan.
> There is no separate "Phase 2"—this is simply completing the original MVP.
>
> This file is retained for historical reference only.

---

**Created:** 2025-10-07
**Status:** 🚧 **IN PROGRESS** (Superseded by Unified Plan)

## Overview

Phase 2 focuses on **API correctness** and achieving a **compilable dry-run matmul** implementation that matches real TT-Metalium structure.

### Goals

1. ✅ **Correct Kernel Loops** - Match actual matmul computation (K-loop)
2. ✅ **Correct Buffer Sizes** - Based on actual matrix dimensions from TileLang
3. ✅ **Correct CB Sizes** - Multiple tiles for pipelining
4. ✅ **Correct Host APIs** - Real Metalium API structure
5. ✅ **Multi-Core Support** - Work distribution and per-core runtime args
6. ✅ **Compilable Dry-Run** - Generates code that compiles (without hardware)

---

## Current Issues (From WS5/WS6)

### 1. Compute Kernel - Missing K Loop
**Problem:** Current kernel only processes 1 tile each of A and B:
```cpp
// WRONG - Current implementation
cb_wait_front(CB_A, 1);
cb_wait_front(CB_B, 1);
// matmul_tiles(...) - single tile
cb_pop_front(CB_A, 1);
cb_pop_front(CB_B, 1);
```

**Should be:** Loop over K dimension:
```cpp
// CORRECT - Matmul computation
// C[m,n] = sum(A[m,k] * B[k,n] for k in K_tiles)
for (uint32_t kt = 0; kt < Kt; ++kt) {
    cb_wait_front(CB_A, 1);  // Wait for A[m, kt]
    cb_wait_front(CB_B, 1);  // Wait for B[kt, n]

    if (kt == 0) {
        matmul_tiles_init(CB_A, CB_B, CB_C);
    }
    matmul_tiles(CB_A, CB_B, CB_C, /* accumulate */ kt > 0);

    cb_pop_front(CB_A, 1);
    cb_pop_front(CB_B, 1);
}
```

### 2. Reader Kernel - Wrong Tile Loading
**Problem:** Current reader loads tiles sequentially without considering matmul structure:
```cpp
// WRONG - Just loads tiles in order
for (uint32_t i = 0; i < num_tiles; ++i) {
    cb_reserve_back(CB_A, 1);
    noc_async_read(...);
    cb_push_back(CB_A, 1);
}
```

**Should be:** Load specific tiles for each output tile:
```cpp
// CORRECT - Matmul tile loading
for (uint32_t out_tile = start_tile; out_tile < end_tile; ++out_tile) {
    uint32_t out_m = out_tile / Nt;
    uint32_t out_n = out_tile % Nt;

    // Load row of A and column of B for this output
    for (uint32_t kt = 0; kt < Kt; ++kt) {
        // Load A[out_m, kt]
        uint32_t tile_a_idx = out_m * Kt + kt;
        noc_async_read_tile(tile_a_idx, a_addr, l1_addr_a);
        cb_push_back(CB_A, 1);

        // Load B[kt, out_n]
        uint32_t tile_b_idx = kt * Nt + out_n;
        noc_async_read_tile(tile_b_idx, b_addr, l1_addr_b);
        cb_push_back(CB_B, 1);
    }
}
```

### 3. Runtime Args - Missing Critical Parameters
**Problem:** Current args only have grid dimensions:
```cpp
// WRONG - Insufficient runtime args
get_arg_val<uint32_t>(0);  // start_id
get_arg_val<uint32_t>(1);  // count
get_arg_val<uint32_t>(2);  // grid_x
get_arg_val<uint32_t>(3);  // grid_y
```

**Should be:**
```cpp
// CORRECT - Matmul runtime args
uint32_t dram_addr_a        = get_arg_val<uint32_t>(0);
uint32_t dram_addr_b        = get_arg_val<uint32_t>(1);
uint32_t dram_addr_c        = get_arg_val<uint32_t>(2);
uint32_t Mt                 = get_arg_val<uint32_t>(3);  // M / TILE_SIZE
uint32_t Kt                 = get_arg_val<uint32_t>(4);  // K / TILE_SIZE
uint32_t Nt                 = get_arg_val<uint32_t>(5);  // N / TILE_SIZE
uint32_t out_tile_start_id  = get_arg_val<uint32_t>(6);
uint32_t num_out_tiles      = get_arg_val<uint32_t>(7);
```

### 4. Host Code - Mock APIs Instead of Real Metalium
**Problem:** Using placeholder mock classes:
```cpp
// WRONG - Mock APIs
class Device {
public:
    static Device* Instance() { static Device dev; return &dev; }
};
```

**Should be:** Real Metalium API structure:
```cpp
// CORRECT - Real Metalium APIs (dry-run compatible)
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

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

### 5. CB Configuration - Wrong Sizes
**Problem:** CB sizes don't account for pipelining:
```cpp
// WRONG - Only 2 pages for all CBs
CircularBufferConfig cb_a(0, TILE_SIZE_FP16, 2);
```

**Should be:** Different sizes for input vs intermediate:
```cpp
// CORRECT - Sized for matmul pipelining
// Input CBs: 2 tiles (double buffer)
CircularBufferConfig cb_src0_config =
    CircularBufferConfig(2 * single_tile_size, {{CB_A, cb_data_format}})
        .set_page_size(CB_A, single_tile_size);

// Intermediate CB for accumulation: 1 tile
CircularBufferConfig cb_output_config =
    CircularBufferConfig(1 * single_tile_size, {{CB_C, cb_data_format}})
        .set_page_size(CB_C, single_tile_size);
```

### 6. Buffer Dimensions - Not from Actual TileLang
**Problem:** Hardcoded to grid dimensions:
```cpp
// WRONG - Just grid_x * 32
constexpr uint32_t M = 256;  // grid_y * 32
constexpr uint32_t N = 256;  // grid_x * 32
```

**Should be:** Read from TileLang PrimFunc buffer parameters:
```cpp
// CORRECT - From actual buffer shapes
auto params = func->params;
auto buffer_A = params[0].as<BufferNode>();
auto shape_A = buffer_A->shape;
uint32_t M = shape_A[0];  // Actual M from TileLang
uint32_t K = shape_A[1];  // Actual K from TileLang
```

---

## Implementation Plan

### WS7: API Correctness & Matmul Implementation

**Priority:** Critical
**Effort:** 8-12 hours

#### Phase 1: Fix Compute Kernel (2-3 hours)
1. Add K-loop for matmul accumulation
2. Add proper matmul tile operations
3. Fix CB wait/reserve/pop patterns
4. Update runtime args to include Mt, Kt, Nt

**Files:**
- `src/target/tt/codegen_tt.cc` - `EmitTTComputeKernel()`

#### Phase 2: Fix Reader/Writer Kernels (2-3 hours)
1. Implement proper tile indexing for matmul
2. Read A[m,k] and B[k,n] tiles correctly
3. Add DRAM address calculations
4. Update runtime args

**Files:**
- `src/target/tt/codegen_tt.cc` - `EmitTTReaderKernel()`, `EmitTTWriterKernel()`

#### Phase 3: Fix Host Code (3-4 hours)
1. Replace mock APIs with real Metalium structure
2. Add proper buffer creation (InterleavedBufferConfig)
3. Add SetRuntimeArgs for each kernel
4. Add work distribution for multi-core
5. Fix CB configuration

**Files:**
- `src/target/tt/codegen_tt.cc` - `EmitTTHostProgram()`

#### Phase 4: Extract Actual Buffer Dimensions (1-2 hours)
1. Read buffer shapes from PrimFunc params
2. Calculate Mt, Kt, Nt from actual dimensions
3. Pass to all codegen functions
4. Update all hardcoded dimensions

**Files:**
- `src/target/tt/codegen_tt.cc` - All emit functions

#### Phase 5: Testing & Validation (2 hours)
1. Update existing tests with correct expectations
2. Add new tests for matmul correctness
3. Verify generated code compiles (dry-run)
4. Check against real Metalium examples

**Files:**
- `testing/python/tt/test_ws7_api_correctness.py` (new)
- Update `test_ws4_codegen.py`, `test_ws5_reader_writer.py`, `test_ws6_host_program.py`

---

## Success Criteria

Phase 2 complete when:
- [ ] Compute kernel has K-loop matching matmul algorithm
- [ ] Reader kernel loads correct A[m,k] and B[k,n] tiles
- [ ] Runtime args include buffer addresses and dimensions
- [ ] Host code uses real Metalium API structure
- [ ] CB sizes match Metalium examples
- [ ] Buffer dimensions read from TileLang code
- [ ] Generated code compiles (dry-run)
- [ ] All tests passing (40+ tests expected)

---

## Reference Examples

### Metalium Matmul Structure
From https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_single_core.html

**Compute Kernel:**
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

**Host Code:**
```cpp
auto reader_kernel = CreateKernel(
    program, "reader.cpp", core,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default
    }
);

SetRuntimeArgs(program, reader_kernel, core, {
    src0_buffer->address(),
    src1_buffer->address(),
    Mt, Kt, Nt,
    start_tile_id,
    num_tiles_per_core
});
```

---

## Timeline

- **Phase 1:** 2-3 hours
- **Phase 2:** 2-3 hours
- **Phase 3:** 3-4 hours
- **Phase 4:** 1-2 hours
- **Phase 5:** 2 hours

**Total:** 10-14 hours

---

## Next Steps

1. Create WS7 status document
2. Implement Phase 1 (Compute kernel K-loop)
3. Implement Phase 2 (Reader/writer tile indexing)
4. Implement Phase 3 (Host APIs)
5. Implement Phase 4 (Buffer dimensions)
6. Test and validate
7. Create PR and merge

---
