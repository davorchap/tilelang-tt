# Workstream 6 Status - Host Program Generation

**Last Updated:** 2025-10-07

## Overview

Workstream 6 focuses on generating the host-side C++ program that:
- Sets up TT device, program, and command queue
- Configures circular buffers for data movement
- Compiles and loads reader/compute/writer kernels
- Allocates DRAM buffers and transfers data
- Launches kernels with runtime arguments
- Reads back results for verification

**Goal:** Generate complete host program for dry-run execution of TT kernels.

**Status:** ⏳ **NOT STARTED** - Next priority after WS5

## Progress Summary

| Task | Status | Priority | Blocker |
|------|--------|----------|---------|
| **WS6 Documentation Structure** | ✅ **COMPLETE** | High | None |
| **Host Program Architecture** | ❌ TODO | **Critical** | None |
| **EmitTTHostProgram Implementation** | ❌ TODO | **Critical** | Architecture |
| **Circular Buffer Config Generation** | ❌ TODO | **Critical** | Architecture |
| **Kernel Compilation & Loading** | ❌ TODO | High | Implementation |
| **DRAM Buffer Management** | ❌ TODO | High | Implementation |
| **Runtime Args & Launch** | ❌ TODO | High | Implementation |
| **Integration Tests** | ❌ TODO | High | Implementation |

**Overall WS6 Progress:** 10% (Planning only)

---

## Implementation Plan

### Phase 1: Host Program Architecture

**File:** `src/target/tt/codegen_tt.cc` (extend existing)

**Objective:** Generate host program that orchestrates the 3-kernel pipeline.

**Host Program Template:**
```cpp
// Host Program (main.cpp)
#include <cstdint>
#include <vector>
#include <iostream>

// Mock TT device APIs for dry-run
class Device {
public:
    static Device* Instance() { static Device dev; return &dev; }
};

class Program {
public:
    void AddKernel(const std::string& name, const std::string& source) {}
    void Build() {}
};

class CommandQueue {
public:
    void EnqueueProgram(Program* prog, bool blocking) {}
    void Finish() {}
};

class CircularBufferConfig {
public:
    CircularBufferConfig(uint32_t cb_id, uint32_t tile_size, uint32_t num_pages) {}
};

// Buffer allocation
template<typename T>
std::vector<T> AllocateDRAMBuffer(size_t num_elements) {
    return std::vector<T>(num_elements);
}

int main() {
    // 1. Device setup
    Device* device = Device::Instance();

    // 2. Circular buffer configuration
    constexpr uint32_t TILE_SIZE = 32 * 32 * sizeof(uint16_t);  // fp16
    CircularBufferConfig cb_a(0, TILE_SIZE, 2);  // CB_A: double buffering
    CircularBufferConfig cb_b(1, TILE_SIZE, 2);  // CB_B: double buffering
    CircularBufferConfig cb_c(2, TILE_SIZE, 2);  // CB_C: double buffering

    // 3. Create program
    Program program;

    // 4. Load kernels (would be loaded from generated .cpp files)
    // program.AddKernel("reader", reader_kernel_source);
    // program.AddKernel("compute", compute_kernel_source);
    // program.AddKernel("writer", writer_kernel_source);
    program.Build();

    // 5. Allocate DRAM buffers
    constexpr uint32_t M = 256, N = 256, K = 256;
    auto dram_a = AllocateDRAMBuffer<uint16_t>(M * K);
    auto dram_b = AllocateDRAMBuffer<uint16_t>(K * N);
    auto dram_c = AllocateDRAMBuffer<uint16_t>(M * N);

    // Initialize input data (example)
    for (size_t i = 0; i < dram_a.size(); ++i) dram_a[i] = static_cast<uint16_t>(i % 256);
    for (size_t i = 0; i < dram_b.size(); ++i) dram_b[i] = static_cast<uint16_t>(i % 256);

    // 6. Set runtime arguments
    constexpr uint32_t GRID_X = 8, GRID_Y = 8;
    constexpr uint32_t NUM_CORES = 64;
    constexpr uint32_t TILES_PER_CORE = 1;

    // 7. Create command queue and launch
    CommandQueue cq;
    cq.EnqueueProgram(&program, true);
    cq.Finish();

    // 8. Read back results
    std::cout << "Kernel execution complete. Results in dram_c." << std::endl;

    return 0;
}
```

**Key Design Decisions:**
- **Dry-Run Focus**: Mock TT device APIs, no actual hardware calls
- **Circular Buffers**: Fixed depth of 2 for double buffering
- **Buffer Layout**: Row-major DRAM allocation
- **Kernel Args**: Pass grid dimensions and tile assignments
- **Synchronization**: Sequential kernel launch (reader → compute → writer)

**Implementation Steps:**
1. Add `EmitTTHostProgram()` function
2. Read buffer shapes and tile counts from WS2 metadata
3. Generate CB config code
4. Generate DRAM buffer allocation
5. Generate kernel launch code with runtime args
6. Emit result readback code

---

### Phase 2: Circular Buffer Configuration

**Objective:** Generate CB allocation code based on buffer metadata.

**CB Config Code Template:**
```cpp
// Circular Buffer Configuration
constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;
constexpr uint32_t TILE_SIZE_FP16 = TILE_H * TILE_W * sizeof(uint16_t);

// CB indices from kernel code
constexpr uint32_t CB_A = 0;
constexpr uint32_t CB_B = 1;
constexpr uint32_t CB_C = 2;

// Double buffering (2 pages per CB)
constexpr uint32_t CB_NUM_PAGES = 2;

CircularBufferConfig cb_config_a = {
    .cb_id = CB_A,
    .tile_size = TILE_SIZE_FP16,
    .num_pages = CB_NUM_PAGES
};

CircularBufferConfig cb_config_b = {
    .cb_id = CB_B,
    .tile_size = TILE_SIZE_FP16,
    .num_pages = CB_NUM_PAGES
};

CircularBufferConfig cb_config_c = {
    .cb_id = CB_C,
    .tile_size = TILE_SIZE_FP16,
    .num_pages = CB_NUM_PAGES
};
```

**Implementation Steps:**
1. Add `EmitTTCBConfig()` helper function
2. Read buffer dtypes from WS2 metadata (determine tile size)
3. Generate CB config structs
4. Emit CB index constants matching kernel code

---

### Phase 3: Kernel Compilation & Loading

**Objective:** Generate code to compile and load the 3 kernels.

**Kernel Loading Template:**
```cpp
// Load kernel sources
extern const char* reader_kernel_source;
extern const char* compute_kernel_source;
extern const char* writer_kernel_source;

Program program;

// Add kernels to program
program.AddKernel("reader_kernel_A", reader_kernel_source);
program.AddKernel("reader_kernel_B", reader_kernel_source);
program.AddKernel("compute_kernel", compute_kernel_source);
program.AddKernel("writer_kernel_C", writer_kernel_source);

// Compile program
program.Build();
```

**Implementation Steps:**
1. Embed kernel sources as string literals
2. Generate program creation code
3. Add all 3 kernels to program
4. Generate build/compile call

---

### Phase 4: DRAM Buffer Management

**Objective:** Generate DRAM buffer allocation and initialization code.

**DRAM Buffer Template:**
```cpp
// Calculate buffer sizes from metadata
constexpr uint32_t M = {grid_y} * TILE_H;
constexpr uint32_t N = {grid_x} * TILE_W;
constexpr uint32_t K = {grid_y} * TILE_H;  // Assume square for now

// Allocate DRAM buffers
std::vector<uint16_t> dram_a(M * K);
std::vector<uint16_t> dram_b(K * N);
std::vector<uint16_t> dram_c(M * N);

// Initialize input data
for (size_t i = 0; i < dram_a.size(); ++i) {
    dram_a[i] = static_cast<uint16_t>(i % 256);
}
for (size_t i = 0; i < dram_b.size(); ++i) {
    dram_b[i] = static_cast<uint16_t>(i % 256);
}
```

**Implementation Steps:**
1. Read buffer shapes from PrimFunc parameters
2. Calculate total buffer sizes
3. Generate allocation code
4. Add optional initialization code

---

### Phase 5: Runtime Args & Kernel Launch

**Objective:** Generate kernel launch code with correct runtime arguments.

**Launch Template:**
```cpp
// Runtime arguments for persistent kernels
struct RuntimeArgs {
    uint32_t start_id;
    uint32_t count;
    uint32_t grid_x;
    uint32_t grid_y;
    void* dram_addr_a;
    void* dram_addr_b;
    void* dram_addr_c;
};

// Launch kernels
CommandQueue cq;

// For each core (simplified - single-core dry-run)
RuntimeArgs args = {
    .start_id = 0,
    .count = {num_tiles},
    .grid_x = {grid_x},
    .grid_y = {grid_y},
    .dram_addr_a = dram_a.data(),
    .dram_addr_b = dram_b.data(),
    .dram_addr_c = dram_c.data()
};

// Enqueue program (blocking for simplicity)
cq.EnqueueProgram(&program, true);
cq.Finish();
```

**Implementation Steps:**
1. Read tt_tiles_per_core from WS2 metadata
2. Generate RuntimeArgs struct
3. Populate args from metadata
4. Generate launch call

---

### Phase 6: Integration with Codegen

**File:** `src/target/tt/codegen_tt.cc`

**Update `CodegenTT()` to emit host program:**
```cpp
std::unordered_map<std::string, std::string> CodegenTT(const IRModule& mod, const std::string& target) {
    std::unordered_map<std::string, std::string> artifacts;

    PrimFunc main_func = /* ... */;

    // Generate all 3 kernels
    artifacts["reader.cpp"] = EmitTTReaderKernel(main_func);
    artifacts["compute.cpp"] = EmitTTComputeKernel(main_func);
    artifacts["writer.cpp"] = EmitTTWriterKernel(main_func);

    // Generate host program
    artifacts["main.cpp"] = EmitTTHostProgram(main_func);

    // Generate metadata
    artifacts["tt.plan.json"] = EmitTTPlanJSON(main_func);

    return artifacts;
}
```

---

## Testing Strategy

### Unit Tests

**File:** `testing/python/tt/test_ws6_host_program.py`

**Test Cases:**
1. **test_emit_host_program_basic()**
   - Verify main.cpp generated
   - Check device setup code present
   - Validate program structure

2. **test_cb_config_generation()**
   - Verify CB config code present
   - Check CB indices match kernel code
   - Validate tile sizes

3. **test_dram_buffer_allocation()**
   - Verify buffer allocation code
   - Check buffer sizes match metadata
   - Validate initialization

4. **test_kernel_launch_args()**
   - Verify RuntimeArgs struct
   - Check start_id/count from WS2
   - Validate grid dimensions

5. **test_full_host_program_structure()**
   - Verify all sections present
   - Check compilation (dry-run)
   - Validate end-to-end structure

---

## Build & Test Instructions

```bash
# Rebuild with host program generation
USE_LLVM=true pip install -e . --no-build-isolation

# Run WS6 tests
pytest testing/python/tt/test_ws6_host_program.py -v
```

---

## Dependencies

- **WS1-5 complete** ✓
- **TVM C++ infrastructure** (codegen framework)
- **Mock TT device APIs** (for dry-run compilation)

---

## Success Criteria

WS6 is complete when:
- [ ] Host program generation working
- [ ] CB config code generated correctly
- [ ] DRAM buffer management code present
- [ ] Kernel launch with correct args
- [ ] 5+ integration tests passing
- [ ] No regressions in existing 28 tests

---

## Key Design Principles

1. **Dry-Run Focus**: Generate compilable C++ without hardware dependencies
2. **Mock APIs**: Use mock TT device/program/queue classes
3. **Single-Core MVP**: Simplify to single-core execution for Phase 0
4. **Extensibility**: Design for future multi-core and real hardware integration

---

## Timeline

**Estimated Effort:** 4-6 hours

- **Hour 1-2**: Implement EmitTTHostProgram() skeleton
- **Hour 3**: CB config and buffer management
- **Hour 4**: Kernel loading and launch
- **Hour 5-6**: Testing and validation

---

## Related Documentation

- [WS5 Status](../workstream5/WS5_STATUS.md) - Reader/Writer kernels (foundation)
- [WS4 Status](../workstream4/WS4_STATUS.md) - Compute kernel
- [Metalium Guide](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md)

---
