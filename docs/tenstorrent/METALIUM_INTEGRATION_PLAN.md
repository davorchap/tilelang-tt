# Metalium API Integration Plan

**Date**: 2025-10-08
**Status**: Planning Phase
**Version**: 1.0
**Context**: Post IR-Driven Codegen Migration (Tasks 1-6 complete, 95 tests passing)

---

## Executive Summary

This document outlines the plan to replace mock TT-Metalium APIs with real hardware APIs, enabling actual execution on Tenstorrent devices (Grayskull/Wormhole). This is the next major milestone after completing IR-driven codegen.

**Current State**: Dry-run only with mock APIs (void functions)
**Target State**: Hardware execution with real Metalium runtime

**Timeline**: Weeks 16-18 (3 weeks estimated)

---

## 1. Current Mock API Surface

### 1.1 Kernel APIs (Generated Code)

Our current IR-driven codegen generates the following mock APIs in kernel code:

#### **Runtime Arguments**
```cpp
// MOCK (Current):
template <typename T>
inline T get_arg_val(uint32_t idx) { return T(); }

// REAL (Target):
#include "ckernel_include.h"  // Provides real get_arg_val
```

#### **Circular Buffer Operations**
```cpp
// MOCK (Current):
inline void cb_wait_front(uint32_t cb_id, uint32_t n_tiles) {}
inline void cb_reserve_back(uint32_t cb_id, uint32_t n_tiles) {}
inline void cb_pop_front(uint32_t cb_id, uint32_t n_tiles) {}
inline void cb_push_back(uint32_t cb_id, uint32_t n_tiles) {}

// REAL (Target):
#include "dataflow_api.h"  // Provides real CB functions
// cb_wait_front(cb_id, n_tiles);     // Blocks until n_tiles available
// cb_reserve_back(cb_id, n_tiles);   // Reserves space for writing
// cb_push_back(cb_id, n_tiles);      // Makes tiles available for consumer
// cb_pop_front(cb_id, n_tiles);      // Releases tiles after consumption
```

#### **NOC Operations**
```cpp
// MOCK (Current):
inline void noc_async_read_tile(uint32_t tile_idx, uint32_t base_addr, uint32_t l1_addr) {}
inline void noc_async_read_barrier() {}
inline void noc_async_write_tile(uint32_t tile_idx, uint32_t l1_addr, uint32_t base_addr) {}
inline void noc_async_write_barrier() {}

// REAL (Target):
#include "dataflow_api.h"
// noc_async_read_tile(tile_idx, dram_noc_addr, l1_write_addr);
// noc_async_read_barrier();  // Wait for all pending reads
// noc_async_write_tile(tile_idx, l1_read_addr, dram_noc_addr);
// noc_async_write_barrier();  // Wait for all pending writes
```

#### **Matmul Intrinsics**
```cpp
// MOCK (Current):
inline void matmul_tiles_init(uint32_t cb_a, uint32_t cb_b, uint32_t cb_c) {}
inline void matmul_tiles(uint32_t cb_a, uint32_t cb_b, uint32_t cb_c, bool accumulate) {}

// REAL (Target):
#include "compute_kernel_api/matmul.h"
// matmul_tiles_init(cb_a, cb_b, cb_c);
// matmul_tiles(cb_a, cb_b, cb_c, ntiles, transpose);  // Note: Different signature!
```

### 1.2 Host APIs (Not Yet Generated)

Currently, we generate a mock host program structure. We need to replace this with real Metalium host APIs:

```cpp
// MOCK (Current): Empty template structure
// REAL (Target): Actual Metalium device/program management

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/program/program.hpp"

using namespace tt::tt_metal;

// Device initialization
auto device = MeshDevice::create_unit_mesh(/*device_id*/0);
CommandQueue& cq = device->mesh_command_queue(/*cq_id*/0);

// Program creation
Program program = CreateProgram();

// Buffer allocation
DeviceLocalBufferConfig dram_config{
    .page_size = tile_size_bytes,
    .buffer_type = BufferType::DRAM
};
ReplicatedBufferConfig buffer_config{
    .size = n_tiles * tile_size_bytes
};
auto a = MeshBuffer::create(buffer_config, dram_config, device.get());
auto b = MeshBuffer::create(buffer_config, dram_config, device.get());
auto c = MeshBuffer::create(buffer_config, dram_config, device.get());

// Circular buffer configuration
CBHandle cb_in0 = CreateCircularBuffer(
    program, core,
    CircularBufferConfig(
        /*total_size*/tiles_per_cb * tile_size_bytes,
        /*data_format_spec*/{{cb_in0_index, tt::DataFormat::Float16_b}})
    .set_page_size(cb_in0_index, tile_size_bytes)
);

// Kernel creation
auto reader = CreateKernel(
    program,
    "path/to/reader_kernel.cpp",
    core,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = reader_args
    }
);

// Program execution
EnqueueProgram(cq, program, /*blocking*/false);
```

---

## 2. API Mapping Specification

### 2.1 Kernel-Side Mapping

| Mock API | Real Metalium API | Header | Notes |
|----------|-------------------|--------|-------|
| `get_arg_val<T>(idx)` | `get_arg_val<T>(idx)` | `ckernel_include.h` | Same signature! |
| `cb_wait_front(id, n)` | `cb_wait_front(id, n)` | `dataflow_api.h` | Same signature! |
| `cb_reserve_back(id, n)` | `cb_reserve_back(id, n)` | `dataflow_api.h` | Same signature! |
| `cb_push_back(id, n)` | `cb_push_back(id, n)` | `dataflow_api.h` | Same signature! |
| `cb_pop_front(id, n)` | `cb_pop_front(id, n)` | `dataflow_api.h` | Same signature! |
| `noc_async_read_tile(...)` | `noc_async_read_tile(...)` | `dataflow_api.h` | Check param order! |
| `noc_async_read_barrier()` | `noc_async_read_barrier()` | `dataflow_api.h` | Same signature! |
| `noc_async_write_tile(...)` | `noc_async_write_tile(...)` | `dataflow_api.h` | Check param order! |
| `noc_async_write_barrier()` | `noc_async_write_barrier()` | `dataflow_api.h` | Same signature! |
| `matmul_tiles_init(a, b, c)` | `matmul_tiles_init(a, b, c)` | `compute_kernel_api/matmul.h` | Same signature! |
| `matmul_tiles(a, b, c, accum)` | `matmul_tiles(a, b, c, ntiles, transpose)` | `compute_kernel_api/matmul.h` | **DIFFERENT!** Need adaptation |

**Key Insight**: Most signatures match! We can replace mock inline functions with `#include` directives.

**Critical Difference**: `matmul_tiles` signature differs:
- Mock: `matmul_tiles(cb_a, cb_b, cb_c, bool accumulate)`
- Real: `matmul_tiles(cb_a, cb_b, cb_c, uint32_t ntiles, bool transpose)`

**Action Required**: Update IR-driven compute visitor to generate correct matmul_tiles calls.

### 2.2 Host-Side Mapping

| Functionality | Mock API | Real Metalium API |
|---------------|----------|-------------------|
| Device Init | Not generated | `MeshDevice::create_unit_mesh(device_id)` |
| Command Queue | Not generated | `device->mesh_command_queue(cq_id)` |
| Program Creation | Not generated | `CreateProgram()` |
| Buffer Allocation | Not generated | `MeshBuffer::create(buffer_config, dram_config, device)` |
| CB Configuration | Not generated | `CreateCircularBuffer(program, core, CircularBufferConfig(...))` |
| Kernel Creation | Not generated | `CreateKernel(program, path, core, DataMovementConfig{...})` |
| Runtime Args | Not generated | `SetRuntimeArgs(program, kernel, core, args)` |
| Execution | Not generated | `EnqueueProgram(cq, program, blocking)` |
| Cleanup | Not generated | `device->close()` |

---

## 3. Integration Strategy

### 3.1 Phase 1: Kernel Headers (Week 16)

**Goal**: Replace mock inline functions with real Metalium headers.

**Tasks**:
1. **Remove mock API definitions** from codegen output
2. **Add real header includes**:
   ```cpp
   // Reader/Writer kernels (dataflow):
   #include "dataflow_api.h"

   // Compute kernels:
   #include "ckernel_include.h"
   #include "compute_kernel_api/matmul.h"
   ```
3. **Update matmul_tiles calls** to use correct signature
4. **Add get_write_ptr/get_read_ptr** for CB address retrieval

**Files to Modify**:
- `src/target/tt/codegen_tt_compute_visitor.cc` - Update preamble generation
- `src/target/tt/codegen_tt_reader_visitor.cc` - Update preamble generation
- `src/target/tt/codegen_tt_writer_visitor.cc` - Update preamble generation
- `src/target/tt/codegen_tt.cc` - Update template-based versions

**Testing Strategy**:
- Unit tests still pass (headers don't affect dry-run)
- Generated code should compile against Metalium headers (if available)

### 3.2 Phase 2: Host Program Generation (Week 17)

**Goal**: Generate real Metalium host program instead of mock structure.

**Tasks**:
1. **Update EmitTTHostProgram()** in `codegen_tt.cc`:
   - Generate device initialization
   - Generate buffer allocation (DRAM interleaved)
   - Generate CB configuration (based on WS2 metadata)
   - Generate kernel creation (reader, compute, writer)
   - Generate runtime argument setup
   - Generate program enqueueing

2. **Extract metadata from IRModule**:
   - Buffer dimensions (from PrimFunc params)
   - Grid dimensions (from `tt_grid_x`, `tt_grid_y`)
   - Core assignments (from `tt_tiles_per_core`)
   - Runtime args schema

3. **Generate proper CB configuration**:
   ```cpp
   // Extract from WS2 metadata:
   // - CB_A size (tiles_per_cb_a)
   // - CB_B size (tiles_per_cb_b)
   // - CB_C size (tiles_per_cb_c)
   // - Data format (Float16_b from buffer dtype)
   ```

**Files to Modify**:
- `src/target/tt/codegen_tt.cc` - EmitTTHostProgram()
- Add new helpers: `GenerateCBConfig()`, `GenerateKernelConfig()`

**Testing Strategy**:
- Generated host.cpp should compile with Metalium headers
- Validate CB sizes match buffer requirements
- Validate runtime args match kernel expectations

### 3.3 Phase 3: Build System Integration (Week 18)

**Goal**: Link against real Metalium libraries, support both mock and real modes.

**Tasks**:
1. **Add CMake FindMetalium.cmake**:
   ```cmake
   find_package(Metalium)
   if (Metalium_FOUND)
       option(USE_REAL_METALIUM "Use real Metalium APIs" ON)
   else()
       option(USE_REAL_METALIUM "Use real Metalium APIs" OFF)
   endif()
   ```

2. **Add compile definitions**:
   ```cmake
   if (USE_REAL_METALIUM)
       target_compile_definitions(tilelang PRIVATE TL_USE_REAL_METALIUM)
       target_link_libraries(tilelang PRIVATE tt_metal)
   endif()
   ```

3. **Conditional codegen**:
   ```cpp
   #ifdef TL_USE_REAL_METALIUM
     // Generate real includes
     code << "#include \"dataflow_api.h\"\n";
   #else
     // Generate mock APIs
     code << "inline void cb_wait_front(...) {}\n";
   #endif
   ```

**Files to Modify**:
- `CMakeLists.txt` - Add Metalium support
- `cmake/FindMetalium.cmake` - New file
- `src/target/tt/codegen_tt.cc` - Add conditional compilation

**Testing Strategy**:
- Mock mode: All existing tests pass
- Real mode: Host program compiles and links (if Metalium installed)

---

## 4. Dependencies

### 4.1 Required Metalium Components

**Headers**:
- `tt_metal/host_api.hpp` - Host-side device/program APIs
- `tt_metal/impl/device/device.hpp` - Device management
- `tt_metal/impl/program/program.hpp` - Program creation
- `dataflow_api.h` - Kernel-side dataflow APIs
- `ckernel_include.h` - Kernel-side compute APIs
- `compute_kernel_api/matmul.h` - Matmul intrinsics

**Libraries**:
- `libtt_metal.so` - Main Metalium runtime
- `libdevice.so` - Device management
- Additional architecture-specific libs (Grayskull/Wormhole)

**Build Requirements**:
- C++17 compiler
- CMake 3.16+
- Metalium SDK installed (TT_METAL_HOME environment variable)

### 4.2 Version Compatibility

**Target Metalium Version**: v0.51.0+ (latest stable as of 2025-10-08)

**Compatibility Strategy**:
- Pin to specific Metalium version in CI
- Use `#if TT_METAL_VERSION >= X` for version-specific code
- Document minimum required version in README

---

## 5. Implementation Phases

### Week 16: Kernel Header Integration

**Milestone**: Generated kernels use real Metalium headers

**Tasks**:
- [x] Document current mock API surface (this document)
- [ ] Update compute visitor preamble generation
- [ ] Update reader visitor preamble generation
- [ ] Update writer visitor preamble generation
- [ ] Fix matmul_tiles signature
- [ ] Add get_write_ptr/get_read_ptr calls
- [ ] Update tests to check for headers instead of mock APIs
- [ ] Verify generated code structure

**Deliverables**:
- Kernels include real headers
- All 95 tests still pass
- Generated code ready for compilation (when Metalium available)

### Week 17: Host Program Generation

**Milestone**: Generated host.cpp creates real Metalium device/program

**Tasks**:
- [ ] Implement GenerateDeviceInit()
- [ ] Implement GenerateBufferAlloc()
- [ ] Implement GenerateCBConfig()
- [ ] Implement GenerateKernelConfig()
- [ ] Implement GenerateRuntimeArgs()
- [ ] Implement GenerateProgramEnqueue()
- [ ] Update EmitTTHostProgram()
- [ ] Add metadata extraction helpers
- [ ] Create test suite for host program generation

**Deliverables**:
- Host program generates real Metalium code
- Metadata correctly extracted from IRModule
- Tests validate host program structure

### Week 18: Build System & Conditional Compilation

**Milestone**: Build system supports both mock and real Metalium

**Tasks**:
- [ ] Create cmake/FindMetalium.cmake
- [ ] Update CMakeLists.txt with USE_REAL_METALIUM option
- [ ] Add TL_USE_REAL_METALIUM compile definition
- [ ] Update codegen to support conditional compilation
- [ ] Add CI job for real Metalium build (if hardware available)
- [ ] Document build process in README
- [ ] Create example: building with/without Metalium

**Deliverables**:
- USE_REAL_METALIUM=ON/OFF builds successfully
- Mock mode: All tests pass (existing behavior)
- Real mode: Code compiles against Metalium headers
- Documentation updated

---

## 6. Testing Strategy

### 6.1 Unit Tests (Mock Mode)

**Existing Tests** (should continue to pass):
- 12 visitor base tests
- 4 compute visitor tests
- 1 reader visitor test
- 1 writer visitor test
- 18 test updates (WS4-6 integration)
- Total: **95 tests passing**

**New Tests** (API integration):
- Header generation tests
- Host program generation tests
- Metadata extraction tests
- Estimated: **+15 tests**

### 6.2 Integration Tests (Real Mode)

**Prerequisite**: Metalium SDK installed, hardware available

**Test Progression**:
1. **Compilation Test**: Generated code compiles
2. **Simulator Test**: Run on TT simulator (if available)
3. **Hardware Test**: Run on Grayskull/Wormhole device

**Test Cases**:
- Simple matmul (64x64)
- Medium matmul (256x256)
- Large matmul (1024x1024)
- Verify correctness against reference CPU implementation

### 6.3 CI Strategy

**Mock Mode CI** (existing):
- Runs on every PR
- Ubuntu + LLVM backend
- All 95+ tests must pass

**Real Mode CI** (new):
- Runs when Metalium SDK available
- Either: Self-hosted runner with hardware
- Or: Skip with clear message "Metalium not available"

---

## 7. Risk Mitigation

### Risk 1: API Signature Mismatches

**Risk**: Real Metalium APIs differ from our mock assumptions

**Mitigation**:
- Incremental integration (headers first, then host APIs)
- Consult Metalium examples before implementing
- Version pin Metalium to avoid breaking changes

**Fallback**: Keep mock mode for development

### Risk 2: Hardware Access

**Risk**: No access to Grayskull/Wormhole hardware for testing

**Mitigation**:
- Develop against simulator if available
- Coordinate with Tenstorrent team for cloud access
- Use mock mode for initial development

**Fallback**: Validate on simulator, defer hardware testing

### Risk 3: Build Complexity

**Risk**: Metalium SDK integration breaks existing builds

**Mitigation**:
- USE_REAL_METALIUM=OFF by default
- Metalium optional dependency
- Clear error messages when Metalium not found

**Fallback**: Always support mock mode

### Risk 4: Performance Issues

**Risk**: Initial hardware execution is slow

**Mitigation**:
- Profile early and often
- Start with small workloads
- Consult Tenstorrent optimization guides

**Fallback**: Document known limitations, optimize in future phases

---

## 8. Success Criteria

### Week 16 Milestone:
- [ ] All kernels use real Metalium headers (no mock inline functions)
- [ ] Generated code structure matches Metalium examples
- [ ] All 95 existing tests pass
- [ ] +15 new header generation tests pass

### Week 18 Milestone:
- [ ] Host program generates real Metalium device setup
- [ ] Build system supports USE_REAL_METALIUM option
- [ ] Mock mode: All tests pass
- [ ] Real mode: Code compiles (if Metalium installed)
- [ ] Documentation complete

### Future Milestone (Weeks 19-22 - Hardware Execution):
- [ ] Matmul executes correctly on hardware
- [ ] Achieves >50% of theoretical peak TFLOPS
- [ ] All tests pass with real runtime

---

## 9. References

### Metalium Documentation
- **Main Guide**: https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md
- **API Reference**: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/index.html
- **GitHub Repo**: https://github.com/tenstorrent/tt-metal

### TileLang Documentation
- **Next Steps**: `docs/tenstorrent/NEXT_STEPS.md`
- **Unified Plan**: `docs/tenstorrent/UNIFIED_MATMUL_MVP_PLAN.md`
- **IR-Driven Codegen**: `docs/tenstorrent/IR_DRIVEN_CODEGEN_PLAN.md`

### Related Issues
- None yet (this is the planning phase)

---

## 10. Document History

- **2025-10-08**: Initial planning document created
- **Version 1.0**: Complete API mapping and integration strategy

---

## Appendix A: Example Generated Code Comparison

### Current (Mock APIs):

```cpp
// compute.cpp (mock)
template <typename T>
inline T get_arg_val(uint32_t idx) { return T(); }

inline void cb_wait_front(uint32_t cb_id, uint32_t n_tiles) {}
inline void matmul_tiles(uint32_t cb_a, uint32_t cb_b, uint32_t cb_c, bool accumulate) {}

void MAIN() {
    uint32_t kt = get_arg_val<uint32_t>(2);
    cb_wait_front(CB_A, 1);
    matmul_tiles(CB_A, CB_B, CB_C, true);
}
```

### Target (Real Metalium):

```cpp
// compute.cpp (real)
#include "ckernel_include.h"
#include "compute_kernel_api/matmul.h"

void MAIN() {
    uint32_t kt = get_arg_val<uint32_t>(2);

    cb_wait_front(CB_A, 1);  // Real API from ckernel_include.h
    matmul_tiles_init(CB_A, CB_B, CB_C);
    matmul_tiles(CB_A, CB_B, CB_C, /*ntiles*/1, /*transpose*/false);
}
```

---

## Appendix B: CMake Integration Example

```cmake
# CMakeLists.txt additions

# Find Metalium SDK
set(TT_METAL_HOME $ENV{TT_METAL_HOME})
if (TT_METAL_HOME)
    message(STATUS "TT-Metalium found: ${TT_METAL_HOME}")
    option(USE_REAL_METALIUM "Use real Metalium APIs" ON)
else()
    message(STATUS "TT-Metalium not found, using mock APIs")
    option(USE_REAL_METALIUM "Use real Metalium APIs" OFF)
endif()

# Add Metalium support
if (USE_REAL_METALIUM)
    add_definitions(-DTL_USE_REAL_METALIUM)

    include_directories(
        ${TT_METAL_HOME}/tt_metal
        ${TT_METAL_HOME}/tt_metal/impl
    )

    link_directories(${TT_METAL_HOME}/build/lib)

    target_link_libraries(tilelang PRIVATE tt_metal device)

    message(STATUS "Building with REAL Metalium APIs")
else()
    message(STATUS "Building with MOCK Metalium APIs (dry-run only)")
endif()
```

---

**End of Integration Plan**
