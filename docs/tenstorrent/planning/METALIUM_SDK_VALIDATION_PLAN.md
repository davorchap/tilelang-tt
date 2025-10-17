# TT-Metalium SDK Validation Plan

**Version**: 1.0
**Date**: 2025-10-08
**Status**: Ready for SDK Access
**Context**: Post Week 16-18 (Kernel Headers + Host Program + CMake Integration)

---

## Executive Summary

This document outlines the **validation plan** for integrating TileLang's Tenstorrent backend with the **real TT-Metalium SDK**.

**Current State**: Code generates correct API structure based on documentation research (Weeks 16-18 complete)
**Next State**: Validate against actual SDK, fix discrepancies, enable hardware execution
**Prerequisites**: Access to TT-Metalium SDK installation (TT_METAL_HOME)

---

## Table of Contents

1. [API Gaps Analysis](#api-gaps-analysis)
2. [Validation Phases](#validation-phases)
3. [Known Issues](#known-issues)
4. [Testing Strategy](#testing-strategy)
5. [Success Criteria](#success-criteria)

---

## API Gaps Analysis

### ✅ Correct (85% Confidence)

Based on docs.tenstorrent.com and github.com/tenstorrent/tt-metal research:

1. **Kernel Creation**:
```cpp
// Our generated code matches real API
auto kernel = CreateKernel(
    program,
    "kernel.cpp",
    core,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default
    }
);
```

2. **Circular Buffer Configuration**:
```cpp
// Matches documented pattern
CircularBufferConfig(size, {{cb_index, DataFormat::Float16_b}})
    .set_page_size(cb_index, tile_size);
```

3. **Program Execution**:
```cpp
// Correct API calls
EnqueueProgram(cq, program, /*blocking*/false);
Finish(cq);
```

### ⚠️ Minor Adjustments Needed

1. **Program Creation** (Low Priority):
```cpp
// Current (works but not idiomatic):
Program program = CreateProgram();

// Preferred (from examples):
Program program{};
```

2. **Namespace Prefixes** (Medium Priority):
```cpp
// Current:
MeshDevice::create_unit_mesh()
MeshBuffer::create()

// Correct:
distributed::MeshDevice::create_unit_mesh()
distributed::MeshBuffer::create()
```

**Impact**: Compilation errors when SDK available
**Fix**: Simple search/replace in `codegen_tt.cc`

### ❌ Critical Gaps (Missing Functionality)

1. **Data Transfer APIs** - Not Implemented:
```cpp
// MISSING: Write input data to device
EnqueueWriteBuffer(cq, buffer_a, host_data_a.data(), /*blocking*/false);
EnqueueWriteBuffer(cq, buffer_b, host_data_b.data(), /*blocking*/false);

// MISSING: Read output data from device
EnqueueReadBuffer(cq, buffer_c, result_data.data(), /*blocking*/true);
```

**Impact**: Cannot transfer data to/from hardware
**Priority**: HIGH (required for hardware execution)

2. **SetRuntimeArgs** - Only Placeholder:
```cpp
// Current: Comment placeholder
// SetRuntimeArgs(program, kernel, core, args);

// Need: Actual implementation
std::vector<uint32_t> compute_args = {
    start_tile_id,
    num_output_tiles,
    Kt
};
SetRuntimeArgs(program, compute_kernel, core, compute_args);
```

**Impact**: Kernels won't receive correct arguments
**Priority**: HIGH (required for hardware execution)

3. **Kernel File Paths** - Placeholder:
```cpp
// Current: Commented placeholders
// auto reader_kernel = CreateKernel(program, "reader.cpp", ...);

// Need: Actual file paths
std::string reader_path = GenerateKernelPath("reader.cpp");
auto reader_kernel = CreateKernel(program, reader_path, core, ...);
```

**Impact**: Kernels cannot be loaded
**Priority**: HIGH (required for hardware execution)

### ❓ Unverified (Need SDK Confirmation)

1. **Include Headers** - Best Guess:
```cpp
// Current (based on documentation):
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/mesh_device.hpp"
#include "tt_metal/impl/buffers/mesh_buffer.hpp"
#include "tt_metal/impl/program/program.hpp"

// May need adjustment based on actual SDK structure
```

**Impact**: Compilation errors
**Priority**: MEDIUM (will discover immediately)

2. **BufferType and DataFormat Enums**:
```cpp
// Assumed:
BufferType::DRAM
tt::DataFormat::Float16_b

// Need verification of exact names
```

---

## Validation Phases

### Phase 1: Dry-Run Compilation (Week 19) ⚠️ BLOCKED BY SDK ACCESS

**Prerequisites**:
- Access to TT-Metalium SDK (TT_METAL_HOME set)
- No hardware required

**CMake Configuration**:
The build system supports both mock and real SDK modes via `cmake/TTMetal.cmake`:
- **Mock mode** (default): `cmake -B build -DUSE_LLVM=ON`
  - Uses stub/mock TT APIs for dry-run testing
  - No SDK required, runs in CI
- **Real mode**: `cmake -B build -DUSE_LLVM=ON -DUSE_REAL_METALIUM=ON`
  - Links against real TT-Metalium SDK
  - Requires `TT_METAL_HOME` environment variable
  - Automatically discovers SDK via `cmake/FindMetalium.cmake`

**Tasks**:
1. Install Metalium SDK (see [METALIUM_SETUP_GUIDE.md](METALIUM_SETUP_GUIDE.md))
2. Set environment: `export TT_METAL_HOME=/path/to/tt-metal`
3. Build with real mode: `cmake -B build -DUSE_LLVM=ON -DUSE_REAL_METALIUM=ON`
4. Attempt compilation of generated artifacts
5. Fix include path errors (if SDK structure differs)
6. Fix namespace errors (e.g., `distributed::MeshDevice`)
7. Fix enum/type mismatches (e.g., `DataFormat`, `BufferType`)

**Expected Issues**:
- Include path adjustments
- Namespace corrections (`distributed::` prefix)
- Minor API signature differences

**Success Criteria**:
- ✅ Code compiles without errors
- ✅ Links against Metalium libraries
- ✅ Generated host program builds

**Estimated Effort**: 2-3 days

---

### Phase 2: API Completion (Week 20) ⚠️ BLOCKED BY PHASE 1

**Prerequisites**:
- Phase 1 complete (code compiles)
- Access to Metalium SDK

**Tasks**:
1. **Implement EnqueueWriteBuffer**:
   - Add to `EmitTTHostProgram()`
   - Write host data to DRAM buffers A, B
   - Before program execution

2. **Implement EnqueueReadBuffer**:
   - Add to `EmitTTHostProgram()`
   - Read result from DRAM buffer C
   - After program execution

3. **Implement SetRuntimeArgs**:
   - Extract runtime args from IR metadata
   - Generate correct arg vectors for each kernel
   - Call SetRuntimeArgs for reader, compute, writer

4. **Implement Kernel File Path Resolution**:
   - Save generated kernels to known paths
   - Pass actual file paths to CreateKernel
   - Verify kernels compile with Metalium compiler

**Example Implementation**:
```cpp
// In EmitTTHostProgram():
code << "    // Write input data to device\\n";
code << "    EnqueueWriteBuffer(cq, buffer_a, host_a.data(), false);\\n";
code << "    EnqueueWriteBuffer(cq, buffer_b, host_b.data(), false);\\n\\n";

code << "    // Execute program\\n";
code << "    EnqueueProgram(cq, program, false);\\n";
code << "    Finish(cq);\\n\\n";

code << "    // Read results\\n";
code << "    std::vector<uint16_t> result(M * N);\\n";
code << "    EnqueueReadBuffer(cq, buffer_c, result.data(), true);\\n";
```

**Success Criteria**:
- ✅ Complete data transfer pipeline
- ✅ Runtime args correctly configured
- ✅ Kernels load successfully

**Estimated Effort**: 3-4 days

---

### Phase 3: Hardware Execution (Week 21-22) ⚠️ BLOCKED BY PHASE 2 + HARDWARE ACCESS

**Prerequisites**:
- Phase 2 complete
- Access to Grayskull or Wormhole device

**Tasks**:
1. **Run Simple Matmul on Hardware**:
   - Start with 64×64 matmul (1 tile per dimension)
   - Verify device detection
   - Verify program compilation
   - Verify execution completes

2. **Verify Correctness**:
   - Compare hardware results with CPU reference
   - Check for numerical accuracy (FP16 tolerance)
   - Verify tile-aligned outputs

3. **Scale Testing**:
   - 128×128 matmul (4×4 tiles)
   - 256×256 matmul (8×8 tiles)
   - 512×512 matmul (16×16 tiles)

4. **Performance Profiling**:
   - Measure execution time
   - Compare with theoretical peak
   - Identify bottlenecks

**Success Criteria**:
- ✅ Matmul executes on hardware
- ✅ Results match CPU reference (within FP16 tolerance)
- ✅ Achieves >50% of theoretical peak performance

**Estimated Effort**: 5-7 days

---

## Known Issues

### Issue 1: Program Creation Syntax

**Severity**: Low
**Current Code**: `Program program = CreateProgram();`
**Correct Code**: `Program program{};`
**Fix**: Line 439 in `codegen_tt.cc`

### Issue 2: Missing Namespace Prefixes

**Severity**: Medium (compilation error)
**Locations**:
- Line 417: `MeshDevice::create_unit_mesh()` → `distributed::MeshDevice::create_unit_mesh()`
- Lines 478-480: `MeshBuffer::create()` → `distributed::MeshBuffer::create()`

**Fix**:
```cpp
// Update all occurrences in EmitTTHostProgram()
code << "    auto device = distributed::MeshDevice::create_unit_mesh(/*device_id*/0);\\n";
code << "    auto buffer_a = distributed::MeshBuffer::create(...);\\n";
```

### Issue 3: Conditional Compilation Inconsistency

**Severity**: Low
**Current**: Mix of runtime `#ifdef` in codegen
**Better**: Consistent pattern for all API differences

**Recommendation**: Create helper functions:
```cpp
std::string GetMeshDeviceCreate() {
#ifdef TL_USE_REAL_METALIUM
    return "distributed::MeshDevice::create_unit_mesh";
#else
    return "Device::Instance";
#endif
}
```

---

## Testing Strategy

### Unit Tests

1. **API Structure Validation**:
   - Parse generated host.cpp
   - Verify all required API calls present
   - Check correct order (device → buffers → program → execute)

2. **Conditional Compilation**:
   - Test both mock and real modes generate valid code
   - Verify `#ifdef TL_USE_REAL_METALIUM` branches work

### Integration Tests

1. **With SDK but No Hardware** (Phase 1):
   - Generated code compiles
   - Links against Metalium libraries
   - Can run dry-run (no-op execution)

2. **With SDK and Hardware** (Phase 3):
   - Simple matmul correctness
   - Multi-size scaling
   - Performance benchmarks

### Regression Tests

1. **Mock Mode Preservation**:
   - All 95 existing tests must pass
   - No changes to default behavior
   - Existing test behavior maintained

---

## Success Criteria

### Phase 1 (Dry-Run Compilation):
- [ ] Code compiles with `-DUSE_REAL_METALIUM=ON`
- [ ] No include path errors
- [ ] No namespace errors
- [ ] Links successfully against `libtt_metal.so`

### Phase 2 (API Completion):
- [ ] EnqueueWriteBuffer implemented and tested
- [ ] EnqueueReadBuffer implemented and tested
- [ ] SetRuntimeArgs generates correct arg vectors
- [ ] Kernel file paths resolved correctly
- [ ] Complete pipeline compiles

### Phase 3 (Hardware Execution):
- [ ] 64×64 matmul runs on hardware
- [ ] 256×256 matmul runs on hardware
- [ ] Results match CPU reference (< 1e-3 error)
- [ ] Achieves >50% of theoretical peak TFLOPS
- [ ] All 95 tests + 10 hardware tests pass

---

## Risk Mitigation

### Risk 1: SDK API Differs from Documentation

**Likelihood**: Medium
**Impact**: High (compilation failures)
**Mitigation**:
- Early validation in Phase 1
- Maintain mock mode as fallback
- Document all API differences discovered

### Risk 2: No Hardware Access

**Likelihood**: Medium
**Impact**: High (cannot validate Phase 3)
**Mitigation**:
- Complete Phases 1-2 without hardware
- Use simulator if available
- Coordinate with Tenstorrent for cloud access

### Risk 3: Performance Issues

**Likelihood**: High (first hardware run)
**Impact**: Medium (functional but slow)
**Mitigation**:
- Start with correctness, optimize later
- Profile to identify bottlenecks
- Consult Tenstorrent optimization guides

---

## Appendix A: Quick Reference

### Environment Setup

```bash
# Install Metalium SDK (see SDK docs)
git clone https://github.com/tenstorrent/tt-metal.git
cd tt-metal
# Follow installation instructions

# Set environment
export TT_METAL_HOME=/path/to/tt-metal

# Build TileLang with real Metalium
cd /path/to/tilelang-tt
cmake -B build -DUSE_LLVM=true -DUSE_REAL_METALIUM=ON
cmake --build build -j4
```

### Build Options

| Option | Default | Purpose | Configured By |
|--------|---------|---------|---------------|
| `TL_TT_BACKEND` | ON | Enable TT backend | `cmake/TTMetal.cmake` |
| `USE_REAL_METALIUM` | OFF | Enable real Metalium APIs | `cmake/TTMetal.cmake` |
| `TT_METAL_HOME` | (env) | Path to Metalium SDK | `cmake/FindMetalium.cmake` |
| `USE_LLVM` | OFF | CPU backend (required for TT) | Root `CMakeLists.txt` |

### Validation Checklist

**Before Requesting SDK Access**:
- [x] Week 16-18 complete (kernel headers, host program, cmake)
- [x] All 95 tests passing in mock mode
- [x] Validation plan documented
- [x] Known issues catalogued

**After Obtaining SDK Access**:
- [ ] Phase 1: Dry-run compilation
- [ ] Phase 2: API completion
- [ ] Phase 3: Hardware execution

---

## Related Documentation

**Programming Model & Kernel Authoring:**
- [GPU_vs_Tenstorrent.md](GPU_vs_Tenstorrent.md) - Execution model, memory hierarchy, and compilation flow comparison
- [kernel_authoring_comparison.md](kernel_authoring_comparison.md) - Side-by-side kernel examples (GEMM, elementwise, reduction)

**Architecture & Implementation:**
- [TT_ARCHITECTURE.md](TT_ARCHITECTURE.md) - Complete TT backend architecture overview
- [v5_pipeline.md](../architecture/v5_pipeline.md) - Authoritative v5 pipeline reference with all 14 passes; see [passes/README.md](../passes/README.md) for pass navigation
- [IR_LOWERING_ANALYSIS.md](IR_LOWERING_ANALYSIS.md) - GPU vs TT compiler pipeline comparison

**Setup & Testing:**
- [METALIUM_SETUP_GUIDE.md](METALIUM_SETUP_GUIDE.md) - SDK installation and configuration
- [local_build_guide.md](local_build_guide.md) - Local build instructions
- [CI.md](CI.md) - Continuous integration workflows

---

## Document History

- **2025-10-08 v1.0**: Initial validation plan created after Week 16-18 completion

---

**END OF VALIDATION PLAN**
