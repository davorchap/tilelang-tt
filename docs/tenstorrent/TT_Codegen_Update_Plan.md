# Tenstorrent Codegen Update Plan

**Date:** 2025-10-15
**Purpose:** Design and implement new codegen (Pass G) that works with our v5 TIR
**Target:** Python implementation for rapid development

---

## Current State Analysis

### What We Have (v5 TIR Output)
After passes A1-F, we have:
1. **Three separate kernels** (reader, compute, writer) as TIR PrimFuncs
2. **Rich metadata** attached to each kernel:
   - `tt.kernel_role`: "reader" | "compute" | "writer"
   - `tt.runtime_args`: Complete argument lists
   - `tt.cb_indices`: CB allocation map
   - `tt.persistent_config`: Loop pattern
   - Protocol insertion markers
3. **Protocol sequences** embedded in TIR:
   - NOC/CB calls (cb_reserve_back, noc_async_read_tile, etc.)
   - Engine init (tt.engine.init_common, tt.fpu.matmul_init, etc.)
   - DST management (tt.dst.acquire, pack_tile, etc.)

### What Existing Codegen Expects
Traditional TVM codegen expects:
- Single kernel function
- Standard buffer arguments
- No Tenstorrent-specific protocols

### Gap Analysis
We need a **completely new codegen** that:
1. Handles 3-kernel architecture
2. Emits Metalium API calls
3. Generates persistent loop structure
4. Creates host launcher

---

## New Codegen Design

### Architecture

```
TIR Module with 3 kernels
         ↓
   CodegenTT (Python)
         ↓
    ┌────┴────┬────────┬────────┐
reader.cpp  compute.cpp  writer.cpp  main.cpp
```

### Components

#### 1. TIR Visitor/Emitter (Python)
- Traverse TIR statements
- Map TIR intrinsics to Metalium APIs
- Handle CB indexing
- Generate C++ code strings

#### 2. Kernel Generator
- One generator per kernel role
- Emit kernel function signature
- Generate persistent loop
- Emit protocol sequences

#### 3. Host Generator
- Create main.cpp
- Set up runtime args
- Launch kernels on cores
- Handle synchronization

#### 4. Build System Integration
- Generate CMakeLists.txt or Makefile
- Link with Metalium libraries
- Handle compilation flags

---

## Implementation Plan

### Phase 1: Core Infrastructure

#### 1.1 Base Codegen Class
```python
class CodegenTT:
    def __init__(self):
        self.kernels = {}  # role -> generated code
        self.host_code = ""
        self.metadata = {}

    def generate(self, mod: IRModule) -> Dict[str, str]:
        """Generate all source files from IR module"""
        pass
```

#### 1.2 TIR to C++ Mapping
Map each TIR intrinsic to Metalium API:

| TIR Intrinsic | Metalium API |
|---------------|--------------|
| `cb_reserve_back(cb, n)` | `cb_reserve_back(cb, n)` |
| `noc_async_read_tile(...)` | `noc_async_read_tile(...)` |
| `tt.dst.acquire()` | `acquire_dst(tt::DstMode::Half)` |
| `pack_tile(dst, cb, idx)` | `pack_tile(dst, cb)` |
| `tt.mm.mma(...)` | `matmul_tiles(...)` |
| `tt.fpu.add(...)` | `add_tiles(...)` |

### Phase 2: Kernel Generation

#### 2.1 Reader Kernel Template
```cpp
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"

void MAIN {
    const uint32_t start_id = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t Mt = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in0 = 0;

    for (uint32_t tile_id = start_id; tile_id < start_id + count; tile_id++) {
        // Generated NOC/CB protocol
        cb_reserve_back(cb_in0, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_in0);
        noc_async_read_tile(tile_id, src_accessor, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);
    }
}
```

#### 2.2 Compute Kernel Template
```cpp
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"

void MAIN {
    const uint32_t Kt = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;
    constexpr uint32_t cb_out = 16;

    // Engine init
    mm_init(cb_in0, cb_in1, cb_out);

    // DST management + compute
    acquire_dst(tt::DstMode::Half);
    for (uint32_t kt = 0; kt < Kt; kt++) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    release_dst(tt::DstMode::Half);
}
```

#### 2.3 Writer Kernel Template
```cpp
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"

void MAIN {
    const uint32_t start_id = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = 16;

    for (uint32_t tile_id = start_id; tile_id < start_id + count; tile_id++) {
        cb_wait_front(cb_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_out);
        noc_async_write_tile(tile_id, dst_accessor, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
```

### Phase 3: Host Generation

#### 3.1 Host Launcher Template
```cpp
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/tt_metal.hpp"

int main(int argc, char **argv) {
    // Device setup
    Device *device = CreateDevice(0);
    CommandQueue& cq = device->command_queue();

    // Create program
    Program program = CreateProgram();

    // Create kernels
    auto reader_kernel = CreateKernel(
        program,
        "reader.cpp",
        CoreRange({0, 0}, {7, 7}),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0}
    );

    auto compute_kernel = CreateKernel(
        program,
        "compute.cpp",
        CoreRange({0, 0}, {7, 7}),
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4}
    );

    auto writer_kernel = CreateKernel(
        program,
        "writer.cpp",
        CoreRange({0, 0}, {7, 7}),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1}
    );

    // Set runtime args
    SetRuntimeArgs(program, reader_kernel, core, {A_addr, start_id, count, Mt, Nt});
    SetRuntimeArgs(program, compute_kernel, core, {Kt});
    SetRuntimeArgs(program, writer_kernel, core, {C_addr, start_id, count, Mt, Nt});

    // Execute
    EnqueueProgram(cq, program, false);
    Finish(cq);

    // Cleanup
    CloseDevice(device);
    return 0;
}
```

### Phase 4: Python Implementation Structure

```python
# codegen_tt.py

class CodegenTT:
    def __init__(self):
        self.indent_level = 0
        self.code_buffer = []

    def generate(self, mod: IRModule) -> Dict[str, str]:
        """Main entry point"""
        outputs = {}

        # Generate each kernel
        for name, func in mod.functions_items():
            if isinstance(func, tir.PrimFunc):
                role = func.attrs.get("tt.kernel_role")
                if role in ["reader", "compute", "writer"]:
                    outputs[f"{role}.cpp"] = self.generate_kernel(func, role)

        # Generate host
        outputs["main.cpp"] = self.generate_host(mod)

        # Generate build files
        outputs["CMakeLists.txt"] = self.generate_cmake(outputs.keys())

        return outputs

    def generate_kernel(self, func: tir.PrimFunc, role: str) -> str:
        """Generate a single kernel"""
        generator = {
            "reader": ReaderKernelGenerator,
            "compute": ComputeKernelGenerator,
            "writer": WriterKernelGenerator
        }[role](func)

        return generator.generate()

    def generate_host(self, mod: IRModule) -> str:
        """Generate host launcher"""
        generator = HostGenerator(mod)
        return generator.generate()
```

---

## Key Design Decisions

### 1. Python vs C++
**Decision:** Implement in Python
**Rationale:**
- Rapid prototyping
- Easy integration with TVM Python API
- Simpler to test and debug
- Can generate C++ code as strings

### 2. Template vs AST-based Generation
**Decision:** Template-based with string interpolation
**Rationale:**
- Simpler implementation
- Easier to understand generated code
- Direct mapping from TIR to C++

### 3. Persistent Loop Generation
**Decision:** Generate in codegen, not in TIR
**Rationale:**
- Keeps TIR clean and analyzable
- Follows v5 specification
- Simpler TIR passes

### 4. CB Management
**Decision:** Use metadata from passes
**Rationale:**
- CB indices already assigned in D1
- Consistent across kernels
- Validated by F pass

---

## Testing Strategy

### Unit Tests
1. Test each generator component
2. Verify intrinsic mapping
3. Check generated code syntax

### Integration Tests
1. Full pipeline test (A1-G)
2. Compile generated code
3. Run on simulator/hardware

### Validation
1. Compare with hand-written kernels
2. Performance benchmarking
3. Correctness verification

---

## Implementation Timeline

### Day 1: Core Infrastructure
- [ ] Base CodegenTT class
- [ ] TIR visitor framework
- [ ] Intrinsic mapping table

### Day 2: Kernel Generators
- [ ] Reader kernel generator
- [ ] Compute kernel generator
- [ ] Writer kernel generator

### Day 3: Host and Build
- [ ] Host launcher generator
- [ ] CMake/build file generation
- [ ] Runtime arg handling

### Day 4: Testing and Polish
- [ ] Unit tests
- [ ] Integration tests
- [ ] Documentation

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Complex TIR patterns | Start with simple patterns, add complexity gradually |
| Metalium API changes | Abstract API calls in mapping layer |
| Performance issues | Profile and optimize hot paths |
| Debugging difficulty | Add verbose logging and code comments |

---

## Success Criteria

1. ✅ Generate compilable C++ code
2. ✅ Support GEMM and element-wise ops
3. ✅ Pass all integration tests
4. ✅ Match hand-written kernel performance (±10%)
5. ✅ Clean, maintainable Python code

---

## Next Steps

1. Review and approve this plan
2. Implement Phase 1 (Core Infrastructure)
3. Implement Phase 2 (Kernel Generation)
4. Implement Phase 3 (Host Generation)
5. Test and validate

This plan provides a clear path to implementing a Python-based codegen that works with our v5 TIR structure and generates efficient Tenstorrent kernels.