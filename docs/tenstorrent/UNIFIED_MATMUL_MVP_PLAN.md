# TileLang Tenstorrent Backend: Unified MVP Plan

**Version:** 2.0 FINAL
**Date:** 2025-10-07
**Status:** ⭐ **AUTHORITATIVE PLAN** - Single Source of Truth

---

## Executive Summary

This document consolidates all previous planning documents into a single authoritative plan with **clear Phase 1 vs Phase 2 separation**.

### 🎉 Phase 2 WS3-Extended COMPLETE (2025-10-07)

**Major Milestone**: All 5 deferred Phase 2 transforms have been successfully implemented!

- ✅ TTShardToCoreMap: Physical core topology mapping (PR #33)
- ✅ MemorySpaceLowerTT: Circular buffer allocation (PR #34)
- ✅ TilePadTT: Padding metadata computation (PR #35)
- ✅ TensorizeTT: Matmul intrinsic annotation (PR #36)
- ✅ VerifyTTIR: IR validation (PR #37)
- ✅ MVP test fixes (PR #38)

**Result**: Complete TIR transformation pipeline with 77 tests (74 verified passing + 3 MVP fixed)

**Status**: Phase 2 WS3-Extended component is **COMPLETE**. Remaining work (IR-driven codegen, real Metalium APIs, hardware execution) deferred to future phases.

**See**: `docs/tenstorrent/PHASE2_COMPLETION_PLAN.md` for detailed completion report.

---

### Key Clarifications

**Phase Separation:**
- **MVP Phase 1** (Current): Template-based dry-run matmul (matmul-specific, mock APIs) - **23 tests passing**
- **MVP Phase 2** (Future): IR-driven production backend (arbitrary kernels, real Metalium APIs) - Target 50+ tests

**Critical Architectural Decision:**
- **Phase 1 uses template-based codegen** (acceptable for POC) - Reads metadata, emits fixed patterns
- **Phase 2 will implement IR-walking codegen** (production) - Analyzes `func->body`, handles arbitrary kernels

**What's "Deferred":**
All deferred work is **MVP Phase 2**, not Phase 1:
- 5 WS3 transforms (TTShardToCoreMap, MemorySpaceLowerTT, TilePadTT, TensorizeTT, VerifyTTIR)
- IR-driven codegen migration (WS4-6-Extended)
- Real Metalium API integration
- Hardware execution

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [MVP Phase 1: Template-Based Dry-Run (CURRENT)](#mvp-phase-1-template-based-dry-run)
3. [MVP Phase 2: IR-Driven Production (FUTURE)](#mvp-phase-2-ir-driven-production)
4. [Detailed Workstream Specifications](#detailed-workstream-specifications)
5. [Success Criteria](#success-criteria)
6. [Implementation Roadmap](#implementation-roadmap)

---

## Architecture Overview

### IR Flow: TileLang → Tenstorrent Artifacts

```
┌─────────────────────────────────────────────────────────────┐
│ TileLang Frontend (User Code)                              │
│ @T.prim_func def matmul(A, B, C):                          │
│     with T.Kernel(8, 8) as (bx, by): ...                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ TVM TIR (Initial)                                           │
│ - Grid-style kernel with blockIdx bindings                 │
│ - T.alloc_fragment, T.copy, T.gemm operations              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼ WS1: Target Registration
┌─────────────────────────────────────────────────────────────┐
│ TIR + TT Default Attributes                                 │
│ attrs = {"tt_schedule_policy": "contiguous", ...}          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼ WS2: Schedule & Sharding Metadata
┌─────────────────────────────────────────────────────────────┐
│ TIR + Schedule + Sharding Metadata                         │
│ attrs = {"tt_grid_x": 8, "tt_tiles_per_core": [...], ...} │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼ WS3: TIR Transforms
┌─────────────────────────────────────────────────────────────┐
│ Persistent Loop TIR                                         │
│ PHASE 1: GridToPersistentTT only                           │
│ PHASE 2: + 5 more transforms                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼ WS4-6: Code Generation
┌─────────────────────────────────────────────────────────────┐
│ Generated Artifacts                                         │
│ compute.cpp, reader.cpp, writer.cpp, main.cpp, plan.json  │
│                                                             │
│ PHASE 1: Template-based (metadata → fixed patterns)        │
│ PHASE 2: IR-driven (walks func->body)                      │
└─────────────────────────────────────────────────────────────┘
```

### CUDA vs Tenstorrent Stack Comparison

| Aspect | CUDA | Tenstorrent (Phase 1) | Tenstorrent (Phase 2) |
|--------|------|----------------------|----------------------|
| **Work Assignment** | Hardware scheduler | Static contiguous | Static + topology |
| **Kernel Style** | Short-lived blocks | Persistent per-core | Persistent per-core |
| **Memory Pipeline** | Single kernel | 3 kernels (R/C/W) | 3 kernels + CBs |
| **Codegen** | IR-driven visitor | **Template-based** | **IR-driven visitor** |
| **APIs** | CUDA Runtime | **Mock TT APIs** | **Real Metalium** |
| **Execution** | Real hardware | **Dry-run only** | **Real hardware** |
| **Kernel Support** | Arbitrary | **Matmul only** | **Arbitrary** |

---

## MVP Phase 1: Template-Based Dry-Run

### Objectives

🎯 **Primary Goal**: Validate end-to-end pipeline for matmul
📦 **Scope**: 256×256 fp16 matmul only
🔧 **Codegen**: Template-based (reads metadata, emits fixed patterns)
🧪 **APIs**: Mock Metalium APIs (dry-run compilation)
✅ **Success**: 23+ tests passing, artifacts compile, POC works

### Current Status (Updated 2025-10-07)

- ✅ **WS1**: Target registration (8 tests passing)
- ✅ **WS2**: Metadata inference (7 tests passing)
- ✅ **WS3**: Complete pipeline (3 foundation + 39 Phase 2 = 42 tests passing)
- ✅ **WS4-6**: Template codegen with K-loop matmul (18 tests + 3 MVP = 21 tests)
- ✅ **Phase 2 WS3-Extended**: All 5 transforms COMPLETE (see below)
- **Total: 77 tests (74 verified passing, 3 MVP fixed)**

### Phase 1 Limitations (By Design)

These are **INTENTIONAL** for Phase 1 POC:

1. ⚠️ **Template-based codegen**: Emits fixed code structure, doesn't walk IR
2. ⚠️ **Matmul-only**: Hardcoded patterns specific to matmul
3. ⚠️ **Mock APIs**: Uses mock Metalium APIs for dry-run
4. ⚠️ **No IR walking**: Reads `func->attrs`, doesn't analyze `func->body`
5. ⚠️ **Partial WS3**: Only GridToPersistentTT (5 transforms deferred)
6. ⚠️ **No hardware execution**: Dry-run compilation only

**These are acceptable for Phase 1 POC and will be addressed in Phase 2.**

### Phase 1 Workstreams

See [Detailed Workstream Specifications](#detailed-workstream-specifications) below.

---

## MVP Phase 2: IR-Driven Production

### Objectives

🎯 **Primary Goal**: Production-ready backend for arbitrary kernels
📦 **Scope**: Any TileLang kernel (not just matmul)
🔧 **Codegen**: IR-walking visitor (analyzes `func->body`)
🧪 **APIs**: Real Metalium runtime
✅ **Success**: 50+ tests, hardware execution, 80%+ peak performance

### Phase 2 Status (Updated 2025-10-07)

**WS3-Extended** ✅ **COMPLETE** (Originally estimated 10-13 weeks):
- ✅ TTShardToCoreMap: CoreRangeSet topology (PR #33, 7 tests)
- ✅ MemorySpaceLowerTT: Circular buffer lowering (PR #34, 8 tests)
- ✅ TilePadTT: Non-aligned buffer padding (PR #35, 8 tests)
- ✅ TensorizeTT: Matmul intrinsic lowering (PR #36, 7 tests)
- ✅ VerifyTTIR: IR validation pass (PR #37, 9 tests)
- ✅ MVP test fixes (PR #38)
- **All 5 transforms implemented, tested, and merged!**

**WS4-6-Extended** ❌ **DEFERRED** (10-13 weeks estimated):
- IR-driven codegen using `StmtExprVisitor` pattern
- Walks `func->body` to generate code
- Supports arbitrary kernel patterns
- Extensible to new operations
- **Note**: Template codegen is working with K-loop matmul

**WS7** ❌ **DEFERRED** (3-4 weeks estimated):
- Real Metalium API integration
- Hardware execution
- Performance validation
- Multi-device support

**Completed Effort**: WS3-Extended (Phase 2, Component 1)
**Remaining Effort**: 13-17 weeks (WS4-6-Extended + WS7)

---

## Detailed Workstream Specifications

### WS1: Target Registration & Default Annotations ✅ COMPLETE

**Status**: 100% (8 tests passing)

**Purpose**: Recognize Tenstorrent target and stamp default attributes

**Files**:
- `tilelang/utils/target.py` - Target registration
- `tilelang/engine/tt/` - Engine adapter
- `tilelang/tt/target.py` - `apply_tt_defaults()`

**TIR Transformation**: Attribute stamping only (body unchanged)

**Input**: Plain PrimFunc from TileLang frontend
```python
@T.prim_func
def matmul(A: T.Buffer[(256, 256), "float16"],
           B: T.Buffer[(256, 256), "float16"],
           C: T.Buffer[(256, 256), "float16"]):
    with T.Kernel(8, 8) as (bx, by):
        # ... kernel body ...
```

**Output**: PrimFunc with TT default attributes
```python
# Body unchanged, only attrs added:
func.attrs = {
    "tt_schedule_policy": StringImm("contiguous"),
    "tt_schedule_order": StringImm("row_major"),
    "tt_layout_type": StringImm("dram_interleaved"),
    "tt_tile_height": IntImm(32),
    "tt_tile_width": IntImm(32),
}
```

**Python API**:
```python
import tilelang.tt as tt

mod = tvm.IRModule({"main": matmul_func})
mod = tt.apply_tt_defaults(mod)  # WS1
```

**Tests**: `testing/python/tt/test_target_registration.py`

---

### WS2: Schedule & Sharding Metadata Inference ✅ COMPLETE

**Status**: 100% (7 tests passing)

**Purpose**: Infer per-core tile assignments and buffer layouts

**Files**:
- `src/transform/tt/infer_tt_schedule.cc` - Schedule inference C++ pass
- `src/transform/tt/infer_tt_shard.cc` - Sharding inference C++ pass
- `tilelang/tt/passes.py` - Python bindings

**TIR Transformation**: Metadata inference (body unchanged)

**Input**: PrimFunc with WS1 attrs + grid from `T.Kernel(8, 8)`

**Output**: PrimFunc with schedule + sharding metadata
```python
func.attrs = {
    # WS1 attrs preserved
    "tt_schedule_policy": "contiguous",
    "tt_layout_type": "dram_interleaved",
    "tt_tile_height": 32,
    "tt_tile_width": 32,

    # WS2: Schedule metadata
    "tt_grid_x": IntImm(8),
    "tt_grid_y": IntImm(8),
    "tt_grid_z": IntImm(1),
    "tt_num_tiles": IntImm(64),  # 8×8
    "tt_num_cores": IntImm(64),  # Grayskull/Wormhole
    "tt_tiles_per_core": Array([
        Array([IntImm(0), IntImm(1)]),   # Core 0: start=0, count=1
        Array([IntImm(1), IntImm(1)]),   # Core 1: start=1, count=1
        # ... 62 more cores ...
        Array([IntImm(63), IntImm(1)]),  # Core 63: start=63, count=1
    ]),

    # WS2: Buffer sharding metadata (per buffer)
    "tt_buffer_A_layout": StringImm("dram_interleaved"),
    "tt_buffer_A_tile_shape": Array([IntImm(32), IntImm(32)]),
    "tt_buffer_A_num_tiles_height": IntImm(8),
    "tt_buffer_A_num_tiles_width": IntImm(8),
    "tt_buffer_A_needs_padding": IntImm(0),

    # Similar for B, C buffers
}
```

**Python API**:
```python
mod = tt.apply_ws2_passes(mod)  # WS2
```

**Tests**: `testing/python/tt/test_ws2_passes.py`

---

### WS3: TIR Transform Pipeline ⚠️ PARTIAL (Phase 1)

**Status**: 50% (3 tests passing - foundation only)

**Purpose**: Transform grid-style kernel to persistent loop

**Phase 1 Scope**: GridToPersistentTT only
**Phase 2 Scope**: + 5 additional transforms

**Files**:
- `src/transform/tt/grid_to_persistent_tt.cc` - GridToPersistentTT pass
- `tilelang/tt/passes.py` - Python bindings

**TIR Transformation**: Grid → Persistent Loop

**Input**: Grid-style TIR with thread bindings
```python
# Conceptual TIR (simplified)
@T.prim_func
def matmul(A, B, C):
    # Thread extent (from T.Kernel(8, 8))
    bx = T.env_thread("blockIdx.x")  # 0..7
    by = T.env_thread("blockIdx.y")  # 0..7
    T.launch_thread(bx, 8)
    T.launch_thread(by, 8)

    # Kernel body uses bx, by
    for k in range(8):
        A_tile = A[by*32:(by+1)*32, k*32:(k+1)*32]
        B_tile = B[k*32:(k+1)*32, bx*32:(bx+1)*32]
        # ...
```

**Output**: Persistent loop TIR
```python
@T.prim_func
def matmul_persistent(A, B, C,
                      tt_start_id: T.int32,  # NEW: Runtime arg
                      tt_count: T.int32,     # NEW: Runtime arg
                      grid_x: T.int32,       # NEW: Runtime arg
                      grid_y: T.int32):      # NEW: Runtime arg

    # NEW: Persistent outer loop (per-core)
    for i in T.serial(tt_count):
        # NEW: Compute tile ID
        tile_id: T.int32 = tt_start_id + i

        # NEW: Recover block indices
        bx: T.int32 = tile_id % grid_x
        by: T.int32 = tile_id / grid_x

        # Original kernel body (bx, by now computed variables)
        for k in range(8):
            A_tile = A[by*32:(by+1)*32, k*32:(k+1)*32]
            B_tile = B[k*32:(k+1)*32, bx*32:(bx+1)*32]
            # ...

# New attributes
func.attrs["tt_persistent_loop"] = IntImm(1)
func.attrs["tt_runtime_args"] = Array([
    StringImm("tt_start_id"),
    StringImm("tt_count"),
    StringImm("grid_x"),
    StringImm("grid_y")
])
```

**Python API**:
```python
mod = tt.apply_ws3_passes(mod)  # WS3 (Phase 1)
```

**Phase 2 Deferred Transforms**:
1. **TTShardToCoreMap**: Map tiles to CoreRangeSet topology
2. **MemorySpaceLowerTT**: Lower allocations to circular buffers
3. **TilePadTT**: Insert padding for non-tile-aligned buffers
4. **TensorizeTT**: Lower matmul loops to intrinsics
5. **VerifyTTIR**: Validate IR correctness

**Tests**: `testing/python/tt/test_ws3_grid_to_persistent.py`

---

### WS4: Compute Kernel Codegen ⚠️ TEMPLATE-BASED (Phase 1)

**Status**: 80% (template works for matmul)

**Purpose**: Generate TT compute kernel C++ code

**Files**:
- `src/target/tt/codegen_tt.cc` - `EmitTTComputeKernel()`

**⚠️ Phase 1 Limitation**: Template-based, not IR-driven

**Input**: PrimFunc with WS1-3 metadata

**Output**: `compute.cpp`
```cpp
// Generated TT Compute Kernel
#include <cstdint>

// Mock TT APIs for dry-run (Phase 1)
template<typename T>
inline T get_arg_val(uint32_t idx) { return T(); }

inline void cb_wait_front(uint32_t cb_id, uint32_t n_tiles) {}
inline void cb_push_back(uint32_t cb_id, uint32_t n_tiles) {}
inline void cb_pop_front(uint32_t cb_id, uint32_t n_tiles) {}
inline void matmul_tiles_init(uint32_t cb_a, uint32_t cb_b, uint32_t cb_c) {}
inline void matmul_tiles(uint32_t cb_a, uint32_t cb_b, uint32_t cb_c, bool acc) {}

constexpr uint32_t CB_A = 0;
constexpr uint32_t CB_B = 1;
constexpr uint32_t CB_C = 2;

void MAIN() {
    // Runtime arguments
    uint32_t out_tile_start_id = get_arg_val<uint32_t>(0);
    uint32_t num_output_tiles = get_arg_val<uint32_t>(1);
    uint32_t Kt = get_arg_val<uint32_t>(2);  // K dimension in tiles

    // Initialize matmul
    matmul_tiles_init(CB_A, CB_B, CB_C);

    // Per-output-tile loop
    for (uint32_t out_tile = 0; out_tile < num_output_tiles; ++out_tile) {
        // K-loop: matmul accumulation
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            cb_wait_front(CB_A, 1);  // Wait for A[m, kt]
            cb_wait_front(CB_B, 1);  // Wait for B[kt, n]

            // Matmul: C += A @ B
            matmul_tiles(CB_A, CB_B, CB_C, /* accumulate */ kt > 0);

            cb_pop_front(CB_A, 1);
            cb_pop_front(CB_B, 1);
        }

        cb_push_back(CB_C, 1);  // Output tile ready
    }
}
```

**Phase 1 Implementation** (Template):
```cpp
std::string EmitTTComputeKernel(const PrimFunc& func) {
    std::ostringstream code;

    // Read metadata from IR attrs
    auto grid_x = func->attrs.GetAttr<Integer>("tt_grid_x");
    auto grid_y = func->attrs.GetAttr<Integer>("tt_grid_y");

    MatmulDims dims = ExtractMatmulDims(func);

    // ⚠️ Template: Emit fixed code structure
    // Does NOT walk func->body

    code << "#include <cstdint>\n\n";
    code << "// Mock TT APIs\n";
    // ... emit mock APIs ...

    code << "void MAIN() {\n";
    code << "    uint32_t out_tile_start_id = get_arg_val<uint32_t>(0);\n";
    code << "    uint32_t num_output_tiles = get_arg_val<uint32_t>(1);\n";
    code << "    uint32_t Kt = get_arg_val<uint32_t>(2);\n";
    code << "    matmul_tiles_init(CB_A, CB_B, CB_C);\n";
    code << "    for (uint32_t out_tile = 0; ...) {\n";
    code << "        for (uint32_t kt = 0; kt < Kt; ++kt) {\n";
    code << "            cb_wait_front(CB_A, 1);\n";
    // ... hardcoded matmul pattern ...
    code << "        }\n";
    code << "    }\n";
    code << "}\n";

    return code.str();
}
```

**Phase 2 Preview** (IR-Driven):
```cpp
class TTComputeCodegen : public StmtExprVisitor {
    void VisitStmt_(const ForNode* op) override {
        code_ << "for (" << op->loop_var << " = " << op->min << "; ...)\n";
        VisitStmt(op->body);  // ✅ Walk IR recursively
    }

    void VisitStmt_(const AttrStmtNode* op) override {
        if (op->attr_key == "tt.matmul_tiles") {
            code_ << "matmul_tiles(...);\n";
        }
        VisitStmt(op->body);
    }
    // ... visit all IR nodes ...
};

std::string EmitTTComputeKernel(const PrimFunc& func) {
    TTComputeCodegen codegen;
    codegen.VisitStmt(func->body);  // ✅ Walk IR!
    return codegen.GetCode();
}
```

**Tests**: `testing/python/tt/test_ws4_codegen.py`

---

### WS5: Reader/Writer Kernel Codegen ⚠️ TEMPLATE-BASED (Phase 1)

**Status**: 80% (template works for matmul)

**Purpose**: Generate data movement kernels

**Files**:
- `src/target/tt/codegen_tt.cc` - `EmitTTReaderKernel()`, `EmitTTWriterKernel()`

**Output**: `reader.cpp`
```cpp
#include <cstdint>

// Mock NOC APIs
inline void cb_reserve_back(uint32_t cb_id, uint32_t n_tiles) {}
inline void cb_push_back(uint32_t cb_id, uint32_t n_tiles) {}
inline uint32_t get_write_ptr(uint32_t cb_id) { return 0; }
inline void noc_async_read_tile(uint32_t tile_idx, uint32_t dram_addr, uint32_t l1_addr) {}
inline void noc_async_read_barrier() {}

void kernel_main() {
    uint32_t dram_addr_a = get_arg_val<uint32_t>(0);
    uint32_t dram_addr_b = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t out_tile_start_id = get_arg_val<uint32_t>(5);
    uint32_t num_out_tiles = get_arg_val<uint32_t>(6);

    for (uint32_t out_tile = 0; out_tile < num_out_tiles; ++out_tile) {
        uint32_t current_tile_id = out_tile_start_id + out_tile;
        uint32_t out_m = current_tile_id / Nt;
        uint32_t out_n = current_tile_id % Nt;

        // Load A[out_m, :] and B[:, out_n] tiles
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            // Read A[out_m, kt] tile
            uint32_t tile_a_idx = out_m * Kt + kt;
            cb_reserve_back(CB_A, 1);
            uint32_t l1_addr_a = get_write_ptr(CB_A);
            noc_async_read_tile(tile_a_idx, dram_addr_a, l1_addr_a);
            noc_async_read_barrier();
            cb_push_back(CB_A, 1);

            // Read B[kt, out_n] tile
            uint32_t tile_b_idx = kt * Nt + out_n;
            cb_reserve_back(CB_B, 1);
            uint32_t l1_addr_b = get_write_ptr(CB_B);
            noc_async_read_tile(tile_b_idx, dram_addr_b, l1_addr_b);
            noc_async_read_barrier();
            cb_push_back(CB_B, 1);
        }
    }
}
```

**Output**: `writer.cpp`
```cpp
void kernel_main() {
    uint32_t dram_addr_c = get_arg_val<uint32_t>(0);
    uint32_t out_tile_start_id = get_arg_val<uint32_t>(1);
    uint32_t num_out_tiles = get_arg_val<uint32_t>(2);

    for (uint32_t out_tile = 0; out_tile < num_out_tiles; ++out_tile) {
        uint32_t tile_idx = out_tile_start_id + out_tile;

        cb_wait_front(CB_C, 1);
        uint32_t l1_addr = get_read_ptr(CB_C);
        noc_async_write_tile(tile_idx, l1_addr, dram_addr_c);
        noc_async_write_barrier();
        cb_pop_front(CB_C, 1);
    }
}
```

**Key Points**:
- ✅ Correct matmul tile indexing: A[m,k], B[k,n]
- ✅ K-loop in reader matches compute K-loop
- ⚠️ Template-based (Phase 1)

**Tests**: `testing/python/tt/test_ws5_reader_writer.py`

---

### WS6: Host Program Codegen ⚠️ MOCK APIS (Phase 1)

**Status**: 80% (mock APIs for dry-run)

**Purpose**: Generate host program

**Files**:
- `src/target/tt/codegen_tt.cc` - `EmitTTHostProgram()`

**Output**: `main.cpp`
```cpp
#include <cstdint>
#include <vector>
#include <iostream>

// Mock TT Device APIs (Phase 1)
class Device {
public:
    static Device* Instance() {
        static Device dev;
        return &dev;
    }
};

class Program {
public:
    void Build() {
        std::cout << "  Program built\n";
    }
};

class CircularBufferConfig {
public:
    CircularBufferConfig(uint32_t cb_id, uint32_t tile_size, uint32_t num_pages) {
        std::cout << "  CB" << cb_id << ": " << num_pages << " pages\n";
    }
};

int main() {
    std::cout << "=== TT Host Program - Dry Run ===\n\n";

    // 1. Device setup
    Device* device = Device::Instance();

    // 2. Circular buffer configuration
    constexpr uint32_t TILE_SIZE_FP16 = 32 * 32 * sizeof(uint16_t);
    CircularBufferConfig cb_a(0, TILE_SIZE_FP16, 2);  // Double-buffer
    CircularBufferConfig cb_b(1, TILE_SIZE_FP16, 2);
    CircularBufferConfig cb_c(2, TILE_SIZE_FP16, 1);  // Single buffer

    // 3. Create program
    Program program;
    program.Build();

    // 4. Allocate DRAM buffers
    constexpr uint32_t M = 256, N = 256, K = 256;
    std::vector<uint16_t> dram_a(M * K);
    std::vector<uint16_t> dram_b(K * N);
    std::vector<uint16_t> dram_c(M * N);

    std::cout << "=== Dry Run Complete ===\n";
    return 0;
}
```

**Phase 2 Preview** (Real Metalium):
```cpp
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt::tt_metal;

int main() {
    // Real device
    Device* device = CreateDevice(0);
    Program program = CreateProgram();

    // Real buffer creation
    InterleavedBufferConfig a_config{
        .device = device,
        .size = M * K * sizeof(uint16_t),
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM
    };
    auto src0_buffer = CreateBuffer(a_config);

    // Real kernel creation
    auto compute_kernel = CreateKernel(
        program, "compute.cpp", core_range,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4}
    );

    // Real execution
    CommandQueue& cq = device->command_queue();
    EnqueueProgram(cq, program, false);
    Finish(cq);

    CloseDevice(device);
    return 0;
}
```

**Tests**: `testing/python/tt/test_ws6_host_program.py`

---

## Success Criteria

### Phase 1 Success Criteria

- [x] WS1: 8 tests passing
- [x] WS2: 7 tests passing
- [x] WS3: 3 tests passing (foundation)
- [x] WS4-6: 5 tests passing (template codegen)
- [ ] POC example runs end-to-end
- [ ] Generated artifacts compile (dry-run)
- [ ] 23+ total tests passing
- [ ] All documentation complete

### Phase 2 Success Criteria (Updated 2025-10-07)

**WS3-Extended Component** ✅ **COMPLETE**:
- [x] WS3-Extended: 5 transforms implemented (39 tests, all passing)
- [x] Complete TIR transformation pipeline
- [x] All transforms tested and integrated
- [x] MVP test fixes applied
- [x] 77 total tests (74 verified + 3 MVP fixed)

**Remaining Components** (Deferred):
- [ ] WS4-6-Extended: IR-driven codegen (10+ tests)
- [ ] Real Metalium APIs integrated
- [ ] Hardware execution working (10+ tests)
- [ ] Performance 80%+ of hand-written kernels
- [ ] 100+ total tests passing
- [ ] Production-ready backend

---

## Implementation Roadmap

### Phase 1: 4-6 Weeks (Template-Based MVP)

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1 | Review/fix WS1-3, verify 18 tests | WS1-3 solid |
| 2 | Review/fix WS4-6, verify 5 tests | WS4-6 working |
| 3 | Create POC example, write TIR specs | Docs complete |
| 4 | E2E testing, CI integration | 23+ tests, MVP done |

### Phase 2: 20-26 Weeks (IR-Driven Production)

| Weeks | Tasks | Deliverables |
|-------|-------|--------------|
| 5-7 | TTShardToCoreMap, MemorySpaceLowerTT | 2 transforms |
| 8-9 | TilePadTT, TensorizeTT | 2 transforms |
| 10 | VerifyTTIR, testing | WS3 complete |
| 11-15 | IR-driven codegen (WS4-6-Extended) | Visitor-based |
| 16-18 | Real Metalium API integration | Real runtime |
| 19-22 | Hardware execution, validation | Hardware tests |
| 23-26 | Optimization, docs, polish | Production-ready |

---

## Related Documentation

### Primary Documents
- **This document** - Authoritative plan (v2.0 FINAL, updated 2025-10-07)
- `PHASE2_COMPLETION_PLAN.md` - Phase 2 WS3-Extended completion report ⭐ NEW
- `TIR_SPECIFICATIONS.md` - Detailed TIR specifications for each workstream

### POC Example
- `examples/tenstorrent/example_matmul_tt_poc.py` - Complete workflow example

### Workstream Status
- `workstream1/WS1_STATUS.md` ✅ Complete
- `workstream2/WS2_STATUS.md` ✅ Complete
- `workstream3/WS3_STATUS.md` ⚠️ Foundation complete
- `workstream4/WS4_STATUS.md` ⚠️ Template complete
- `workstream5/WS5_STATUS.md` ⚠️ Template complete
- `workstream6/WS6_STATUS.md` ⚠️ Template complete

### Architecture Docs
- `GPU_vs_Tenstorrent.md` - Architecture comparison
- `kernel_authoring_comparison.md` - Authoring guide
- `local_build_guide.md` - Build instructions
- `CI.md` - CI/CD documentation

---

## Appendix: Key Decisions

### Decision 1: Template vs IR-Driven Codegen

**Decision**: Phase 1 uses template-based codegen, Phase 2 migrates to IR-driven

**Rationale**:
- Template approach is faster to implement for POC
- Validates end-to-end pipeline quickly
- Matmul-only scope is acceptable for MVP
- IR-driven is complex, better suited for Phase 2 after foundation solid

**Trade-offs**:
- ✅ Fast MVP delivery
- ✅ Validates architecture
- ❌ Limited to matmul
- ❌ Requires Phase 2 refactor

### Decision 2: Mock vs Real APIs

**Decision**: Phase 1 uses mock Metalium APIs, Phase 2 integrates real runtime

**Rationale**:
- Dry-run compilation sufficient for Phase 1 validation
- Real APIs require TT-Metalium SDK installation
- Hardware execution better suited for Phase 2
- Mock APIs allow development without hardware

**Trade-offs**:
- ✅ No hardware dependency
- ✅ Faster iteration
- ❌ No real execution validation
- ❌ API structure may differ from real Metalium

### Decision 3: Partial WS3 in Phase 1

**Decision**: Only GridToPersistentTT in Phase 1, defer 5 transforms to Phase 2

**Rationale**:
- GridToPersistentTT sufficient for template codegen
- Other transforms blocked on IR-driven codegen
- Reduces Phase 1 scope and risk
- Clear Phase 2 backlog

**Trade-offs**:
- ✅ Unblocks WS4-6
- ✅ Reduces Phase 1 complexity
- ❌ Incomplete WS3
- ❌ Phase 2 dependency

---

**Document Version**: 2.1 (Phase 2 WS3-Extended COMPLETE)
**Author**: Claude Code
**Last Updated**: 2025-10-07
**Status**: ⭐ AUTHORITATIVE - Single Source of Truth

**Change Log**:
- v2.1 (2025-10-07): Phase 2 WS3-Extended completion update
- v2.0 (2025-10-07): FINAL unified plan with Phase 1/2 separation
- v1.x: Previous planning iterations (deprecated)

---

**END OF UNIFIED MVP PLAN**
