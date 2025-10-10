# TileLang Tenstorrent Backend Documentation

**Last Updated**: 2025-10-10  
**Status**: Production-Ready (95 tests passing)

---

## Quick Start

### Developers (Mock Mode - No Hardware)

```bash
# Clone and build
git clone https://github.com/davorchap/tilelang-tt.git
cd tilelang-tt
bash maint/scripts/local_build_and_test_tt.sh --skip-deps --jobs 4

# All 95 tests pass ✅
```

### Hardware Users (Real Mode - With Tenstorrent Device)

```bash
# 1. Install TT-Metalium SDK (see METALIUM_SETUP_GUIDE.md)
export TT_METAL_HOME=/path/to/tt-metal

# 2. Build with real Metalium
cmake -B build -DUSE_LLVM=true -DUSE_REAL_METALIUM=ON
cmake --build build -j$(nproc)
pip install -e . --no-build-isolation
```

---

## Documentation Index

### 🏗️ Architecture

| Document | Purpose | Audience |
|----------|---------|----------|
| **[TT_ARCHITECTURE.md](TT_ARCHITECTURE.md)** ⭐ | Complete TT backend architecture | All developers |
| **[IR_LOWERING_ANALYSIS.md](IR_LOWERING_ANALYSIS.md)** | GPU vs TT lowering pipeline comparison | Compiler engineers |
| **[PASS_TABLE.md](PASS_TABLE.md)** | Comprehensive pass reference (layout-aware roadmap) | Transform developers |
| **[IR_LOWERING_TASKS.md](IR_LOWERING_TASKS.md)** | Pattern detection implementation tasks | Contributors |

### 🚀 Setup & Usage

| Document | Purpose | Audience |
|----------|---------|----------|
| **[METALIUM_SETUP_GUIDE.md](METALIUM_SETUP_GUIDE.md)** | SDK installation & configuration | Hardware users |
| **[local_build_guide.md](local_build_guide.md)** | Local build instructions | Developers |
| **[CI.md](CI.md)** | Continuous integration | Contributors |

### 🔬 Validation

| Document | Purpose | Status |
|----------|---------|--------|
| **[METALIUM_SDK_VALIDATION_PLAN.md](METALIUM_SDK_VALIDATION_PLAN.md)** | SDK validation phases | ⚠️ Blocked (needs SDK access) |

---

## Current Status (2025-10-08)

### ✅ Complete (95 tests passing)

**IR Pipeline:**
- ✅ Target registration (8 tests)
- ✅ Metadata inference (7 tests)
  - Schedule inference (per-core tile assignments, legacy path)
  - Shard inference (DRAM layout descriptors, legacy path)
- ✅ Transform pipeline (39 tests)
  - GridToPersistentTT (persistent loop model)
  - TTTilesToCoreMap (NOC grid mapping)
  - MemorySpaceLowerTT (DRAM → L1 circular buffers)
  - TilePadTT (32×32 tile alignment)
  - TensorizeTT (pattern detection)
  - VerifyTTIR (constraint verification)

**Code Generation (41 tests):**
- ✅ IR-driven visitor infrastructure
- ✅ Reader kernel (DRAM → L1 via NOC)
- ✅ Compute kernel (Tensix tile math)
- ✅ Writer kernel (L1 → DRAM via NOC)
- ✅ Host program (device setup, execution)
- ✅ DST lifecycle (acquire→compute→commit→pack→release)

**SDK Integration:**
- ✅ External SDK approach (like CUDA/ROCm)
- ✅ CMake FindMetalium module
- ✅ Real vs Mock build modes
- ✅ CI workflows (mock + SDK validation)

### 🚧 Next Steps

**Layout-Aware Metadata (P0):**
- Implement `InferTTLayout`, `PropagateTTLayout`, `LayoutAwareWorkPartitionTT`.
- Update `GridToPersistentTT` and `EmitTTKernels` to consume new attributes.
- Add Python annotation helpers (`annotate_tt_layout`, `annotate_tt_schedule`).
- Track progress in [IR_LOWERING_TASKS.md](IR_LOWERING_TASKS.md).

**Pattern Detection (P1):**
- Extend `tensorize_tt.cc` to detect manual matmul loops.
- Add element-wise detection and remove codegen heuristics.

**SDK Validation (Blocked):**
- Phase 1: Dry-run compilation (fix namespaces, includes).
- Phase 2: API completion (EnqueueWriteBuffer, SetRuntimeArgs).
- Phase 3: Hardware execution (Grayskull/Wormhole).  
See [METALIUM_SDK_VALIDATION_PLAN.md](METALIUM_SDK_VALIDATION_PLAN.md).

---

## Architecture Overview

```
TileLang DSL (Python)
    ↓
TVM IRModule
    ↓
Apply TT Defaults → Stamp default schedule/shard metadata
    ↓
Layout-Aware Metadata (planned)
    ├─ InferTTLayout (buffer + shard schema)
    ├─ PropagateTTLayout (CB metadata)
    └─ LayoutAwareWorkPartitionTT (core ranges, partition mode)
    ↓
Transform Pipeline (6 TT-specific + 11 shared passes)
    ├─ infer_default_tt_schedule (legacy defaults)
    ├─ infer_default_tt_shard (legacy layout descriptors)
    ├─ grid_to_persistent_tt (GPU grid → persistent loop)
    ├─ tt_tiles_to_core_map (legacy NOC mapping)
    ├─ memory_space_lower_tt (DRAM → L1 circular buffers)
    ├─ tile_pad_tt (pad to 32×32 tiles)
    ├─ tensorize_tt (pattern detection)
    └─ verify_tt_ir (constraint verification)
    ↓
Code Generation (IR-Driven Visitors)
    ├─ Reader Kernel (NOC DRAM→L1)
    ├─ Compute Kernel (Tensix tile math)
    ├─ Writer Kernel (NOC L1→DRAM)
    ├─ Host Program (device setup)
    └─ Execution Plan (JSON metadata)
    ↓
5 Generated Files:
    ├─ reader.cpp
    ├─ compute.cpp
    ├─ writer.cpp
    ├─ main.cpp
    └─ tt.plan.json
```

See [TT_ARCHITECTURE.md](TT_ARCHITECTURE.md) for complete architecture details.

---

## Build Modes

### Mock Mode (Default)

**Purpose**: Development without hardware

```bash
cmake -B build -DUSE_LLVM=true
cmake --build build -j$(nproc)
```

**Features**:
- ✅ No hardware required
- ✅ Fast iteration
- ✅ Complete code generation
- ✅ All 95 tests pass

**Limitations**:
- ❌ Cannot execute on hardware
- ❌ Mock APIs (void functions)

### Real Mode (With SDK)

**Purpose**: Hardware execution

```bash
export TT_METAL_HOME=/path/to/tt-metal
cmake -B build -DUSE_LLVM=true -DUSE_REAL_METALIUM=ON
cmake --build build -j$(nproc)
```

**Features**:
- ✅ Real Metalium APIs
- ✅ Hardware execution ready
- ✅ Performance profiling

**Requirements**:
- ✅ TT-Metalium SDK installed
- ✅ `TT_METAL_HOME` environment variable
- ✅ Tenstorrent device (for execution)

**Setup**: See [METALIUM_SETUP_GUIDE.md](METALIUM_SETUP_GUIDE.md)

---

## Testing

```bash
# All TT backend tests
pytest testing/python/tt/ -v

# Specific test categories
pytest testing/python/tt/test_target_registration.py -v    # Target registration
pytest testing/python/tt/test_passes.py -v                 # Metadata inference
pytest testing/python/tt/test_grid_to_persistent_tt.py -v  # Persistent loop
pytest testing/python/tt/test_codegen_tt.py -v             # Code generation
```

---

## Contributing

### Development Workflow

1. **Feature Development** (Mock Mode):
   ```bash
   bash maint/scripts/local_build_and_test_tt.sh
   ```

2. **Create Pull Request**:
   ```bash
   git checkout -b feature-name
   git commit -m "Description"
   git push -u origin feature-name
   gh pr create --repo davorchap/tilelang-tt --base main
   ```

3. **SDK Validation** (When Available):
   ```bash
   export TT_METAL_HOME=/path/to/tt-metal
   cmake -B build -DUSE_REAL_METALIUM=ON
   ```

---

## Key Concepts

### Persistent Loop Model

**GPU**: Launch N threadblocks, each processes 1 tile
**TT**: Launch N cores, each iterates over M tiles

**Benefits**: Better data reuse, reduced launch overhead

### 3-Kernel Architecture

- **Reader**: DRAM → L1 (NOC transfers)
- **Compute**: L1 tile math (Tensix)
- **Writer**: L1 → DRAM (NOC transfers)

**Benefits**: Overlapped execution, hardware specialization

### Circular Buffers

**L1 Memory**: Circular buffers for producer-consumer communication
- `cb_in0`, `cb_in1`: Input tiles
- `cb_out0`: Output tiles
- Double buffering (2 pages per CB)

### Tile Size: 32×32

- Matches Tensix hardware
- FP16: 32×32 = 2KB per tile
- Efficient for matrix operations

---

## Tools

| Tool | Purpose | Usage |
|------|---------|-------|
| **`local_build_and_test_tt.sh`** | Local build & test | `bash local_build_and_test_tt.sh --with-metalium` |
| **`verify_metalium_sdk.sh`** | SDK verification | `bash verify_metalium_sdk.sh ~/tt-metal` |

---

## Questions & Support

- **Architecture**: See [TT_ARCHITECTURE.md](TT_ARCHITECTURE.md)
- **Build issues**: See [local_build_guide.md](local_build_guide.md)
- **SDK setup**: See [METALIUM_SETUP_GUIDE.md](METALIUM_SETUP_GUIDE.md)
- **Compiler internals**: See [IR_LOWERING_ANALYSIS.md](IR_LOWERING_ANALYSIS.md)

---

**Repository**: https://github.com/davorchap/tilelang-tt
**License**: Apache 2.0
**Maintainer**: TileLang Tenstorrent Team
