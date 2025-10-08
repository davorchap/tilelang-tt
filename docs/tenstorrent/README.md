# TileLang Tenstorrent Backend Documentation

**Last Updated**: 2025-10-08
**Status**: IR-Driven Backend Complete, SDK Integration Ready

---

## Quick Start

### For Developers (Mock Mode - No Hardware)

```bash
# Clone and build
git clone https://github.com/davorchap/tilelang-tt.git
cd tilelang-tt
bash maint/scripts/local_build_and_test_tt.sh --skip-deps --jobs 4

# All 95 tests pass ✅
```

### For Hardware Users (Real Mode - With Tenstorrent Device)

```bash
# 1. Install TT-Metalium SDK
# See: METALIUM_SETUP_GUIDE.md

# 2. Point to SDK
export TT_METAL_HOME=/path/to/tt-metal

# 3. Build with real Metalium
cmake -B build -DUSE_LLVM=true -DUSE_REAL_METALIUM=ON
cmake --build build -j$(nproc)
pip install -e . --no-build-isolation
```

---

## Documentation Index

### 🚀 Getting Started

| Document | Purpose | Audience |
|----------|---------|----------|
| **[METALIUM_SETUP_GUIDE.md](METALIUM_SETUP_GUIDE.md)** | Install & configure TT-Metalium SDK | Hardware users |
| **[local_build_guide.md](local_build_guide.md)** | Local build instructions | Developers |
| **[CI.md](CI.md)** | Continuous integration setup | Contributors |

### 📋 Architecture & Design

| Document | Purpose | Status |
|----------|---------|--------|
| **[UNIFIED_MATMUL_MVP_PLAN.md](UNIFIED_MATMUL_MVP_PLAN.md)** | Original MVP specification | ✅ Complete (historical reference) |
| **[IR_LOWERING_ANALYSIS.md](IR_LOWERING_ANALYSIS.md)** ⭐ | GPU vs TT lowering pipeline comparison | 📘 Current (2025-10-08) |
| **[PASS_TABLE.md](PASS_TABLE.md)** ⭐ | Comprehensive pass reference (60+ passes) | 📘 Current (2025-10-08) |
| **[IR_LOWERING_TASKS.md](IR_LOWERING_TASKS.md)** | Pattern detection implementation tasks | 🚧 Next steps (tensorize_tt extension) |
| **[IR_LOWERING_VALIDATION.md](IR_LOWERING_VALIDATION.md)** | Tasks 1-8 validation results | ✅ Complete (95/95 tests passing) |
| **[GPU_vs_Tenstorrent.md](GPU_vs_Tenstorrent.md)** | Architecture comparison | 📘 Reference (older) |
| **[kernel_authoring_comparison.md](kernel_authoring_comparison.md)** | Kernel development patterns | 📘 Reference |
| **[TIR_SPECIFICATIONS.md](TIR_SPECIFICATIONS.md)** | TIR transformation specs | 📘 Reference |

### 🔧 SDK Integration

| Document | Purpose | Status |
|----------|---------|--------|
| **[METALIUM_SETUP_GUIDE.md](METALIUM_SETUP_GUIDE.md)** ⭐ | SDK installation & build guide | ✅ Current (external SDK approach) |
| **[METALIUM_SDK_VALIDATION_PLAN.md](METALIUM_SDK_VALIDATION_PLAN.md)** | API validation phases 1-3 | ⚠️ Next steps (blocked by SDK access) |
| **[EXTERNAL_SDK_IMPLEMENTATION_PLAN.md](EXTERNAL_SDK_IMPLEMENTATION_PLAN.md)** | External SDK implementation plan | ✅ Complete (Tasks 1-7) |

### 🛠️ Tools & Scripts

| Tool | Purpose | Usage |
|------|---------|-------|
| **`maint/scripts/local_build_and_test_tt.sh`** | Local build & test script | `bash local_build_and_test_tt.sh --with-metalium` |
| **`maint/scripts/verify_metalium_sdk.sh`** | SDK verification tool | `bash verify_metalium_sdk.sh ~/tt-metal` |

### 📂 Phase-Driven Development (Active)

| Directory | Purpose | Status |
|-----------|---------|--------|
| **[phases/](phases/)** | 6-Phase TileLang→Metalium implementation | 🚧 Phase 1: 37% (1/3 examples) |
| **[phases/PHASES_STATUS.md](phases/PHASES_STATUS.md)** | Master tracking document | 📊 Overall: 17% complete |

### 📂 Workstream Details (Archived)

| Directory | Purpose | Status |
|-----------|---------|--------|
| **[archive/workstream1/](archive/workstream1/)** | Target registration & defaults | ✅ Complete (8 tests) |
| **[archive/workstream2/](archive/workstream2/)** | Metadata inference | ✅ Complete (7 tests) |
| **[archive/workstream3/](archive/workstream3/)** | Transform pipeline | ✅ Complete (39 tests) |
| **[archive/workstream4/](archive/workstream4/)** | Compute kernel codegen | ✅ Complete (IR-driven) |
| **[archive/workstream5/](archive/workstream5/)** | Reader/writer codegen | ✅ Complete (IR-driven) |
| **[archive/workstream6/](archive/workstream6/)** | Host program codegen | ✅ Complete (conditional compilation) |

---

## Current Status (2025-10-08)

### ✅ Completed

**IR-Driven Backend** (95 tests passing):
- ✅ WS1: Target registration (8 tests)
- ✅ WS2: Metadata inference (7 tests)
- ✅ WS3-Extended: Full transform pipeline (39 tests)
  - GridToPersistentTT, TTShardToCoreMap, MemorySpaceLowerTT
  - TilePadTT, TensorizeTT, VerifyTTIR
- ✅ IR-Driven Codegen: Visitor infrastructure (41 tests)
  - Base visitor, Compute visitor, Reader/Writer visitors
  - Full WS4-6 integration

**Metalium Integration** (Weeks 16-18):
- ✅ Week 16: Kernel headers with conditional compilation
- ✅ Week 17: Host program generation (real/mock modes)
- ✅ Week 18: CMake build system integration
- ✅ FindMetalium.cmake module (external SDK detection)

**External SDK Implementation** (Complete):
- ✅ CMake: find_package(TT-Metalium) with TT::Metalium target
- ✅ Local build: `--with-metalium` flag for local_build_and_test_tt.sh
- ✅ CI workflow: `.github/workflows/tenstorrent-sdk-ci.yml` (manual/weekly)
- ✅ SDK verification: `verify_metalium_sdk.sh` script

### ⚠️ Next Steps (Blocked by SDK Access)

**SDK Validation** (Weeks 19-22):
- Phase 1: Dry-run compilation (fix namespaces, includes)
- Phase 2: API completion (EnqueueWriteBuffer, SetRuntimeArgs)
- Phase 3: Hardware execution (Grayskull/Wormhole)

See **[METALIUM_SDK_VALIDATION_PLAN.md](METALIUM_SDK_VALIDATION_PLAN.md)** for details.

---

## Build Modes

### Mock Mode (Default)

**Purpose**: Development without hardware

```bash
cmake -B build -DUSE_LLVM=true
cmake --build build -j$(nproc)

# Generates dry-run code with mock APIs
# All 95 tests pass
```

**Features**:
- ✅ No hardware required
- ✅ Fast iteration
- ✅ Complete code generation
- ✅ Test coverage

**Limitations**:
- ❌ Cannot execute on hardware
- ❌ Mock APIs (void functions)

---

### Real Mode (With SDK)

**Purpose**: Hardware execution

```bash
export TT_METAL_HOME=/path/to/tt-metal
cmake -B build -DUSE_LLVM=true -DUSE_REAL_METALIUM=ON
cmake --build build -j$(nproc)

# Generates code with real Metalium APIs
# Links against libtt_metal.so
```

**Features**:
- ✅ Real Metalium APIs
- ✅ Hardware execution ready
- ✅ Performance profiling

**Requirements**:
- ✅ TT-Metalium SDK installed
- ✅ `TT_METAL_HOME` environment variable set
- ✅ Tenstorrent device (for execution)

**Setup**: See **[METALIUM_SETUP_GUIDE.md](METALIUM_SETUP_GUIDE.md)**

---

## Architecture Overview

```
TileLang DSL (Python)
    ↓
TVM IRModule
    ↓
WS1: apply_tt_defaults() → Stamp TT attributes
    ↓
WS2: infer_tt_schedule(), infer_tt_shard() → Compute metadata
    ↓
WS3: Transform Pipeline (6 passes)
    ├─ GridToPersistentTT
    ├─ TTShardToCoreMap
    ├─ MemorySpaceLowerTT
    ├─ TilePadTT
    ├─ TensorizeTT
    └─ VerifyTTIR
    ↓
WS4-6: Code Generation (IR-Driven Visitors)
    ├─ Compute Kernel (matmul intrinsics)
    ├─ Reader Kernel (NOC DRAM → L1)
    ├─ Writer Kernel (L1 → NOC DRAM)
    └─ Host Program (device setup, execution)
    ↓
Generated Artifacts:
    ├─ compute_kernel.cpp (TT kernel code)
    ├─ reader_kernel.cpp  (TT kernel code)
    ├─ writer_kernel.cpp  (TT kernel code)
    └─ main.cpp           (Host program)
```

---

## Design Philosophy

### External SDK Approach (Like CUDA/ROCm)

TT-Metalium is treated as an **external hardware SDK**, not a submodule:

**Why External**:
- Hardware dependency (requires Tenstorrent device)
- Complex system requirements (drivers, firmware)
- Independent build system
- User controls SDK version

**How It Works**:
1. User installs tt-metal separately
2. Sets `TT_METAL_HOME` environment variable
3. CMake's `FindMetalium.cmake` locates SDK
4. TileLang links against SDK libraries

**Benefits**:
- ✅ Lightweight repository
- ✅ No build conflicts
- ✅ Standard practice (matches CUDA/ROCm)
- ✅ Mock mode works without SDK

---

## Contributing

### Development Workflow

1. **Feature Development** (Mock Mode):
   ```bash
   # Work in mock mode (fast iteration)
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
   # Test with real SDK
   export TT_METAL_HOME=/path/to/tt-metal
   cmake -B build -DUSE_REAL_METALIUM=ON
   ```

### Testing

```bash
# Run all TT backend tests
pytest testing/python/tt/ -v

# Run specific workstream tests
pytest testing/python/tt/test_ws1_*.py -v
pytest testing/python/tt/test_ws2_*.py -v
pytest testing/python/tt/test_ws3_*.py -v
```

---

## Questions & Support

- **Build issues**: See `local_build_guide.md`
- **SDK setup**: See `METALIUM_SETUP_GUIDE.md`
- **Architecture**: See `GPU_vs_Tenstorrent.md`
- **API gaps**: See `METALIUM_SDK_VALIDATION_PLAN.md`

---

**Last Updated**: 2025-10-08
**Maintainer**: TileLang Tenstorrent Team
**Repository**: https://github.com/davorchap/tilelang-tt
