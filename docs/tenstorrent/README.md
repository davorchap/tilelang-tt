# TileLang Tenstorrent Backend Documentation

**Last Updated**: 2025-10-16
**Status**: v5 pipeline complete (14 passes, Python-only). Old pipeline removed. SDK validation awaiting hardware access.

---

## Quick Start

### Developers (Mock Mode - No Hardware)

```bash
git clone https://github.com/davorchap/tilelang-tt.git
cd tilelang-tt
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
bash maint/scripts/local_build_and_test_tt.sh --skip-deps --jobs 4
```

### Hardware Users (Real Mode - With Tenstorrent Device)

```bash
export TT_METAL_HOME=/path/to/tt-metal   # See METALIUM_SETUP_GUIDE.md
bash maint/scripts/local_build_and_test_tt.sh --with-metalium --skip-deps --jobs 4
```

---

## Documentation Index

### 🏗️ Architecture & Design

| Document | Purpose | Audience |
|----------|---------|----------|
| **[architecture/TT_ARCHITECTURE.md](architecture/TT_ARCHITECTURE.md)** ⭐ | Complete TT backend architecture | All developers |
| **[architecture/v5_pipeline.md](architecture/v5_pipeline.md)** ⭐ | Authoritative v5 pipeline reference (14 passes) | Backend developers |
| **[architecture/GPU_vs_Tenstorrent_Analysis.md](architecture/GPU_vs_Tenstorrent_Analysis.md)** ⭐ | GPU vs TT architecture & compiler comparison | Compiler engineers |
| **[architecture/TileLang_TT_TIR_Lowering_Guide_v5.md](architecture/TileLang_TT_TIR_Lowering_Guide_v5.md)** | V5 pass pipeline lowering guide | Backend developers |
| **[architecture/RUNTIME_PLAN.md](architecture/RUNTIME_PLAN.md)** | Runtime plan specification (tt.plan.json) | Backend developers |

### 📚 Development Guides

| Document | Purpose | Audience |
|----------|---------|----------|
| **[guides/TIR_BASICS.md](guides/TIR_BASICS.md)** | TensorIR primer and TT lowering concepts | All developers |
| **[guides/TT_Python_Implementation_Quickstart.md](guides/TT_Python_Implementation_Quickstart.md)** | Quick-start for Python pass development | Contributors |
| **[guides/kernel_authoring_comparison.md](guides/kernel_authoring_comparison.md)** | DSL vs SDK kernel comparison | Kernel developers |

### 🔧 Setup & Configuration

| Document | Purpose | Audience |
|----------|---------|----------|
| **[setup/METALIUM_SETUP_GUIDE.md](setup/METALIUM_SETUP_GUIDE.md)** | SDK installation & configuration | Hardware users |
| **[setup/CI.md](setup/CI.md)** | Continuous integration + local parity | Contributors |
| **[setup/local_build_guide.md](setup/local_build_guide.md)** | Detailed local build walkthrough | Developers |

### 📋 Planning & Status

| Document | Purpose | Audience |
|----------|---------|----------|
| **[planning/TT_Pass_Status.md](planning/TT_Pass_Status.md)** | v5 pipeline implementation status (historical) | Contributors |
| **[planning/TT_Implementation_Plan.md](planning/TT_Implementation_Plan.md)** | v5 implementation roadmap (historical, superseded) | Project team |
| **[planning/TT_BACKEND_TASKS.md](planning/TT_BACKEND_TASKS.md)** | Backend consolidation tasks (historical) | Contributors |
| **[planning/METALIUM_SDK_VALIDATION_PLAN.md](planning/METALIUM_SDK_VALIDATION_PLAN.md)** | SDK validation phases | Hardware team |
| **[archive/pre-v5/](archive/pre-v5/)** | Pre-v5 planning docs, progress reports, old pipeline | Historical reference |

### 📖 Reference

| Document | Purpose | Audience |
|----------|---------|----------|
| **[reference/PASS_TABLE_SHARED.md](reference/PASS_TABLE_SHARED.md)** | Shared lowering/optimization passes | Transform developers |
| **[reference/PASS_TABLE_GPU.md](reference/PASS_TABLE_GPU.md)** | CUDA/ROCm-only pass reference | GPU backend developers |
| **[reference/PASS_TABLE_TT.md](reference/PASS_TABLE_TT.md)** | Tenstorrent pass reference | TT backend developers |
| **[reference/TT_Pass_Specifications.md](reference/TT_Pass_Specifications.md)** | Detailed pass specifications | Implementation team |

### 🔄 Pass Documentation

| Document | Purpose |
|----------|---------|
| **[passes/](passes/)** | Individual pass implementation docs |

---

## Current Status (2025-10-16)

### ✅ Completed (v5 Pipeline)
- **v5 Pipeline Complete**: 14 passes in stages A-E, all Python implementation
- **Old Pipeline Removed**: Original 5-pass pipeline deleted (PR #135)
- **Python-Only Architecture**: All TT backend passes remain in Python for maintainability
- Target registration and Python orchestration (`tilelang/tenstorrent`)
- Layout-aware metadata pipeline generating canonical runtime-argument schemas
- Grid-to-persistent transformation with shard-aware guardrails
- IR-driven reader/compute/writer/host codegen visitors
- Mock-mode CI parity via `maint/scripts/local_build_and_test_tt.sh`
- Runtime plan generation (`tt.plan.json`) for host-device coordination
- Grid extraction from T.Kernel IR structure
- Proper JSON serialization for TVM container types
- **Test Suite**: 120 passing, 21 skipped (85.1% pass rate)

### 🎯 Next Steps
- SDK-backed hardware validation (awaiting device access)
- Performance profiling and optimization
- Additional diagnostics (halo hints, L1 capacity checks)

### ⏸️ Blocked
- Real hardware validation and performance profiling (see [planning/METALIUM_SDK_VALIDATION_PLAN.md](planning/METALIUM_SDK_VALIDATION_PLAN.md))

---

## Architecture Overview

```
TileLang DSL (Python)
    ↓
TVM IRModule
    ↓
Apply TT Defaults
    ↓
V5 Transform Pipeline (14 Passes, Stages A-E)
    ├─ Stage A: Metadata (infer_tt_layout_v5, propagate_tt_layout_v5, attach_tensor_accessor_tt)
    ├─ Stage B: Partitioning (layout_aware_work_partition_tt_v5, grid_to_core_grid_v5)
    ├─ Stage C: Protocol-less Lowering (lower_shared_to_cb_v5, lower_tt_tile_intrinsics_v5, build_tile_dfg_tt)
    ├─ Stage D: Late Split & Protocol (split_device_kernel, configure_tensor_accessor_tt, lower_cb_intrinsics, insert_compute_init_tt, insert_dst_management_tt)
    └─ Stage E: Finalization (finalize_persistent_signature_tt)
    ↓
Code Generation (IR-Driven Visitors)
    ├─ Reader Kernel (NOC DRAM→L1)
    ├─ Compute Kernel (Tensix tile math)
    ├─ Writer Kernel (NOC L1→DRAM)
    ├─ Host Metadata Summary (per-core runtime tables)
    └─ Execution Plan (JSON metadata)
    ↓
5 Generated Files:
    ├─ reader.cpp
    ├─ compute.cpp
    ├─ writer.cpp
    ├─ main.cpp
    └─ tt.plan.json
```

See [architecture/TT_ARCHITECTURE.md](architecture/TT_ARCHITECTURE.md) for complete architecture details.

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
- ✅ 120 tests passing (21 skipped)

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
# All TT backend tests (120 passing, 21 skipped)
pytest testing/python/tenstorrent/ -v

# Quick summary
pytest testing/python/tenstorrent/ --tb=no -q

# Specific test categories
pytest testing/python/tenstorrent/test_target_registration.py -v     # Target registration
pytest testing/python/tenstorrent/test_v5_passes_integration.py -v   # v5 pipeline integration
pytest testing/python/tenstorrent/test_codegen_pipeline.py -v        # Code generation
pytest testing/python/tenstorrent/test_jit_decorator.py -v           # JIT decorator
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

- **Architecture**: See [architecture/TT_ARCHITECTURE.md](architecture/TT_ARCHITECTURE.md)
- **v5 Pipeline**: See [architecture/v5_pipeline.md](architecture/v5_pipeline.md)
- **Build issues**: See [setup/local_build_guide.md](setup/local_build_guide.md)
- **SDK setup**: See [setup/METALIUM_SETUP_GUIDE.md](setup/METALIUM_SETUP_GUIDE.md)
- **GPU vs TT comparison**: See [architecture/GPU_vs_Tenstorrent_Analysis.md](architecture/GPU_vs_Tenstorrent_Analysis.md)
- **Pass status**: See [planning/TT_Pass_Status.md](planning/TT_Pass_Status.md)
- **Historical docs**: See [archive/pre-v5/](archive/pre-v5/)

---

**Repository**: https://github.com/davorchap/tilelang-tt
**License**: Apache 2.0
**Maintainer**: TileLang Tenstorrent Team
