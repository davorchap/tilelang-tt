# TileLang Tenstorrent Backend Documentation

**Last Updated**: 2025-10-10  
**Status**: Layout-aware metadata + shard-aware runtime integration ready; SDK validation awaiting hardware access.

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
| **[architecture/IR_LOWERING_ANALYSIS.md](architecture/IR_LOWERING_ANALYSIS.md)** | GPU vs TT comparison & execution models | Compiler engineers |
| **[architecture/TileLang_TT_TIR_Lowering_Guide_v5.md](architecture/TileLang_TT_TIR_Lowering_Guide_v5.md)** | V5 pass pipeline specification | Backend developers |
| **[architecture/RUNTIME_PLAN.md](architecture/RUNTIME_PLAN.md)** | Runtime plan specification (tt.plan.json) | Backend developers |

### 📚 Development Guides

| Document | Purpose | Audience |
|----------|---------|----------|
| **[guides/TIR_BASICS.md](guides/TIR_BASICS.md)** | TensorIR primer and TT lowering concepts | All developers |
| **[guides/TT_Python_Implementation_Quickstart.md](guides/TT_Python_Implementation_Quickstart.md)** | Quick-start for Python pass development | Contributors |
| **[guides/TT_Python_to_CPP_Migration.md](guides/TT_Python_to_CPP_Migration.md)** | Python→C++ migration guide | Backend developers |
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
| **[planning/TT_Implementation_Plan.md](planning/TT_Implementation_Plan.md)** | 4-week implementation roadmap | Project team |
| **[planning/TT_Pass_Status.md](planning/TT_Pass_Status.md)** | Current pass implementation status | Contributors |
| **[planning/TT_BACKEND_TASKS.md](planning/TT_BACKEND_TASKS.md)** | Task breakdown & assignments | Contributors |
| **[planning/METALIUM_SDK_VALIDATION_PLAN.md](planning/METALIUM_SDK_VALIDATION_PLAN.md)** | SDK validation phases | Hardware team |

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

## Current Status (2025-10-14)

### ✅ Completed
- Target registration and Python orchestration (`tilelang/tenstorrent`).
- V5 metadata-driven pipeline with 14 passes organized in stages A-E.
- Layout-aware metadata pipeline (v5 passes) generating canonical runtime-argument schemas.
- Grid-to-persistent transformation with shard-aware guardrails and per-core runtime metadata tables in host artifacts.
- IR-driven reader/compute/writer visitors aligned with the new runtime contract.
- Mock-mode CI parity via `maint/scripts/local_build_and_test_tt.sh`.
- Runtime plan generation (`tt.plan.json`) for host-device coordination.
- Grid extraction from T.Kernel IR structure.
- Proper JSON serialization for TVM container types.

### 🚧 In Progress
- Additional diagnostics for halo hints, L1 capacity checks, and documentation refreshes.

### ⏸️ Blocked
- Real hardware validation and performance profiling (see [METALIUM_SDK_VALIDATION_PLAN.md](METALIUM_SDK_VALIDATION_PLAN.md)).

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
pytest testing/python/tenstorrent/ -v

# Specific test categories
pytest testing/python/tenstorrent/test_target_registration.py -v    # Target registration
pytest testing/python/tenstorrent/test_metadata_inference.py -v     # Metadata inference
pytest testing/python/tenstorrent/test_persistent_lowering.py -v    # Persistent loop
pytest testing/python/tenstorrent/test_codegen_pipeline.py -v       # Code generation
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
- **Build issues**: See [setup/local_build_guide.md](setup/local_build_guide.md)
- **SDK setup**: See [setup/METALIUM_SETUP_GUIDE.md](setup/METALIUM_SETUP_GUIDE.md)
- **Compiler internals**: See [architecture/IR_LOWERING_ANALYSIS.md](architecture/IR_LOWERING_ANALYSIS.md)
- **Implementation plan**: See [planning/TT_Implementation_Plan.md](planning/TT_Implementation_Plan.md)

---

**Repository**: https://github.com/davorchap/tilelang-tt
**License**: Apache 2.0
**Maintainer**: TileLang Tenstorrent Team
