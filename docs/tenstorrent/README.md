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

### 🏗️ Architecture

| Document | Purpose | Audience |
|----------|---------|----------|
| **[NEW_LOWERING_ARCHITECTURE.md](NEW_LOWERING_ARCHITECTURE.md)** 🆕 | New metadata-driven Grid → Persistent lowering | All developers |
| **[RUNTIME_PLAN.md](RUNTIME_PLAN.md)** 🆕 | Runtime plan specification (tt.plan.json) | Backend developers |
| **[passes/NEW_PIPELINE.md](passes/NEW_PIPELINE.md)** 🆕 | New pass pipeline documentation | Transform developers |
| **[TT_ARCHITECTURE.md](TT_ARCHITECTURE.md)** ⭐ | Complete TT backend architecture | All developers |
| **[IR_LOWERING_ANALYSIS.md](IR_LOWERING_ANALYSIS.md)** | GPU vs TT lowering pipeline comparison | Compiler engineers |
| **[PASS_TABLE_SHARED.md](PASS_TABLE_SHARED.md)** | Shared lowering/optimization passes | Transform developers |
| **[PASS_TABLE_GPU.md](PASS_TABLE_GPU.md)** | CUDA/ROCm-only pass reference | GPU backend developers |
| **[PASS_TABLE_TT.md](PASS_TABLE_TT.md)** | Tenstorrent pass reference (layout-aware roadmap) | TT backend developers |
| **[TT_BACKEND_TASKS.md](TT_BACKEND_TASKS.md)** | Pattern detection implementation tasks | Contributors |

### 🚀 Setup & Usage

| Document | Purpose | Audience |
|----------|---------|----------|
| **[CI.md](CI.md)** | Continuous integration + local parity steps | Contributors |
| **[local_build_guide.md](local_build_guide.md)** | Detailed local build walkthrough | Developers |
| **[METALIUM_SETUP_GUIDE.md](METALIUM_SETUP_GUIDE.md)** | SDK installation & configuration | Hardware users |

### 🔬 Validation

| Document | Purpose | Status |
|----------|---------|--------|
| **[METALIUM_SDK_VALIDATION_PLAN.md](METALIUM_SDK_VALIDATION_PLAN.md)** | SDK validation phases | ⚠️ Blocked (needs SDK access) |

---

## Current Status (2025-10-10)

### ✅ Completed
- Target registration and Python orchestration (`tilelang/tenstorrent`).
- Layout-aware metadata pipeline (`InferTTLayout`, `PropagateTTLayout`, `LayoutAwareWorkPartitionTT`) generating canonical runtime-argument schemas.
- Grid-to-persistent transformation with shard-aware guardrails and per-core runtime metadata tables in host artifacts.
- IR-driven reader/compute/writer visitors aligned with the new runtime contract.
- Mock-mode CI parity via `maint/scripts/local_build_and_test_tt.sh`.
- **NEW**: Refactored metadata-driven lowering pipeline with cleaner abstractions and centralized attribute definitions.
- **NEW**: Migration to pure Python pass implementation complete (2025-10-14) - removed all legacy C++ passes.

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
Apply TT Defaults → Stamp legacy schedule/shard metadata (compatibility path)
    ↓
Layout-Aware Metadata
    ├─ InferTTLayout (buffer + shard schema)
    ├─ PropagateTTLayout (CB metadata)
    └─ LayoutAwareWorkPartitionTT (core ranges, partition mode, runtime args)
    ↓
Transform Pipeline (New Python Implementation)
    ├─ InferTTLayout (extract grid, infer layouts)
    ├─ PropagateTTLayout (normalize and distribute)
    ├─ TTTilesToCoreMap (work partition generation)
    ├─ LowerTTTileIntrinsics (tile op lowering)
    └─ GridToPersistentTT (final lowering + plan emission)
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

- **Architecture**: See [TT_ARCHITECTURE.md](TT_ARCHITECTURE.md)
- **Build issues**: See [local_build_guide.md](local_build_guide.md)
- **SDK setup**: See [METALIUM_SETUP_GUIDE.md](METALIUM_SETUP_GUIDE.md)
- **Compiler internals**: See [IR_LOWERING_ANALYSIS.md](IR_LOWERING_ANALYSIS.md)

---

**Repository**: https://github.com/davorchap/tilelang-tt
**License**: Apache 2.0
**Maintainer**: TileLang Tenstorrent Team
