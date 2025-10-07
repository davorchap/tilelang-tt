# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

TileLang is a domain-specific language for developing high-performance GPU/CPU kernels (GEMM, FlashAttention, etc.) and accelerators including **Tenstorrent AI architecture**. Built on TVM with a Pythonic syntax, it enables productivity without sacrificing low-level optimizations.

This repository (`tilelang-tt`) is a **public fork** focused on adding first-class **Tenstorrent TT-Metalium backend** support alongside existing NVIDIA CUDA, AMD ROCm, and Huawei Ascend targets.

## Repository Information

**⚠️ CRITICAL: This is a fork. Always target the correct repository for PRs! ⚠️**

- **This repository:** `davorchap/tilelang-tt` (public fork)
- **Upstream (DO NOT PR HERE):** `tile-ai/tilelang` (original repository)

### Pull Request Workflow

**🚨 IMPORTANT: ALL PULL REQUESTS MUST TARGET `davorchap/tilelang-tt` 🚨**

**DO NOT** create pull requests against `tile-ai/tilelang`. This is a fork and all work stays in the fork.

**Correct PR settings:**
- ✅ **Base repository:** `davorchap/tilelang-tt`
- ✅ **Base branch:** `main`
- ✅ **Head branch:** your feature branch
- ❌ **NEVER use:** `tile-ai/tilelang` as base

**Creating PRs:**
```bash
# 1. Create feature branch
git checkout -b feature-name

# 2. Make changes and commit
git add .
git commit -m "Your commit message"

# 3. Push to origin (this pushes to davorchap/tilelang-tt)
git push -u origin feature-name

# 4. Create PR - use the correct URL format:
# https://github.com/davorchap/tilelang-tt/compare/main...feature-name
#
# OR if you have gh CLI with auth:
gh pr create --repo davorchap/tilelang-tt --base main --head feature-name
```

**Verifying PR target:**
When creating a PR on GitHub's web UI, verify:
- Base repository shows: `davorchap/tilelang-tt`
- Base branch shows: `main`
- If you see `tile-ai/tilelang`, you're targeting the WRONG repository!

**Protected branches:**
- **CRITICAL:** Changes to `main` require pull requests (direct pushes NOT allowed)
- **NEVER push directly to main** - always create a feature branch first
- All PRs must pass CI checks before merging

**Workflow summary:**
```bash
# ALWAYS work on a feature branch:
git checkout -b feature-branch-name
# Make changes, commit
git add .
git commit -m "Your changes"
# Push feature branch
git push -u origin feature-branch-name
# Create PR via gh CLI or GitHub web UI
gh pr create --repo davorchap/tilelang-tt --base main --head feature-branch-name
```

## Build System

### Environment Variables

- `USE_LLVM=true` - Enable LLVM backend (CPU-only builds, required for Tenstorrent CI)
- `USE_ROCM=true` - Enable AMD ROCm backend (requires `ROCM_HOME`)
- `USE_CUDA=true` - Default; requires `CUDA_HOME` (automatically detected)
- `DEBUG_MODE=true` - Build with debug symbols and logging
- `WITH_COMMITID=true` - Include git commit ID in wheel filename (default for non-PyPI builds)
- `PYPI_BUILD=true` - Build for PyPI distribution (clean version strings)

### Building TileLang

**Quick start - Automated local build (recommended for Tenstorrent development):**
```bash
# Builds with LLVM backend, runs tests, ~1 minute with ccache
bash maint/scripts/local_build_and_test_tt.sh --skip-deps --jobs 4
```

The automated script:
- Automatically initializes git submodules if needed
- Sets up Python virtual environment (`.venv`)
- Builds with LLVM backend (required for Tenstorrent)
- Installs TVM and TileLang in editable mode
- Runs Tenstorrent backend tests with proper environment
- See `docs/tenstorrent/local_build_guide.md` for full documentation

**Standard CUDA build:**
```bash
python setup.py build_ext --inplace
pip install -e .
```

**LLVM-only build (for CPU/Tenstorrent development):**
```bash
USE_LLVM=true pip install -e .
```

**ROCm build:**
```bash
USE_ROCM=true pip install -e .
```

The build system:
- Uses CMake + Ninja for C++/CUDA compilation
- Automatically downloads LLVM 10.0.1 if system llvm-config unavailable
- Compiles TVM from `3rdparty/tvm` submodule (unless `TVM_PREBUILD_PATH` set)
- Generates `libtvm.so`, `libtvm_runtime.so`, `libtilelang.so`, `libtilelang_module.so`
- Supports incremental builds via ccache (CI uses this heavily)

### Testing

**Run all tests:**
```bash
pytest testing/python/ -v
```

**Run Tenstorrent tests (automated via build script):**
```bash
bash maint/scripts/local_build_and_test_tt.sh --skip-deps --skip-build
```

**Run Tenstorrent tests manually:**
```bash
# Quick one-liner
source .venv/bin/activate && export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH && cd testing/python/tt && pytest test_target_registration.py -v

# Or step by step
source .venv/bin/activate
export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH
pytest testing/python/tt/test_target_registration.py -v
```

**Expected test results:**
- 8 tests pass (Workstream 1 complete)
  - 5 target registration and engine adapter tests
  - 3 default annotation helper tests

**Run specific test category:**
```bash
pytest testing/python/kernel/ -v          # Kernel tests
pytest testing/python/language/ -v        # Language tests
pytest testing/python/autotune/ -v        # Autotuner tests
```

Note: Set `LD_LIBRARY_PATH` to include `build/tvm` for tests to find TVM shared libraries.

### Code Formatting

**Check formatting:**
```bash
bash format.sh
```

This runs:
- `yapf` for Python formatting
- `ruff` for Python linting
- `codespell` for spelling checks
- `clang-format` for C++ code (if `.clang-format` exists)

**Auto-format (if supported):**
The format script will show diffs; manually apply changes or use auto-formatting tools.

## Code Architecture

### Tenstorrent Backend Design

**Goal:** Map TileLang's GPU-style grid kernels to Tenstorrent's persistent, tile-based execution model.

**Key concept:** Users write grid-style kernels with `T.Kernel(grid_x, grid_y)` using block indices `(bx, by)`. The backend generates a **persistent outer loop** for each core that iterates over assigned tiles, recovering `(bx, by)` from a static schedule.

**Components:**

1. **Default Annotation Helper** (`tilelang/tt/target.py`):
   - ✅ **Implemented (WS1)** - `apply_tt_defaults()` function
   - Stamps default TT attributes on PrimFuncs when user doesn't specify them
   - Default schedule: `contiguous` policy with `row_major` order
   - Default layout: 32×32 DRAM interleaved tilization
   - Ensures backward compatibility for GPU-style kernels

2. **Annotations API** (`python/tilelang_tt/annotations.py`):
   - `T.annotate_tt_schedule()` - Control static scheduling (contiguous/strided/rect)
   - `T.annotate_tt_sharding()` - Specify tensor sharding/layout on TT cores

3. **Compiler Passes** (`src/tt/passes/`):
   - `GridToPersistentTT` - Wraps grid kernel body in per-core scheduler loop
   - `TTShardToCoreMap` - Translates sharding annotations to CoreRangeSet
   - `TilePadTT` - Handles non-tile-multiple shapes (32×32 tiles)
   - `MemorySpaceLowerTT` - Lower DRAM↔L1 moves, circular buffers
   - `TensorizeTT` - Map tile operations to TT micro-kernels

4. **Codegen** (`src/tt/codegen/`):
   - `EmitTTKernels` - Generate compute/reader/writer C++ kernels and host stubs

5. **Target Registration & Engine** (`tilelang/engine/tt/`):
   - ✅ **Implemented (WS1)** - Target registration hooks for TVM integration
   - ✅ **Implemented (WS1)** - Engine adapter with lowering entry point
   - Integrates default annotation helper into lowering pipeline

### Directory Structure

```
tilelang-tt/
├── 3rdparty/
│   ├── tvm/                    # TVM submodule (compiler infrastructure)
│   ├── cutlass/                # NVIDIA CUTLASS for CUDA kernels
│   └── composable_kernel/      # AMD CK for ROCm kernels
├── src/
│   ├── ir.cc                   # IR definitions
│   ├── layout/                 # Layout transformations
│   ├── op/                     # Operator implementations
│   ├── runtime/                # CUDA runtime utilities
│   ├── target/                 # Code generators (CUDA, HIP, WebGPU, C++)
│   │   ├── codegen_cuda.cc
│   │   ├── codegen_hip.cc
│   │   ├── rt_mod_cuda.cc      # CUDA runtime module
│   │   └── rt_mod_hip.cc       # ROCm runtime module
│   ├── tl_templates/           # Kernel templates
│   └── transform/              # IR transformation passes
├── tilelang/
│   ├── engine/                 # Backend engines
│   │   └── tt/                 # Tenstorrent engine adapter (WS1 complete)
│   ├── tt/                     # Tenstorrent utilities (WS1 complete)
│   │   └── target.py           # Default annotation helper
│   ├── language/               # TileLang DSL (Python API)
│   ├── autotuner/              # Auto-tuning framework
│   ├── jit/                    # JIT compilation
│   │   └── adapter/cython/     # Cython wrapper for performance
│   ├── primitives/             # Primitive operations
│   └── transform/              # Python-level transforms
├── testing/python/
│   ├── tt/                     # Tenstorrent tests
│   ├── kernel/                 # Kernel tests
│   ├── language/               # Language tests
│   └── autotune/               # Autotuner tests
├── examples/                   # Example kernels (GEMM, attention, etc.)
└── docs/tenstorrent/           # Tenstorrent backend documentation
```

## Coding Standards

### Python (from copilot-instructions.md)

- Follow PEP 8 standards
- Use type hints for all functions
- Include docstrings for public APIs
- Security-conscious dependency updates

### C++

- Follow clang-format rules (run `format.sh`)
- Ensure compatibility with TVM coding style

## CI/CD

### Workflows

1. **`tenstorrent-ci.yml`** - Tenstorrent backend CI:
   - Triggers on PRs modifying `tilelang/engine/tt/`, `testing/python/tt/`, or workflow files
   - Runs on GitHub-hosted runners (Ubuntu + Python 3.10)
   - Uses LLVM backend (not CUDA) for lightweight CPU-only tests
   - **Caching strategy:**
     - TVM build cache (keyed by submodule commit) - saves ~5-6 min
     - ccache (keyed by CMakeLists.txt) - fast incremental compilation
     - pip packages (keyed by requirements files)
   - Jobs: lint-and-format, build-and-test, static-analysis (mypy)
   - Tests currently `continue-on-error: true` (backend incomplete)

2. **`ci.yml`** - Main CI:
   - Self-hosted NVIDIA runners
   - Full CUDA build and test suite

3. **`amd_ci.yml`** - AMD ROCm CI

### Running CI Locally

```bash
# Lint and format
bash format.sh

# Build and test (mimics Tenstorrent CI)
USE_LLVM=true pip install -e .
LD_LIBRARY_PATH=build/tvm pytest testing/python/tt/ -v
```

## Development Workflow

### For Tenstorrent Backend Development

1. **Branch naming:** Use `ws1-*` prefix for workstream 1 tasks (auto-triggers CI)

2. **Key files to modify:**
   - `tilelang/engine/tt/` - Python-level target registration and engine
   - `src/tt/` - C++ passes and codegen (when ready for Phase 0)
   - `testing/python/tt/` - Tests for Tenstorrent backend

3. **Testing strategy:**
   - Start with target registration tests (`test_target_registration.py`)
   - Add compile-only tests before hardware tests
   - Use "dry-run" mode to emit kernel sources without execution

4. **Documentation:**
   - Update `docs/tenstorrent/` with design decisions
   - Follow phased approach (Phase 0: GEMM, Phase 1: SDPA, Phase 2: Ergonomics)

## Key Technical Details

### Tenstorrent Execution Model

- **Persistent kernels:** Each core runs a long-lived kernel iterating over assigned tiles
- **Tile size:** 32×32 elements (dtype determines bytes per tile)
- **Memory hierarchy:** DRAM ↔ L1 circular buffers ↔ Compute
- **Static partitioning:** Host assigns `(start_id, count)` per core before launch

### Default Behavior (Backward Compatibility)

When no Tenstorrent annotations provided:
- Schedule: `policy="contiguous"`, `order="row_major"`
- Layout: Row-major 32×32 DRAM tilization
- L1 circular buffers auto-generated around `T.copy` sites

This allows existing GPU-style kernels to run on TT with minimal changes (subject to tile padding).

## Related Documentation

- [GPU vs Tenstorrent Architecture](docs/tenstorrent/GPU_vs_Tenstorrent.md)
- [Kernel Authoring Comparison](docs/tenstorrent/kernel_authoring_comparison.md)
- [CI Documentation](docs/tenstorrent/CI.md)
- [Installation Guide](docs/get_started/Installation.md)

## Important Notes

- **LLVM vs CUDA builds:** For Tenstorrent development, use `USE_LLVM=true` to avoid CUDA dependency
- **System LLVM preferred:** CI uses system LLVM (via apt) to avoid libtinfo.so.5 linking issues with downloaded LLVM 10.0.1
- **TVM library path:** Always set `LD_LIBRARY_PATH=build/tvm` when running tests
- **Submodules:** Run `git submodule update --init --recursive` after fresh clone
- **Cython JIT adapter:** Auto-compiles on first use with caching in `.cycache/`
