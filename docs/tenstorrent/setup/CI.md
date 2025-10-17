# Tenstorrent Backend CI/CD

This document describes the CI setup for the Tenstorrent backend in TileLang.

---

## Table of Contents

1. [Overview](#overview)
2. [CI Jobs](#ci-jobs)
3. [Two-Tier CI Strategy](#two-tier-ci-strategy)
4. [Caching Strategy](#caching-strategy)
5. [Running Locally](#running-locally)
6. [CI Flow Diagrams](#ci-flow-diagrams)
7. [Performance Characteristics](#performance-characteristics)

---

## Overview

The Tenstorrent backend uses a **two-tier CI approach** to balance speed and validation:

- **Tier 1 (Mock Mode):** Fast, lightweight CI that runs on every PR (2-3 minutes)
- **Tier 2 (Real SDK):** Full SDK validation CI that now runs on every PR (~15 minutes first run, 3-5 minutes cached)

**Primary workflow:** `.github/workflows/tenstorrent-ci.yml`
**Secondary workflow:** `.github/workflows/tenstorrent-sdk-ci.yml` (when SDK available)

---

## CI Jobs

### Job 1: Lint and Format (`lint-and-format`)

**Environment:** Ubuntu runner with Python 3.10

**Purpose:** Ensure code formatting and style consistency

**Steps:**
1. Checkout repository with submodules
2. Set up Python with pip caching (caches `requirements-lint.txt` dependencies)
3. Install lint dependencies: yapf, ruff, codespell, clang-format
4. Run `format.sh` to check formatting compliance
   - If formatting issues are found, the job fails and shows the diff

**Caching:**
- Pip packages are cached based on `requirements-lint.txt` hash
- Subsequent runs with unchanged dependencies skip pip installation

---

### Job 2: Build and Test (`build-and-test`)

**Environment:** Ubuntu runner with Python 3.10

**Purpose:** Build TileLang with LLVM backend and run Tenstorrent tests

**Note:** Currently builds with LLVM backend (not CUDA) since we only run CPU tests at this stage. This keeps the CI lightweight and fast. GPU/CUDA testing will be added in future when needed.

**Steps:**
1. Checkout repository with submodules
2. Set up Python with pip caching (caches `requirements-test.txt` dependencies)
3. Install system dependencies: cmake, ninja, llvm, build-essential, libedit-dev, libxml2-dev, zlib1g-dev
4. Install Python dependencies from requirements-test.txt
5. **Enable ccache:**
   - Uses `hendrikmuhs/ccache-action` for compiler caching
   - Cache key based on CMakeLists.txt hash + OS + version
   - Max size: 2G
   - Creates symlinks for automatic use by CMake
6. **TVM Build Caching:**
   - Generate cache key based on TVM submodule commit hash
   - Restore cached TVM build artifacts if available (uses `actions/cache/restore@v4`)
   - Caches: `build/tvm/` (contains libtvm*.so), `build/libtilelang*.so`, and `build/3rdparty/`
   - Save TVM artifacts after build completes (uses `actions/cache/save@v4` with `if: always()`)
   - Cache is saved even if job fails, preventing redundant TVM rebuilds
   - Only rebuilds TVM when the submodule is updated
7. Build TileLang with LLVM backend (ccache-enabled)
   - Uses Ninja build system with ccache as compiler launcher
   - Limited to 2 parallel jobs to avoid OOM on GitHub runners
   - LLVM backend is sufficient for CPU-only testing
   - Uses system LLVM packages instead of downloading LLVM 10.0.1
8. Install TileLang and TVM Python packages
   - Install TVM Python package from `3rdparty/tvm/python` with `TVM_LIBRARY_PATH` set
   - Install TileLang with `USE_LLVM=true` to enable LLVM backend
   - setup.py checks for nvcc availability before trying to use it
   - Gracefully skips CUDA version detection if nvcc is not found
9. Print ccache statistics (with availability check)
10. Run Tenstorrent target registration tests
    - Sets `LD_LIBRARY_PATH` to include `build/tvm` for TVM library discovery
    - Continue-on-error enabled for graceful handling
11. Run all Tenstorrent Python tests (CPU-only)
    - Sets `LD_LIBRARY_PATH` to include `build/tvm` for TVM library discovery
    - All 95/95 tests passing ✅

---

### Job 3: Static Analysis (`static-analysis`)

**Environment:** Ubuntu runner with Python 3.10

**Purpose:** Type checking with mypy

**Steps:**
1. Checkout repository
2. Set up Python with pip caching (caches `requirements-mypy.txt` dependencies)
3. Install mypy from requirements-mypy.txt
4. Run mypy on `tilelang/engine/tenstorrent/` (currently set to continue-on-error)

**Caching:**
- Pip packages are cached based on `requirements-mypy.txt` hash
- Ensures consistent caching behavior across CI runs

---

## Two-Tier CI Strategy

### Tier 1: Mock Mode CI (Default - Fast)

**Purpose:** Fast validation for every PR

**Characteristics:**
- ✅ Runs on every PR/commit
- ✅ No hardware required
- ✅ No SDK installation needed
- ✅ Tests: 95/95 passing
- ✅ Build time: ~2-3 minutes (with ccache)
- ✅ **This runs 95% of the time**

**Triggers:**
- Pull requests modifying:
  - `tilelang/engine/tenstorrent/**`
  - `src/target/tenstorrent/**`
  - `src/transform/tenstorrent/**`
  - `testing/python/tenstorrent/**`
  - `tilelang/utils/target.py`
  - `.github/workflows/tenstorrent-ci.yml`
- Pushes to `main` or `ws*-**` branches

---

### Tier 2: Real SDK CI (SDK Validation)

**Purpose:** Validate against real TT-Metalium SDK APIs

**Characteristics:**
- ✅ Runs on every PR (in addition to manual `workflow_dispatch` and weekly cron)
- ⚠️ Requires SDK installation in CI
- ✅ SDK cached for fast subsequent runs
- ✅ Tests hardware-specific code paths
- ⏱️ Build time: ~10-15 minutes (first time), ~3-5 minutes (cached)

**Workflow:** `.github/workflows/tenstorrent-sdk-ci.yml`

**Steps:**
1. Checkout with submodules
2. Restore tt-metal SDK cache (keyed by SDK version)
3. Install SDK if cache miss (~15 minutes first time)
4. Build TileLang with `-DUSE_REAL_METALIUM=ON`
5. Run tests with real API validation
6. Save SDK cache for next run

---

## Caching Strategy

The CI uses multiple layers of caching for efficiency:

| Job | What's Cached | Cache Key | Benefit |
|-----|---------------|-----------|---------|
| lint-and-format | Pip packages | requirements-lint.txt hash | Fast linter installation |
| build-and-test | TVM build artifacts | TVM submodule commit + OS | Avoid rebuilding TVM (~5-6 min), saved even on failure |
| build-and-test | ccache compiler cache | CMakeLists.txt hash + OS + version | Fast recompilation of unchanged files |
| build-and-test | Pip packages | requirements-test.txt hash | Fast pytest install |
| static-analysis | Pip packages | requirements-mypy.txt hash | Fast mypy installation |
| **sdk-ci** | **tt-metal SDK** | **TT_METAL_VERSION + OS** | **Avoid rebuilding SDK (~15 min)** |

### Cache Effectiveness

**Tier 1 (Mock Mode)**:
```
Cache Layers:
1. ccache (compiler cache) ──────► ~90% hit rate
2. TVM build artifacts ───────────► ~95% hit rate (TVM rarely changes)
3. pip packages ─────────────────► ~99% hit rate

Result: 2-3 min builds consistently
```

**Tier 2 (Real SDK Mode)**:
```
Cache Layers:
1. tt-metal SDK (~1.5GB) ────────► ~99% hit rate (version pinned)
2. ccache (compiler cache) ──────► ~90% hit rate
3. TVM build artifacts ──────────► ~95% hit rate

First run:  15-20 min (downloads+builds SDK)
Cached run: 3-5 min (SDK cached, only build TileLang)
```

---

## Running Locally

### Mock Mode (Default)

To ensure your changes will pass CI:

```bash
# Run formatting checks
bash format.sh

# If format.sh makes changes, review and commit them
git diff
git add .
git commit -m "Apply formatting"

# Run tests (requires TileLang built with TVM)
cd testing/python/tenstorrent
pytest test_target_registration.py -v

# Or use automated local build script
bash maint/scripts/local_build_and_test_tt.sh --skip-deps --jobs 4
```

### Real SDK Mode

To test with real Metalium SDK:

```bash
# One-time SDK setup
git clone https://github.com/tenstorrent/tt-metal.git ~/tt-metal
cd ~/tt-metal && ./build_metal.sh
export TT_METAL_HOME=~/tt-metal

# Build TileLang with real Metalium
cd /path/to/tilelang-tt
bash maint/scripts/local_build_and_test_tt.sh --with-metalium --skip-deps --jobs 4

# Or manual build:
cmake -B build -DUSE_LLVM=true -DUSE_REAL_METALIUM=ON
cmake --build build -j$(nproc)
pip install -e . --no-build-isolation

# Run tests
export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH
pytest testing/python/tenstorrent/ -v
```

See [`METALIUM_SETUP_GUIDE.md`](METALIUM_SETUP_GUIDE.md) for detailed SDK installation instructions.

---

## CI Flow Diagrams

### Tier 1: Mock Mode CI Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                   GitHub PR / Push to main                       │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │  .github/workflows/               │
         │  tenstorrent-ci.yml triggers      │
         └───────────────┬───────────────────┘
                         │
         ┌───────────────┴───────────────────────────────────┐
         │                                                    │
         ▼                                                    ▼
┌────────────────────┐                          ┌──────────────────────┐
│  Job 1: Lint       │                          │ Job 2: Build & Test  │
│  ────────────      │                          │ ─────────────────    │
│  • yapf            │                          │ ✅ NO SDK INSTALL    │
│  • ruff            │                          │ ✅ NO TT_METAL_HOME  │
│  • codespell       │                          │ ✅ Mock mode only    │
└────────────────────┘                          └──────────────────────┘
                                                           │
                                                           ▼
                                        ┌──────────────────────────────┐
                                        │ 1. Checkout (with submodules)│
                                        │    - tilelang-tt code        │
                                        │    - 3rdparty/tvm (submodule)│
                                        │    ❌ NOT tt-metal           │
                                        └──────────────────────────────┘
                                                           │
                                                           ▼
                                        ┌──────────────────────────────┐
                                        │ 2. Restore ccache            │
                                        │    Key: CMakeLists.txt hash  │
                                        │    Size: Up to 2GB           │
                                        └──────────────────────────────┘
                                                           │
                                                           ▼
                                        ┌──────────────────────────────┐
                                        │ 3. Restore TVM build cache   │
                                        │    Key: TVM submodule commit │
                                        │    Path: build/tvm/*.so      │
                                        └──────────────────────────────┘
                                                           │
                                                           ▼
                                        ┌──────────────────────────────┐
                                        │ 4. Build with LLVM           │
                                        │    cmake -DUSE_LLVM=true     │
                                        │    ❌ NO -DUSE_REAL_METALIUM │
                                        │    Builds TVM + TileLang     │
                                        │    Time: ~2-3 min (cached)   │
                                        └──────────────────────────────┘
                                                           │
                                                           ▼
                                        ┌──────────────────────────────┐
                                        │ 5. Install Python packages   │
                                        │    pip install -e .          │
                                        │    (TVM + TileLang)          │
                                        └──────────────────────────────┘
                                                           │
                                                           ▼
                                        ┌──────────────────────────────┐
                                        │ 6. Run Tenstorrent tests     │
                                        │    pytest testing/python/tenstorrent/ │
                                        │    ✅ 95/95 tests pass       │
                                        │    Time: ~30 seconds         │
                                        └──────────────────────────────┘
                                                           │
                                                           ▼
                                        ┌──────────────────────────────┐
                                        │ 7. Save caches (if updated)  │
                                        │    - ccache stats            │
                                        │    - TVM build artifacts     │
                                        └──────────────────────────────┘
                                                           │
                                                           ▼
                                                  ┌────────────────┐
                                                  │  ✅ CI PASS    │
                                                  │  Total: ~5 min │
                                                  └────────────────┘
```

**Key Points**:
- ✅ **No SDK needed** - Mock mode only
- ✅ **Fast** - Caches TVM build and ccache
- ✅ **Runs on every PR** - Quick feedback
- ✅ **95/95 tests pass** - Full coverage

---

### Tier 2: Real SDK CI Flow

```
┌──────────────────────────────────────────────────────────────────┐
│         Manual workflow_dispatch OR Weekly schedule              │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │  .github/workflows/               │
         │  tenstorrent-sdk-ci.yml triggers  │
         └───────────────┬───────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │ 1. Checkout (with submodules)     │
         │    - tilelang-tt code             │
         │    - 3rdparty/tvm (submodule)     │
         └───────────────┬───────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │ 2. Restore tt-metal SDK cache     │
         │    Key: TT_METAL_VERSION          │
         │    Path: ~/tt-metal/              │
         │    Size: ~1.5GB                   │
         └───────────────┬───────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │ 3. Install tt-metal SDK (if miss) │
         │    git clone tt-metal             │
         │    ./build_metal.sh               │
         │    Time: ~15 min (first time)     │
         │    Time: ~0 sec (cached)          │
         └───────────────┬───────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │ 4. Build TileLang with real SDK   │
         │    export TT_METAL_HOME=~/tt-metal│
         │    cmake -DUSE_REAL_METALIUM=ON   │
         │    FindMetalium.cmake runs ✅     │
         │    Links against libtt_metal.so   │
         │    Time: ~3-5 min                 │
         └───────────────┬───────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │ 5. Run SDK integration tests      │
         │    pytest testing/python/tenstorrent/ -v   │
         │    ✅ 95/95 tests pass            │
         │    + SDK-specific tests (future)  │
         └───────────────┬───────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │ 6. Save tt-metal SDK cache        │
         │    Subsequent runs fast ✅        │
         └───────────────┬───────────────────┘
                         │
                         ▼
                ┌────────────────┐
                │  ✅ CI PASS    │
                │  Total: ~20min │
                │  (first run)   │
                │  Total: ~5min  │
                │  (cached)      │
                └────────────────┘
```

**Key Points**:
- ⚠️ **SDK installed in CI** - But cached for reuse
- ⚠️ **Slower first run** - But subsequent runs fast
- ✅ **Validates real APIs** - Catches SDK compatibility issues
- ✅ **Always-on coverage** - Runs alongside mock CI on every PR

---

## Performance Characteristics

### Build Times

| Scenario | Time | Notes |
|----------|------|-------|
| **Mock CI (cold cache)** | 6-7 min | Builds TVM from scratch with ccache |
| **Mock CI (warm cache)** | 2-3 min | TVM cached, incremental build |
| **Real SDK CI (cold cache)** | 15-20 min | Downloads+builds SDK + TileLang |
| **Real SDK CI (warm cache)** | 3-5 min | SDK cached, builds TileLang only |

### Cache Sizes

| Cache | Size | Lifetime |
|-------|------|----------|
| ccache | Up to 2GB | Evicted after 7 days inactive |
| TVM build artifacts | ~400MB | Evicted after 7 days inactive |
| pip packages | ~50MB | Evicted after 7 days inactive |
| **tt-metal SDK** | **~1.5GB** | **Evicted after 7 days inactive** |

**Total cache storage:** GitHub Actions provides up to 10GB cache per repository

---

## Key Design Decisions

1. **System LLVM vs Downloaded LLVM:** Uses system LLVM packages (installed via apt) instead of downloading LLVM 10.0.1. This avoids compatibility issues with newer Ubuntu versions, which do not include `libtinfo.so.5` by default—causing runtime linking errors when using the downloaded LLVM 10.0.1 binaries.

2. **Separate TVM Python Installation:** TVM Python package is installed separately before TileLang to ensure proper library path configuration.

3. **LD_LIBRARY_PATH for Tests:** Tests require `LD_LIBRARY_PATH` to be set to `build/tvm` so Python can find the TVM shared libraries at runtime.

4. **Cache Split (Restore/Save):** Using separate `actions/cache/restore` and `actions/cache/save` with `if: always()` ensures TVM cache is saved even when the job fails, preventing redundant rebuilds on retry.

5. **Two-Tier CI:** Separates fast mock CI (every PR) from slow SDK validation (on-demand) to optimize developer experience while maintaining quality.

6. **External SDK Approach:** SDK is installed separately (not as Git submodule) to keep repository lightweight and enable flexible SDK version management.

---

## Future Improvements

Potential optimizations:
- Add CUDA build and GPU testing when needed (will require NVIDIA container or GPU runners)
- Custom Docker image with pre-built TVM (eliminates TVM build entirely)
- Parallel test execution with pytest-xdist
- Separate workflow for expensive builds (only on main/release branches)
- Hardware execution tests (Phase 3 - when Tenstorrent hardware available)

---

## Related Documentation

- **SDK Setup:** [`METALIUM_SETUP_GUIDE.md`](METALIUM_SETUP_GUIDE.md)
- **SDK Validation:** [`sdk-validation-plan.md`](sdk-validation-plan.md)
- **External SDK Implementation:** [`EXTERNAL_SDK_IMPLEMENTATION_PLAN.md`](EXTERNAL_SDK_IMPLEMENTATION_PLAN.md)
- **Local Build Guide:** [`local_build_guide.md`](local_build_guide.md)
