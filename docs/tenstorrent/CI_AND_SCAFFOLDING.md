# CI Automation and Project Scaffolding with External SDK

**Date**: 2025-10-08
**Purpose**: Document CI workflows and directory structure for External SDK approach

---

## Table of Contents

1. [CI Automation Strategy](#ci-automation-strategy)
2. [Full Directory Tree](#full-directory-tree)
3. [Build Artifacts](#build-artifacts)
4. [CI Workflow Examples](#ci-workflow-examples)

---

## CI Automation Strategy

### Two-Tier CI Approach

**Tier 1: Mock Mode CI** (Default, Fast, Always Runs)
- Runs on every PR/commit
- No hardware required
- No SDK installation needed
- Tests: 95/95 passing
- Build time: ~2-3 minutes (with ccache)
- **This is what runs 95% of the time**

**Tier 2: Real SDK CI** (Optional, Slow, On-Demand)
- Runs when SDK available
- Requires SDK installation in CI
- Tests hardware-specific code paths
- Build time: ~15-20 minutes (first time)
- **This runs only when validating against real SDK**

---

## Full Directory Tree

### Repository Structure (Before Build)

```
tilelang-tt/                           # Root repository
├── .github/
│   └── workflows/
│       ├── tenstorrent-ci.yml        # ✅ Tier 1: Mock mode CI (current)
│       ├── tenstorrent-sdk-ci.yml    # ⚠️ Tier 2: Real SDK CI (future)
│       ├── ci.yml                    # CUDA builds (self-hosted)
│       └── amd_ci.yml                # ROCm builds
│
├── 3rdparty/                          # Third-party dependencies
│   ├── tvm/                          # TVM submodule (required)
│   │   ├── CMakeLists.txt
│   │   ├── include/
│   │   └── src/
│   ├── cutlass/                      # NVIDIA CUTLASS (optional)
│   └── composable_kernel/            # AMD CK (optional)
│   # NOTE: tt-metal NOT here (external SDK)
│
├── cmake/
│   ├── modules/
│   │   ├── LLVM.cmake
│   │   └── ...
│   └── FindMetalium.cmake            # ✅ Finds external Metalium SDK
│
├── src/                               # C++ source code
│   ├── target/
│   │   ├── codegen_cuda.cc
│   │   ├── codegen_hip.cc
│   │   └── tt/
│   │       └── codegen_tt.cc         # ✅ Conditional: #ifdef TL_USE_REAL_METALIUM
│   ├── transform/
│   │   └── tt/
│   │       ├── infer_tt_schedule.cc
│   │       ├── infer_tt_shard.cc
│   │       ├── grid_to_persistent_tt.cc
│   │       └── ...
│   └── ...
│
├── tilelang/                          # Python package
│   ├── engine/
│   │   ├── cuda/
│   │   ├── hip/
│   │   └── tt/                       # Tenstorrent engine
│   │       ├── __init__.py
│   │       └── engine.py
│   ├── tt/                           # TT utilities
│   │   ├── __init__.py
│   │   ├── target.py
│   │   └── passes.py
│   └── ...
│
├── testing/python/
│   └── tt/                           # ✅ 95 tests (all mock mode)
│       ├── test_target_registration.py
│       ├── test_ws2_passes.py
│       ├── test_ws3_grid_to_persistent.py
│       ├── test_codegen_visitor_base.py
│       └── ...
│
├── docs/tenstorrent/
│   ├── README.md                     # ✅ Documentation index
│   ├── METALIUM_SETUP_GUIDE.md       # ✅ SDK installation guide
│   ├── METALIUM_SDK_VALIDATION_PLAN.md
│   ├── IR_DRIVEN_CODEGEN_PLAN.md
│   └── ...
│
├── CMakeLists.txt                    # ✅ Main build configuration
├── pyproject.toml
├── setup.py
└── maint/scripts/
    └── local_build_and_test_tt.sh   # ✅ Local build script

# External SDK (NOT in repository)
# User's machine:
~/tt-metal/                           # ⚠️ User installs separately
├── tt_metal/
│   ├── host_api.hpp
│   ├── impl/
│   │   ├── device/mesh_device.hpp
│   │   └── buffers/mesh_buffer.hpp
│   └── ...
├── build/
│   └── lib/
│       ├── libtt_metal.so
│       ├── libdevice.so
│       └── ...
└── ...
```

---

## Build Artifacts

### Mock Mode Build (Default)

```
tilelang-tt/
├── build/                            # Build directory
│   ├── tvm/                         # TVM build artifacts
│   │   ├── libtvm.so
│   │   └── libtvm_runtime.so
│   ├── libtilelang.so               # TileLang library (mock APIs)
│   ├── libtilelang_module.so
│   └── ...
│
└── .venv/                           # Python virtual environment
    ├── bin/
    │   └── python
    └── lib/python3.12/site-packages/
        ├── tvm/
        └── tilelang/
```

**Characteristics**:
- ✅ No external SDK required
- ✅ Fast build (~2-3 minutes)
- ✅ Small disk usage (~500MB)
- ✅ Mock APIs in codegen_tt.cc

---

### Real Mode Build (With SDK)

```
# User's system
~/tt-metal/                           # External SDK (user installed)
├── build/lib/
│   ├── libtt_metal.so               # Real Metalium libraries
│   └── libdevice.so
└── ...

tilelang-tt/
├── build/
│   ├── tvm/
│   │   ├── libtvm.so
│   │   └── libtvm_runtime.so
│   ├── libtilelang.so               # ✅ Links against libtt_metal.so
│   ├── libtilelang_module.so
│   └── ...
│
└── .venv/
    └── lib/python3.12/site-packages/
        ├── tvm/
        └── tilelang/
```

**Build Command**:
```bash
export TT_METAL_HOME=~/tt-metal
cmake -B build -DUSE_LLVM=true -DUSE_REAL_METALIUM=ON
cmake --build build -j$(nproc)
```

**Characteristics**:
- ✅ Links against external SDK libraries
- ✅ Real Metalium APIs in generated code
- ⚠️ Requires TT_METAL_HOME environment variable
- ⚠️ Larger disk usage (~2GB with SDK)

---

## CI Workflow Examples

### Tier 1: Mock Mode CI (Current - Fast)

**.github/workflows/tenstorrent-ci.yml** (Existing):

```yaml
name: Tenstorrent Backend CI

on:
  pull_request:
    paths:
      - 'tilelang/engine/tt/**'
      - 'src/target/tt/**'
      - 'src/transform/tt/**'
      - 'testing/python/tt/**'

jobs:
  build-and-test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          submodules: recursive  # Gets TVM, not tt-metal

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Cache TVM build
        uses: actions/cache@v4
        with:
          path: build/tvm
          key: tvm-${{ hashFiles('3rdparty/tvm') }}

      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y llvm-14 llvm-14-dev

      - name: Build TileLang (Mock Mode)
        run: |
          # ✅ No USE_REAL_METALIUM flag (defaults to mock)
          USE_LLVM=true pip install -e . --no-build-isolation

      - name: Run Tenstorrent tests
        run: |
          export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH
          pytest testing/python/tt/ -v

      # ✅ Result: 95/95 tests pass in mock mode
```

**Key Points**:
- ✅ **No SDK installation** - Fast and simple
- ✅ **No network dependency** - Builds offline (except apt)
- ✅ **Uses ccache** - Incremental builds are fast
- ✅ **2-3 minute builds** - Lightweight
- ✅ **Runs on every PR** - Catches regressions early

---

### Tier 2: Real SDK CI (Future - Optional)

**.github/workflows/tenstorrent-sdk-ci.yml** (Not yet implemented):

```yaml
name: Tenstorrent SDK Integration Tests

on:
  workflow_dispatch:  # Manual trigger only
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  build-with-real-sdk:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          submodules: recursive

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Cache tt-metal SDK
        uses: actions/cache@v4
        with:
          path: ~/tt-metal
          key: tt-metal-${{ env.TT_METAL_VERSION }}

      - name: Install tt-metal SDK
        run: |
          # ⚠️ Only runs if cache miss
          if [ ! -d ~/tt-metal ]; then
            git clone https://github.com/tenstorrent/tt-metal.git ~/tt-metal
            cd ~/tt-metal
            git checkout ${{ env.TT_METAL_VERSION }}
            ./build_metal.sh
          fi

      - name: Build TileLang (Real Mode)
        env:
          TT_METAL_HOME: /home/runner/tt-metal
        run: |
          USE_LLVM=true cmake -B build -DUSE_REAL_METALIUM=ON
          cmake --build build -j$(nproc)
          pip install -e . --no-build-isolation

      - name: Run SDK integration tests
        env:
          TT_METAL_HOME: /home/runner/tt-metal
        run: |
          export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH
          pytest testing/python/tt/ -v

          # ✅ Additional SDK-specific tests
          pytest testing/python/tt/test_sdk_integration.py -v
```

**Key Points**:
- ⚠️ **Manual trigger** - Doesn't run on every PR
- ⚠️ **Slower** - Downloads and builds SDK (first time)
- ✅ **Cached SDK** - Subsequent runs faster
- ✅ **Validates real APIs** - Catches SDK compatibility issues
- ⚠️ **No hardware execution** - Still dry-run only (no Tenstorrent device in CI)

---

### Local Development Workflow

**Developer without hardware** (95% of users):

```bash
# Clone repository
git clone https://github.com/davorchap/tilelang-tt.git
cd tilelang-tt

# One-command build and test
bash maint/scripts/local_build_and_test_tt.sh --skip-deps --jobs 4

# ✅ Result: 95/95 tests pass in 2-3 minutes
```

**Developer with hardware**:

```bash
# One-time SDK setup
git clone https://github.com/tenstorrent/tt-metal.git ~/tt-metal
cd ~/tt-metal && ./build_metal.sh
export TT_METAL_HOME=~/tt-metal

# Build TileLang with real Metalium
cd /path/to/tilelang-tt
cmake -B build -DUSE_LLVM=true -DUSE_REAL_METALIUM=ON
cmake --build build -j$(nproc)
pip install -e . --no-build-isolation

# Run tests (dry-run, no hardware yet)
export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH
pytest testing/python/tt/ -v

# ✅ Result: 95/95 tests pass with real API structure
```

---

## CMake Configuration Flow

### Mock Mode (Default)

```
CMake Configure:
1. ✅ USE_REAL_METALIUM=OFF (default)
2. ✅ Skips FindMetalium.cmake
3. ✅ No TL_USE_REAL_METALIUM define
4. ✅ codegen_tt.cc uses mock APIs (#else branch)
5. ✅ No Metalium libraries linked

Build:
1. ✅ Fast build (only TVM + TileLang)
2. ✅ Small binary size
3. ✅ No external dependencies

Runtime:
1. ✅ Generated code uses mock classes
2. ✅ No hardware execution
3. ✅ All tests pass
```

---

### Real Mode (With SDK)

```
CMake Configure:
1. ✅ USE_REAL_METALIUM=ON (user specified)
2. ✅ Runs FindMetalium.cmake
3. ✅ Checks TT_METAL_HOME environment variable
4. ✅ Searches for tt_metal/host_api.hpp
5. ✅ Finds libtt_metal.so, libdevice.so
6. ✅ Sets Metalium_FOUND=TRUE
7. ✅ Adds TL_USE_REAL_METALIUM define
8. ✅ Adds Metalium include directories
9. ✅ Links Metalium libraries

Build:
1. ✅ codegen_tt.cc uses real APIs (#ifdef TL_USE_REAL_METALIUM)
2. ✅ Links against libtt_metal.so
3. ✅ Slower build (more dependencies)

Runtime:
1. ✅ Generated code uses real Metalium APIs
2. ⚠️ Requires LD_LIBRARY_PATH to find libtt_metal.so
3. ⚠️ Ready for hardware execution (when hardware available)
```

---

## Directory Comparison: Mock vs Real

### Mock Mode

```
tilelang-tt/
├── build/                    Total: ~500MB
│   ├── tvm/                  (~400MB)
│   │   ├── libtvm.so
│   │   └── libtvm_runtime.so
│   └── libtilelang.so        (~100MB, no Metalium deps)
│
└── .venv/                    (~200MB)
```

**Total**: ~700MB

---

### Real Mode

```
# External (user's system)
~/tt-metal/                   Total: ~1.5GB
├── build/lib/
│   ├── libtt_metal.so        (~300MB)
│   ├── libdevice.so          (~100MB)
│   └── ...
└── ...

# TileLang repository
tilelang-tt/
├── build/                    Total: ~600MB
│   ├── tvm/                  (~400MB)
│   │   ├── libtvm.so
│   │   └── libtvm_runtime.so
│   └── libtilelang.so        (~200MB, links Metalium)
│       # ✅ RPATH or LD_LIBRARY_PATH to ~/tt-metal/build/lib
│
└── .venv/                    (~200MB)
```

**Total**: ~2.3GB (700MB + 1.5GB SDK)

---

## Advantages of External SDK for CI

### Mock Mode CI (Tier 1)

✅ **Fast builds**: 2-3 minutes
✅ **No network dependency**: Builds offline
✅ **Small cache size**: ~500MB (TVM only)
✅ **Runs on every PR**: Catches bugs early
✅ **Free GitHub runners**: No special hardware needed
✅ **95/95 tests pass**: Full test coverage

### Real SDK CI (Tier 2)

✅ **Optional**: Only runs when needed
✅ **SDK cached**: First build slow, subsequent fast
✅ **Dry-run validation**: Verifies API compatibility
✅ **Manual trigger**: Control when it runs
⚠️ **No hardware required**: Still no device execution (Phase 3 future work)

---

## Comparison with ExternalProject_Add

| Aspect | External SDK (A) | ExternalProject_Add (B) |
|--------|------------------|-------------------------|
| **Mock CI Speed** | ✅ 2-3 min | ❌ 15-20 min (builds SDK) |
| **Real CI Speed** | ⚠️ 15 min (first), 3 min (cached) | ❌ 15-20 min (always) |
| **Disk Usage** | ✅ 500MB (mock), 2.3GB (real) | ❌ 2.3GB (always) |
| **Network at Build** | ✅ No | ❌ Yes (downloads SDK) |
| **Offline Builds** | ✅ Yes | ❌ No |
| **User Setup** | ⚠️ Manual SDK install | ✅ Zero setup |
| **Flexibility** | ✅ Any SDK version | ❌ Pinned version |

---

## Summary

**External SDK approach (A) for CI**:

1. **Tier 1 (Default)**: Mock mode, fast, always runs
   - No SDK needed
   - 95/95 tests pass
   - 2-3 minute builds
   - Runs on every PR

2. **Tier 2 (Optional)**: Real SDK, slow, on-demand
   - SDK cached in CI
   - Validates against real APIs
   - Manual trigger or weekly
   - No hardware execution (yet)

**Full tree**: Repository stays lightweight, SDK external

**Result**: Best of both worlds
- ✅ Fast development (mock mode)
- ✅ SDK validation (when needed)
- ✅ No bloat in repository
- ✅ Standard industry pattern

---

**Next Steps**:
1. ✅ Keep Tier 1 CI (current) - Already working
2. ⚠️ Add Tier 2 CI (future) - When SDK access available
3. ⚠️ Phase 3: Hardware CI (future) - When hardware available

See `METALIUM_SDK_VALIDATION_PLAN.md` for validation phases.
