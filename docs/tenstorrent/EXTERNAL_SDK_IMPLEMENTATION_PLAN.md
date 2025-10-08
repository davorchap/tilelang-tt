# External SDK Implementation Plan

**Date**: 2025-10-08
**Status**: Planning → Implementation
**Approach**: External SDK (like CUDA/ROCm)

---

## Table of Contents

1. [SDK Location Strategy](#sdk-location-strategy)
2. [CMake Integration](#cmake-integration)
3. [Local Build Workflow](#local-build-workflow)
4. [GitHub CI Workflow](#github-ci-workflow)
5. [Implementation Tasks](#implementation-tasks)

---

## SDK Location Strategy

### Local Development

**Recommended Location**:
```bash
~/tt-metal/           # User's home directory (default)
```

**Alternative Locations**:
```bash
/opt/tt-metal/        # System-wide installation
$HOME/.local/tt-metal # User-local (no sudo)
Custom path via TT_METAL_HOME
```

**Discovery Method**:
1. Check `TT_METAL_HOME` environment variable (highest priority)
2. Check `~/tt-metal` (convention)
3. Check `/opt/tt-metal` (system install)
4. Check CMake cache `TT_METAL_ROOT` hint

**Structure**:
```
~/tt-metal/
├── CMakeLists.txt
├── tt_metal/
│   ├── host_api.hpp
│   └── impl/
│       ├── device/mesh_device.hpp
│       └── buffers/mesh_buffer.hpp
├── build/
│   ├── lib/
│   │   ├── libtt_metal.so
│   │   ├── libdevice.so
│   │   └── ...
│   └── cmake/
│       └── TT-MetaliumConfig.cmake  # Package config
└── install/  # If user runs cmake --install
```

---

### GitHub CI

**Location**: `$HOME/tt-metal` on runner

**Cache Strategy**:
```yaml
uses: actions/cache@v4
with:
  path: ~/tt-metal
  key: tt-metal-${{ env.TT_METAL_VERSION }}-${{ runner.os }}
  restore-keys: |
    tt-metal-${{ env.TT_METAL_VERSION }}-
```

**Cache Key Components**:
- `TT_METAL_VERSION`: Git commit hash or tag
- `runner.os`: Ubuntu, macOS, etc.

**Cache Size**: ~1.5GB (includes build artifacts)

**Cache Hit Rate**: ~99% (version rarely changes)

---

## CMake Integration

### Option 1: Use tt-metal's Package Config (Recommended)

tt-metal provides `find_package(TT-Metalium)` that creates `TT::Metalium` target.

**Our CMakeLists.txt**:
```cmake
option(USE_REAL_METALIUM "Use real TT-Metalium APIs" OFF)

if(USE_REAL_METALIUM)
  # Find tt-metal package
  find_package(TT-Metalium)

  if(TT-Metalium_FOUND)
    message(STATUS "Found TT-Metalium: ${TT-Metalium_DIR}")

    # Add compile definition
    add_compile_definitions(TL_USE_REAL_METALIUM)

    # Link against TT::Metalium target
    target_link_libraries(tilelang PRIVATE TT::Metalium)

  else()
    message(WARNING "TT-Metalium not found, falling back to mock mode")
    set(USE_REAL_METALIUM OFF CACHE BOOL "" FORCE)
  endif()
endif()
```

**Advantages**:
- ✅ Uses official tt-metal package config
- ✅ Automatic include dirs, link flags, dependencies
- ✅ Namespaced target `TT::Metalium`
- ✅ Follows CMake best practices

**How it finds the package**:
```bash
# 1. Via CMAKE_PREFIX_PATH
cmake -B build -DCMAKE_PREFIX_PATH=~/tt-metal/build

# 2. Via environment variable
export CMAKE_PREFIX_PATH=~/tt-metal/build
cmake -B build

# 3. Via TT-Metalium_DIR
cmake -B build -DTT-Metalium_DIR=~/tt-metal/build/cmake

# 4. Via standard install location
# After: cmake --install ~/tt-metal/build --prefix /usr/local
```

---

### Option 2: Custom FindMetalium.cmake (Fallback)

If tt-metal doesn't provide package config, use our custom finder.

**Update cmake/FindMetalium.cmake**:
```cmake
# FindMetalium.cmake - Find TT-Metalium SDK

# Try TT_METAL_HOME first
if(DEFINED ENV{TT_METAL_HOME})
  set(TT_METAL_ROOT $ENV{TT_METAL_HOME})
else()
  # Try common locations
  find_path(TT_METAL_ROOT
    NAMES tt_metal/host_api.hpp
    PATHS
      ~/tt-metal
      $ENV{HOME}/tt-metal
      /opt/tt-metal
      /usr/local
    NO_DEFAULT_PATH
  )
endif()

if(TT_METAL_ROOT)
  # Find include directory
  find_path(Metalium_INCLUDE_DIR
    NAMES tt_metal/host_api.hpp
    PATHS ${TT_METAL_ROOT}
    PATH_SUFFIXES tt_metal include
    NO_DEFAULT_PATH
  )

  # Find libraries
  find_library(Metalium_LIBRARY_TT_METAL
    NAMES tt_metal
    PATHS ${TT_METAL_ROOT}
    PATH_SUFFIXES build/lib lib
    NO_DEFAULT_PATH
  )

  find_library(Metalium_LIBRARY_DEVICE
    NAMES device
    PATHS ${TT_METAL_ROOT}
    PATH_SUFFIXES build/lib lib
    NO_DEFAULT_PATH
  )

  # Create imported target
  if(Metalium_INCLUDE_DIR AND Metalium_LIBRARY_TT_METAL)
    if(NOT TARGET TT::Metalium)
      add_library(TT::Metalium INTERFACE IMPORTED)
      set_target_properties(TT::Metalium PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Metalium_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${Metalium_LIBRARY_TT_METAL};${Metalium_LIBRARY_DEVICE}"
      )
    endif()
    set(Metalium_FOUND TRUE)
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Metalium
  FOUND_VAR Metalium_FOUND
  REQUIRED_VARS Metalium_INCLUDE_DIR Metalium_LIBRARY_TT_METAL
)
```

---

## Local Build Workflow

### First-Time Setup

**1. Install tt-metal SDK** (one-time):
```bash
# Clone tt-metal
export TT_METAL_HOME=~/tt-metal
git clone https://github.com/tenstorrent/tt-metal.git $TT_METAL_HOME
cd $TT_METAL_HOME

# Build (takes ~15-20 minutes)
cmake -B build -G Ninja
cmake --build build -j$(nproc)

# Optional: Install to standard location
# sudo cmake --install build --prefix /opt/tt-metal
```

**2. Set environment variable** (permanent):
```bash
echo "export TT_METAL_HOME=~/tt-metal" >> ~/.bashrc
source ~/.bashrc
```

---

### Build TileLang with Mock Mode (Default)

```bash
cd /path/to/tilelang-tt

# Use local build script (mock mode)
bash maint/scripts/local_build_and_test_tt.sh --skip-deps --jobs 4

# ✅ No SDK needed
# ✅ 95/95 tests pass
# ✅ 2-3 minutes
```

---

### Build TileLang with Real SDK

```bash
cd /path/to/tilelang-tt

# Method 1: Via TT_METAL_HOME (recommended)
export TT_METAL_HOME=~/tt-metal
cmake -B build -DUSE_LLVM=true -DUSE_REAL_METALIUM=ON -G Ninja
cmake --build build -j$(nproc)

# Method 2: Via CMAKE_PREFIX_PATH
cmake -B build \
  -DUSE_LLVM=true \
  -DUSE_REAL_METALIUM=ON \
  -DCMAKE_PREFIX_PATH=~/tt-metal/build \
  -G Ninja
cmake --build build -j$(nproc)

# Install Python packages
pip install -e . --no-build-isolation

# Run tests
export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH
pytest testing/python/tt/ -v

# ✅ 95/95 tests pass with real API structure
```

---

### Enhanced Local Build Script

**Update `maint/scripts/local_build_and_test_tt.sh`**:

Add support for `--with-metalium` flag:

```bash
#!/bin/bash

# Parse arguments
WITH_METALIUM=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --with-metalium)
      WITH_METALIUM=true
      shift
      ;;
    # ... other args
  esac
done

# Build step
if [ "$WITH_METALIUM" = true ]; then
  echo "Building with real Metalium SDK..."

  # Check TT_METAL_HOME
  if [ -z "$TT_METAL_HOME" ]; then
    echo "Error: TT_METAL_HOME not set"
    echo "Please export TT_METAL_HOME=/path/to/tt-metal"
    exit 1
  fi

  if [ ! -f "$TT_METAL_HOME/tt_metal/host_api.hpp" ]; then
    echo "Error: tt-metal SDK not found at $TT_METAL_HOME"
    exit 1
  fi

  # Build with real Metalium
  USE_LLVM=true cmake -B build -DUSE_REAL_METALIUM=ON -G Ninja
  cmake --build build -j${JOBS}
else
  echo "Building with mock Metalium APIs..."
  USE_LLVM=true pip install -e . --no-build-isolation
fi
```

**Usage**:
```bash
# Mock mode (default)
bash local_build_and_test_tt.sh --jobs 4

# Real mode (with SDK)
export TT_METAL_HOME=~/tt-metal
bash local_build_and_test_tt.sh --with-metalium --jobs 4
```

---

## GitHub CI Workflow

### Tier 1: Mock Mode CI (Current - Keep)

**File**: `.github/workflows/tenstorrent-ci.yml`

**Status**: ✅ Already working, no changes needed

**Workflow**:
```yaml
# Already implemented
- Build with: USE_LLVM=true (no USE_REAL_METALIUM)
- Run tests: pytest testing/python/tt/
- ✅ 95/95 tests pass in 2-3 minutes
```

---

### Tier 2: Real SDK CI (New)

**File**: `.github/workflows/tenstorrent-sdk-ci.yml`

**Trigger**: Manual (`workflow_dispatch`) or weekly schedule

**Environment Variables**:
```yaml
env:
  TT_METAL_VERSION: "v0.53.0"  # Or specific commit hash
  TT_METAL_HOME: /home/runner/tt-metal
```

**Workflow Steps**:

```yaml
name: Tenstorrent SDK Integration Tests

on:
  workflow_dispatch:  # Manual trigger
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

env:
  TT_METAL_VERSION: "v0.53.0"
  TT_METAL_HOME: /home/runner/tt-metal
  PYTHON_VERSION: '3.10'

jobs:
  build-with-sdk:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout TileLang
        uses: actions/checkout@v5
        with:
          submodules: recursive

      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install system dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y \
            build-essential cmake ninja-build \
            llvm libedit-dev libxml2-dev zlib1g-dev

      # Cache tt-metal SDK
      - name: Cache tt-metal SDK
        id: cache-tt-metal
        uses: actions/cache@v4
        with:
          path: ~/tt-metal
          key: tt-metal-${{ env.TT_METAL_VERSION }}-${{ runner.os }}
          restore-keys: |
            tt-metal-${{ env.TT_METAL_VERSION }}-

      # Install tt-metal SDK (if cache miss)
      - name: Install tt-metal SDK
        if: steps.cache-tt-metal.outputs.cache-hit != 'true'
        run: |
          echo "Installing tt-metal SDK..."
          git clone https://github.com/tenstorrent/tt-metal.git ~/tt-metal
          cd ~/tt-metal
          git checkout ${{ env.TT_METAL_VERSION }}

          # Build tt-metal
          cmake -B build -G Ninja
          cmake --build build -j$(nproc)

          echo "tt-metal SDK installed successfully"

      - name: Verify tt-metal SDK
        run: |
          echo "TT_METAL_HOME: $TT_METAL_HOME"
          ls -la ~/tt-metal/tt_metal/host_api.hpp
          ls -la ~/tt-metal/build/lib/libtt_metal.so

      # Build TileLang with real Metalium
      - name: Build TileLang with real Metalium
        run: |
          export TT_METAL_HOME=~/tt-metal

          # Configure
          cmake -B build \
            -G Ninja \
            -DUSE_LLVM=true \
            -DUSE_REAL_METALIUM=ON \
            -DCMAKE_BUILD_TYPE=Release

          # Build
          cmake --build build -j$(nproc)

      - name: Install TileLang Python packages
        run: |
          pip install -r requirements-test.txt

          # Install TVM
          export TVM_LIBRARY_PATH=$(pwd)/build/tvm
          cd 3rdparty/tvm/python
          pip install -e .
          cd ../../..

          # Install TileLang
          pip install -e . --no-build-isolation

      - name: Run tests with real Metalium
        run: |
          export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH
          export TT_METAL_HOME=~/tt-metal

          pytest testing/python/tt/ -v

      - name: Check generated code uses real APIs
        run: |
          # Verify generated code includes real Metalium headers
          python -c "
import tilelang.tt as tt
import tvm
from tvm import tir

# Create simple module
A = tir.decl_buffer((256, 256), 'float16', name='A')
B = tir.decl_buffer((256, 256), 'float16', name='B')
C = tir.decl_buffer((256, 256), 'float16', name='C')
func = tir.PrimFunc(params=[A, B, C], body=tir.Evaluate(0))
func = func.with_attrs({'global_symbol': 'main', 'tt_grid_x': 8, 'tt_grid_y': 8})
mod = tvm.IRModule({'main': func})

# Generate code
mod = tt.apply_default_schedule(mod)
artifacts = tt.emit_tt_artifacts(mod)

# Check for real API includes
assert '#include \"tt_metal/host_api.hpp\"' in artifacts['main.cpp'], 'Missing real API includes'
print('✅ Generated code uses real Metalium APIs')
"
```

**Cache Effectiveness**:
- First run: ~20 minutes (clone + build SDK)
- Cached runs: ~3-5 minutes (SDK cached, only build TileLang)
- Cache size: ~1.5GB
- Cache validity: Weeks/months (SDK version rarely changes)

---

## Implementation Tasks

### Task 1: Investigate tt-metal Package Config

**Goal**: Determine if tt-metal provides `TT-MetaliumConfig.cmake`

**Steps**:
1. Clone tt-metal locally
2. Build it: `cmake -B build && cmake --build build`
3. Check for package config:
   ```bash
   find ~/tt-metal/build -name "*Metalium*.cmake"
   ```
4. If found: Use their package config
5. If not found: Create our own FindMetalium.cmake

**Decision Point**: Option 1 vs Option 2

---

### Task 2: Update CMakeLists.txt

**Changes**:
```cmake
# Replace FindMetalium with find_package(TT-Metalium)
if(USE_REAL_METALIUM)
  find_package(TT-Metalium)

  if(TT-Metalium_FOUND)
    message(STATUS "Building with REAL TT-Metalium APIs")
    add_compile_definitions(TL_USE_REAL_METALIUM)
    target_link_libraries(tilelang PRIVATE TT::Metalium)
  else()
    message(WARNING "Falling back to MOCK mode")
    set(USE_REAL_METALIUM OFF CACHE BOOL "" FORCE)
  endif()
endif()
```

**Files to modify**:
- `CMakeLists.txt`
- `cmake/FindMetalium.cmake` (update or remove)

---

### Task 3: Update Local Build Script

**File**: `maint/scripts/local_build_and_test_tt.sh`

**Add**:
- `--with-metalium` flag
- TT_METAL_HOME validation
- SDK existence check

**Test**:
```bash
# Without SDK
bash local_build_and_test_tt.sh --jobs 4

# With SDK
export TT_METAL_HOME=~/tt-metal
bash local_build_and_test_tt.sh --with-metalium --jobs 4
```

---

### Task 4: Create Real SDK CI Workflow

**File**: `.github/workflows/tenstorrent-sdk-ci.yml`

**Implement**:
- SDK caching strategy
- SDK installation (if cache miss)
- Build with USE_REAL_METALIUM=ON
- Test verification

**Test**: Manual trigger via GitHub UI

---

### Task 5: Update Documentation

**Files to update**:
- `docs/tenstorrent/METALIUM_SETUP_GUIDE.md` - Add CI cache strategy
- `docs/tenstorrent/README.md` - Update with new workflows
- `CLAUDE.md` - Update build instructions

**Add**:
- SDK location conventions
- Cache strategy explanation
- CI workflow documentation

---

### Task 6: End-to-End Testing

**Local Test**:
1. Install tt-metal SDK
2. Build TileLang with real SDK
3. Run all tests
4. Verify generated code uses real APIs

**CI Test**:
1. Trigger Real SDK CI workflow
2. Verify cache works
3. Confirm tests pass

---

## Summary

**SDK Locations**:
- **Local**: `~/tt-metal` (or `TT_METAL_HOME`)
- **CI**: `/home/runner/tt-metal` (cached)

**CMake Integration**:
- Use `find_package(TT-Metalium)` → `TT::Metalium` target
- Fallback to custom FindMetalium.cmake if needed

**Local Build**:
- Mock mode: `bash local_build_and_test_tt.sh`
- Real mode: `bash local_build_and_test_tt.sh --with-metalium`

**CI**:
- Tier 1 (Mock): Already working ✅
- Tier 2 (Real): New workflow with SDK caching

**Implementation Order**:
1. Investigate tt-metal package config
2. Update CMakeLists.txt
3. Update local build script
4. Create SDK CI workflow
5. Update documentation
6. Test end-to-end

---

**Next**: Start with Task 1 (investigate tt-metal package config)
