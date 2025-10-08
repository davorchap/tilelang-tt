# External SDK Implementation Status

**Date**: 2025-10-08
**Approach**: External SDK (like CUDA/ROCm) ✅ CONFIRMED

---

## Completed Tasks ✅

### Task 1: Investigate tt-metal CMake Package Config ✅

**Findings**:
- tt-metal provides official CMake package: `TT-Metalium`
- Creates namespaced target: `TT::Metalium`
- Package config location: `${CMAKE_INSTALL_LIBDIR}/cmake/tt-metalium/tt-metalium-config.cmake`
- Target alias: `TT::Metalium` (modern), `Metalium::Metal` (backwards compat)

**Evidence**:
```cmake
# From tt-metal/tt_metal/CMakeLists.txt:
add_library(tt_metal)
add_library(TT::Metalium ALIAS tt_metal)
install(EXPORT Metalium NAMESPACE TT:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tt-metalium)
```

**Example Usage** (from tt-metal/programming_examples):
```cmake
find_package(TT-Metalium REQUIRED)
target_link_libraries(myapp PUBLIC TT::Metalium)
```

---

### Task 2: Update CMakeLists.txt ✅

**Changes Made**:

**File**: `CMakeLists.txt` (lines 212-254)

**Before**:
```cmake
find_package(Metalium)  # Custom FindMetalium.cmake
if(Metalium_FOUND)
    # Manual include dirs and libraries
    list(APPEND TILE_LANG_INCLUDES ${Metalium_INCLUDE_DIRS})
    set(METALIUM_LINK_LIBRARIES ${Metalium_LIBRARIES})
endif()
```

**After**:
```cmake
# Try official package first, fallback to custom FindMetalium.cmake
find_package(TT-Metalium QUIET)
if(NOT TT-Metalium_FOUND)
    find_package(Metalium QUIET)  # Fallback for TT_METAL_HOME
endif()

if(TT-Metalium_FOUND OR TARGET TT::Metalium)
    add_compile_definitions(TL_USE_REAL_METALIUM)
    set(METALIUM_FOUND TRUE)
    # Improved error messages with setup instructions
endif()
```

**Linking** (lines 272-277):
```cmake
if(METALIUM_FOUND AND TARGET TT::Metalium)
    target_link_libraries(tilelang PRIVATE TT::Metalium)
    target_link_libraries(tilelang_static PRIVATE TT::Metalium)
    message(STATUS "Linked TileLang against TT::Metalium target")
endif()
```

**Benefits**:
- ✅ Uses official tt-metal package config when available
- ✅ Falls back to custom FindMetalium.cmake for custom locations
- ✅ Standard modern CMake target-based linking
- ✅ Better error messages with setup instructions

---

### Task 3: Update FindMetalium.cmake ✅

**Changes Made**:

**File**: `cmake/FindMetalium.cmake` (lines 101-121)

**Added**:
```cmake
# Create imported target TT::Metalium (matches official package)
if(Metalium_FOUND AND NOT TARGET TT::Metalium)
    add_library(TT::Metalium INTERFACE IMPORTED)
    set_target_properties(TT::Metalium PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Metalium_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${Metalium_LIBRARIES}"
    )
endif()
```

**Purpose**:
- Provides `TT::Metalium` target even when using custom SDK location
- Compatible with both official package and `TT_METAL_HOME` approach
- Seamless fallback mechanism

---

## Remaining Tasks

### Task 4: Update Local Build Script 🚧

**File**: `maint/scripts/local_build_and_test_tt.sh`

**Required Changes**:

**Add flag parsing**:
```bash
WITH_METALIUM=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --with-metalium)
            WITH_METALIUM=true
            shift
            ;;
        # ... existing flags
    esac
done
```

**Update build section**:
```bash
if [ "$WITH_METALIUM" = true ]; then
    echo "Building with real Metalium SDK..."

    # Validate TT_METAL_HOME
    if [ -z "$TT_METAL_HOME" ]; then
        echo "Error: TT_METAL_HOME not set"
        echo "Please: export TT_METAL_HOME=~/tt-metal"
        exit 1
    fi

    # Check SDK exists
    if [ ! -f "$TT_METAL_HOME/tt_metal/host_api.hpp" ]; then
        echo "Error: tt-metal SDK not found at $TT_METAL_HOME"
        exit 1
    fi

    # Build with real SDK
    cmake .. \
        -G Ninja \
        -DUSE_LLVM=true \
        -DUSE_REAL_METALIUM=ON \
        -DCMAKE_PREFIX_PATH="$TT_METAL_HOME/build" \
        # ... other flags
else
    # Mock mode (existing code)
    cmake .. -G Ninja # ... existing flags
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

### Task 5: Create Real SDK CI Workflow 🚧

**File**: `.github/workflows/tenstorrent-sdk-ci.yml` (NEW)

**Template**: See `EXTERNAL_SDK_IMPLEMENTATION_PLAN.md` for complete workflow

**Key Features**:
- Manual trigger (`workflow_dispatch`) or weekly schedule
- SDK caching (~1.5GB, key: `TT_METAL_VERSION`)
- First run: ~20 min (install SDK)
- Cached run: ~5 min (SDK cached)

**Environment**:
```yaml
env:
  TT_METAL_VERSION: "v0.53.0"  # Or commit hash
  TT_METAL_HOME: /home/runner/tt-metal
```

**Cache Strategy**:
```yaml
- name: Cache tt-metal SDK
  uses: actions/cache@v4
  with:
    path: ~/tt-metal
    key: tt-metal-${{ env.TT_METAL_VERSION }}-${{ runner.os }}
```

---

### Task 6: Update Documentation 🚧

**Files to Update**:

1. **`docs/tenstorrent/METALIUM_SETUP_GUIDE.md`**:
   - Add section on `find_package(TT-Metalium)`
   - Document `CMAKE_PREFIX_PATH` vs `TT_METAL_HOME`
   - Add SDK location best practices

2. **`docs/tenstorrent/README.md`**:
   - Update quick start with both methods
   - Add CI workflow documentation
   - Update status to reflect completed tasks

3. **`CLAUDE.md`**:
   - Update build instructions with `--with-metalium` flag
   - Document SDK locations (local and CI)
   - Add Real SDK CI workflow to CI section

---

### Task 7: Create SDK Verification Script 🚧

**File**: `maint/scripts/verify_metalium_sdk.sh` (NEW)

**Purpose**: Help users verify their tt-metal SDK installation

**Script**:
```bash
#!/bin/bash
# Verify TT-Metalium SDK Installation

set -e

echo "=== TT-Metalium SDK Verification ==="
echo ""

# Check TT_METAL_HOME
if [ -z "$TT_METAL_HOME" ]; then
    echo "❌ TT_METAL_HOME not set"
    echo "Please: export TT_METAL_HOME=/path/to/tt-metal"
    exit 1
else
    echo "✅ TT_METAL_HOME=$TT_METAL_HOME"
fi

# Check directory exists
if [ ! -d "$TT_METAL_HOME" ]; then
    echo "❌ Directory not found: $TT_METAL_HOME"
    exit 1
else
    echo "✅ Directory exists"
fi

# Check headers
if [ -f "$TT_METAL_HOME/tt_metal/host_api.hpp" ]; then
    echo "✅ Headers found"
else
    echo "❌ Headers not found: $TT_METAL_HOME/tt_metal/host_api.hpp"
    exit 1
fi

# Check libraries
if [ -f "$TT_METAL_HOME/build/lib/libtt_metal.so" ]; then
    echo "✅ Libraries found"
else
    echo "❌ Libraries not found: $TT_METAL_HOME/build/lib/libtt_metal.so"
    echo "Run: cd $TT_METAL_HOME && cmake -B build && cmake --build build"
    exit 1
fi

# Check package config
if [ -f "$TT_METAL_HOME/build/cmake/tt-metalium-config.cmake" ]; then
    echo "✅ Package config found"
elif [ -f "$TT_METAL_HOME/build/lib/cmake/tt-metalium/tt-metalium-config.cmake" ]; then
    echo "✅ Package config found (installed location)"
else
    echo "⚠️  Package config not found (will use FindMetalium.cmake fallback)"
fi

echo ""
echo "=== SDK Ready! ==="
echo "To build TileLang with real SDK:"
echo "  export CMAKE_PREFIX_PATH=$TT_METAL_HOME/build"
echo "  cmake -B build -DUSE_LLVM=true -DUSE_REAL_METALIUM=ON"
echo "  cmake --build build -j\$(nproc)"
```

---

## SDK Location Summary

### Local Development

**Recommended**:
```bash
~/tt-metal/                 # User's home directory
├── build/
│   ├── lib/
│   │   ├── libtt_metal.so
│   │   └── libdevice.so
│   └── cmake/              # Or lib/cmake/
│       └── tt-metalium/
│           └── tt-metalium-config.cmake
└── tt_metal/
    └── host_api.hpp
```

**Environment**:
```bash
export TT_METAL_HOME=~/tt-metal
export CMAKE_PREFIX_PATH=~/tt-metal/build
```

---

### GitHub CI

**Location**: `/home/runner/tt-metal`

**Cache**: GitHub Actions cache, ~1.5GB

**Key**: `tt-metal-{{ env.TT_METAL_VERSION }}-{{ runner.os }}`

---

## Testing Plan

### Mock Mode (Default) ✅

```bash
# Already working
bash maint/scripts/local_build_and_test_tt.sh --skip-deps --jobs 4
# ✅ 95/95 tests pass
```

### Real SDK Mode (After Task 4)

```bash
# Install SDK
export TT_METAL_HOME=~/tt-metal
git clone https://github.com/tenstorrent/tt-metal.git $TT_METAL_HOME
cd $TT_METAL_HOME && cmake -B build && cmake --build build

# Build TileLang
cd /path/to/tilelang-tt
bash maint/scripts/local_build_and_test_tt.sh --with-metalium --jobs 4

# Should output:
# "Building with REAL TT-Metalium APIs"
# "Found TT-Metalium package config: ~/tt-metal/build/cmake/tt-metalium"
# "Linked TileLang against TT::Metalium target"
# ✅ 95/95 tests pass
```

---

## Next Steps

1. ✅ **Commit current work** (Tasks 1-3 complete)
2. 🚧 **Complete Task 4**: Update local build script
3. 🚧 **Complete Task 5**: Create Real SDK CI workflow
4. 🚧 **Complete Task 6**: Update documentation
5. 🚧 **Complete Task 7**: Create verification script
6. 🧪 **Test end-to-end**: Build with real SDK locally
7. 🧪 **Test CI**: Trigger Real SDK CI workflow

---

## Commit Message Template

```
Implement External SDK Integration for TT-Metalium

Part 1: CMake Integration (Tasks 1-3)

This commit implements the foundational CMake integration for external
TT-Metalium SDK, following industry standard patterns (like CUDA/ROCm).

Changes:
- Updated CMakeLists.txt to use find_package(TT-Metalium)
- Enhanced FindMetalium.cmake to create TT::Metalium target
- Support both official package config and TT_METAL_HOME fallback
- Improved error messages with setup instructions

Technical Details:
- find_package(TT-Metalium) → uses tt-metal's package config
- find_package(Metalium) → fallback for custom locations
- TT::Metalium target → modern CMake imported target
- Graceful fallback to mock mode if SDK not found

Testing:
- Mock mode verified working (default)
- Ready for real SDK testing (requires tt-metal installed)

Next: Local build script and CI workflow (Tasks 4-7)

See docs/tenstorrent/EXTERNAL_SDK_IMPLEMENTATION_PLAN.md
See docs/tenstorrent/EXTERNAL_SDK_STATUS.md
```

---

**Status**: 3/7 tasks complete, 4 remaining
**Progress**: ~43%
**Blocked**: No, ready to proceed with Tasks 4-7
