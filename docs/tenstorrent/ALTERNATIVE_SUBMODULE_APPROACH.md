# Alternative: Metalium as Git Submodule

**Status**: NOT RECOMMENDED
**Date**: 2025-10-08

---

## ⚠️ Why This is NOT Recommended

1. **Hardware Required**: tt-metal requires Tenstorrent device to build/use
2. **Complex Dependencies**: Drivers, firmware, system requirements
3. **Build Conflicts**: Their build_metal.sh vs our CMake
4. **Repository Bloat**: Large repo with own submodules
5. **Not Standard Practice**: Hardware SDKs are typically external

**Current Approach (Recommended)**: Treat like CUDA/ROCm - external SDK

---

## If You Proceed Anyway

### Step 1: Add Submodule

```bash
# Add tt-metal as submodule
git submodule add https://github.com/tenstorrent/tt-metal.git 3rdparty/tt-metal
git submodule update --init --recursive
```

### Step 2: Update CMakeLists.txt

```cmake
# Option to build with bundled Metalium
option(USE_BUNDLED_METALIUM "Use bundled tt-metal submodule" OFF)

if(USE_BUNDLED_METALIUM)
    # Add tt-metal subdirectory
    set(TT_METAL_HOME ${CMAKE_SOURCE_DIR}/3rdparty/tt-metal)

    # Option 1: Use their build script (messy)
    execute_process(
        COMMAND ${TT_METAL_HOME}/build_metal.sh
        WORKING_DIRECTORY ${TT_METAL_HOME}
    )

    # Option 2: Try to integrate their CMake (may conflict)
    # add_subdirectory(3rdparty/tt-metal EXCLUDE_FROM_ALL)

    # Set include/library paths
    set(Metalium_INCLUDE_DIRS
        ${TT_METAL_HOME}/tt_metal
        ${TT_METAL_HOME}/tt_metal/impl
    )
    set(Metalium_LIBRARIES
        ${TT_METAL_HOME}/build/lib/libtt_metal.so
        ${TT_METAL_HOME}/build/lib/libdevice.so
    )

    # Enable real mode automatically
    set(USE_REAL_METALIUM ON)
    add_compile_definitions(TL_USE_REAL_METALIUM)
else()
    # Use existing FindMetalium.cmake approach (recommended)
    if(USE_REAL_METALIUM)
        find_package(Metalium)
    endif()
endif()
```

### Step 3: Update .gitmodules

```ini
[submodule "3rdparty/tt-metal"]
    path = 3rdparty/tt-metal
    url = https://github.com/tenstorrent/tt-metal.git
    branch = main
```

### Issues You'll Face

1. **Build Script Incompatibility**:
   - Their `build_metal.sh` expects specific environment
   - Hard to integrate with our CMake build
   - May create build artifacts in unexpected places

2. **Circular Dependencies**:
   - tt-metal may have conflicting dependencies with TVM
   - Both have complex build systems

3. **Hardware Requirements**:
   - Still can't build/test without Tenstorrent device
   - CI won't work without hardware

4. **Repository Size**:
   - Clone time increases significantly
   - Disk space issues for developers

5. **Version Conflicts**:
   - Hard to update (their submodules + ours)
   - Breaking changes harder to manage

---

## Better Alternative: Provide Setup Script

Instead of submodule, provide a convenience script:

```bash
#!/bin/bash
# scripts/setup_metalium.sh

echo "Setting up TT-Metalium SDK..."

# Option 1: Clone and build
METALIUM_DIR="${METALIUM_DIR:-$HOME/tt-metal}"

if [ ! -d "$METALIUM_DIR" ]; then
    echo "Cloning tt-metal to $METALIUM_DIR..."
    git clone https://github.com/tenstorrent/tt-metal.git "$METALIUM_DIR" --recurse-submodules
    cd "$METALIUM_DIR"
    ./build_metal.sh
else
    echo "tt-metal already exists at $METALIUM_DIR"
fi

# Set environment variable
export TT_METAL_HOME="$METALIUM_DIR"
echo "export TT_METAL_HOME=$METALIUM_DIR" >> ~/.bashrc

echo "✅ Metalium SDK setup complete!"
echo "Now run: cmake -B build -DUSE_REAL_METALIUM=ON"
```

---

## Conclusion

**Recommendation**: Keep the current environment-based approach (like CUDA/ROCm)

**If you must bundle it**: Use setup script, not submodule

**Why**: Hardware SDKs are external by nature - this is industry standard
