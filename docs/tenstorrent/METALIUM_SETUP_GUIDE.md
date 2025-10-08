# TT-Metalium SDK Setup Guide

**Version**: 1.0
**Date**: 2025-10-08
**Status**: Official Setup Method

---

## Overview

TileLang's Tenstorrent backend supports two build modes:

1. **Mock Mode** (default): Dry-run code generation, no hardware required
2. **Real Mode**: Actual Metalium SDK integration for hardware execution

This guide covers setting up **Real Mode** with the TT-Metalium SDK.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Methods](#installation-methods)
3. [Building TileLang with Metalium](#building-tilelang-with-metalium)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

- **Tenstorrent Device**: Grayskull or Wormhole accelerator card
- **System**: Linux with glibc 2.34+ (Ubuntu 22.04 recommended)
- **Python**: 3.10 or later

### Software Requirements

Before installing Metalium SDK:

```bash
# Update system
sudo apt-get update

# Install build essentials
sudo apt-get install -y build-essential cmake ninja-build

# Install Python 3.10+
sudo apt-get install -y python3 python3-pip python3-venv

# Install git
sudo apt-get install -y git
```

---

## Installation Methods

### Method 1: Official Installation (Recommended)

**Step 1: Clone tt-metal Repository**

```bash
# Choose installation directory
export METALIUM_INSTALL_DIR="$HOME/tt-metal"

# Clone with submodules
git clone https://github.com/tenstorrent/tt-metal.git "$METALIUM_INSTALL_DIR" --recurse-submodules
cd "$METALIUM_INSTALL_DIR"
```

**Step 2: Build tt-metal**

```bash
# Use official build script
./build_metal.sh

# This will:
# - Build tt_metal libraries
# - Build device kernels
# - Install Python packages
# - Set up environment
```

**Step 3: Set Environment Variable**

```bash
# Set TT_METAL_HOME (required for TileLang to find SDK)
export TT_METAL_HOME="$METALIUM_INSTALL_DIR"

# Make permanent (add to ~/.bashrc)
echo "export TT_METAL_HOME=$METALIUM_INSTALL_DIR" >> ~/.bashrc
source ~/.bashrc
```

**Verify Installation**:

```bash
# Check environment variable
echo $TT_METAL_HOME
# Should output: /home/youruser/tt-metal

# Check libraries exist
ls -la $TT_METAL_HOME/build/lib/libtt_metal.so
ls -la $TT_METAL_HOME/build/lib/libdevice.so
```

---

### Method 2: Using TT-Installer (If Available)

For systems with Tenstorrent hardware:

```bash
# Install using TT-Installer (if available)
# This installs drivers, firmware, and SDK
sudo tt-installer install

# Set environment
export TT_METAL_HOME="/opt/tt-metal"  # Or wherever installer places it
```

---

### Method 3: System-Wide Installation

If you prefer system-wide installation:

```bash
# Build and install to /usr/local
cd "$METALIUM_INSTALL_DIR"
./build_metal.sh
sudo cmake --install build --prefix /usr/local

# Set environment
export TT_METAL_HOME="/usr/local"
```

---

## Building TileLang with Metalium

Once TT-Metalium SDK is installed:

### Quick Build (Automated Script)

```bash
cd /path/to/tilelang-tt

# Set TT_METAL_HOME if not already set
export TT_METAL_HOME="$HOME/tt-metal"

# Build with LLVM and Real Metalium
cmake -B build \
    -DUSE_LLVM=true \
    -DUSE_REAL_METALIUM=ON \
    -G Ninja

cmake --build build -j$(nproc)

# Install
pip install -e . --no-build-isolation
```

**Expected Output**:

```
-- Found TT_METAL_HOME: /home/user/tt-metal
-- Building with REAL TT-Metalium APIs
-- Metalium version: X.Y.Z
-- Metalium includes: /home/user/tt-metal/tt_metal;...
-- Metalium libraries: /home/user/tt-metal/build/lib/libtt_metal.so;...
```

---

### Manual Build (Step by Step)

**Step 1: Configure CMake**

```bash
cd /path/to/tilelang-tt

# Create build directory
mkdir -p build
cd build

# Configure with Metalium
cmake .. \
    -DUSE_LLVM=true \
    -DUSE_REAL_METALIUM=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -G Ninja

# CMake will:
# 1. Search for $TT_METAL_HOME
# 2. Run FindMetalium.cmake
# 3. Locate headers and libraries
# 4. Apply -DTL_USE_REAL_METALIUM compile flag
```

**Step 2: Build**

```bash
# Build C++ components
ninja -j$(nproc)

# Or use cmake
cmake --build . -j$(nproc)
```

**Step 3: Install Python Package**

```bash
cd /path/to/tilelang-tt

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e . --no-build-isolation
```

---

### Build Without Metalium (Mock Mode)

If you don't have hardware or SDK:

```bash
# Default build (mock mode)
cmake -B build -DUSE_LLVM=true
cmake --build build -j$(nproc)

# USE_REAL_METALIUM defaults to OFF
# Will output: "Building with MOCK TT-Metalium APIs (dry-run only)"
```

---

## Verification

### Verify SDK Detection

```bash
cd /path/to/tilelang-tt/build

# Check CMake cache for Metalium variables
cmake -L | grep Metalium

# Expected output:
# Metalium_FOUND:BOOL=TRUE
# Metalium_INCLUDE_DIR:PATH=/home/user/tt-metal/tt_metal
# Metalium_LIBRARY_TT_METAL:FILEPATH=/home/user/tt-metal/build/lib/libtt_metal.so
# USE_REAL_METALIUM:BOOL=ON
```

### Verify Compilation

```bash
# Check that TL_USE_REAL_METALIUM is defined
cd /path/to/tilelang-tt/build

# Look for the define in compile commands
grep "TL_USE_REAL_METALIUM" compile_commands.json

# Should find: -DTL_USE_REAL_METALIUM
```

### Run Tests

```bash
cd /path/to/tilelang-tt

# Activate environment
source .venv/bin/activate

# Set library path
export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH

# Run Tenstorrent tests
pytest testing/python/tt/ -v

# All 95 tests should pass (mock APIs used in tests)
```

### Verify Generated Code Uses Real APIs

```bash
# Generate a simple matmul
python examples/tenstorrent/example_matmul_tt_poc.py

# Check generated host program
cat /tmp/tt_artifacts/main.cpp | head -20

# Should see:
# #include "tt_metal/host_api.hpp"
# #include "tt_metal/impl/device/mesh_device.hpp"
# auto device = distributed::MeshDevice::create_unit_mesh(0);
```

---

## Troubleshooting

### Issue 1: TT_METAL_HOME Not Found

**Error**:
```
-- TT_METAL_HOME environment variable not set
-- USE_REAL_METALIUM is ON but Metalium SDK not found
-- Falling back to MOCK Metalium APIs (dry-run only)
```

**Solution**:
```bash
# Set environment variable
export TT_METAL_HOME="/path/to/tt-metal"

# Reconfigure CMake
cd /path/to/tilelang-tt
rm -rf build
cmake -B build -DUSE_LLVM=true -DUSE_REAL_METALIUM=ON
```

---

### Issue 2: Libraries Not Found

**Error**:
```
-- Could NOT find Metalium: Found unsuitable version "", but required is at least "0"
```

**Solution**:
```bash
# Verify libraries exist
ls -la $TT_METAL_HOME/build/lib/libtt_metal.so
ls -la $TT_METAL_HOME/build/lib/libdevice.so

# If missing, rebuild tt-metal
cd $TT_METAL_HOME
./build_metal.sh
```

---

### Issue 3: Include Headers Not Found

**Error** (during compilation):
```
fatal error: tt_metal/host_api.hpp: No such file or directory
```

**Solution**:
```bash
# Verify headers exist
ls -la $TT_METAL_HOME/tt_metal/host_api.hpp

# Check CMake found correct path
cmake -L | grep Metalium_INCLUDE

# Reconfigure if needed
cd /path/to/tilelang-tt
rm -rf build
cmake -B build -DUSE_LLVM=true -DUSE_REAL_METALIUM=ON
```

---

### Issue 4: Version Mismatch

**Error**:
```
-- Metalium version: unknown
```

**Solution**:
```bash
# Check if VERSION file exists
cat $TT_METAL_HOME/VERSION

# If missing, update tt-metal
cd $TT_METAL_HOME
git pull
./build_metal.sh
```

---

### Issue 5: Link Errors

**Error** (during linking):
```
undefined reference to `tt::tt_metal::CreateDevice()`
```

**Solution**:
```bash
# Check library path
echo $TT_METAL_HOME/build/lib

# Verify libraries are linked
ldd /path/to/tilelang-tt/build/libtilelang.so | grep tt_metal

# Should show: libtt_metal.so => /path/to/tt-metal/build/lib/libtt_metal.so
```

---

## Environment Setup Examples

### For Development

```bash
# ~/.bashrc or ~/.zshrc

# TT-Metalium SDK
export TT_METAL_HOME="$HOME/tt-metal"

# TileLang development
export TILELANG_HOME="$HOME/tilelang-tt"
export LD_LIBRARY_PATH="$TILELANG_HOME/build/tvm:$LD_LIBRARY_PATH"

# Python virtual environment
alias tl-activate="source $TILELANG_HOME/.venv/bin/activate"
```

### For Production

```bash
# System-wide installation

# /etc/environment
TT_METAL_HOME=/opt/tt-metal
TILELANG_HOME=/opt/tilelang

# /etc/ld.so.conf.d/tilelang.conf
/opt/tilelang/build/tvm
/opt/tt-metal/build/lib

# Update cache
sudo ldconfig
```

---

## Build Options Reference

| Option | Default | Description |
|--------|---------|-------------|
| `USE_LLVM` | OFF | Enable LLVM backend (required for TT) |
| `USE_REAL_METALIUM` | OFF | Use real Metalium SDK instead of mock |
| `TT_METAL_HOME` | (env) | Path to tt-metal installation |

### Example Build Configurations

**Development (Mock Mode)**:
```bash
cmake -B build -DUSE_LLVM=true
# Fast builds, no hardware needed
```

**Testing (Real Mode)**:
```bash
export TT_METAL_HOME="$HOME/tt-metal"
cmake -B build -DUSE_LLVM=true -DUSE_REAL_METALIUM=ON
# Test with real SDK, requires hardware
```

**Production (Optimized)**:
```bash
export TT_METAL_HOME="/opt/tt-metal"
cmake -B build \
    -DUSE_LLVM=true \
    -DUSE_REAL_METALIUM=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/tilelang
cmake --build build -j$(nproc)
cmake --install build
```

---

## Next Steps

After successful setup:

1. **Validate SDK Integration**: See `METALIUM_SDK_VALIDATION_PLAN.md`
   - Phase 1: Dry-run compilation
   - Phase 2: API completion
   - Phase 3: Hardware execution

2. **Run Examples**:
   ```bash
   python examples/tenstorrent/example_matmul_tt_poc.py
   ```

3. **Hardware Testing**: See hardware validation guide (coming soon)

---

## Summary

**Setup Steps**:
1. ✅ Install TT-Metalium SDK (`git clone` + `./build_metal.sh`)
2. ✅ Set `TT_METAL_HOME` environment variable
3. ✅ Build TileLang with `-DUSE_REAL_METALIUM=ON`
4. ✅ Verify with `cmake -L | grep Metalium`

**Key Points**:
- Metalium is **external dependency** (like CUDA/ROCm)
- CMake **finds SDK automatically** via `FindMetalium.cmake`
- **Graceful fallback** to mock mode if SDK not found
- **No submodules** - clean separation

**Documentation**:
- This guide: Setup instructions
- `FindMetalium.cmake`: CMake module (automatic)
- `METALIUM_SDK_VALIDATION_PLAN.md`: Validation phases
- `CLAUDE.md`: Build reference

---

**Questions?** See troubleshooting section or check SDK documentation at:
https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/installing.html

---

**END OF SETUP GUIDE**
