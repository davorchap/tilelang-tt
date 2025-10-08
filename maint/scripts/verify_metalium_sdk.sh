#!/bin/bash
# TT-Metalium SDK Verification Script
#
# This script verifies that a TT-Metalium SDK installation is valid and ready
# to use with TileLang.
#
# Usage:
#   bash maint/scripts/verify_metalium_sdk.sh [TT_METAL_HOME]
#
# Examples:
#   bash maint/scripts/verify_metalium_sdk.sh              # Uses $TT_METAL_HOME
#   bash maint/scripts/verify_metalium_sdk.sh ~/tt-metal  # Uses provided path

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TT-Metalium SDK Verification${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Determine SDK path
if [ -n "$1" ]; then
    SDK_PATH="$1"
    echo -e "${YELLOW}Using provided path: $SDK_PATH${NC}"
elif [ -n "$TT_METAL_HOME" ]; then
    SDK_PATH="$TT_METAL_HOME"
    echo -e "${YELLOW}Using TT_METAL_HOME: $SDK_PATH${NC}"
else
    echo -e "${RED}Error: No SDK path provided${NC}"
    echo -e "${YELLOW}Usage: $0 [path_to_tt_metal]${NC}"
    echo -e "${YELLOW}Or set TT_METAL_HOME environment variable${NC}"
    echo ""
    echo -e "${YELLOW}Example:${NC}"
    echo -e "  export TT_METAL_HOME=~/tt-metal"
    echo -e "  $0"
    echo ""
    exit 1
fi

echo ""

# Check 1: Directory exists
echo -e "${BLUE}[1/6] Checking SDK directory...${NC}"
if [ ! -d "$SDK_PATH" ]; then
    echo -e "${RED}❌ Directory not found: $SDK_PATH${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Directory exists${NC}"
fi

# Check 2: Headers
echo -e "${BLUE}[2/6] Checking headers...${NC}"
REQUIRED_HEADERS=(
    "tt_metal/host_api.hpp"
    "tt_metal/impl/device/mesh_device.hpp"
    "tt_metal/impl/buffers/mesh_buffer.hpp"
)

HEADERS_OK=true
for header in "${REQUIRED_HEADERS[@]}"; do
    if [ -f "$SDK_PATH/$header" ]; then
        echo -e "${GREEN}  ✅ $header${NC}"
    else
        echo -e "${RED}  ❌ $header${NC}"
        HEADERS_OK=false
    fi
done

if [ "$HEADERS_OK" = false ]; then
    echo -e "${RED}❌ Some headers are missing${NC}"
    exit 1
else
    echo -e "${GREEN}✅ All required headers found${NC}"
fi

# Check 3: Libraries
echo -e "${BLUE}[3/6] Checking libraries...${NC}"
REQUIRED_LIBS=(
    "build/lib/libtt_metal.so"
    "build/lib/libdevice.so"
)

LIBS_OK=true
for lib in "${REQUIRED_LIBS[@]}"; do
    if [ -f "$SDK_PATH/$lib" ]; then
        SIZE=$(ls -lh "$SDK_PATH/$lib" | awk '{print $5}')
        echo -e "${GREEN}  ✅ $lib ($SIZE)${NC}"
    else
        echo -e "${RED}  ❌ $lib${NC}"
        LIBS_OK=false
    fi
done

if [ "$LIBS_OK" = false ]; then
    echo -e "${RED}❌ Libraries not built${NC}"
    echo -e "${YELLOW}Please build tt-metal:${NC}"
    echo -e "  cd $SDK_PATH"
    echo -e "  cmake -B build -G Ninja"
    echo -e "  cmake --build build -j\$(nproc)"
    exit 1
else
    echo -e "${GREEN}✅ All required libraries found${NC}"
fi

# Check 4: Package config (optional but recommended)
echo -e "${BLUE}[4/6] Checking CMake package config...${NC}"
PACKAGE_CONFIG_PATHS=(
    "build/lib/cmake/tt-metalium/tt-metalium-config.cmake"
    "build/cmake/tt-metalium-config.cmake"
)

PACKAGE_CONFIG_FOUND=false
for config_path in "${PACKAGE_CONFIG_PATHS[@]}"; do
    if [ -f "$SDK_PATH/$config_path" ]; then
        echo -e "${GREEN}  ✅ Found: $config_path${NC}"
        PACKAGE_CONFIG_FOUND=true
        break
    fi
done

if [ "$PACKAGE_CONFIG_FOUND" = false ]; then
    echo -e "${YELLOW}  ⚠️  Package config not found (will use FindMetalium.cmake fallback)${NC}"
else
    echo -e "${GREEN}✅ Package config found${NC}"
fi

# Check 5: Version (if available)
echo -e "${BLUE}[5/6] Checking SDK version...${NC}"
if [ -f "$SDK_PATH/VERSION" ]; then
    VERSION=$(cat "$SDK_PATH/VERSION")
    echo -e "${GREEN}  ✅ Version: $VERSION${NC}"
elif [ -d "$SDK_PATH/.git" ]; then
    cd "$SDK_PATH"
    GIT_VERSION=$(git describe --tags --always 2>/dev/null || echo "unknown")
    echo -e "${GREEN}  ✅ Git version: $GIT_VERSION${NC}"
    cd - > /dev/null
else
    echo -e "${YELLOW}  ⚠️  Version file not found${NC}"
fi

# Check 6: Test CMake can find it
echo -e "${BLUE}[6/6] Testing CMake discovery...${NC}"

# Create temporary test directory
TEST_DIR=$(mktemp -d)
cd "$TEST_DIR"

# Create minimal CMakeLists.txt
cat > CMakeLists.txt << EOF
cmake_minimum_required(VERSION 3.18)
project(tt_metalium_test)

# Try to find TT-Metalium package
find_package(TT-Metalium QUIET)

if(TT-Metalium_FOUND OR TARGET TT::Metalium)
    message(STATUS "SUCCESS: Found TT-Metalium via package config")
    message(STATUS "TT-Metalium_DIR: \${TT-Metalium_DIR}")
else()
    # Try fallback
    list(APPEND CMAKE_MODULE_PATH "\${CMAKE_CURRENT_SOURCE_DIR}")
    find_package(Metalium QUIET)

    if(Metalium_FOUND OR TARGET TT::Metalium)
        message(STATUS "SUCCESS: Found Metalium via FindMetalium.cmake")
    else()
        message(FATAL_ERROR "FAILED: Could not find TT-Metalium")
    endif()
endif()
EOF

# Run CMake test
if CMAKE_PREFIX_PATH="$SDK_PATH/build" TT_METAL_HOME="$SDK_PATH" \
   cmake . > cmake_output.txt 2>&1; then
    if grep -q "SUCCESS:" cmake_output.txt; then
        echo -e "${GREEN}  ✅ CMake can find SDK${NC}"
        grep "SUCCESS:" cmake_output.txt | sed 's/^/  /'
    else
        echo -e "${YELLOW}  ⚠️  CMake configured but no success message${NC}"
    fi
else
    echo -e "${RED}  ❌ CMake failed to find SDK${NC}"
    echo -e "${YELLOW}  CMake output:${NC}"
    cat cmake_output.txt | sed 's/^/    /'
fi

# Cleanup
cd - > /dev/null
rm -rf "$TEST_DIR"

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ SDK Verification Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}SDK Location: $SDK_PATH${NC}"
echo ""
echo -e "${YELLOW}To build TileLang with this SDK:${NC}"
echo ""
echo -e "  ${BLUE}# Option 1: Using TT_METAL_HOME${NC}"
echo -e "  export TT_METAL_HOME=$SDK_PATH"
echo -e "  cmake -B build -DUSE_LLVM=true -DUSE_REAL_METALIUM=ON"
echo -e "  cmake --build build -j\$(nproc)"
echo ""
echo -e "  ${BLUE}# Option 2: Using CMAKE_PREFIX_PATH${NC}"
echo -e "  export CMAKE_PREFIX_PATH=$SDK_PATH/build"
echo -e "  cmake -B build -DUSE_LLVM=true -DUSE_REAL_METALIUM=ON"
echo -e "  cmake --build build -j\$(nproc)"
echo ""
echo -e "  ${BLUE}# Option 3: Using build script${NC}"
echo -e "  export TT_METAL_HOME=$SDK_PATH"
echo -e "  bash maint/scripts/local_build_and_test_tt.sh --with-metalium --jobs 4"
echo ""
echo -e "${YELLOW}For more information:${NC}"
echo -e "  See docs/tenstorrent/METALIUM_SETUP_GUIDE.md"
echo ""
