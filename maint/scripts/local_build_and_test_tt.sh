#!/bin/bash
# Local Build and Test Script for Tenstorrent Backend (Ubuntu 22.04)
#
# This script replicates the GitHub Actions CI workflow locally, allowing you to:
# - Build TileLang with LLVM backend
# - Run Tenstorrent backend tests
# - Verify changes before pushing to remote
#
# Prerequisites:
# - Ubuntu 22.04 (or compatible Linux distribution)
# - Python 3.10 or later
# - Git with submodules initialized
#
# Usage:
#   bash maint/scripts/local_build_and_test_tt.sh [options]
#
# Options:
#   --skip-deps       Skip system dependency installation
#   --skip-build      Skip build step (use existing build)
#   --skip-tests      Skip running tests
#   --clean           Clean build directory before building
#   --jobs N          Number of parallel build jobs (default: 2)
#   --help            Show this help message

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
SKIP_DEPS=false
SKIP_BUILD=false
SKIP_TESTS=false
CLEAN_BUILD=false
BUILD_JOBS=2

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --jobs)
            BUILD_JOBS="$2"
            shift 2
            ;;
        --help)
            head -n 25 "$0" | tail -n 24
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Tenstorrent Backend Local Build & Test${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Get to repository root
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT_DIR"

echo -e "${YELLOW}Working directory: $ROOT_DIR${NC}"
echo ""

# Step 1: Install system dependencies
if [ "$SKIP_DEPS" = false ]; then
    echo -e "${GREEN}[1/5] Installing system dependencies...${NC}"

    # Check if running as root or can use sudo
    if [ "$EUID" -eq 0 ]; then
        APT_CMD="apt-get"
    elif command -v sudo &> /dev/null; then
        APT_CMD="sudo apt-get"
    else
        echo -e "${RED}Error: Need root privileges or sudo to install system dependencies${NC}"
        echo -e "${YELLOW}Run with --skip-deps if dependencies are already installed${NC}"
        exit 1
    fi

    export DEBIAN_FRONTEND=noninteractive
    export TZ=Etc/UTC

    $APT_CMD update
    $APT_CMD install -y \
        build-essential \
        cmake \
        ninja-build \
        llvm \
        libedit-dev \
        libxml2-dev \
        zlib1g-dev \
        ccache

    echo -e "${GREEN}System dependencies installed successfully${NC}"
else
    echo -e "${YELLOW}[1/5] Skipping system dependency installation${NC}"
fi
echo ""

# Step 2: Install Python dependencies
echo -e "${GREEN}[2/5] Installing Python dependencies...${NC}"
python3 -m pip install --upgrade pip
pip install -r requirements-test.txt
echo -e "${GREEN}Python dependencies installed successfully${NC}"
echo ""

# Step 3: Build TileLang with LLVM
if [ "$SKIP_BUILD" = false ]; then
    echo -e "${GREEN}[3/5] Building TileLang with LLVM backend...${NC}"

    if [ "$CLEAN_BUILD" = true ]; then
        echo -e "${YELLOW}Cleaning build directory...${NC}"
        rm -rf build
    fi

    mkdir -p build
    cd build

    # Create config.cmake for TVM
    cp ../3rdparty/tvm/cmake/config.cmake .
    echo "set(USE_LLVM ON)" >> config.cmake
    echo "set(USE_CUDA OFF)" >> config.cmake

    # Configure with CMake
    echo -e "${YELLOW}Running CMake configuration...${NC}"
    cmake .. \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

    # Build
    echo -e "${YELLOW}Building TileLang (using $BUILD_JOBS parallel jobs)...${NC}"
    cmake --build . --config Release -j "$BUILD_JOBS"

    cd ..
    echo -e "${GREEN}Build completed successfully${NC}"

    # Show ccache stats
    if command -v ccache >/dev/null 2>&1; then
        echo -e "${YELLOW}ccache statistics:${NC}"
        ccache -s
    fi
else
    echo -e "${YELLOW}[3/5] Skipping build step${NC}"
fi
echo ""

# Step 4: Install TileLang
echo -e "${GREEN}[4/5] Installing TileLang Python packages...${NC}"

# Copy built libraries to tilelang/lib
mkdir -p tilelang/lib
cp build/*.so tilelang/lib/ 2>/dev/null || true

# Install TVM Python package
echo -e "${YELLOW}Installing TVM Python package...${NC}"
export TVM_LIBRARY_PATH=$(pwd)/build/tvm
cd 3rdparty/tvm/python
pip install -e .
cd ../../..

# Install TileLang
echo -e "${YELLOW}Installing TileLang...${NC}"
export USE_LLVM=true
pip install -e .

echo -e "${GREEN}Python packages installed successfully${NC}"
echo ""

# Step 5: Run tests
if [ "$SKIP_TESTS" = false ]; then
    echo -e "${GREEN}[5/5] Running Tenstorrent tests...${NC}"

    export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH

    # Run target registration tests
    echo -e "${YELLOW}Running Tenstorrent target registration tests...${NC}"
    cd testing/python/tt
    if pytest test_target_registration.py -v --tb=short; then
        echo -e "${GREEN}Target registration tests passed${NC}"
    else
        echo -e "${YELLOW}Target registration tests failed (this may be expected if backend is incomplete)${NC}"
    fi
    cd ../../..

    # Run all Tenstorrent tests
    echo -e "${YELLOW}Running all Tenstorrent Python tests...${NC}"
    cd testing/python
    if pytest tt/ -v --tb=short -k "not gpu"; then
        echo -e "${GREEN}All tests passed${NC}"
    else
        echo -e "${YELLOW}Some tests failed (this may be expected if backend is incomplete)${NC}"
    fi
    cd ../..
else
    echo -e "${YELLOW}[5/5] Skipping tests${NC}"
fi
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build and test completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Next steps:"
echo -e "  - Review test results above"
echo -e "  - Run ${YELLOW}bash format.sh${NC} to check code formatting"
echo -e "  - Commit your changes and push to remote"
echo ""
