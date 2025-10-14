# Local Build and Test Guide for Tenstorrent Backend

This guide explains how to build and test the TileLang Tenstorrent backend locally using the automated build script.

## Overview

The `maint/scripts/local_build_and_test_tt.sh` script automates the entire build and test process, replicating the GitHub Actions CI workflow on your local machine. It handles:

1. System dependency installation
2. Python virtual environment setup
3. TileLang compilation with LLVM backend
4. Tenstorrent backend tests

## Prerequisites

- **Operating System:** Ubuntu 22.04 or 24.04 (or compatible Linux distribution)
- **Python:** 3.10 or later
- **Git:** With submodules initialized
- **Disk Space:** ~2-3 GB for build artifacts and dependencies
- **RAM:** Minimum 4 GB recommended (8+ GB for faster parallel builds)

## Quick Start

```bash
# Clone the repository (if not already done)
git clone https://github.com/davorchap/tilelang-tt.git
cd tilelang-tt

# Run the build script (automatically initializes submodules)
bash maint/scripts/local_build_and_test_tt.sh
```

The script will:
1. Check and initialize git submodules (if needed)
2. Install system dependencies (requires sudo)
3. Create a Python virtual environment (`.venv`)
4. Build TileLang with LLVM backend
5. Run Tenstorrent tests

**Total time:** ~5-10 minutes on first run, ~1 minute on subsequent runs (with ccache)

**Note:** The script automatically checks and initializes submodules if they haven't been set up yet. No manual initialization is required.

## What Gets Installed

### System-Wide Dependencies (requires sudo)

The following packages are installed system-wide via `apt-get`:

| Package | Purpose |
|---------|---------|
| `build-essential` | C/C++ compiler toolchain (gcc, g++, make) |
| `cmake` | Build system generator |
| `ninja-build` | Fast build system (used by CMake) |
| `llvm` | LLVM compiler infrastructure and tools |
| `libedit-dev` | Command-line editing library (TVM dependency) |
| `libxml2-dev` | XML parsing library (TVM dependency) |
| `zlib1g-dev` | Compression library (TVM dependency) |
| `ccache` | Compiler cache for faster incremental builds |
| `python3-venv` | Python virtual environment support |

**Total disk space:** ~600 MB

### Virtual Environment Dependencies

All Python packages are installed in an isolated virtual environment (`.venv/`) and do **not** affect system Python:

#### Core Dependencies (from `requirements-test.txt`)

- **Build tools:** `Cython`, `cmake`, `wheel`, `setuptools`
- **Testing frameworks:** `pytest`, `pytest-xdist`, `pytest-timeout`, `pytest-durations`
- **Linting tools:** `yapf`, `ruff`, `codespell`, `cpplint`
- **Scientific computing:** `numpy`, `scipy`, `torch`, `einops`
- **Utilities:** `tqdm`, `PyYAML`, `packaging`, `requests`, `cloudpickle`, `psutil`, `tabulate`
- **Type checking:** `typing_extensions`, `ml_dtypes`
- **Config parsing:** `cffi`, `dtlib`
- **Documentation:** `docutils`
- **Network:** `tornado`

> **Note on PyTorch:** The Tenstorrent backend works with CPU-only PyTorch (no CUDA required). You can save ~5GB of disk space by using:
> ```bash
> pip uninstall -y torch triton
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```
> CPU-only PyTorch provides all functionality needed for TT backend development (tensor operations, NumPy interop, DLPack conversion). See `testing/python/tenstorrent/test_pytorch_cpu_compatibility.py` for verification tests.

#### TVM and TileLang (editable installs)

- **TVM:** Installed from `3rdparty/tvm/python` (editable mode)
- **TileLang:** Installed from repository root (editable mode)

**Total virtual environment size:** ~1-2 GB

### Build Artifacts

Build artifacts are stored in the `build/` directory:

- `build/tvm/` - TVM shared libraries (`libtvm.so`, `libtvm_runtime.so`)
- `build/libtilelang*.so` - TileLang shared libraries
- `build/.ninja_*` - Ninja build files
- `build/CMakeFiles/` - CMake metadata

**Total build artifacts:** ~500 MB - 1 GB

## Script Options

```bash
bash maint/scripts/local_build_and_test_tt.sh [options]
```

### Available Options

| Option | Description |
|--------|-------------|
| `--skip-deps` | Skip system dependency installation (use if already installed) |
| `--skip-build` | Skip build step (use existing build artifacts) |
| `--skip-tests` | Skip running tests (build only) |
| `--clean` | Clean build directory before building (force fresh build) |
| `--jobs N` | Number of parallel build jobs (default: 2, recommended: 4-8) |
| `--help` | Show help message |

### Example Usage

**Fast incremental build (dependencies already installed):**
```bash
bash maint/scripts/local_build_and_test_tt.sh --skip-deps --jobs 8
```

**Clean rebuild:**
```bash
bash maint/scripts/local_build_and_test_tt.sh --skip-deps --clean --jobs 4
```

**Build only (no tests):**
```bash
bash maint/scripts/local_build_and_test_tt.sh --skip-tests
```

**Run tests only (using existing build):**
```bash
bash maint/scripts/local_build_and_test_tt.sh --skip-deps --skip-build
```

## Build Process Details

### Step 1: System Dependencies

The script installs system packages via `apt-get`. This step requires sudo privileges.

**Skip this step if dependencies are already installed:**
```bash
bash maint/scripts/local_build_and_test_tt.sh --skip-deps
```

### Step 2: Python Virtual Environment

The script creates a virtual environment at `.venv/` if it doesn't exist. This isolates Python dependencies from the system.

**Virtual environment benefits:**
- No conflicts with system Python packages
- Clean, reproducible environment
- Easy to delete and recreate (`rm -rf .venv`)

**To manually activate the virtual environment later:**
```bash
source .venv/bin/activate
```

**To deactivate:**
```bash
deactivate
```

### Step 3: Build TileLang with LLVM

The script:
1. Configures CMake with LLVM backend (no CUDA required)
2. Builds TVM from submodule
3. Builds TileLang C++ libraries
4. Uses ccache for faster incremental compilation

**Build configuration:**
- Backend: LLVM (CPU-only)
- Build type: Release
- Compiler cache: ccache (enabled)
- Build system: Ninja
- Parallel jobs: Configurable (default: 2)

**Build artifacts location:**
- TVM libraries: `build/tvm/`
- TileLang libraries: `build/`

### Step 4: Install Python Packages

The script installs TVM and TileLang Python packages in editable mode within the virtual environment:

```bash
# TVM (editable install)
cd 3rdparty/tvm/python
pip install -e .

# TileLang (editable install with LLVM)
export USE_LLVM=true
pip install -e .
```

**Editable install benefits:**
- Changes to Python code take effect immediately
- No need to reinstall after code changes
- Development-friendly workflow

### Step 5: Run Tests

The script runs Tenstorrent backend tests:

1. **Target registration tests:**
   ```bash
   pytest testing/python/tenstorrent/test_target_registration.py -v
   ```

2. **All Tenstorrent tests:**
   ```bash
   pytest testing/python/tenstorrent/ -v --tb=short
   ```

**Test environment:**
- `LD_LIBRARY_PATH` set to `build/tvm` for TVM library discovery
- Tests run with `continue-on-error` behavior (backend incomplete)

**Expected test results:**
- 4 tests pass
- 1 test marked as `xfail` (expected failure: target registration not yet implemented)

### Running Individual Tests Manually

To run individual tests outside the build script:

```bash
# Activate virtual environment and set library path
source .venv/bin/activate
export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH

# Run a specific test file
cd testing/python/tenstorrent
pytest test_target_registration.py -v

# Run a single test function
pytest test_target_registration.py::test_available_targets_contains_tt -v

# Run with more verbose output
pytest test_target_registration.py -vv --tb=long
```

**Quick one-liner for running tests:**
```bash
source .venv/bin/activate && export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH && cd testing/python/tenstorrent && pytest test_target_registration.py -v
```

## Troubleshooting

### Issue: "ccache not found"

**Solution:**
```bash
sudo apt-get install ccache
```

### Issue: "python3-venv not found"

**Solution:**
```bash
sudo apt-get install python3-venv
```

### Issue: "Build fails with memory errors"

**Solution:** Reduce parallel jobs:
```bash
bash maint/scripts/local_build_and_test_tt.sh --jobs 2
```

### Issue: "TVM library not found during tests"

**Cause:** `LD_LIBRARY_PATH` not set correctly

**Solution:** The script automatically sets this. If running tests manually:
```bash
export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH
pytest testing/python/tenstorrent/ -v
```

### Issue: "Virtual environment creation fails"

**Solution:** Ensure `python3-venv` is installed:
```bash
sudo apt-get install python3-venv
python3 -m venv .venv
```

### Issue: "CMake configuration fails"

**Solution:** Ensure submodules are initialized:
```bash
git submodule update --init --recursive
```

## Manual Build (Without Script)

If you prefer to build manually:

```bash
# 1. Install system dependencies
sudo apt-get install build-essential cmake ninja-build llvm \
    libedit-dev libxml2-dev zlib1g-dev ccache python3-venv

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install --upgrade pip
pip install -r requirements-test.txt

# 4. Initialize submodules
git submodule update --init --recursive

# 5. Configure and build with CMake
mkdir -p build
cd build
cp ../3rdparty/tvm/cmake/config.cmake .
echo "set(USE_LLVM ON)" >> config.cmake
echo "set(USE_CUDA OFF)" >> config.cmake
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build . --config Release -j 4
cd ..

# 6. Install TVM Python package
export TVM_LIBRARY_PATH=$(pwd)/build/tvm
cd 3rdparty/tvm/python
pip install -e .
cd ../../..

# 7. Install TileLang
export USE_LLVM=true
pip install -e .

# 8. Run tests
export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH
pytest testing/python/tenstorrent/ -v
```

## Cleaning Up

### Remove build artifacts only
```bash
rm -rf build/
```

### Remove virtual environment only
```bash
rm -rf .venv/
```

### Full cleanup
```bash
rm -rf build/ .venv/ tilelang/lib/
```

### Remove system dependencies (optional)
```bash
sudo apt-get remove cmake ninja-build llvm ccache \
    libedit-dev libxml2-dev zlib1g-dev python3-venv
sudo apt-get autoremove
```

## Performance Tips

### Faster Builds

1. **Use ccache:** Automatically enabled by the script
   ```bash
   # Check ccache statistics
   ccache -s
   ```

2. **Increase parallel jobs:** Use more CPU cores
   ```bash
   bash maint/scripts/local_build_and_test_tt.sh --jobs 8
   ```

3. **Incremental builds:** Don't use `--clean` unless necessary

### Faster Dependency Installation

After first run, use `--skip-deps` to avoid reinstalling system packages:
```bash
bash maint/scripts/local_build_and_test_tt.sh --skip-deps
```

## CI Comparison

This script replicates the GitHub Actions CI workflow defined in `.github/workflows/tenstorrent-ci.yml`:

| CI Feature | Local Script |
|------------|--------------|
| System dependencies | ✅ Same packages via apt-get |
| Python environment | ✅ Virtual environment (CI uses Ubuntu runner) |
| LLVM backend | ✅ Same configuration |
| ccache | ✅ Enabled |
| Build parallelism | ✅ Configurable (CI uses 2 jobs) |
| TVM caching | ❌ Not implemented (CI caches build artifacts) |
| Tests | ✅ Same pytest commands |

## Next Steps After Building

1. **Verify build succeeded:**
   ```bash
   ls -lh build/*.so
   ls -lh build/tvm/*.so
   ```

2. **Run tests manually:**
   ```bash
   source .venv/bin/activate
   export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH
   pytest testing/python/tenstorrent/test_target_registration.py -v
   ```

3. **Check code formatting:**
   ```bash
   bash format.sh
   ```

4. **Make code changes and rebuild:**
   - For Python changes: No rebuild needed (editable install)
   - For C++ changes: Rerun script with `--skip-deps`

## Additional Resources

- [Tenstorrent CI Documentation](CI.md)
- [Main README](../../README.md)
- [CLAUDE.md](../../CLAUDE.md) - AI assistant guidance

## Support

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review build logs in terminal output
3. Open an issue on GitHub with:
   - Ubuntu version (`lsb_release -a`)
   - Python version (`python3 --version`)
   - Error messages
   - Build command used
