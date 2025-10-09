# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

TileLang is a domain-specific language for developing high-performance GPU/CPU kernels (GEMM, FlashAttention, etc.) and accelerators including **Tenstorrent AI architecture**. Built on TVM with a Pythonic syntax, it enables productivity without sacrificing low-level optimizations.

This repository (`tilelang-tt`) is a **public fork** focused on adding first-class **Tenstorrent TT-Metalium backend** support alongside existing NVIDIA CUDA, AMD ROCm, and Huawei Ascend targets.

### Current Status (2025-10-08)

**✅ IR-Driven Backend Complete (95 tests passing)**
- ✅ Target registration (8 tests)
- ✅ Metadata inference (7 tests)
  - Schedule inference (per-core tile assignments)
  - Shard inference (DRAM layout descriptors)
- ✅ Transform pipeline (39 tests)
  - GridToPersistentTT (persistent loop model)
  - TTShardToCoreMap (NOC grid mapping)
  - MemorySpaceLowerTT (DRAM → L1 circular buffers)
  - TilePadTT (32×32 tile alignment)
  - TensorizeTT (pattern detection)
  - VerifyTTIR (constraint verification)
- ✅ Code generation (41 tests)
  - IR-driven visitor infrastructure
  - Reader/Compute/Writer kernels
  - Host program generation
  - DST lifecycle management

**✅ SDK Integration Complete**
- ✅ External SDK approach (like CUDA/ROCm)
- ✅ CMake FindMetalium module
- ✅ Real vs Mock build modes
- ✅ CI workflows (mock + SDK validation)

**⚠️ Next Steps**
- Extend `tensorize_tt.cc` for pattern detection
- SDK validation (blocked - needs SDK access)

**Key Documents:**
- 🏗️ **Architecture**: [TT_ARCHITECTURE.md](docs/tenstorrent/TT_ARCHITECTURE.md) - Complete TT backend architecture
- 📊 **Compiler Internals**: [IR_LOWERING_ANALYSIS.md](docs/tenstorrent/IR_LOWERING_ANALYSIS.md) - GPU vs TT pipeline comparison
- 📋 **Pass Reference**: [PASS_TABLE.md](docs/tenstorrent/PASS_TABLE.md) - All 60+ transform passes
- 🚧 **Next Tasks**: [IR_LOWERING_TASKS.md](docs/tenstorrent/IR_LOWERING_TASKS.md) - Pattern detection roadmap
- ⚙️ **SDK Setup**: [METALIUM_SETUP_GUIDE.md](docs/tenstorrent/METALIUM_SETUP_GUIDE.md) - SDK installation
- 🔬 **SDK Validation**: [METALIUM_SDK_VALIDATION_PLAN.md](docs/tenstorrent/METALIUM_SDK_VALIDATION_PLAN.md) - Validation phases

**Build Modes:**
- **Mock mode** (default): All tests pass, dry-run code generation
- **Real mode**: `cmake -DUSE_REAL_METALIUM=ON` (requires TT_METAL_HOME)

---

## 6-Phase TileLang→Metalium Compiler Implementation (2025-10-08+)

**⭐ CURRENT ACTIVE WORK:** Systematic implementation of all TileLang patterns to Metalium

**Directive**: Implement all Tensor IR transforms and codegen to generate all TileLang GPU examples to Metalium (from simple elementwise to FlashAttention). Use case-driven development: implement transforms only when examples need them. All work in mock/dry-run mode first, hardware validation after all examples complete.

### Phase Progress (Overall: 17%)

| Phase | Status | Examples | PRs | Duration |
|-------|--------|----------|-----|----------|
| **Phase 1: Foundation** | 🟡 37% | 1/3 partial | 3 merged | 2 weeks |
| **Phase 2: Optimizations** | ⏳ 0% | 0/3 | 0 | 2 weeks |
| **Phase 3: Advanced** | ⏳ 0% | 0/3 | 0 | 2 weeks |
| **Phase 4: Attention** | ⏳ 0% | 0/5 | 0 | 2 weeks |
| **Phase 5: Specialized** | ⏳ 0% | 0/3 | 0 | 2 weeks |
| **Phase 6: Complex** | ⏳ 0% | 0/3 | 0 | 2 weeks |

**Master Plan**: [docs/tenstorrent/phases/PHASES_STATUS.md](docs/tenstorrent/phases/PHASES_STATUS.md)

### Key Achievement: DST Double Buffering ✅

**PR #56 (MERGED)**: DST (Destination Register) double buffering fully working
- Pattern 1 (Element-wise): acquire→compute→commit→pack→release per tile
- Pattern 3 (K-loop GEMM): acquire→K-loop→commit→pack→release
- Both patterns tested and generating correct DST lifecycle

### Phase 1 Details (Foundation - 37% Complete)

**Completed**:
- ✅ DST double buffering infrastructure (PRs #53, #54, #56)
- ✅ Mock APIs for all intrinsics (add_tiles, matmul_tiles, pack_tile)
- ✅ Loop pattern detection (element-wise vs K-loop)
- ✅ Proper TileLang IR structure for examples
- ✅ EmitElementwiseAddIntrinsic() and EmitMatmulIntrinsic() methods

**In Progress**:
- 🟡 Pattern recognition for T.copy and T.gemm operations
- 🟡 Intrinsic emission (replacing "unsupported call" placeholders)
- 🟡 CB management for inputs (cb_wait_front, cb_pop_front)

**Examples Status**:
- 1.1 Elementwise Add: 40% (DST ✅, intrinsics ⏳)
- 1.2 Multi-operand: 0% (blocked by 1.1)
- 1.3 Simple GEMM: 30% (DST ✅, K-loop ✅, intrinsics ⏳)

### Autonomous Development Process

Per user directive: **No human-in-the-loop, no interruptions**. For each example:
1. Craft plan → write transform spec (if needed)
2. Document requirements
3. Implement transform/codegen
4. Make PR → merge PR
5. Move to next example

Work continuously through all 20 examples across 6 phases until complete.

**Current Task**: Complete intrinsic emission for Phase 1.1 Elementwise Add

### Phase Specifications

All phases fully specified (PR #55):
- `docs/tenstorrent/phases/PHASE1_FOUNDATION_SPEC.md`
- `docs/tenstorrent/phases/PHASE2_OPTIMIZATIONS_SPEC.md`
- `docs/tenstorrent/phases/PHASE3_ADVANCED_SPEC.md`
- `docs/tenstorrent/phases/PHASE4_ATTENTION_SPEC.md`
- `docs/tenstorrent/phases/PHASE5_SPECIALIZED_SPEC.md`
- `docs/tenstorrent/phases/PHASE6_COMPLEX_SPEC.md`
- `docs/tenstorrent/phases/PHASES_STATUS.md` (master tracking)

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
- `USE_REAL_METALIUM=true` - Enable real TT-Metalium SDK integration (requires `TT_METAL_HOME`)
- `DEBUG_MODE=true` - Build with debug symbols and logging
- `WITH_COMMITID=true` - Include git commit ID in wheel filename (default for non-PyPI builds)
- `PYPI_BUILD=true` - Build for PyPI distribution (clean version strings)

### Building TileLang

**Quick start - Automated local build (recommended for Tenstorrent development):**
```bash
# Mock mode (default) - Builds with LLVM backend, runs tests, ~1 minute with ccache
bash maint/scripts/local_build_and_test_tt.sh --skip-deps --jobs 4

# Real SDK mode - Builds with TT-Metalium SDK integration
export TT_METAL_HOME=/path/to/tt-metal
bash maint/scripts/local_build_and_test_tt.sh --with-metalium --skip-deps --jobs 4
```

The automated script:
- Automatically initializes git submodules if needed
- Sets up Python virtual environment (`.venv`)
- Builds with LLVM backend (required for Tenstorrent)
- Supports `--with-metalium` flag for real SDK integration
- Validates SDK installation when using `--with-metalium`
- Installs TVM and TileLang in editable mode
- Runs Tenstorrent backend tests with proper environment
- See `docs/tenstorrent/local_build_guide.md` for full documentation

**Standard CUDA build:**
```bash
python setup.py build_ext --inplace
pip install -e .
```

**LLVM-only build (for CPU/Tenstorrent development - mock mode):**
```bash
USE_LLVM=true pip install -e .
```

**ROCm build:**
```bash
USE_ROCM=true pip install -e .
```

**Tenstorrent with Real Metalium SDK:**
```bash
# See docs/tenstorrent/METALIUM_SETUP_GUIDE.md for SDK installation

# Option 1: Automated build script (recommended)
export TT_METAL_HOME=/path/to/tt-metal
bash maint/scripts/local_build_and_test_tt.sh --with-metalium --skip-deps --jobs 4

# Option 2: Manual CMake build
export TT_METAL_HOME=/path/to/tt-metal
cmake -B build -DUSE_LLVM=true -DUSE_REAL_METALIUM=ON -DCMAKE_PREFIX_PATH="$TT_METAL_HOME/build"
cmake --build build -j$(nproc)
pip install -e . --no-build-isolation

# Verify SDK installation
bash maint/scripts/verify_metalium_sdk.sh $TT_METAL_HOME
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
- 8 tests pass (Phase 1 complete)
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

**Current Status (2025-10-08):**
- ✅ **IR-Driven Backend Complete (100%)** - All transforms and codegen operational
- ✅ **Metalium Integration Complete** - Weeks 16-18 (conditional compilation ready)
- ⚠️ **SDK Validation Pending** - Weeks 19-22 (blocked by SDK access)

**Architecture Details:** See [TT_ARCHITECTURE.md](docs/tenstorrent/TT_ARCHITECTURE.md) for complete details.

**Key Components:**

1. **Target Registration** ✅ COMPLETE
   - `tilelang/utils/target.py` - Target registration
   - `tilelang/engine/tt/` - Engine adapter
   - `tilelang/tt/target.py` - `apply_tt_defaults()` function
   - Stamps default TT attributes (schedule, shard)
   - 8 tests passing

2. **Metadata Inference** ✅ COMPLETE
   - `src/transform/tt/infer_tt_schedule.cc` - Per-core tile assignments
   - `src/transform/tt/infer_tt_shard.cc` - DRAM layout descriptors
   - `tilelang/tt/passes.py` - Python bindings
   - 7 tests passing

3. **Transform Pipeline** ✅ COMPLETE
   - `src/transform/tt/grid_to_persistent_tt.cc` - Persistent loop model
   - `src/transform/tt/tt_shard_to_core_map.cc` - NOC grid mapping
   - `src/transform/tt/memory_space_lower_tt.cc` - DRAM → L1 CB
   - `src/transform/tt/tile_pad_tt.cc` - Tile alignment
   - `src/transform/tt/tensorize_tt.cc` - Pattern detection (incomplete)
   - `src/transform/tt/verify_tt_ir.cc` - Constraint verification
   - 39 tests passing

4. **Code Generation** ✅ COMPLETE (IR-Driven)
   - `src/target/tt/codegen_tt.cc` - Main codegen framework
   - `src/target/tt/codegen_tt_visitor_base.cc` - IR walking infrastructure
   - `src/target/tt/codegen_tt_compute_visitor.cc` - Compute kernel
   - `src/target/tt/codegen_tt_reader_visitor.cc` - Reader kernel
   - `src/target/tt/codegen_tt_writer_visitor.cc` - Writer kernel
   - Conditional compilation (real/mock modes)
   - 41 tests passing

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
│   │   └── tt/                 # Tenstorrent engine adapter (Target Registration complete)
│   ├── tt/                     # Tenstorrent utilities (Target Registration complete)
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

1. **`tenstorrent-ci.yml`** - Tenstorrent backend CI (mock mode):
   - Triggers on PRs modifying `tilelang/engine/tt/`, `testing/python/tt/`, or workflow files
   - Runs on GitHub-hosted runners (Ubuntu + Python 3.10)
   - Uses LLVM backend (not CUDA) for lightweight CPU-only tests
   - **Caching strategy:**
     - TVM build cache (keyed by submodule commit) - saves ~5-6 min
     - ccache (keyed by CMakeLists.txt) - fast incremental compilation
     - pip packages (keyed by requirements files)
   - Jobs: lint-and-format, build-and-test, static-analysis (mypy)
   - All 95 tests passing ✅

2. **`tenstorrent-sdk-ci.yml`** - Tenstorrent SDK integration CI (real mode):
   - Triggers on manual dispatch (`workflow_dispatch`) or weekly schedule
   - Builds with real TT-Metalium SDK (`-DUSE_REAL_METALIUM=ON`)
   - **SDK caching strategy:**
     - tt-metal SDK cache (~1.5GB, keyed by TT_METAL_VERSION)
     - First run: ~20 min (clone + build SDK)
     - Cached runs: ~5 min (SDK cached, only build TileLang)
   - Verifies generated code uses real Metalium APIs
   - Tests SDK integration workflow

3. **`ci.yml`** - Main CI:
   - Self-hosted NVIDIA runners
   - Full CUDA build and test suite

4. **`amd_ci.yml`** - AMD ROCm CI

### Running CI Locally

```bash
# Lint and format
bash format.sh

# Build and test - Mock mode (mimics tenstorrent-ci.yml)
bash maint/scripts/local_build_and_test_tt.sh --skip-deps --jobs 4

# Build and test - Real SDK mode (mimics tenstorrent-sdk-ci.yml)
export TT_METAL_HOME=/path/to/tt-metal
bash maint/scripts/local_build_and_test_tt.sh --with-metalium --skip-deps --jobs 4

# Manual build and test (alternative)
USE_LLVM=true pip install -e .
LD_LIBRARY_PATH=build/tvm pytest testing/python/tt/ -v
```

## Development Workflow

### For Tenstorrent Backend Development

**Primary Reference:** [docs/tenstorrent/METALIUM_SDK_VALIDATION_PLAN.md](docs/tenstorrent/METALIUM_SDK_VALIDATION_PLAN.md)

1. **Current Status:** IR-Driven Backend Complete (95 tests)
   - ✅ All transforms implemented and tested
   - ✅ IR-driven codegen with visitor pattern
   - ✅ Metalium API integration (conditional compilation)
   - ⚠️ SDK validation pending (blocked by SDK access)

2. **Current Priority:** SDK Validation (Weeks 19-22)
   - **Phase 1:** Dry-run compilation with real SDK
     - Fix include paths and namespace issues
     - Verify API signatures match documentation
   - **Phase 2:** Complete missing APIs
     - Implement EnqueueWriteBuffer/ReadBuffer
     - Complete SetRuntimeArgs implementation
   - **Phase 3:** Hardware execution
     - Run matmul on Grayskull/Wormhole
     - Validate correctness and performance

3. **Branch naming:** Use `phase<N>-*` or `sdk-*` prefix for validation tasks

4. **Key files for SDK validation:**
   - `src/target/tt/codegen_tt.cc` - Host program EmitTTHostProgram()
   - `src/target/tt/codegen_tt_*_visitor.cc` - Kernel generation visitors
   - `cmake/FindMetalium.cmake` - SDK detection (fallback for TT_METAL_HOME)
   - `CMakeLists.txt` - Build configuration (find_package(TT-Metalium))
   - `maint/scripts/local_build_and_test_tt.sh` - Local build script with --with-metalium flag
   - `maint/scripts/verify_metalium_sdk.sh` - SDK verification tool
   - `.github/workflows/tenstorrent-sdk-ci.yml` - Real SDK CI workflow

5. **Testing strategy:**
   - Mock mode: 95 tests passing ✅
   - Real mode: Pending SDK access
   - Hardware tests: Pending Phase 3

6. **Documentation:**
   - See METALIUM_SDK_VALIDATION_PLAN.md for complete roadmap
   - Update status after each validation phase
   - Document any API discrepancies discovered

## Autonomous Phase Execution Framework

This section provides a systematic approach for AI agents (like Claude Code) to autonomously complete phases from planning through implementation, testing, PR creation, and merge.

### Phase Lifecycle Overview

Each phase follows this lifecycle:
```
Plan → Implement → Test → Document → PR → Merge → Next WS
```

### Phase 1: Planning (Before Code)

**Objective:** Fully understand requirements and create detailed implementation plan.

**Steps:**
1. **Read phase documentation:**
   ```bash
   # Read status document
   cat docs/tenstorrent/phase{N}/WS{N}_STATUS.md

   # Read all related design docs
   find docs/tenstorrent/phase{N}/ -name "*.md" -exec cat {} \;
   ```

2. **Understand dependencies:**
   - Verify prerequisite phases are complete
   - Check which components from previous WS are needed
   - Identify new components to build

3. **Create implementation plan:**
   - List all files to create/modify
   - Define C++ classes/functions needed
   - Define Python modules/functions needed
   - List test cases to write
   - Estimate task breakdown (use TodoWrite tool)

4. **Document the plan:**
   - Update WS{N}_STATUS.md with detailed plan
   - Add any architecture decisions to design docs
   - Commit plan before implementation begins

**Planning checklist:**
- [ ] Read all WS documentation
- [ ] Understand input/output of this WS
- [ ] List all files to modify
- [ ] Design API signatures (C++ and Python)
- [ ] Plan test strategy
- [ ] Update STATUS.md with plan

### Phase 2: Implementation

**Objective:** Implement all components according to plan.

**Workflow:**
1. **Create feature branch:**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b ws{N}-{feature-description}
   ```

2. **Implement in order:**
   - Start with C++ infrastructure (if needed)
   - Add to CMakeLists.txt if new files
   - Implement core logic
   - Add Python bindings
   - Update `__init__.py` exports

3. **Build incrementally:**
   ```bash
   # After each significant change
   USE_LLVM=true pip install -e . --no-build-isolation

   # Verify no compilation errors
   ```

4. **Follow coding standards:**
   - Use existing patterns from similar files
   - Match naming conventions (e.g., `InferDefaultTT*` for TT passes)
   - Add comprehensive docstrings
   - Use type hints in Python

**Implementation checklist:**
- [ ] Feature branch created
- [ ] C++ files implemented (if applicable)
- [ ] CMakeLists.txt updated (if new files)
- [ ] Build succeeds without errors
- [ ] Python bindings implemented
- [ ] Exports added to `__init__.py`
- [ ] Code follows style guide

### Phase 3: Testing

**Objective:** Verify implementation works correctly with comprehensive tests.

**Test strategy:**
1. **Create test file:**
   ```bash
   # Create test file following naming convention
   touch testing/python/tt/test_ws{N}_{feature}.py
   ```

2. **Write comprehensive tests:**
   - Unit tests for individual functions
   - Integration tests for full pipeline
   - Edge case tests (boundary conditions)
   - Error handling tests

3. **Run tests locally:**
   ```bash
   source .venv/bin/activate
   export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH
   pytest testing/python/tt/test_ws{N}_*.py -v
   ```

4. **Verify all existing tests still pass:**
   ```bash
   pytest testing/python/tt/ -v
   ```

5. **Run format checks:**
   ```bash
   bash format.sh
   ```

**Testing checklist:**
- [ ] Test file created
- [ ] Unit tests written and passing
- [ ] Integration tests written and passing
- [ ] Edge cases covered
- [ ] All existing tests still pass
- [ ] Code formatting passes

### Phase 4: Documentation

**Objective:** Update all documentation to reflect changes.

**Documentation updates:**
1. **Update STATUS document:**
   ```bash
   # Update WS{N}_STATUS.md
   # - Mark tasks complete
   # - Add test results
   # - Document any issues resolved
   # - Update progress percentage
   ```

2. **Update CLAUDE.md (if needed):**
   - Add new components to architecture section
   - Update directory structure if new paths
   - Update test count expectations
   - Add any new important notes

3. **Add inline documentation:**
   - Docstrings for all public functions
   - Comments for complex logic
   - File-level docstrings explaining purpose

**Documentation checklist:**
- [ ] WS{N}_STATUS.md updated
- [ ] Test results documented
- [ ] CLAUDE.md updated (if needed)
- [ ] All functions have docstrings
- [ ] Complex logic has comments

### Phase 5: Create Pull Request

**Objective:** Create well-documented PR for review.

**PR creation steps:**
1. **Commit all changes:**
   ```bash
   git add .
   git commit -m "$(cat <<'EOF'
   Complete WS{N}: {Brief description}

   This PR implements Phase {N}: {Full description}

   **What's included:**
   - C++ implementation: {list files}
   - Python bindings: {list files}
   - Tests: {X} tests, all passing
   - Documentation updates

   **Test results:**
   ```
   {paste test output}
   ```

   **Key changes:**
   - {bullet point 1}
   - {bullet point 2}
   - {bullet point 3}

   Closes #{issue_number} (if applicable)

   🤖 Generated with [Claude Code](https://claude.com/claude-code)
   EOF
   )"
   ```

2. **Push feature branch:**
   ```bash
   git push -u origin ws{N}-{feature-description}
   ```

3. **Create PR:**
   ```bash
   gh pr create \
     --repo davorchap/tilelang-tt \
     --base main \
     --head ws{N}-{feature-description} \
     --title "Complete WS{N}: {Brief description}" \
     --body "$(cat <<'EOF'
   ## Summary

   This PR implements Phase {N}: {Full description}.

   ## Changes

   ### C++ Implementation
   - `src/transform/tt/{file}.cc` - {description}
   - {list other C++ files}

   ### Python Bindings
   - `tilelang/tt/{file}.py` - {description}
   - {list other Python files}

   ### Tests
   - `testing/python/tt/test_ws{N}_{feature}.py` - {X} tests

   ### Documentation
   - Updated `docs/tenstorrent/phase{N}/WS{N}_STATUS.md`
   - {list other doc updates}

   ## Test Results

   All {X} tests passing:
   ```
   {paste test output}
   ```

   ## Checklist

   - [x] C++ implementation complete
   - [x] Python bindings complete
   - [x] Tests written and passing
   - [x] Documentation updated
   - [x] Code formatted (format.sh)
   - [x] All existing tests still pass

   ## Related Issues

   Closes #{issue_number} (if applicable)

   ---

   🤖 Generated with [Claude Code](https://claude.com/claude-code)
   EOF
   )"
   ```

**PR checklist:**
- [ ] All changes committed
- [ ] Branch pushed to origin
- [ ] PR created with comprehensive description
- [ ] PR targets correct repo (davorchap/tilelang-tt)
- [ ] PR targets correct base branch (main)
- [ ] Test results included in PR description

### Phase 6: Merge Pull Request

**Objective:** Merge PR after verification.

**Merge steps:**
1. **Verify PR checks pass:**
   ```bash
   # Check PR status
   gh pr view --repo davorchap/tilelang-tt
   ```

2. **Review the PR:**
   - Check CI status (if applicable)
   - Review diff one more time
   - Ensure no conflicts with main

3. **Merge the PR:**
   ```bash
   # Merge via GitHub CLI
   gh pr merge --repo davorchap/tilelang-tt --squash --delete-branch

   # Or merge via web UI:
   # Visit PR URL and click "Squash and merge"
   ```

4. **Update local main:**
   ```bash
   git checkout main
   git pull origin main
   ```

5. **Verify merge:**
   ```bash
   # Verify changes are in main
   git log --oneline -5

   # Run tests from main
   pytest testing/python/tt/ -v
   ```

**Merge checklist:**
- [ ] PR checks passed (or reviewed)
- [ ] No merge conflicts
- [ ] PR merged successfully
- [ ] Feature branch deleted
- [ ] Local main updated
- [ ] Tests pass on main branch

### Phase 7: Move to Next Phase

**Objective:** Seamlessly transition to next phase.

**Transition steps:**
1. **Verify current WS complete:**
   ```bash
   # Check status document
   cat docs/tenstorrent/phase{N}/WS{N}_STATUS.md | grep "Progress:"
   ```

2. **Identify next phase:**
   ```bash
   # Read project plan
   cat docs/tenstorrent/UNIFIED_MATMUL_MVP_PLAN.md

   # Check next WS directory
   ls -la docs/tenstorrent/phase{N+1}/
   ```

3. **Read next WS documentation:**
   - Start Phase 1 (Planning) for next WS
   - Read WS{N+1}_STATUS.md
   - Understand dependencies

4. **Begin next phase:**
   - Go back to Phase 1 with new WS
   - Follow same lifecycle

**Transition checklist:**
- [ ] Current WS marked complete in STATUS.md
- [ ] All PRs for current WS merged
- [ ] Next WS identified
- [ ] Next WS documentation read
- [ ] Ready to begin Phase 1 for next WS

### Automation Guidelines

**When to use TodoWrite:**
- Beginning of each phase (track all tasks)
- During implementation (track subtasks)
- Mark tasks complete as you finish them
- Update task status in real-time

**When to commit:**
- After completing each major component
- Before switching contexts
- After passing test suite
- Before creating PR

**Error handling:**
- If tests fail, debug before proceeding
- If build fails, fix compilation errors immediately
- If FFI registration issues, check library loading
- Don't move to next phase until current phase complete

**Quality gates:**
- No phase 2 without completed phase 1 plan
- No phase 3 without successful build
- No phase 4 without all tests passing
- No phase 5 without updated documentation
- No phase 7 without successful merge

### Key Success Metrics

For each phase, verify:
- ✅ All planned components implemented
- ✅ All tests passing (aim for 100% test pass rate)
- ✅ Documentation complete and accurate
- ✅ PR merged to main branch
- ✅ No breaking changes to existing functionality

### Example: Autonomous Execution of Metadata Inference

```
Phase 1: Read docs/tenstorrent/phase2/*.md → Create plan → Update STATUS.md
Phase 2: Implement infer_tt_schedule.cc, infer_tt_shard.cc, passes.py → Build succeeds
Phase 3: Write test_ws2_passes.py → 7/7 tests pass → All existing tests pass
Phase 4: Update Metadata Inference_STATUS.md with results → Update CLAUDE.md
Phase 5: Commit → Push ws2-schedule-shard-inference → Create PR
Phase 6: Review PR → Merge to main → Update local main
Phase 7: Read Transform Pipeline docs → Start Phase 1 for Transform Pipeline
```

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

### Core Documentation
- **[TT_ARCHITECTURE.md](docs/tenstorrent/TT_ARCHITECTURE.md)** ⭐ - Complete TT backend architecture
- **[IR_LOWERING_ANALYSIS.md](docs/tenstorrent/IR_LOWERING_ANALYSIS.md)** - GPU vs TT pipeline comparison
- **[PASS_TABLE.md](docs/tenstorrent/PASS_TABLE.md)** - Comprehensive pass reference (60+ passes)
- **[IR_LOWERING_TASKS.md](docs/tenstorrent/IR_LOWERING_TASKS.md)** - Pattern detection implementation tasks

### Setup & Usage
- [METALIUM_SETUP_GUIDE.md](docs/tenstorrent/METALIUM_SETUP_GUIDE.md) - SDK installation and configuration
- [local_build_guide.md](docs/tenstorrent/local_build_guide.md) - Local build instructions
- [CI.md](docs/tenstorrent/CI.md) - Continuous integration

### Validation
- [METALIUM_SDK_VALIDATION_PLAN.md](docs/tenstorrent/METALIUM_SDK_VALIDATION_PLAN.md) - SDK validation phases

### Phase Development (Active)
- [phases/PHASES_STATUS.md](docs/tenstorrent/phases/PHASES_STATUS.md) - 6-phase implementation tracking


## Important Notes

- **LLVM vs CUDA builds:** For Tenstorrent development, use `USE_LLVM=true` to avoid CUDA dependency
- **System LLVM preferred:** CI uses system LLVM (via apt) to avoid libtinfo.so.5 linking issues with downloaded LLVM 10.0.1
- **TVM library path:** Always set `LD_LIBRARY_PATH=build/tvm` when running tests
- **Submodules:** Run `git submodule update --init --recursive` after fresh clone
- **Cython JIT adapter:** Auto-compiles on first use with caching in `.cycache/`
