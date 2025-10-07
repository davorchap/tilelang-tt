# Workstream 1 Status - Frontend Integration & Target Selection

**Last Updated:** 2025-10-07

## Overview

Workstream 1 focuses on enabling TileLang to recognize `target="tenstorrent"`, synthesize default TT annotations, and route modules through a TT-specific lowering stack.

**Goal:** Complete frontend integration so TileLang can target Tenstorrent without impacting existing CUDA/HIP/CPU backends.

## Progress Summary

| Task | Status | Priority | Blocker |
|------|--------|----------|---------|
| Target Registration | ✅ Complete | High | None |
| Engine Adapter | ✅ Complete | High | None |
| Target Registration Tests | ✅ Complete (All 8 passing) | High | None |
| TVM Target Registration | ✅ Complete | High | None |
| Default Annotation Helper | ✅ Complete | High | None |
| **Lower Hook** | ✅ **Complete** | **High** | None |

**Overall WS1 Progress:** 🎉 100% complete (All 6 tasks complete!)

## Completed

### ✅ Target Registration (`ws1_target_registration.md`)
**Status:** Complete
**Merged:** PR #19
**Location:** `tilelang/utils/target.py`

**Implemented:**
- Added `TENSTORRENT_TARGET = "tenstorrent"` constant
- Added "tenstorrent" to `AVALIABLE_TARGETS` list
- Implemented `_is_tenstorrent_backend_enabled()` check
- Added target determination logic in `determine_target()` function
- Raises error if TT backend requested but not enabled

**What it does:**
- Allows users to specify `target="tenstorrent"` in TileLang code
- Explicit opt-in only (no auto-detection)
- Does not impact CUDA/HIP auto-detection logic

---

### ✅ Engine Adapter (`ws1_engine_adapter.md`)
**Status:** Complete
**Merged:** PR #19
**Location:** `tilelang/engine/tt/`

**Implemented:**
- Created `tilelang/engine/tt/__init__.py`
- Created `tilelang/engine/tt/lower.py` with basic lowering stub
- Stub raises `NotImplementedError` with clear message

**What it does:**
- Provides TT-specific lowering entry point
- Isolates TT backend logic from generic TileLang frontend
- Ready to be wired into main lowering orchestration

---

### ✅ Target Registration Tests (`ws1_target_registration_test.md`)
**Status:** Complete
**Merged:** PR #19
**Location:** `testing/python/tt/test_target_registration.py`

**Test Results:**
- ✅ `test_available_targets_contains_tt` - PASSED
- ✅ `test_determine_target_returns_target_when_backend_enabled` - PASSED
- ✅ `test_determine_target_raises_when_backend_disabled` - PASSED
- ✅ `test_tenstorrent_engine_lower_raises_not_implemented` - PASSED
- ✅ `test_tenstorrent_engine_lower_validates_target` - PASSED

**Current status:** All 5 tests passing ✅

---

### ✅ TVM Target Registration
**Status:** Complete
**Merged:** PR #19
**Location:** `3rdparty/tvm` (submodule updated to davorchap/tvm fork)

**Implemented:**
- Registered "tenstorrent" as a valid target kind in TVM's C++ target registry
- Updated TVM submodule to point to davorchap/tvm fork with TT support
- Removed xfail marker from test - all tests now passing

**What it does:**
- Allows creation of `Target("tenstorrent")` objects
- Enables `target.kind.name == "tenstorrent"` checks
- Provides foundation for TT-specific target attributes in future

---

### ✅ Default Annotation Helper (`ws1_default_annotation_helper.md`)
**Status:** Complete
**Location:** `tilelang/tt/target.py`

**Implemented:**
- Created `tilelang/tt/target.py` with `apply_tt_defaults()` function
- Adds default TT attributes to PrimFuncs when not already present
- Ensures idempotency - doesn't override existing user annotations
- Added comprehensive tests (3 new tests, all passing)

**Default behavior applied:**
- **Schedule:** `policy="contiguous"`, `order="row_major"`
- **Layout:** Row-major 32×32 DRAM interleaved tilization
  - `tt_tile_height=32`, `tt_tile_width=32`
  - `tt_layout_type="dram_interleaved"`

**What it does:**
Allows existing GPU-style kernels to run on TT with minimal changes by providing sensible defaults when users don't specify TT-specific annotations.

### ✅ Lower Hook (`ws1_lower_hook.md`)
**Status:** Complete
**Location:** `tilelang/engine/tt/lower.py`

**Implemented:**
- Wired `apply_tt_defaults()` into TT lowering entry point
- Called immediately after target validation, before raising NotImplementedError
- Only applies to TT target (doesn't impact other backends)
- Updated documentation and comments to reflect integration

**Integration:**
The TT lowering function now:
1. Validates the target is "tenstorrent"
2. Applies default TT annotations via `apply_tt_defaults(mod)`
3. Returns prepared module (full pipeline to be added in Workstream 2)

**Why it's needed:**
Ensures backward compatibility - GPU-style kernels can target TT without modification.

---

## WS1 Complete! 🎉

All Workstream 1 tasks are now complete:
- ✅ Target registration and validation
- ✅ Engine adapter with lowering entry point
- ✅ TVM C++ target registration
- ✅ Default annotation helper
- ✅ Lower hook integration
- ✅ Comprehensive test coverage (8 tests, all passing)

**What this means:**
TileLang can now recognize `target="tenstorrent"` and automatically apply sensible defaults for TT execution. The foundation is in place for Workstream 2 (Schedule & Sharding Metadata) to build upon.

---

## Dependency Graph

```
✅ TVM Target Registration (COMPLETE)
    ↓
    └─→ ✅ Test suite passing (8/8 tests)

✅ Default Annotation Helper (COMPLETE)
    ↓
    └─→ ✅ Lower Hook (COMPLETE)
         ↓
         └─→ ✅ WS1 Complete! Ready for WS2
```

**Critical path:**
1. ✅ TVM target registration - COMPLETE
2. ✅ Annotation helper - COMPLETE
3. ✅ Lower hook - COMPLETE
4. ✅ All tasks complete - Ready for Workstream 2!

---

## Testing Strategy

### Current Test Coverage (8/8 passing)
- ✅ Target string "tenstorrent" recognized
- ✅ Error handling when backend disabled
- ✅ Engine adapter validates target
- ✅ TVM Target creation works correctly
- ✅ Default annotations applied to unannotated modules
- ✅ User annotations preserved when present
- ✅ Idempotency of annotation helper

### Future Test Coverage (Workstream 2+)
- [ ] Lower hook integrates correctly in full pipeline
- [ ] End-to-end dry-run of simple GEMM with TT target
- [ ] TT-specific compiler passes work correctly

---

## Build & Test Instructions

### Running WS1 Tests

**Quick one-liner:**
```bash
source .venv/bin/activate && export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH && cd testing/python/tt && pytest test_target_registration.py -v
```

**Step by step:**
```bash
# Build with automated script
bash maint/scripts/local_build_and_test_tt.sh --skip-deps --jobs 4

# Or run tests manually
source .venv/bin/activate
export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH
pytest testing/python/tt/test_target_registration.py -v
```

**Expected results:**
- All 8 tests pass ✅ (5 original + 3 annotation helper tests)

### Building for Development

See `docs/tenstorrent/local_build_guide.md` for complete build instructions.

---

## Next Workstream Preview

**Workstream 2: Schedule & Sharding Metadata**

Once WS1 is complete, WS2 will focus on:
- Implementing C++ compiler passes for TT schedule inference
- Adding sharding metadata for DRAM interleaved tilization
- Core-level scheduling and partitioning logic

**Cannot start until:** WS1 is fully complete (all tasks done).

---

## Related Documentation

- [Project Plan](../project_1.md) - Overall TT backend MVP plan
- [Local Build Guide](../local_build_guide.md) - Build and test instructions
- [WS1 Target Registration](ws1_target_registration.md) - Detailed task breakdown
- [WS1 Engine Adapter](ws1_engine_adapter.md) - Engine implementation details
- [WS1 Target Registration Test](ws1_target_registration_test.md) - Test specification

---

## Questions or Issues?

- Check the [CLAUDE.md](../../CLAUDE.md) for development workflow
- Review individual task `.md` files in this directory for detailed specs
- Run the build script for automated setup and testing
