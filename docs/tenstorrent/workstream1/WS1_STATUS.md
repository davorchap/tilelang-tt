# Workstream 1 Status - Frontend Integration & Target Selection

**Last Updated:** 2025-10-06

## Overview

Workstream 1 focuses on enabling TileLang to recognize `target="tenstorrent"`, synthesize default TT annotations, and route modules through a TT-specific lowering stack.

**Goal:** Complete frontend integration so TileLang can target Tenstorrent without impacting existing CUDA/HIP/CPU backends.

## Progress Summary

| Task | Status | Priority | Blocker |
|------|--------|----------|---------|
| Target Registration | ✅ In Review | High | None |
| Engine Adapter | ✅ In Review | High | None |
| Target Registration Tests | ⚠️ In Review (1 xfail) | High | TVM target registration |
| **TVM Target Registration** | ❌ **TODO** | **🔥 CRITICAL** | **Blocks test passing** |
| Default Annotation Helper | ❌ TODO | Medium | Target registration |
| Lower Hook | ❌ TODO | Medium | Annotation helper |

**Overall WS1 Progress:** ~50% complete (3 of 6 tasks in review/done)

## Completed/In Review

### ✅ Target Registration (`ws1_target_registration.md`)
**Status:** In Review
**Branch:** `tt-matmul-mvp-plan`
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
**Status:** In Review
**Branch:** `ws1-engine-adapter`
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

### ⚠️ Target Registration Tests (`ws1_target_registration_test.md`)
**Status:** In Review (1 test marked xfail)
**Branch:** `tt-matmul-mvp-plan`
**Location:** `testing/python/tt/test_target_registration.py`

**Test Results:**
- ✅ `test_available_targets_contains_tt` - PASSED
- ❌ `test_determine_target_returns_target_when_backend_enabled` - **XFAIL** (expected failure)
- ✅ `test_determine_target_raises_when_backend_disabled` - PASSED
- ✅ `test_tenstorrent_engine_lower_raises_not_implemented` - PASSED
- ✅ `test_tenstorrent_engine_lower_validates_target` - PASSED

**Current status:** 4 passed, 1 xfailed

**Why xfail:** The failing test expects to create a TVM `Target` object with `kind="tenstorrent"`, but this target kind hasn't been registered in TVM's C++ target registry yet.

**Error message:**
```
ValueError: Target kind "tenstorrent" is not defined. Target creation from string failed: tenstorrent
```

**To fix:** Need to register the target in TVM (see TODO section below)

---

## TODO - Next Steps

### 🔥 CRITICAL: TVM Target Registration
**Priority:** HIGHEST
**Blocks:** Test suite passing, downstream WS1 tasks

**What needs to be done:**
Register "tenstorrent" as a valid target kind in TVM's C++ target registry.

**Likely locations:**
- `3rdparty/tvm/src/target/target.cc` - Main target registration
- `3rdparty/tvm/src/target/target_kind.cc` - Target kind definitions
- `3rdparty/tvm/include/tvm/target/target.h` - Target API

**Implementation approach:**
1. Add "tenstorrent" to TVM's registered target kinds
2. Define basic target attributes (similar to LLVM/CUDA targets)
3. Register target with TVM's target registry
4. Rebuild TVM submodule
5. Verify test passes (remove xfail marker)

**Dependencies:** None (can start immediately)

**Success criteria:**
- `test_determine_target_returns_target_when_backend_enabled` passes
- Can create `Target("tenstorrent")` without errors
- `target.kind.name == "tenstorrent"` works

---

### Default Annotation Helper (`ws1_default_annotation_helper.md`)
**Priority:** Medium
**Depends on:** TVM target registration
**Status:** TODO

**What needs to be done:**
Implement `python/tilelang_tt/target.py` with a helper that stamps default TT schedule/sharding attributes when users omit them.

**Default behavior:**
- **Schedule:** `policy="contiguous"`, `order="row_major"`
- **Layout:** Row-major 32×32 DRAM interleaved tilization
- **Sharding:** DRAM interleaved tensors via TensorAccessor

**Implementation:**
- Create `python/tilelang_tt/target.py`
- Implement `synthesize_default_tt_annotations(mod: IRModule) -> IRModule`
- Add function attributes for schedule/sharding metadata
- Only apply when attributes are missing (don't override user annotations)

**Why it's needed:**
Allows existing GPU-style kernels to run on TT with minimal changes by providing sensible defaults.

---

### Lower Hook (`ws1_lower_hook.md`)
**Priority:** Medium
**Depends on:** Default annotation helper
**Status:** TODO

**What needs to be done:**
Wire the default annotation helper into the main lowering entry point.

**Implementation:**
- Update `tilelang/engine/lower.lower` to call annotation helper
- Call after target determination, before main lowering
- Only invoke for TT target (don't impact other backends)
- Ensure helper is called before TT-specific passes

**Integration point:**
```python
# In tilelang/engine/lower.lower
if target == "tenstorrent":
    mod = synthesize_default_tt_annotations(mod)
# Continue with normal lowering...
```

---

## Dependency Graph

```
TVM Target Registration (CRITICAL BLOCKER)
    ↓
    └─→ Fixes test_target_registration.py xfail

Default Annotation Helper
    ↓
    └─→ Lower Hook
         ↓
         └─→ Complete WS1 ✅
```

**Critical path:**
1. TVM target registration must complete first
2. Annotation helper can start after target registration
3. Lower hook requires annotation helper
4. All must complete before Workstream 2 begins

---

## Testing Strategy

### Current Test Coverage
- ✅ Target string "tenstorrent" recognized
- ✅ Error handling when backend disabled
- ✅ Engine adapter validates target
- ⚠️ TVM Target creation (xfail - needs TVM registration)

### Needed Test Coverage (after completing TODO items)
- [ ] Default annotations applied to unannotated modules
- [ ] User annotations preserved when present
- [ ] Lower hook integrates correctly in full pipeline
- [ ] End-to-end dry-run of simple GEMM with TT target

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
- 4 tests pass
- 1 test xfail (until TVM target registration is complete)

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
