# Metalium API Fix Implementation Plan

**Date**: 2025-10-08
**Objective**: Fix TileLang compiler to generate correct Metalium-compatible code
**Reference**: METALIUM_API_ANALYSIS.md

---

## Overview

**Current Status**: Compiler generates non-functional Metalium code with wrong API names and incorrect control flow.

**Target**: Generate code matching real Metalium examples:
- `tiles_add.cpp` (element-wise add)
- `mm.cpp` (matmul multi-core)
- `eltwise_sfpu.cpp` (SFPU operations)

**Estimated Effort**: 3-5 days of development + testing

---

## Phase 1: API Renaming (Critical - Day 1)

### Task 1.1: Rename Tile Register APIs

**Files**:
- `src/target/tt/codegen_tt_compute_visitor.cc`
- `src/target/tt/codegen_tt_compute_visitor.h`

**Changes**:
```cpp
// OLD (Wrong)
inline void acquire_dst() {}
inline void commit_dst() {}
inline void wait_for_tile() {}
inline void release_dst() {}

// NEW (Correct)
inline void tile_regs_acquire() {}
inline void tile_regs_commit() {}
inline void tile_regs_wait() {}
inline void tile_regs_release() {}
```

**Method Renaming**:
- `EmitDSTAcquire()` → `EmitTileRegsAcquire()`
- `EmitDSTCommit()` → `EmitTileRegsCommit()`
- Add `EmitTileRegsWait()`
- `EmitDSTRelease()` → `EmitTileRegsRelease()`

**Test**:
```bash
# Verify generated code uses correct API names
python examples/tenstorrent/example_elementwise_add_tt.py | grep "tile_regs_acquire"
```

**Acceptance Criteria**:
- ✅ All generated code uses `tile_regs_*` names
- ✅ No references to `acquire_dst`, `commit_dst`, etc.
- ✅ 95 existing tests still pass

---

### Task 1.2: Fix CB Index Format

**Files**:
- `src/target/tt/codegen_tt_compute_visitor.cc`
- `src/target/tt/codegen_tt_reader_visitor.cc`
- `src/target/tt/codegen_tt_writer_visitor.cc`

**Changes**:
```cpp
// OLD (Wrong)
constexpr uint32_t CB_A = 0;
constexpr uint32_t CB_B = 1;
constexpr uint32_t CB_C = 2;

// NEW (Correct)
constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;
```

**Implementation**:
1. Add CB index mapping function:
   ```cpp
   std::string GetCBIndex(int buffer_id) {
       if (buffer_id == 2) return "tt::CBIndex::c_16";  // Output
       return "tt::CBIndex::c_" + std::to_string(buffer_id);
   }
   ```

2. Update all CB API calls to use proper format

**Test**:
```bash
# Verify CB index format
python examples/tenstorrent/example_elementwise_add_tt.py | grep "tt::CBIndex"
```

**Acceptance Criteria**:
- ✅ All CB references use `tt::CBIndex::c_N` format
- ✅ Input buffers use c_0, c_1, etc.
- ✅ Output buffer uses c_16
- ✅ No hardcoded `CB_A`, `CB_B`, `CB_C` constants

---

## Phase 2: Fix Control Flow (Critical - Day 2)

### Task 2.1: Fix Element-wise Control Flow

**File**: `src/target/tt/codegen_tt_compute_visitor.cc`

**Current (Wrong)**:
```cpp
acquire_dst();  // OUTSIDE loop

for (tile in tiles) {
    cb_wait_front(...);
    add_tiles_init();  // WRONG: Should be before loop
    add_tiles(...);
    cb_pop_front(...);
}

commit_dst();  // OUTSIDE loop
pack_tile(...);
release_dst();
```

**Target (Correct)**:
```cpp
// INITIALIZATION (before loop)
binary_op_init_common(cb_in0, cb_in1, cb_out);
add_tiles_init(cb_in0, cb_in1);

// MAIN LOOP
for (tile in tiles) {
    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);

    tile_regs_acquire();
    add_tiles(cb_in0, cb_in1, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();

    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);

    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);

    tile_regs_release();
}
```

**Implementation Steps**:
1. Add pattern detection for element-wise operations
2. Generate initialization calls before loop
3. Move tile_regs_acquire/commit/wait/release inside loop
4. Move pack_tile/cb_push_back inside loop

**Test**:
Create test that validates structure:
```python
def validate_elementwise_structure(code):
    lines = code.split('\n')
    # Find init calls before loop
    assert "binary_op_init_common" in code
    assert code.index("add_tiles_init") < code.index("for (")
    # Find tile_regs calls inside loop
    loop_start = code.index("for (")
    loop_body = code[loop_start:]
    assert "tile_regs_acquire" in loop_body
    assert "tile_regs_release" in loop_body
```

**Acceptance Criteria**:
- ✅ Initialization calls appear before loop
- ✅ `tile_regs_acquire/commit/wait/release` inside loop
- ✅ `pack_tile` inside loop
- ✅ Generated code matches `tiles_add.cpp` structure

---

### Task 2.2: Fix Matmul Control Flow

**File**: `src/target/tt/codegen_tt_compute_visitor.cc`

**Current (Wrong)**:
```cpp
acquire_dst();  // OUTSIDE outer loop

for (out_tile in output_tiles) {
    matmul_tiles_init(...);  // WRONG: Doesn't exist
    for (k in Kt) {
        cb_wait_front(...);
        matmul_tiles(...);  // WRONG signature
        cb_pop_front(...);
    }
    cb_reserve_back(...);
}

commit_dst();
pack_tile(...);
release_dst();
```

**Target (Correct)**:
```cpp
// INITIALIZATION (before loops)
mm_init(cb_in0, cb_in1, cb_out);

// OUTER LOOP (output tiles)
for (out_tile in output_tiles) {
    tile_regs_acquire();  // ONCE per output tile

    // INNER LOOP (K dimension)
    for (kt in Kt) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);

        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }

    tile_regs_commit();
    tile_regs_wait();

    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);

    tile_regs_release();
}
```

**Implementation Steps**:
1. Detect matmul pattern (has K-loop)
2. Generate `mm_init()` before loops
3. Remove `matmul_tiles_init()` (doesn't exist)
4. Move `tile_regs_acquire` inside outer loop, before K-loop
5. Move `tile_regs_commit/wait` inside outer loop, after K-loop
6. Move `pack_tile/cb_push_back` inside outer loop
7. Move `tile_regs_release` inside outer loop

**Test**:
```python
def validate_matmul_structure(code):
    # mm_init before loops
    assert code.index("mm_init") < code.index("for (")
    # tile_regs_acquire after outer loop start, before K-loop
    outer_loop = code.index("for (uint32_t i")
    k_loop = code.index("for (uint32_t k")
    acquire = code.index("tile_regs_acquire")
    assert outer_loop < acquire < k_loop
    # tile_regs_commit after K-loop
    commit = code.index("tile_regs_commit")
    assert k_loop < commit
```

**Acceptance Criteria**:
- ✅ `mm_init()` appears before loops
- ✅ No `matmul_tiles_init()` calls
- ✅ `tile_regs_acquire` inside outer loop, before K-loop
- ✅ `tile_regs_commit/wait/release` inside outer loop, after K-loop
- ✅ Generated code matches `mm.cpp` structure

---

## Phase 3: Fix API Signatures (Critical - Day 3)

### Task 3.1: Fix matmul_tiles Signature

**File**: `src/target/tt/codegen_tt_compute_visitor.cc`

**Current (Wrong)**:
```cpp
matmul_tiles(CB_A, CB_B, CB_C, accumulate);  // 4 params
```

**Target (Correct)**:
```cpp
matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);  // 6 params
```

**Parameters**:
1. `tt::CBIndex cb_in0` - Input CB for A
2. `tt::CBIndex cb_in1` - Input CB for B
3. `uint32_t tile_idx_in0` - Tile index in CB for A (usually 0)
4. `uint32_t tile_idx_in1` - Tile index in CB for B (usually 0)
5. `uint32_t dst_tile_idx` - Destination tile index (usually 0)
6. `bool transpose` - Transpose flag (false for normal accumulation)

**Implementation**:
```cpp
void EmitMatmulTiles(...) {
    ss_ << "matmul_tiles(";
    ss_ << GetCBIndex(0) << ", ";     // cb_in0
    ss_ << GetCBIndex(1) << ", ";     // cb_in1
    ss_ << "0, ";                      // tile_idx_in0
    ss_ << "0, ";                      // tile_idx_in1
    ss_ << "0, ";                      // dst_tile_idx
    ss_ << "false";                    // transpose/accumulate
    ss_ << ");";
}
```

**Note**: The `false` parameter means "don't transpose" - the API automatically accumulates across K iterations.

**Test**:
```bash
# Verify 6-parameter signature
python examples/tenstorrent/example_simple_gemm_tt.py | grep "matmul_tiles"
# Should show: matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0, false)
```

**Acceptance Criteria**:
- ✅ `matmul_tiles` uses 6 parameters
- ✅ Parameters match real Metalium signature
- ✅ No `matmul_tiles_init()` calls

---

### Task 3.2: Fix Initialization Calls

**File**: `src/target/tt/codegen_tt_compute_visitor.cc`

**Add Method**:
```cpp
void EmitInitialization(OpType op_type) {
    if (op_type == OpType::kElementwise) {
        EmitLine("binary_op_init_common(cb_in0, cb_in1, cb_out);");
        EmitLine("add_tiles_init(cb_in0, cb_in1);");
    } else if (op_type == OpType::kMatmul) {
        EmitLine("mm_init(cb_in0, cb_in1, cb_out);");
    } else if (op_type == OpType::kSFPU) {
        EmitLine("init_sfpu(cb_in, cb_out);");
        // Add specific SFPU init based on operation
    }
}
```

**Call Before Loop**:
```cpp
void VisitComputeKernel(...) {
    // ... CB declarations ...

    // Detect operation type
    OpType op_type = DetectOperationType(func);

    // Emit initialization
    EmitInitialization(op_type);

    // Emit main loop
    EmitMainLoop(...);
}
```

**Acceptance Criteria**:
- ✅ `binary_op_init_common` and `add_tiles_init` for element-wise
- ✅ `mm_init` for matmul
- ✅ Initialization appears before loop

---

## Phase 4: Update Includes & Namespace (Day 4)

### Task 4.1: Add Proper Metalium Headers

**File**: `src/target/tt/codegen_tt_compute_visitor.cc`

**Current (Mock Mode)**:
```cpp
// Mock TT circular buffer APIs for dry-run
inline void cb_wait_front(...) {}
// ... etc
```

**Target (Real Mode)**:
```cpp
#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api.h"
```

**Implementation**:
Add mode flag:
```cpp
bool use_real_metalium_headers = false;  // Set via env var or param

if (use_real_metalium_headers) {
    EmitRealIncludes();
} else {
    EmitMockAPIs();
}
```

**Acceptance Criteria**:
- ✅ Mock mode still works (for dry-run)
- ✅ Real mode generates proper includes
- ✅ Mode selectable via parameter

---

### Task 4.2: Add Namespace Wrapper

**Current**:
```cpp
void MAIN() {
    // ...
}
```

**Target**:
```cpp
namespace NAMESPACE {
void MAIN {
    // ...
}
}  // namespace NAMESPACE
```

**Implementation**:
```cpp
void EmitComputeKernel(...) {
    if (use_real_metalium_headers) {
        EmitLine("namespace NAMESPACE {");
    }

    EmitLine("void MAIN {");
    // ... kernel body ...
    EmitLine("}");

    if (use_real_metalium_headers) {
        EmitLine("}  // namespace NAMESPACE");
    }
}
```

**Acceptance Criteria**:
- ✅ Real mode wraps kernel in namespace
- ✅ Mock mode omits namespace (for simplicity)

---

## Phase 5: Update Example Validations (Day 5)

### Task 5.1: Update All Example Validation Scripts

**Files**:
- `examples/tenstorrent/example_elementwise_add_tt.py`
- `examples/tenstorrent/example_simple_gemm_tt.py`
- `examples/tenstorrent/example_gemm_cb_pipeline_tt.py`
- All Phase 2-6 examples

**Changes**:
```python
# OLD validation
has_acquire = "acquire_dst()" in compute
has_commit = "commit_dst()" in compute

# NEW validation
has_acquire = "tile_regs_acquire()" in compute
has_commit = "tile_regs_commit()" in compute
has_wait = "tile_regs_wait()" in compute
has_release = "tile_regs_release()" in compute

# Add CB format check
has_proper_cb_format = "tt::CBIndex::c_" in compute

# Add initialization check
has_init = ("binary_op_init_common" in compute or
            "mm_init" in compute)
```

**Acceptance Criteria**:
- ✅ All examples validate correct API names
- ✅ All examples validate CB index format
- ✅ All examples validate initialization calls
- ✅ All examples validate control flow structure

---

### Task 5.2: Create Real Metalium Comparison Tests

**New File**: `testing/python/tt/test_metalium_comparison.py`

```python
def test_elementwise_matches_metalium():
    """Compare generated element-wise add with real tiles_add.cpp"""
    # Generate code
    artifacts = generate_elementwise_add()
    generated = artifacts['compute.cpp']

    # Load real Metalium example
    real_metalium = load_metalium_example('tiles_add.cpp')

    # Compare structure
    assert has_same_initialization(generated, real_metalium)
    assert has_same_loop_structure(generated, real_metalium)
    assert has_same_api_calls(generated, real_metalium)

def test_matmul_matches_metalium():
    """Compare generated matmul with real mm.cpp"""
    # Similar to above
    ...
```

**Acceptance Criteria**:
- ✅ Element-wise comparison test passes
- ✅ Matmul comparison test passes
- ✅ Tests catch any regressions

---

## Phase 6: Documentation & Cleanup (Day 5)

### Task 6.1: Update Documentation

**Files**:
- Update `METALIUM_API_ANALYSIS.md` with fixes applied
- Update `PHASES_STATUS.md` with Metalium API fix completion
- Update examples with corrected validation

**Acceptance Criteria**:
- ✅ All documentation reflects correct APIs
- ✅ Examples show correct usage

---

### Task 6.2: Final Validation

**Run Full Test Suite**:
```bash
# All existing tests should pass
pytest testing/python/tt/ -v

# All examples should validate correctly
for ex in examples/tenstorrent/*.py; do
    python "$ex" || echo "FAILED: $ex"
done
```

**Manual Comparison**:
```bash
# Generate element-wise add
python examples/tenstorrent/example_elementwise_add_tt.py > /tmp/gen_add.cpp

# Compare with real Metalium (structure, not exact match)
diff -u /tmp/real_tiles_add.cpp /tmp/gen_add.cpp
```

**Acceptance Criteria**:
- ✅ All 95 tests pass
- ✅ All 10 enhanced examples validate correctly
- ✅ Generated code structure matches real Metalium
- ✅ No mock API names in generated code
- ✅ Proper CB index format throughout

---

## Testing Matrix

| Test Type | Coverage | Status |
|-----------|----------|--------|
| Unit Tests (API Names) | tile_regs_* naming | ⏳ |
| Unit Tests (CB Format) | tt::CBIndex::c_N | ⏳ |
| Unit Tests (Signatures) | matmul_tiles params | ⏳ |
| Integration (Element-wise) | Full pattern match | ⏳ |
| Integration (Matmul) | Full pattern match | ⏳ |
| Regression | 95 existing tests | ⏳ |
| Validation | 10 example scripts | ⏳ |

---

## Risk Assessment

### High Risk

1. **Breaking Changes**: Renaming APIs will break all existing validation
   - Mitigation: Update all examples simultaneously
   - Estimated Impact: 2-3 hours

2. **Control Flow Complexity**: Matmul control flow is intricate
   - Mitigation: Careful step-by-step implementation with tests
   - Estimated Impact: 1 day

### Medium Risk

3. **CB Index Format**: Many files to update
   - Mitigation: Systematic search-and-replace with verification
   - Estimated Impact: 3-4 hours

4. **Initialization Calls**: Pattern detection may be fragile
   - Mitigation: Conservative detection, expand later
   - Estimated Impact: Half day

### Low Risk

5. **Namespace Wrapper**: Simple addition
   - Mitigation: Optional feature, well-scoped
   - Estimated Impact: 1 hour

---

## Timeline

**Day 1**: Phase 1 (API Renaming)
- Morning: Task 1.1 (Tile Register APIs)
- Afternoon: Task 1.2 (CB Index Format)
- Evening: Update all example validations for new names

**Day 2**: Phase 2 (Control Flow)
- Morning: Task 2.1 (Element-wise Control Flow)
- Afternoon: Task 2.2 (Matmul Control Flow)
- Evening: Test and validate structure

**Day 3**: Phase 3 (API Signatures)
- Morning: Task 3.1 (matmul_tiles Signature)
- Afternoon: Task 3.2 (Initialization Calls)
- Evening: Integration testing

**Day 4**: Phase 4 (Includes & Namespace)
- Morning: Task 4.1 (Metalium Headers)
- Afternoon: Task 4.2 (Namespace Wrapper)
- Evening: Test both mock and real modes

**Day 5**: Phase 5-6 (Validation & Documentation)
- Morning: Task 5.1 (Update Example Validations)
- Afternoon: Task 5.2 (Comparison Tests)
- Evening: Task 6.1-6.2 (Documentation & Final Validation)

**Total**: 5 days (can be compressed to 3-4 days if focused)

---

## Success Criteria

**Must Have**:
- ✅ All tile register APIs use `tile_regs_*` naming
- ✅ All CB references use `tt::CBIndex::c_N` format
- ✅ Element-wise control flow matches `tiles_add.cpp`
- ✅ Matmul control flow matches `mm.cpp`
- ✅ `matmul_tiles` uses correct 6-parameter signature
- ✅ Initialization calls present before loops
- ✅ All 95 tests pass
- ✅ All 10 examples validate correctly

**Nice to Have**:
- ✅ Real Metalium header mode (for actual hardware compilation)
- ✅ Namespace wrapper
- ✅ Comparison tests against real Metalium code
- ✅ Comprehensive documentation

**Stretch Goals**:
- Generate code that compiles with real Metalium headers
- Hardware execution validation
- Performance benchmarking

---

## Next Actions

1. Review this plan with stakeholders
2. Set up task tracking (TodoWrite or GitHub issues)
3. Begin Phase 1: API Renaming
4. Daily progress updates and demos
5. Continuous integration testing

**Status**: Ready to begin implementation
