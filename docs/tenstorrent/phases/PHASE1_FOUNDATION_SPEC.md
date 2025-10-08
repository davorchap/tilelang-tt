# Phase 1: Foundation - Elementwise & Simple GEMM

**Timeline**: Weeks 19-20 (2 weeks)
**Priority**: CRITICAL - Foundation for all subsequent phases
**Status**: 🟡 In Progress (DST foundation complete, examples pending)

---

## Overview

Phase 1 establishes the foundational patterns for all TileLang→Metalium compilation:
- **Pattern 1 (Element-wise)**: Single-tile operations with per-tile DST lifecycle
- **Pattern 3 (K-loop GEMM)**: Multi-tile accumulation with DST spanning K iterations

These patterns form the basis for all compute kernels in Phases 2-6.

---

## Examples Covered

### 1.1 Elementwise Add ✅ **START HERE**
**File**: `examples/elementwise/example_elementwise_add.py` → `examples/tenstorrent/example_elementwise_add_tt.py`

**Why First**:
- Simplest TileLang pattern
- Tests basic CB management
- Tests DST acquire/release for element-wise ops
- No K-loop, no accumulation

**Required Transforms**:
- ✅ None (existing transforms sufficient)

**Required Codegen**:
- ✅ DST double buffering for element-wise ops (DONE - PR #53, #54)
  - `acquire_dst()` before computation
  - `commit_dst()` after computation
  - `release_dst()` to free registers
- 🔧 CB management for A, B inputs (IN PROGRESS)
- 🔧 CB management for C output (IN PROGRESS)
- ⏳ Element-wise intrinsic annotation (TODO)
- ⏳ Tile indexing for element-wise pattern (TODO)

**Expected Generated Compute Kernel**:
```cpp
void MAIN() {
    uint32_t out_tile_start_id = get_arg_val<uint32_t>(0);
    uint32_t num_output_tiles = get_arg_val<uint32_t>(1);

    for (uint32_t i = 0; i < num_output_tiles; ++i) {
        uint32_t tid = out_tile_start_id + i;
        uint32_t by = tid / grid_x;
        uint32_t bx = tid % grid_x;

        acquire_dst();

        // Wait for inputs
        cb_wait_front(CB_A, 1);
        cb_wait_front(CB_B, 1);

        // Compute C = A + B
        add_tiles_init();
        add_tiles(CB_A, CB_B, 0, 0, 0);

        // Commit and pack result
        cb_reserve_back(CB_C, 1);
        commit_dst();
        pack_tile(0, CB_C);
        cb_push_back(CB_C, 1);

        // Release inputs and DST
        cb_pop_front(CB_A, 1);
        cb_pop_front(CB_B, 1);
        release_dst();
    }
}
```

**Test Strategy**:
- Generate mock code, verify DST handshake
- Verify CB operations match Metalium examples
- Check tile indexing recovery (bx, by from linear tid)

**Acceptance Criteria**:
- ✅ DST lifecycle complete (acquire → commit → release)
- ⏳ CB wait/pop/push for all buffers
- ⏳ Element-wise intrinsic emitted
- ⏳ Proper tile indexing

---

### 1.2 Elementwise (Multi-operand)
**Files**: `examples/elementwise/*.py` (all variants)

**What's New**:
- Multiple inputs (A + B + C, A * B + C, etc.)
- More complex element-wise expressions
- Fusion opportunities

**Required Transforms**:
- ⏳ None (reuse existing)

**Required Codegen**:
- ⏳ Extend DST management for multi-operand ops
- ⏳ Support for multiple CB inputs
- ⏳ Expression tree visiting for fused element-wise ops

**Expected Pattern**:
```cpp
// For C = A + B * D
acquire_dst();
cb_wait_front(CB_A, 1);
cb_wait_front(CB_B, 1);
cb_wait_front(CB_D, 1);

mul_tiles_init();
mul_tiles(CB_B, CB_D, 0, 0, 0);  // Intermediate in DST
add_tiles_init();
add_tiles(CB_A, DST, 0, 0, 0);   // Final result

cb_reserve_back(CB_C, 1);
commit_dst();
pack_tile(0, CB_C);
cb_push_back(CB_C, 1);

cb_pop_front(CB_A, 1);
cb_pop_front(CB_B, 1);
cb_pop_front(CB_D, 1);
release_dst();
```

**Acceptance Criteria**:
- ⏳ Multi-input CB management
- ⏳ Fused expression evaluation
- ⏳ Correct DST usage for intermediate results

---

### 1.3 Simple GEMM (no K-loop pipelining)
**File**: `examples/gemm/example_tilelang_gemm_simple.py` (create if doesn't exist)

**What's New**:
- K-loop with accumulation
- DST double buffering for matmul tiles
- Packer waits for all K iterations

**Required Transforms**:
- ⏳ None (reuse existing)

**Required Codegen**:
- ✅ DST double buffering for matmul (FOUNDATION DONE)
  - Acquire DST at start of (bx, by) computation
  - For k in K-loop: matmul_tiles (accumulate in DST)
  - Commit DST after K-loop complete
  - Wait for packer
  - Release DST
- ⏳ CB pipelining for A, B tiles
- ⏳ K-loop structure with proper bounds
- ⏳ Matmul intrinsic emission

**Expected Generated Compute Kernel**:
```cpp
void MAIN() {
    uint32_t out_tile_start_id = get_arg_val<uint32_t>(0);
    uint32_t num_output_tiles = get_arg_val<uint32_t>(1);
    uint32_t Kt = get_arg_val<uint32_t>(2);

    for (uint32_t tile_idx = 0; tile_idx < num_output_tiles; ++tile_idx) {
        uint32_t tid = out_tile_start_id + tile_idx;
        uint32_t out_m = tid / Nt;
        uint32_t out_n = tid % Nt;

        acquire_dst();  // ✓ Before K-loop
        matmul_tiles_init(CB_A, CB_B, CB_C);

        for (uint32_t k = 0; k < Kt; ++k) {
            cb_wait_front(CB_A, 1);
            cb_wait_front(CB_B, 1);

            matmul_tiles(CB_A, CB_B, 0, 0, 0, false /* transpose */);

            cb_pop_front(CB_A, 1);
            cb_pop_front(CB_B, 1);
        }

        cb_reserve_back(CB_C, 1);
        commit_dst();  // ✓ After K-loop
        pack_tile(0, CB_C);
        cb_push_back(CB_C, 1);
        release_dst();  // ✓
    }
}
```

**Acceptance Criteria**:
- ✅ DST held across K-loop (acquire before, release after)
- ⏳ K-loop bounds from runtime args
- ⏳ Matmul intrinsic properly emitted
- ⏳ CB synchronization correct

---

## Transform Requirements

### None Required for Phase 1

Phase 1 uses existing transforms from WS1-3:
- ✅ `apply_tt_defaults()` - Stamp TT attributes
- ✅ `infer_tt_schedule()` - Compute tile assignments
- ✅ `infer_tt_shard()` - DRAM layout descriptors
- ✅ `grid_to_persistent_tt()` - Convert grid to persistent loop

**Why no new transforms?**
- Element-wise and simple GEMM patterns map directly to existing IR
- DST lifecycle is a codegen concern, not IR transform
- CB management is handled in codegen visitors

---

## Codegen Enhancements

### 1. Element-wise Intrinsic Support

**File**: `src/target/tt/codegen_tt_compute_visitor.cc`

**What to add**:
```cpp
void TTComputeCodegenVisitor::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == "tt.elementwise_add") {
    EmitElementwiseAdd(op);
  } else if (op->attr_key == "tt.elementwise_mul") {
    EmitElementwiseMul(op);
  } else if (op->attr_key == "tt.matmul_intrinsic") {
    EmitMatmulIntrinsic(op);
  }
  // ... existing cases
}

void TTComputeCodegenVisitor::EmitElementwiseAdd(const AttrStmtNode* op) {
  EmitLine("// Element-wise add");
  EmitLine("add_tiles_init();");
  EmitLine("add_tiles(CB_A, CB_B, 0, 0, 0);");
}
```

### 2. K-loop Bounds from Runtime Args

**File**: `src/target/tt/codegen_tt_compute_visitor.cc`

**What to add**:
- Extract Kt from IR attributes
- Emit runtime arg extraction
- Use Kt in K-loop bounds

### 3. Tile Indexing Recovery

**File**: `src/target/tt/codegen_tt_compute_visitor.cc`

**What to add**:
```cpp
void TTComputeCodegenVisitor::EmitTileIndexRecovery() {
  auto grid_x = func_->attrs.GetAttr<Integer>("tt_grid_x");

  EmitLine("uint32_t tid = out_tile_start_id + i;");
  EmitLine("uint32_t by = tid / " + std::to_string(grid_x.value()->value) + ";");
  EmitLine("uint32_t bx = tid % " + std::to_string(grid_x.value()->value) + ";");
}
```

---

## Implementation Checklist

### 1.1 Elementwise Add
- [x] DST foundation (PR #53)
- [x] Element-wise pattern support (PR #54)
- [ ] Element-wise intrinsic annotation in IR
- [ ] CB management for inputs/outputs
- [ ] Tile indexing recovery
- [ ] Test with mock execution
- [ ] PR: "Implement Elementwise Add for TT Backend"

### 1.2 Multi-operand Elementwise
- [ ] Multi-input CB management
- [ ] Expression tree visiting
- [ ] Fused operation support
- [ ] Test with multiple operators
- [ ] PR: "Support Multi-operand Elementwise Operations"

### 1.3 Simple GEMM
- [ ] K-loop bounds extraction
- [ ] Matmul intrinsic emission
- [ ] CB pipelining for A, B
- [ ] Test with simple matmul
- [ ] PR: "Implement Simple GEMM with K-loop"

---

## Success Metrics

**Code Quality**:
- ✅ All generated code follows Metalium examples
- ✅ DST lifecycle balanced (no acquire without release)
- ✅ CB operations properly synchronized

**Test Coverage**:
- ⏳ 3 new tests (one per example)
- ⏳ All tests pass in mock mode
- ⏳ Generated code inspectable and correct

**Documentation**:
- ✅ This spec document
- ✅ DST_DOUBLE_BUFFERING_SPEC.md
- ✅ TILELANG_TO_TT_EXAMPLES_PLAN.md

---

## Timeline Estimate

| Task | Duration | Dependencies |
|------|----------|--------------|
| 1.1 Elementwise Add | 4-6 hours | DST foundation (done) |
| 1.2 Multi-operand | 2-4 hours | 1.1 complete |
| 1.3 Simple GEMM | 4-6 hours | DST foundation (done) |
| **Total Phase 1** | **10-16 hours** | **2-3 days** |

---

## Next Phase Preview

**Phase 2: Optimizations** will build on Phase 1 foundation:
- CB double-buffering (depth=2, 4, 8)
- K-loop software pipelining
- Type conversion operations
- Reduction patterns

Phase 2 requires Phase 1 examples working to validate pipelining optimizations.

---

**Status**: 🟡 In Progress
**Last Updated**: 2025-10-08
**Next Milestone**: Complete 1.1 Elementwise Add
