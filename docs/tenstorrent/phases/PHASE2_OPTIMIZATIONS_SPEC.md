# Phase 2: Optimizations - CB Pipelining, Cast, Reductions

**Timeline**: Weeks 21-22 (2 weeks)
**Priority**: HIGH - Performance critical optimizations
**Status**: ⏳ Not Started (Blocked by Phase 1)

---

## Overview

Phase 2 builds on Phase 1 foundation to add performance-critical optimizations:
- **CB Double-Buffering**: Overlap compute with data movement (depth=2, 4, 8)
- **K-loop Software Pipelining**: Prefetch next iteration while computing current
- **Type Conversion**: Data type transformations (FP32→BF16, INT8→FP16, etc.)
- **Reduction Operations**: sum, max, mean across dimensions

These optimizations are essential for achieving competitive performance on Tenstorrent hardware.

---

## Examples Covered

### 2.1 GEMM with CB Double-Buffering
**File**: `examples/gemm/*.py` (optimized variants)

**What's New**:
- K-loop software pipelining
- Double-buffered CBs (depth=2 minimum)
- Overlap compute with data movement

**Required Transforms**:
- 🆕 **NEW**: CB depth inference
  - Analyze K-loop structure
  - Insert cb_reserve_back/cb_push_back for producer (reader)
  - Insert cb_wait_front/cb_pop_front for consumer (compute)
  - Determine optimal depth (2, 4, or 8 tiles)

**Transform Spec**: See `docs/tenstorrent/transforms/CB_DEPTH_INFERENCE_SPEC.md` (to be created)

**Required Codegen**:
- CB double-buffering in reader kernel
- CB synchronization in compute kernel
- Prefetch next K iteration while computing current

**Expected Pattern**:

**Reader Kernel**:
```cpp
for (uint32_t out_tile = 0; out_tile < num_out_tiles; ++out_tile) {
    for (uint32_t k = 0; k < Kt; ++k) {
        // Double-buffered: can push while compute consumes previous
        cb_reserve_back(CB_A, 1);
        noc_async_read_tile(tile_a_idx, dram_addr_a, l1_write_addr_a);
        noc_async_read_barrier();
        cb_push_back(CB_A, 1);

        cb_reserve_back(CB_B, 1);
        noc_async_read_tile(tile_b_idx, dram_addr_b, l1_write_addr_b);
        noc_async_read_barrier();
        cb_push_back(CB_B, 1);
    }
}
```

**Compute Kernel** (with pipelining):
```cpp
for (uint32_t tile_idx = 0; tile_idx < num_output_tiles; ++tile_idx) {
    acquire_dst();
    matmul_tiles_init(CB_A, CB_B, CB_C);

    for (uint32_t k = 0; k < Kt; ++k) {
        // Wait for current iteration data
        cb_wait_front(CB_A, 1);
        cb_wait_front(CB_B, 1);

        // Compute current iteration
        matmul_tiles(CB_A, CB_B, 0, 0, 0, false);

        // Pop current (frees buffer for next iteration)
        cb_pop_front(CB_A, 1);
        cb_pop_front(CB_B, 1);
    }

    cb_reserve_back(CB_C, 1);
    commit_dst();
    pack_tile(0, CB_C);
    cb_push_back(CB_C, 1);
    release_dst();
}
```

**Acceptance Criteria**:
- CB depth = 2 or higher (enables pipelining)
- Reader can push while compute consumes
- Verify overlap via timing analysis (mock mode)
- No deadlocks (proper synchronization)

---

### 2.2 Cast / Type Conversion
**File**: `examples/cast/*.py`

**What's New**:
- Data type conversions (FP32 → BF16, INT8 → FP16, etc.)
- May use DST for conversion or direct packer path
- Different tile sizes for different dtypes

**Required Transforms**:
- None (handled in codegen)

**Required Codegen**:
- Type conversion in DST
- Pack to different output type
- Handle different tile sizes (FP32 tiles are 2x larger than FP16)

**Expected Pattern**:
```cpp
// FP32 → BF16 conversion
for (uint32_t i = 0; i < num_tiles; ++i) {
    acquire_dst();

    cb_wait_front(CB_IN_FP32, 1);

    // Convert FP32 → BF16 in DST
    convert_fp32_to_bf16_tiles_init();
    convert_fp32_to_bf16_tiles(CB_IN_FP32, 0, 0);

    cb_reserve_back(CB_OUT_BF16, 1);
    commit_dst();
    pack_tile(0, CB_OUT_BF16);
    cb_push_back(CB_OUT_BF16, 1);

    cb_pop_front(CB_IN_FP32, 1);
    release_dst();
}
```

**Acceptance Criteria**:
- Correct tile size calculations for different dtypes
- Type conversion intrinsics emitted
- CB configurations match dtype requirements

---

### 2.3 Reduction Operations
**File**: `examples/norm/*.py`, `examples/online_softmax/*.py`

**What's New**:
- Reduction patterns (sum, max, mean, etc.)
- Potential multi-stage reductions
- Synchronization across cores for global reductions

**Required Transforms**:
- 🆕 **NEW**: Reduction pattern recognition
  - Identify reduction axis (row, col, or full)
  - Plan multi-core reduction strategy
  - Insert reduction intrinsic annotations

**Transform Spec**: See `docs/tenstorrent/transforms/REDUCTION_PATTERN_SPEC.md` (to be created)

**Required Codegen**:
- Reduction in DST (local reduction per core)
- Inter-core reduction via NoC (if needed for global)
- Support for different reduction ops (sum, max, mean)

**Expected Pattern (Row Reduction)**:
```cpp
// Reduce across row: C[m] = sum(A[m, :])
for (uint32_t row_idx = 0; row_idx < num_rows; ++row_idx) {
    acquire_dst();

    // Initialize accumulator to zero
    reduce_init_tiles(CB_OUT);

    // Accumulate across row tiles
    for (uint32_t col_idx = 0; col_idx < num_cols; ++col_idx) {
        cb_wait_front(CB_IN, 1);
        reduce_sum_tiles(CB_IN, 0, 0);
        cb_pop_front(CB_IN, 1);
    }

    cb_reserve_back(CB_OUT, 1);
    commit_dst();
    pack_tile(0, CB_OUT);
    cb_push_back(CB_OUT, 1);
    release_dst();
}
```

**Acceptance Criteria**:
- Local reductions work correctly
- Multi-core reductions (if needed) properly synchronized
- Support for sum, max, mean operations

---

## New Transform Requirements

### Transform 1: CB Depth Inference

**Purpose**: Automatically determine optimal circular buffer depths for pipelining

**Input**: TIR with K-loop structure
**Output**: TIR with CB depth annotations

**Algorithm**:
1. Identify producer-consumer relationships (reader→compute→writer)
2. Analyze K-loop trip count
3. Determine optimal depth:
   - depth=2 for simple double-buffering
   - depth=4 for deeper pipelines
   - depth=8 for very deep pipelines (if L1 allows)
4. Annotate buffers with depth attribute

**Example**:
```python
# Before
A_tile = T.alloc_buffer((32, 32), "float16")

# After
A_tile = T.alloc_buffer((32, 32), "float16")
A_tile.attrs["tt_cb_depth"] = 2  # Double-buffered
```

**Implementation File**: `src/transform/tt/cb_depth_inference.cc`

**Test File**: `testing/python/tt/test_cb_depth_inference.py`

---

### Transform 2: Reduction Pattern Recognition

**Purpose**: Identify and annotate reduction operations

**Input**: TIR with reduction loops
**Output**: TIR with reduction intrinsic annotations

**Algorithm**:
1. Detect reduction patterns (loops with accumulation)
2. Identify reduction axis (row, col, or full)
3. Determine reduction operation (sum, max, mean, etc.)
4. Insert reduction intrinsic annotation

**Example**:
```python
# Detect pattern
for i in range(M):
    acc = 0
    for j in range(N):
        acc += A[i, j]
    C[i] = acc

# Annotate as reduction
T.attr_stmt("tt.reduction_sum", axis=1)
```

**Implementation File**: `src/transform/tt/reduction_pattern_recognition.cc`

**Test File**: `testing/python/tt/test_reduction_pattern.py`

---

## Codegen Enhancements

### 1. CB Depth Configuration in Host Program

**File**: `src/target/tt/codegen_tt.cc` (host program generator)

**What to add**:
```cpp
// Extract CB depth from buffer attributes
int cb_depth = 2;  // default
if (buffer->attrs.count("tt_cb_depth")) {
    cb_depth = buffer->attrs["tt_cb_depth"].as<IntImm>()->value;
}

// Generate CB config with depth
std::ostringstream config;
config << "CircularBufferConfig cb_" << cb_id << "(";
config << cb_id << ", " << tile_size << ", " << cb_depth << ");";
EmitLine(config.str());
```

### 2. Type Conversion Intrinsics

**File**: `src/target/tt/codegen_tt_compute_visitor.cc`

**What to add**:
```cpp
void TTComputeCodegenVisitor::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == "tt.convert_fp32_to_bf16") {
    EmitTypeConversion(op, "fp32", "bf16");
  } else if (op->attr_key == "tt.convert_int8_to_fp16") {
    EmitTypeConversion(op, "int8", "fp16");
  }
  // ... existing cases
}

void TTComputeCodegenVisitor::EmitTypeConversion(
    const AttrStmtNode* op, const std::string& from, const std::string& to) {
  EmitLine("// Type conversion: " + from + " → " + to);
  EmitLine("convert_" + from + "_to_" + to + "_tiles_init();");
  EmitLine("convert_" + from + "_to_" + to + "_tiles(CB_IN, 0, 0);");
}
```

### 3. Reduction Intrinsics

**File**: `src/target/tt/codegen_tt_compute_visitor.cc`

**What to add**:
```cpp
void TTComputeCodegenVisitor::EmitReductionIntrinsic(const AttrStmtNode* op) {
  std::string reduce_op = GetReduceOp(op);  // "sum", "max", "mean"

  EmitLine("// Reduction: " + reduce_op);
  EmitLine("reduce_init_tiles(CB_OUT);");

  // Visit reduction loop body
  VisitStmt(op->body);

  // Finalize reduction if needed
  if (reduce_op == "mean") {
    EmitLine("reduce_mean_finalize(CB_OUT, " + std::to_string(count) + ");");
  }
}
```

---

## Implementation Checklist

### 2.1 GEMM with CB Double-Buffering
- [ ] Implement CB depth inference transform
- [ ] Update host program CB config generation
- [ ] Update reader kernel for double-buffered push
- [ ] Update compute kernel for pipelined consume
- [ ] Test with various depths (2, 4, 8)
- [ ] PR: "Add CB Double-Buffering for GEMM K-loop"

### 2.2 Cast / Type Conversion
- [ ] Add type conversion intrinsic annotations
- [ ] Implement type conversion codegen
- [ ] Handle different tile sizes per dtype
- [ ] Test FP32→BF16, INT8→FP16 conversions
- [ ] PR: "Support Type Conversion Operations"

### 2.3 Reduction Operations
- [ ] Implement reduction pattern recognition transform
- [ ] Add reduction intrinsic codegen
- [ ] Support sum, max, mean operations
- [ ] Test row/col/full reductions
- [ ] PR: "Implement Reduction Operations"

---

## Success Metrics

**Performance**:
- CB pipelining shows overlap in timing (mock mode analysis)
- No CB deadlocks or stalls
- Reduction operations correctly accumulate

**Code Quality**:
- CB depths configurable and correct
- Type conversions handle all dtype combinations
- Reductions work for all axes

**Test Coverage**:
- 3 new examples tested (GEMM pipeline, cast, reduction)
- All tests pass in mock mode
- Generated code matches Metalium patterns

---

## Timeline Estimate

| Task | Duration | Dependencies |
|------|----------|--------------|
| 2.1 CB Double-Buffering | 6-8 hours | Phase 1.3 complete |
| 2.2 Cast / Type Conversion | 3-4 hours | Phase 1.1 complete |
| 2.3 Reduction Operations | 6-8 hours | Phase 1.1 complete |
| **Total Phase 2** | **15-20 hours** | **3-4 days** |

---

## Next Phase Preview

**Phase 3: Advanced Patterns** will build on Phases 1-2:
- GEMV (matrix-vector multiplication)
- Convolution (im2col or direct)
- Split-K GEMM (multi-core K-dimension split)

Phase 3 requires CB pipelining (Phase 2.1) to be efficient.

---

**Status**: ⏳ Not Started
**Blocked By**: Phase 1 completion
**Last Updated**: 2025-10-08
**Next Milestone**: Complete Phase 1 first
