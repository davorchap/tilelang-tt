# IR Lowering Implementation Tasks

**Document Version:** 1.0
**Date:** 2025-10-08
**Status:** Active

## Overview

Based on the IR lowering analysis, this document tracks the implementation tasks needed to complete the Tenstorrent backend pattern detection and tensorization.

## Current Status

**Problem:** TT codegen tries to do pattern detection (heuristics) AND code emission, which is incomplete and fragile.

**Generated Code Issues:**
- K-loop structure detected ✅
- Scaffolding emitted (mm_init, tile_regs_acquire) ✅
- **Body still has raw array operations** ❌
- Missing intrinsic calls (cb_wait_front, matmul_tiles, cb_pop_front) ❌

**Root Cause:** Pattern detection in codegen visitor instead of transform pass.

## Tasks

### Task 1: Extend TensorizeTT Transform Pass

**File:** `src/transform/tt/tensorize_tt.cc`

**Current Implementation:**
```cpp
// Only handles T.gemm() intrinsic calls
Stmt VisitStmt_(const AttrStmtNode* op) override {
  if (op->attr_key == "pragma_gemm" || op->attr_key == "tl.gemm") {
    // Annotate with tt.matmul_intrinsic
    return AttrStmt(op->node, "tt.matmul_intrinsic", matmul_id, new_body);
  }
}
```

**What to Add:**

```cpp
class TensorizeTTMutator : public StmtMutator {
 private:
  int matmul_count_ = 0;
  int eltwise_count_ = 0;

 public:
  Stmt VisitStmt_(const ForNode* op) override {
    std::string loop_var = op->loop_var->name_hint;

    // Pattern 1: K-loop for matmul
    if (IsKLoop(loop_var)) {
      MatmulPatternMatcher matcher;
      if (matcher.Match(op->body)) {
        // Extract A, B, C buffer names from pattern
        Array<Var> input_bufs = {matcher.GetBufferA(), matcher.GetBufferB()};
        Var output_buf = matcher.GetBufferC();

        // Create nested annotations
        Stmt annotated = op->body;

        // Annotate output buffer
        annotated = AttrStmt(
          output_buf,
          "tt.output_buffer",
          StringImm(output_buf->name_hint),
          annotated
        );

        // Annotate input buffers
        annotated = AttrStmt(
          input_bufs[0],
          "tt.input_buffers",
          Array<PrimExpr>{input_bufs[0], input_bufs[1]},
          annotated
        );

        // Annotate K-loop
        PrimExpr matmul_id = IntImm(DataType::Int(32), matmul_count_++);
        annotated = AttrStmt(
          op->loop_var,
          "tt.matmul_k_loop",
          matmul_id,
          annotated
        );

        // Return new For with annotated body
        return For(
          op->loop_var,
          op->min,
          op->extent,
          op->kind,
          annotated
        );
      }
    }

    // Pattern 2: Element-wise tile operations
    if (IsOuterTileLoop(op)) {
      ElementwisePatternMatcher matcher;
      if (matcher.Match(op->body)) {
        // Annotate element-wise operation
        PrimExpr eltwise_id = IntImm(DataType::Int(32), eltwise_count_++);
        Stmt annotated = AttrStmt(
          op->loop_var,
          "tt.elementwise_op",
          eltwise_id,
          op->body
        );

        return For(
          op->loop_var,
          op->min,
          op->extent,
          op->kind,
          annotated
        );
      }
    }

    return StmtMutator::VisitStmt_(op);
  }

 private:
  bool IsKLoop(const std::string& var_name) const {
    return var_name == "kt" || var_name == "k" ||
           var_name.find("_k") != std::string::npos;
  }

  bool IsOuterTileLoop(const ForNode* op) const {
    // Check if this is the outer loop of a tile operation
    // (typically 'i' iterating 0 to 32 or similar)
    return op->extent.as<IntImmNode>() &&
           op->extent.as<IntImmNode>()->value == 32;
  }
};
```

**Pattern Matchers:**

```cpp
class MatmulPatternMatcher : public StmtExprVisitor {
 public:
  bool Match(const Stmt& stmt) {
    VisitStmt(stmt);
    return matched_;
  }

  Var GetBufferA() const { return buffer_a_; }
  Var GetBufferB() const { return buffer_b_; }
  Var GetBufferC() const { return buffer_c_; }

 private:
  void VisitStmt_(const BufferStoreNode* op) override {
    // Look for: C[m,n] = ... or C[m,n] += ...
    buffer_c_ = op->buffer->data;

    // Analyze RHS to find A[m,k] * B[k,n] pattern
    if (auto* add = op->value.as<AddNode>()) {
      if (auto* mul = add->b.as<MulNode>()) {
        // Found += pattern
        AnalyzeMultiply(mul);
      }
    } else if (auto* mul = op->value.as<MulNode>()) {
      // Found = pattern
      AnalyzeMultiply(mul);
    }
  }

  void AnalyzeMultiply(const MulNode* mul) {
    // Check if LHS is A[m,k] and RHS is B[k,n]
    if (auto* load_a = mul->a.as<BufferLoadNode>()) {
      if (auto* load_b = mul->b.as<BufferLoadNode>()) {
        buffer_a_ = load_a->buffer->data;
        buffer_b_ = load_b->buffer->data;
        matched_ = true;
      }
    }
  }

  bool matched_ = false;
  Var buffer_a_;
  Var buffer_b_;
  Var buffer_c_;
};
```

**Subtasks:**
- [ ] Implement MatmulPatternMatcher
- [ ] Implement ElementwisePatternMatcher
- [ ] Add annotation creation logic
- [ ] Add tests for pattern detection
- [ ] Verify annotations appear in IR dump

**Estimated Effort:** 2-3 days

---

### Task 2: Update Compute Visitor to Read Annotations

**File:** `src/target/tt/codegen_tt_compute_visitor.cc`

**Current Implementation:**
```cpp
Stmt VisitStmt_(const ForNode* op) override {
  std::string loop_var = op->loop_var->name_hint;
  bool is_k_loop = (loop_var == "kt" || loop_var.find("kt") != std::string::npos);

  if (is_k_loop) {
    // Emit scaffolding
    EmitLine("for (...) {");
    VisitStmt(op->body);  // ❌ Emits raw IR
    EmitLine("}");
  }
}
```

**What to Change:**

```cpp
Stmt VisitStmt_(const ForNode* op) override {
  // Check if body has tt.matmul_k_loop annotation
  if (auto* attr = op->body.as<AttrStmtNode>()) {
    if (attr->attr_key == "tt.matmul_k_loop") {
      EmitMatmulKLoop(op, attr);
      return;  // Don't visit body!
    }
    else if (attr->attr_key == "tt.elementwise_op") {
      EmitElementwiseOp(op, attr);
      return;  // Don't visit body!
    }
  }

  // Fallback: visit normally (for non-annotated loops)
  return StmtMutator::VisitStmt_(op);
}

void EmitMatmulKLoop(const ForNode* loop, const AttrStmtNode* attr) {
  std::string loop_var = loop->loop_var->name_hint;

  // Emit DST lifecycle (if not already emitted)
  if (!dst_acquired_) {
    EmitLine("// Acquire tile registers for matmul accumulation");
    EmitTileRegsAcquire();
  }

  if (!matmul_init_emitted_) {
    EmitLine("// Initialize matmul (once before all loops)");
    EmitLine("mm_init(cb_in0, cb_in1, cb_out0);");
    matmul_init_emitted_ = true;
  }

  // Emit K-loop header
  EmitLine("// K-loop: C[m,n] += sum(A[m,k] * B[k,n] for k in Kt)");
  EmitLine("for (uint32_t " + loop_var + " = 0; " +
           loop_var + " < Kt; ++" + loop_var + ") {");
  indent_++;

  // ✅ Emit intrinsics instead of visiting body
  EmitLine("// Wait for input tiles in circular buffers");
  EmitLine("cb_wait_front(cb_in0, 1);");
  EmitLine("cb_wait_front(cb_in1, 1);");
  EmitLine("");
  EmitLine("// Perform tile matmul: accumulate if k > 0");
  EmitLine("bool accumulate = (" + loop_var + " > 0);");
  EmitLine("matmul_tiles(cb_in0, cb_in1, 0, 0, 0, accumulate);");
  EmitLine("");
  EmitLine("// Pop consumed tiles");
  EmitLine("cb_pop_front(cb_in0, 1);");
  EmitLine("cb_pop_front(cb_in1, 1);");

  indent_--;
  EmitLine("}");

  // After K-loop: commit and pack result
  EmitLine("");
  EmitLine("// Commit tile register computation");
  EmitLine("tile_regs_commit();");
  EmitLine("// Wait for tile register computation to complete");
  EmitLine("tile_regs_wait();");
  EmitLine("// Pack result to output circular buffer");
  EmitLine("cb_reserve_back(cb_out0, 1);");
  EmitLine("pack_tile(0, cb_out0);");
  EmitLine("cb_push_back(cb_out0, 1);");
  EmitLine("// Release tile registers");
  EmitLine("tile_regs_release();");
}

void EmitElementwiseOp(const ForNode* loop, const AttrStmtNode* attr) {
  // Similar implementation for element-wise operations
  // ...
}
```

**Subtasks:**
- [ ] Add annotation checking in VisitStmt_(ForNode*)
- [ ] Implement EmitMatmulKLoop()
- [ ] Implement EmitElementwiseOp()
- [ ] Remove heuristic-based K-loop detection
- [ ] Update tests to verify intrinsic emission

**Estimated Effort:** 1-2 days

---

### Task 3: Add Tests for Annotated IR

**File:** `testing/python/tt/test_tensorize_tt.py` (new)

**Test Cases:**

```python
def test_tensorize_matmul_k_loop():
    """Test that tensorize_tt annotates K-loops correctly."""

    @T.prim_func
    def matmul_manual(
        A: T.Buffer((256, 256), "float16"),
        B: T.Buffer((256, 256), "float16"),
        C: T.Buffer((256, 256), "float16")
    ):
        with T.Kernel(8, 8) as (bx, by):
            for kt in T.serial(8):  # K-loop
                for i, j in T.Parallel(32, 32):
                    C[bx*32+i, by*32+j] += A[bx*32+i, kt*32+j] * B[kt*32+i, by*32+j]

    # Apply TT lowering pipeline through tensorize_tt
    mod = IRModule({"main": matmul_manual})
    mod = apply_tt_defaults(mod)
    # ... (other passes)
    mod = tensorize_tt(mod)

    # Check IR contains annotation
    ir_str = str(mod)
    assert "tt.matmul_k_loop" in ir_str
    assert "tt.input_buffers" in ir_str
    assert "tt.output_buffer" in ir_str


def test_tensorize_elementwise():
    """Test that tensorize_tt annotates element-wise ops correctly."""

    @T.prim_func
    def elementwise_add(
        A: T.Buffer((256, 256), "float16"),
        B: T.Buffer((256, 256), "float16"),
        C: T.Buffer((256, 256), "float16")
    ):
        with T.Kernel(8, 8) as (bx, by):
            for i, j in T.Parallel(32, 32):
                C[bx*32+i, by*32+j] = A[bx*32+i, by*32+j] + B[bx*32+i, by*32+j]

    mod = IRModule({"main": elementwise_add})
    mod = apply_tt_defaults(mod)
    # ... (other passes)
    mod = tensorize_tt(mod)

    # Check IR contains annotation
    ir_str = str(mod)
    assert "tt.elementwise_op" in ir_str


def test_codegen_emits_intrinsics():
    """Test that codegen emits Metalium intrinsics for annotated IR."""

    @T.prim_func
    def matmul_manual(...):
        # Same as above
        pass

    # Full pipeline
    mod = IRModule({"main": matmul_manual})
    artifacts = compile_to_tt(mod)

    compute_kernel = artifacts["compute.cpp"]

    # Verify intrinsics in generated code
    assert "mm_init(cb_in0, cb_in1, cb_out0)" in compute_kernel
    assert "cb_wait_front(cb_in0, 1)" in compute_kernel
    assert "cb_wait_front(cb_in1, 1)" in compute_kernel
    assert "matmul_tiles(cb_in0, cb_in1, 0, 0, 0, accumulate)" in compute_kernel
    assert "cb_pop_front(cb_in0, 1)" in compute_kernel
    assert "cb_pop_front(cb_in1, 1)" in compute_kernel

    # Verify NO raw array operations inside K-loop
    assert "C[((((((((int64)" not in compute_kernel
```

**Subtasks:**
- [ ] Create test file
- [ ] Add test for K-loop annotation
- [ ] Add test for element-wise annotation
- [ ] Add test for intrinsic emission
- [ ] Add test for NO raw array ops in output

**Estimated Effort:** 1 day

---

### Task 4: Update Test Matmul to Use Real Operations

**File:** `testing/python/tt/test_matmul_codegen.py`

**Current Workaround:**
```python
# Uses for kt in T.serial(1) with element-wise add to trigger K-loop detection
```

**What to Change:**
```python
@T.prim_func
def matmul_256x256_real(
    A: T.Buffer((256, 256), "float16"),
    B: T.Buffer((256, 256), "float16"),
    C: T.Buffer((256, 256), "float16")
):
    """Real matmul with accumulation loop."""
    with T.Kernel(T.ceildiv(256, 32), T.ceildiv(256, 32)) as (bx, by):
        # Allocate accumulator
        C_tile = T.alloc_fragment((32, 32), "float16", scope="local")

        # Clear accumulator
        for i, j in T.Parallel(32, 32):
            C_tile[i, j] = T.float16(0)

        # K-loop with accumulation
        for kt in T.serial(T.ceildiv(256, 32)):  # 8 iterations
            for i, j in T.Parallel(32, 32):
                # Load A tile
                a_val = A[bx * 32 + i, kt * 32 + j]
                # Load B tile
                b_val = B[kt * 32 + i, by * 32 + j]
                # Accumulate
                C_tile[i, j] += a_val * b_val

        # Store result
        for i, j in T.Parallel(32, 32):
            C[bx * 32 + i, by * 32 + j] = C_tile[i, j]
```

**Subtasks:**
- [ ] Remove workaround comment
- [ ] Use real K-loop with accumulation
- [ ] Verify pattern detection works
- [ ] Update validation checks

**Estimated Effort:** 0.5 days

---

## Implementation Order

1. **Task 3** (Tests) - Write tests first (TDD approach)
2. **Task 1** (TensorizeTT) - Implement pattern detection
3. **Task 2** (Codegen) - Update codegen to read annotations
4. **Task 4** (Test Update) - Use real matmul operations

## Success Criteria

- [ ] `tensorize_tt` pass annotates K-loops with `tt.matmul_k_loop`
- [ ] `tensorize_tt` pass annotates element-wise ops with `tt.elementwise_op`
- [ ] Compute visitor emits Metalium intrinsics for annotated loops
- [ ] Generated compute kernel has NO raw array operations
- [ ] Generated compute kernel has correct intrinsic calls:
  - `mm_init()`
  - `cb_wait_front()`
  - `matmul_tiles()`
  - `cb_pop_front()`
- [ ] All 95+ tests pass
- [ ] New tests for tensorization pass

## Timeline

**Total Estimated Effort:** 5-6 days

- Day 1: Task 3 (write tests)
- Day 2-3: Task 1 (pattern detection in tensorize_tt)
- Day 4: Task 2 (update codegen)
- Day 5: Task 4 (update test matmul)
- Day 6: Integration and debugging

## Related Documents

- [IR Lowering Analysis](./IR_LOWERING_ANALYSIS.md) - Detailed analysis of GPU vs TT
- [PASS_TABLE.md](./PASS_TABLE.md) - Complete pass reference (60+ passes)
- [Unified Matmul MVP Plan](./UNIFIED_MATMUL_MVP_PLAN.md) - Original MVP plan
