# Pattern Detection Refactor Implementation Plan

**Priority:** HIGH
**Created:** 2025-10-08
**Status:** Planning

## Problem Statement

The current codegen detects operation patterns based solely on loop variable names, causing incorrect pattern emission:

```cpp
// Current (INCORRECT):
bool is_k_loop = (loop_var == "k" || loop_var.find("_k") != std::string::npos);

if (is_k_loop) {
    // ALWAYS emits matmul patterns, even for reduction/GEMV!
    EmitLine("mm_init(cb_in0, cb_in1, cb_out0);");
}
```

**Impact:**
- ❌ Reduction operations emit `mm_init()` (should emit `reduce_tiles_init()`)
- ❌ GEMV operations emit `mm_init()` (should emit vector-specific patterns)
- ❌ Element-wise operations in K-loops emit wrong patterns

## Correct Solution Architecture

### Approach: Two-Pass Pattern Detection

**Pass 1: Pattern Analysis**
- Traverse loop body BEFORE emitting code
- Detect operation types: T.gemm(), T.reduce(), element-wise, etc.
- Build pattern metadata

**Pass 2: Code Emission**
- Use pattern metadata to emit correct initialization
- Emit pattern-specific loop bodies
- Emit pattern-specific cleanup

### Pattern Types to Detect

| Pattern | Detection Criteria | Init Code | Compute Code |
|---------|-------------------|-----------|--------------|
| **Matmul** | `T.gemm()` call in loop | `mm_init()` | `matmul_tiles()` |
| **Reduction** | Accumulation of same variable | `reduce_tiles_init()` | `reduce_tiles()` |
| **Element-wise** | Independent tile operations | `binary_op_init_common()` | `add_tiles()`, etc. |
| **GEMV** | Matrix-vector multiply pattern | `gemv_init()` | `gemv_tiles()` |

## Implementation Plan

### Phase 1: Add Pattern Analysis Infrastructure

**Files to Modify:**
- `src/target/tt/codegen_tt_compute_visitor.h`
- `src/target/tt/codegen_tt_compute_visitor.cc`

**Changes:**

1. **Add Pattern Enum** (codegen_tt_compute_visitor.h):
```cpp
enum class ComputePattern {
  UNKNOWN,
  MATMUL,        // T.gemm() - A @ B
  REDUCTION,     // sum(A[i, :]) - accumulate into same variable
  ELEMENTWISE,   // C = A + B - independent operations
  GEMV,          // y = A @ x - matrix-vector multiply
  CUSTOM         // User-defined patterns
};
```

2. **Add Pattern Detector** (codegen_tt_compute_visitor.h):
```cpp
class PatternDetector {
 public:
  // Analyze loop body to detect pattern type
  static ComputePattern DetectPattern(const For* loop);

 private:
  // Helper methods
  static bool HasGemmCall(const Stmt* body);
  static bool HasReductionPattern(const Stmt* body);
  static bool HasElementwisePattern(const Stmt* body);
  static bool HasGemvPattern(const Stmt* body);
};
```

3. **Add Pattern Metadata Storage** (codegen_tt_compute_visitor.h):
```cpp
class TTComputeCodeGenVisitor {
  // ... existing members ...

  // Pattern detection
  std::map<const For*, ComputePattern> loop_patterns_;
  ComputePattern current_pattern_ = ComputePattern::UNKNOWN;
};
```

### Phase 2: Implement Pattern Detection Logic

**File:** `src/target/tt/codegen_tt_compute_visitor.cc`

**Implementation:**

```cpp
ComputePattern PatternDetector::DetectPattern(const For* loop) {
  // 1. Check for T.gemm() call
  if (HasGemmCall(loop->body.get())) {
    return ComputePattern::MATMUL;
  }

  // 2. Check for reduction pattern
  //    - Accumulation: y[i] = y[i] + ...
  //    - Same variable on both sides
  if (HasReductionPattern(loop->body.get())) {
    return ComputePattern::REDUCTION;
  }

  // 3. Check for GEMV pattern
  //    - Matrix load + vector load
  //    - Multiply-accumulate into vector
  if (HasGemvPattern(loop->body.get())) {
    return ComputePattern::GEMV;
  }

  // 4. Default to element-wise if independent operations
  if (HasElementwisePattern(loop->body.get())) {
    return ComputePattern::ELEMENTWISE;
  }

  return ComputePattern::UNKNOWN;
}

bool PatternDetector::HasGemmCall(const Stmt* body) {
  // Look for CallNode with name "tl.gemm" or similar
  class GemmDetector : public StmtExprVisitor {
   public:
    bool found_gemm = false;
    void VisitExpr_(const CallNode* op) override {
      // Check if call is T.gemm()
      if (op->op.as<OpNode>() && op->op.as<OpNode>()->name == "tl.gemm") {
        found_gemm = true;
      }
      StmtExprVisitor::VisitExpr_(op);
    }
  };

  GemmDetector detector;
  detector.VisitStmt(body);
  return detector.found_gemm;
}

bool PatternDetector::HasReductionPattern(const Stmt* body) {
  // Look for pattern: var[i] = var[i] + expr
  // This indicates accumulation (reduction)
  class ReductionDetector : public StmtExprVisitor {
   public:
    bool found_reduction = false;
    void VisitStmt_(const StoreNode* op) override {
      // Check if store target appears in RHS
      if (const LoadNode* load = op->value.as<LoadNode>()) {
        if (load->buffer_var.same_as(op->buffer_var)) {
          found_reduction = true;
        }
      }
      StmtExprVisitor::VisitStmt_(op);
    }
  };

  ReductionDetector detector;
  detector.VisitStmt(body);
  return detector.found_reduction;
}

// Similar for HasElementwisePattern, HasGemvPattern...
```

### Phase 3: Refactor VisitStmt_(const For* op) to Use Pattern Detection

**File:** `src/target/tt/codegen_tt_compute_visitor.cc`

**Changes:**

```cpp
void TTComputeCodeGenVisitor::VisitStmt_(const For* op) {
  // ... existing intrinsic emission code ...

  // === NEW: Pattern detection ===
  ComputePattern pattern = PatternDetector::DetectPattern(op);
  loop_patterns_[op] = pattern;
  current_pattern_ = pattern;

  // Detect K-loop (but don't assume pattern)
  bool is_k_loop = (loop_var == "kt" || loop_var.find("kt") != std::string::npos ||
                    loop_var == "k" || loop_var.find("_k") != std::string::npos);

  // Detect outer tile loop
  bool is_outer_loop = (loop_depth_ == 1);

  // === NEW: Emit pattern-specific initialization ===
  if (is_k_loop) {
    // Acquire tile registers before K-loop (all patterns need this)
    if (!dst_acquired_) {
      EmitLine("// Acquire tile registers for computation");
      EmitTileRegsAcquire();
      EmitLine("");
    }

    // Emit pattern-specific comment and initialization
    switch (pattern) {
      case ComputePattern::MATMUL:
        EmitLine("// K-loop: C[m,n] += sum(A[m,k] * B[k,n] for k in Kt)");
        if (!matmul_init_emitted_) {
          EmitLine("// Initialize matmul (once before all loops)");
          EmitLine("mm_init(cb_in0, cb_in1, cb_out0);");
          EmitLine("");
          matmul_init_emitted_ = true;
        }
        break;

      case ComputePattern::REDUCTION:
        EmitLine("// K-loop: Reduction - accumulate across dimension");
        if (!reduction_init_emitted_) {
          EmitLine("// Initialize reduction (once before all loops)");
          EmitLine("reduce_tiles_init();");
          EmitLine("");
          reduction_init_emitted_ = true;
        }
        break;

      case ComputePattern::GEMV:
        EmitLine("// K-loop: Matrix-Vector multiply - y[m] = sum(A[m,k] * x[k])");
        if (!gemv_init_emitted_) {
          EmitLine("// Initialize GEMV (once before all loops)");
          EmitLine("gemv_init();");
          EmitLine("");
          gemv_init_emitted_ = true;
        }
        break;

      case ComputePattern::ELEMENTWISE:
        EmitLine("// K-loop: Element-wise operations");
        // Element-wise init already handled per-tile
        break;

      default:
        EmitLine("// K-loop: Generic accumulation");
        break;
    }

    k_loop_var_ = loop_var;
  }

  // ... rest of existing code ...
}
```

### Phase 4: Update Pattern-Specific Emission

**Changes to VisitExpr_(const CallNode* op):**

```cpp
void TTComputeCodeGenVisitor::VisitExpr_(const CallNode* op) {
  // Detect operation type
  std::string op_name = GetCallName(op);

  if (op_name == "tl.gemm") {
    // Emit matmul pattern
    EmitGemmOperation(op);
  } else if (op_name == "tl.reduce") {
    // Emit reduction pattern
    EmitReduceOperation(op);
  } else if (IsElementwiseOp(op_name)) {
    // Emit element-wise pattern
    EmitElementwiseOperation(op);
  }
  // ... etc
}
```

### Phase 5: Add New Helper Methods

```cpp
// In TTComputeCodeGenVisitor class

void EmitGemmOperation(const CallNode* op) {
  EmitLine("// Wait for input tiles from reader");
  EmitLine("cb_wait_front(cb_in0, 1);");
  EmitLine("cb_wait_front(cb_in1, 1);");
  EmitLine("");

  EmitLine("// Matmul: process tile (accumulation automatic)");
  EmitLine("matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);");
  EmitLine("");

  EmitLine("// Release input tiles");
  EmitLine("cb_pop_front(cb_in0, 1);");
  EmitLine("cb_pop_front(cb_in1, 1);");
  EmitLine("");
}

void EmitReduceOperation(const CallNode* op) {
  EmitLine("// Wait for input tile from reader");
  EmitLine("cb_wait_front(cb_in0, 1);");
  EmitLine("");

  bool accumulate = (current_k_iter_ > 0);
  std::string acc_flag = accumulate ? "true" : "false";

  EmitLine("// Reduce: accumulate tile");
  EmitLine("reduce_tiles(cb_in0, 0, " + acc_flag + ");");
  EmitLine("");

  EmitLine("// Release input tile");
  EmitLine("cb_pop_front(cb_in0, 1);");
  EmitLine("");
}

// Similar for other patterns...
```

## Testing Strategy

### Unit Tests

**File:** `testing/python/tt/test_pattern_detection.py`

```python
def test_pattern_detection_matmul():
    """Test that T.gemm() is detected as MATMUL pattern."""
    # Create module with T.gemm()
    # Apply codegen
    # Assert mm_init() present, no reduce_tiles_init()

def test_pattern_detection_reduction():
    """Test that reduction loop is detected correctly."""
    # Create module with accumulation pattern
    # Apply codegen
    # Assert reduce_tiles_init() present, no mm_init()

def test_pattern_detection_gemv():
    """Test that matrix-vector multiply is detected."""
    # Create module with GEMV pattern
    # Apply codegen
    # Assert gemv_init() present

def test_pattern_detection_elementwise():
    """Test that element-wise ops are detected."""
    # Create module with element-wise operations
    # Apply codegen
    # Assert binary_op_init_common() present
```

### Integration Tests

Update existing examples to verify correct patterns:
- `example_simple_gemm_tt.py` - Should emit matmul pattern only
- `example_reduction_sum_tt.py` - Should emit reduction pattern only (NO matmul)
- `example_gemv_tt.py` - Should emit GEMV pattern only (NO matmul)

## Implementation Timeline

**Week 1: Pattern Analysis Infrastructure**
- Day 1-2: Add pattern enums, detector class structure
- Day 3-4: Implement HasGemmCall detection
- Day 5: Unit tests for GEMM detection

**Week 2: Additional Pattern Detection**
- Day 1-2: Implement HasReductionPattern
- Day 3-4: Implement HasGemvPattern, HasElementwisePattern
- Day 5: Unit tests for all patterns

**Week 3: Integration and Code Emission**
- Day 1-2: Refactor VisitStmt_(const For*) to use pattern detection
- Day 3-4: Update pattern-specific emission methods
- Day 5: Integration tests with examples

**Week 4: Validation and Cleanup**
- Day 1-2: Run all phase examples, verify correct patterns
- Day 3: Update documentation
- Day 4: Code review and fixes
- Day 5: PR creation and merge

## Success Criteria

- ✅ Phase 1.3 (GEMM): Still emits mm_init() and matmul_tiles()
- ✅ Phase 2.3 (Reduction): Emits reduce_tiles_init() INSTEAD of mm_init()
- ✅ Phase 3.1 (GEMV): Emits gemv_init() INSTEAD of mm_init()
- ✅ All 95 tests still pass
- ✅ No pattern mixing in generated code
- ✅ Pattern detection extensible for future operations

## Future Enhancements

After basic pattern detection is working:
1. **Multi-pattern loops**: Support loops with multiple operation types
2. **Pattern optimization**: Fuse compatible patterns
3. **Custom patterns**: User-defined pattern registration
4. **Pattern verification**: Static analysis to catch pattern violations

## References

- Current buggy code: `src/target/tt/codegen_tt_compute_visitor.cc:209-244`
- Phase status: `docs/tenstorrent/PHASE_STATUS_SUMMARY.md`
- Example outputs: `examples/tenstorrent/example_*_tt.py`
