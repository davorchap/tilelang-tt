# TensorizeTT Pass

**Status**: 🟡 Partial
**Priority**: HIGH
**File**: `src/transform/tt/tensorize_tt.cc`

---

## Purpose

Detect manual matmul loops and element-wise operations in TIR and annotate them with Metalium intrinsic metadata, enabling the code generator to emit efficient tile-based operations.

---

## Why Needed

**Problem**: Codegen currently uses heuristics (like variable name "kt") to detect patterns, which is:
- Fragile (fails if variables renamed)
- Incomplete (misses complex patterns)
- Violates separation of concerns (codegen should be "dumb")

**Solution**: Transform pass detects patterns and adds IR annotations that codegen simply reads.

---

## Current Implementation

**What Works**:
- Detects `T.gemm()` intrinsic calls
- Annotates with `tt.matmul_intrinsic` attribute

**What's Missing**:
- Manual matmul loop detection (3 nested loops with accumulation)
- Element-wise operation detection (tile grid operations)
- Input/output buffer annotations
- K-loop annotations for matmul

---

## Specification

### Pattern 1: K-Loop Matmul

**Input IR** (manual matmul):
```python
for tile_idx in range(num_output_tiles):
    for kt in range(Kt):  # K-loop
        for i, j in T.grid(32, 32):
            C[tile_m*32+i, tile_n*32+j] += A[tile_m*32+i, kt*32+j] * B[kt*32+i, tile_n*32+j]
```

**Output IR** (annotated):
```python
AttrStmt("tt.matmul_k_loop", matmul_id=0):
  AttrStmt("tt.input_buffers", [A, B]):
    AttrStmt("tt.output_buffer", C):
      for kt in range(Kt):
        # loop body unchanged
```

### Pattern 2: Element-Wise Operations

**Input IR**:
```python
for tile_idx in range(num_tiles):
    for i, j in T.grid(32, 32):
        C[bx*32+i, by*32+j] = A[bx*32+i, by*32+j] + B[bx*32+i, by*32+j]
```

**Output IR** (annotated):
```python
AttrStmt("tt.elementwise_op", op_id=0):
  AttrStmt("tt.op_type", "add"):
    AttrStmt("tt.input_buffers", [A, B]):
      AttrStmt("tt.output_buffer", C):
        # loop body unchanged
```

---

## Implementation Tasks

### Task 1.1: Matmul Pattern Matcher

**File**: `src/transform/tt/tensorize_tt.cc`

**Add**:
```cpp
class MatmulPatternMatcher {
public:
  bool Match(const Stmt& body);
  Var GetBufferA();
  Var GetBufferB();
  Var GetBufferC();
  bool IsAccumulation();  // Check for += pattern
};
```

**Status**: ❌ Not Implemented

### Task 1.2: Element-Wise Pattern Matcher

**Add**:
```cpp
class ElementwisePatternMatcher {
public:
  bool Match(const Stmt& body);
  std::string GetOpType();  // "add", "mul", etc.
  Array<Var> GetInputBuffers();
  Var GetOutputBuffer();
};
```

**Status**: ❌ Not Implemented

### Task 1.3: Annotation Logic

**Add to** `TensorizeTTMutator::VisitStmt_(const ForNode* op)`:
- Detect K-loop via pattern matching (not variable name)
- Detect element-wise via `T.grid(32, 32)` pattern
- Add nested `AttrStmt` nodes with buffer info

**Status**: ❌ Not Implemented

---

## Tests

**Test File**: `testing/python/tt/test_tensorize_tt.py`

**Current**: 8 tests (basic intrinsic detection)

**Needed**:
- Test manual matmul loop detection
- Test element-wise detection
- Test annotation structure
- Test multiple patterns in same function

**Status**: ⏳ Pending implementation

---

## Dependencies

**Depends On**:
- None (standalone transform)

**Depended On By**:
- `codegen_tt_compute_visitor.cc` - Reads annotations to emit intrinsics
- All downstream code generation

---

## Related Files

- `src/transform/tt/tensorize_tt.cc` - Implementation
- `src/target/tt/codegen_tt_compute_visitor.cc` - Consumes annotations
- `testing/python/tt/test_tensorize_tt.py` - Tests

---

## Success Criteria

- [ ] Detects manual matmul loops (no `T.gemm()` call needed)
- [ ] Detects element-wise operations
- [ ] Generates correct nested annotations
- [ ] Codegen emits intrinsics based on annotations (not heuristics)
- [ ] All existing tests pass
- [ ] New pattern detection tests pass

---

## Timeline

**Estimated Effort**: 2-3 days
- Day 1: Pattern matchers
- Day 2: Annotation logic
- Day 3: Tests and integration

---

**Last Updated**: 2025-10-09
