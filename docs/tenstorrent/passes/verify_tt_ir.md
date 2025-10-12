# VerifyTTIR Pass

**Status**: ✅ Complete
**Priority**: MEDIUM
**File**: `src/transform/tenstorrent/verify_tt_ir.cc`

---

## Purpose

Verify that the transformed TIR satisfies all Tenstorrent backend constraints before code generation.

---

## Why Needed

Catch errors early before codegen:
- Invalid grid sizes (must fit hardware)
- Non-tile-aligned dimensions
- Missing required metadata
- Invalid CB configurations
- Resource limits exceeded

---

## Verification Checks

### Check 1: Grid Size Limits

**Constraint**: Grid must fit on available cores
- Grayskull: 12×9 = 108 cores max
- Wormhole: 8×10 = 80 cores max

**Error if**: `grid_x * grid_y > num_available_cores`

### Check 2: Tile Alignment

**Constraint**: All buffer dimensions must be multiples of 32

**Error if**: Any `dim % 32 != 0`

### Check 3: Required Metadata

**Constraint**: All buffers must have shard metadata

**Error if**: Missing `tt.shard` or `tt.schedule` attributes

### Check 4: Circular Buffer Limits

**Constraint**:
- Max 32 CBs per core
- Total CB size < 1MB L1 memory

**Error if**: Too many CBs or excessive memory usage

### Check 5: Data Format Support

**Constraint**: Only supported dtypes (FP16, BF16, FP32, INT32)

**Error if**: Unsupported dtype detected

---

## Implementation

**Pass Structure**:
```cpp
class VerifyTTIR : public IRVisitor {
  void VisitStmt_(const AllocateNode* op) override {
    // Check tile alignment
    // Check CB limits
  }

  void VisitStmt_(const AttrStmtNode* op) override {
    // Verify metadata presence and correctness
  }

  void VisitExpr_(const CallNode* op) override {
    // Verify intrinsic usage
  }
};
```

**Error Handling**:
- Throws `CompileError` with descriptive message
- Indicates which constraint was violated
- Suggests fix when possible

---

## Tests

**File**: `testing/python/tenstorrent/test_verify_tt_ir.py`
**Status**: ✅ 8 tests passing

Tests cover:
- Valid IR (no errors)
- Grid size violations
- Missing metadata
- Non-aligned dimensions
- CB limit violations
- Error message correctness

---

## Dependencies

**Depends On**:
- All previous transform passes (runs last)

**Depended On By**:
- Code generation (assumes verified IR)

---

## Related Files

- `src/transform/tenstorrent/verify_tt_ir.cc` - Implementation
- `tilelang/tenstorrent/passes.py` - Python binding
- `testing/python/tenstorrent/test_verify_tt_ir.py` - Tests

---

## Success Criteria

- [x] Catches all constraint violations
- [x] Clear error messages
- [x] No false positives (valid IR passes)
- [x] All tests passing (8/8)

---

**Last Updated**: 2025-10-09
