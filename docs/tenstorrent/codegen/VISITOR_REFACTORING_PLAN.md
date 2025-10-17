# TileLang Tenstorrent CodeGen Visitor Refactoring Plan

**Date:** 2025-10-17
**Status:** Planning Phase
**Priority:** HIGH - Architecture Cleanup

---

## Executive Summary

The current TileLang Tenstorrent CodeGen implementation is a **hybrid of hardcoded templates and IR visitors**. This violates the intended architecture where **all code generation should be driven by IR traversal**. This plan outlines a 3-phase approach to refactor to a pure visitor-based architecture.

**Root Cause:** The v5 pipeline may not be producing complete IR, causing codegen to fall back to hardcoded templates.

**Solution:** Fix IR completeness + Remove all templates + Fail loudly when IR is incomplete.

---

## 1. Current State Analysis

### 1.1 Architecture Problems

#### Problem 1: Hybrid Template/Visitor Code Generation

**File:** `tilelang/tenstorrent/codegen/kernel_generators.py`

**Evidence:**
```python
# Lines 221-249: ReaderKernelGenerator fallback template
if not reader_body or is_empty_ir(reader_body):
    # FALLBACK: Generate template-based reader
    code = self._generate_template_reader()

# Lines 399-419: ComputeKernelGenerator fallback template
if not has_compute_ops(compute_body):
    # FALLBACK: Generate template-based compute
    code = self._generate_template_compute()
```

**Problem:** When IR is incomplete, codegen silently falls back to hardcoded templates instead of failing loudly.

#### Problem 2: Scattered Intrinsic Mappings

**Evidence:**
- NOC intrinsics defined in 3+ places:
  - `kernel_generators.py`: `noc_async_read_tile()`, `noc_async_write_tile()`
  - `intrinsics.py`: CB intrinsics like `cb_reserve_back()`
  - `compute_visitor.py`: Compute intrinsics like `mm_init()`

**Problem:** No single source of truth for IR�C++ intrinsic mappings. Duplication causes maintenance burden.

#### Problem 3: Missing IR Validation

**Evidence:**
- No pass validates that split kernels have proper IR bodies
- No checks that reader has NOC ops
- No checks that compute has tile math ops
- No checks that writer has NOC ops

**Problem:** Incomplete IR flows to codegen, which compensates with templates.

### 1.2 File Inventory

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `tilelang/tenstorrent/codegen/kernel_generators.py` | 650 | Main codegen orchestration | **NEEDS REFACTOR** |
| `tilelang/tenstorrent/codegen/intrinsics.py` | 200 | Intrinsic mappings (partial) | **NEEDS CONSOLIDATION** |
| `tilelang/tenstorrent/codegen/visitor_base.py` | - | **MISSING** | **TO CREATE** |
| `src/target/tenstorrent/codegen_tenstorrent.cc` | 800 | C++ codegen | **REVIEW NEEDED** |
| `src/target/tenstorrent/reader_visitor.cc` | - | **MISSING** | **TO CREATE** |
| `src/target/tenstorrent/compute_visitor.cc` | - | **MISSING** | **TO CREATE** |
| `src/target/tenstorrent/writer_visitor.cc` | - | **MISSING** | **TO CREATE** |

### 1.3 Expected vs Actual Flow

**Expected Flow (Correct Architecture):**
```
TileLang DSL
  �
v5 Pipeline (14 passes) � Complete IR with all ops
  �
IR Validation � Verify IR completeness
  �
Pure IR Visitors � Traverse IR, emit C++
  �
Generated Code (reader.cpp, compute.cpp, writer.cpp)
```

**Actual Flow (Current Broken State):**
```
TileLang DSL
  �
v5 Pipeline (14 passes) � Incomplete IR?
  �
NO Validation � IR flows unchecked
  �
Hybrid Visitors � Traverse IR OR fallback to templates
  �
Generated Code (mix of IR-driven + templates)
```

---

## 2. Target Architecture

### 2.1 Pure Visitor Design

**Principle:** Code generation is **100% driven by IR traversal**. No templates, no fallbacks, no hardcoded patterns.

```python
class TTCodegenVisitor(IRVisitor):
    """Base visitor for all TT codegen"""

    def __init__(self, primfunc: tir.PrimFunc):
        self.func = primfunc
        self.validate_ir()  # Fail if IR incomplete

    def validate_ir(self):
        """Ensure IR is complete before codegen"""
        if not self.func.body:
            raise ValueError("IR body is empty - v5 pipeline failed")
        # More validation...

    def visit_call(self, call: tir.Call):
        """Map IR calls to C++ intrinsics"""
        intrinsic = INTRINSIC_REGISTRY.get(call.op.name)
        if not intrinsic:
            raise ValueError(f"Unknown intrinsic: {call.op.name}")
        return intrinsic.emit(call.args)
```

### 2.2 Single Intrinsic Registry

**File:** `tilelang/tenstorrent/codegen/intrinsics.py`

```python
class Intrinsic:
    """Single intrinsic definition"""
    def __init__(self, name: str, cpp_template: str):
        self.name = name
        self.cpp_template = cpp_template

    def emit(self, args: List) -> str:
        return self.cpp_template.format(*args)

# Single registry
INTRINSIC_REGISTRY = {
    # NOC ops
    "tir.noc_async_read_tile": Intrinsic(
        "noc_async_read_tile",
        "noc_async_read_tile({0}, {1}, {2})"
    ),
    "tir.noc_async_write_tile": Intrinsic(
        "noc_async_write_tile",
        "noc_async_write_tile({0}, {1}, {2})"
    ),

    # CB ops
    "tir.cb_reserve_back": Intrinsic(
        "cb_reserve_back",
        "cb_reserve_back({0}, {1})"
    ),
    "tir.cb_push_back": Intrinsic(
        "cb_push_back",
        "cb_push_back({0}, {1})"
    ),

    # Compute ops
    "tir.mm_init": Intrinsic("mm_init", "mm_init()"),
    "tir.matmul_tiles": Intrinsic(
        "matmul_tiles",
        "matmul_tiles({0}, {1}, {2}, {3})"
    ),
}
```

### 2.3 Specialized Visitors

Each kernel type has a specialized visitor:

```python
class ReaderVisitor(TTCodegenVisitor):
    """Generate reader.cpp from reader kernel IR"""
    def generate(self) -> str:
        # Pure IR traversal - no templates
        return self.visit(self.func.body)

class ComputeVisitor(TTCodegenVisitor):
    """Generate compute.cpp from compute kernel IR"""
    def generate(self) -> str:
        return self.visit(self.func.body)

class WriterVisitor(TTCodegenVisitor):
    """Generate writer.cpp from writer kernel IR"""
    def generate(self) -> str:
        return self.visit(self.func.body)
```

### 2.4 Fail Loudly on Incomplete IR

**No silent fallbacks:**

```python
def codegen_tenstorrent(mod: IRModule) -> Dict[str, str]:
    """Entry point for TT codegen"""

    # Validate IR completeness BEFORE codegen
    validate_split_kernels(mod)  # NEW PASS

    # Extract kernels
    reader = get_kernel(mod, "reader")
    compute = get_kernel(mod, "compute")
    writer = get_kernel(mod, "writer")

    # Each visitor validates IR and fails if incomplete
    reader_code = ReaderVisitor(reader).generate()
    compute_code = ComputeVisitor(compute).generate()
    writer_code = WriterVisitor(writer).generate()

    return {
        "reader.cpp": reader_code,
        "compute.cpp": compute_code,
        "writer.cpp": writer_code,
    }
```

---

## 3. Implementation Plan

### Phase 1: Fix v5 Pipeline Integration (Week 1 - CRITICAL)

**Goal:** Ensure v5 pipeline produces complete IR before codegen.
**Status:** ✅ COMPLETED (2025-10-17)
**Findings:** See [PHASE_1_FINDINGS.md](../planning/PHASE_1_FINDINGS.md)

#### Task 1.1: Add IR Validation Pass [✅ COMPLETED]

**File:** `tilelang/tenstorrent/passes/validate_split_kernels.py` (NEW)

**Implementation:**
```python
@tvm.tir.transform.prim_func_pass(opt_level=0)
def ValidateSplitKernels():
    """Validate that split kernels have complete IR"""

    def transform(func, mod, ctx):
        kernel_type = func.attrs.get("kernel_type")

        if kernel_type == "reader":
            validate_reader_ir(func)
        elif kernel_type == "compute":
            validate_compute_ir(func)
        elif kernel_type == "writer":
            validate_writer_ir(func)

        return func

    return transform

def validate_reader_ir(func):
    """Ensure reader has NOC ops"""
    has_noc_read = find_intrinsic(func, "tir.noc_async_read_tile")
    if not has_noc_read:
        raise ValueError("Reader IR missing NOC read operations")

def validate_compute_ir(func):
    """Ensure compute has tile math ops"""
    has_matmul = find_intrinsic(func, "tir.matmul_tiles")
    if not has_matmul:
        raise ValueError("Compute IR missing matmul operations")

def validate_writer_ir(func):
    """Ensure writer has NOC ops"""
    has_noc_write = find_intrinsic(func, "tir.noc_async_write_tile")
    if not has_noc_write:
        raise ValueError("Writer IR missing NOC write operations")
```

**Integration:** Add to v5 pipeline after Stage E:
```python
# tilelang/tenstorrent/transform.py
seq = tvm.transform.Sequential([
    # ... Stage A, B, C, D, E passes ...
    ValidateSplitKernels(),  # NEW
    # ... Common optimizations ...
])
```

**Effort:** 4 hours

#### Task 1.2: Debug Empty IR Bodies

**Goal:** Investigate why IR bodies may be empty after Stage D (split_device_kernel).

**Steps:**
1. Add debug logging to `split_device_kernel.py`
2. Dump IR after each pass in Stage D
3. Identify which pass fails to produce complete IR
4. Fix the root cause

**Effort:** 8 hours

#### Task 1.3: Add IR Dump Utility

**File:** `tilelang/tenstorrent/utils/debug_ir.py` (NEW)

```python
def dump_ir_after_each_pass(mod: IRModule, output_dir: str):
    """Dump IR after each pass for debugging"""
    # Implementation for debugging
    pass
```

**Effort:** 2 hours

---

### Phase 2: Refactor Python CodeGen (Week 2)

**Goal:** Replace all template code with pure IR visitors.

#### Task 2.1: Create Visitor Base Class

**File:** `tilelang/tenstorrent/codegen/visitor_base.py` (NEW)

**Implementation:**
```python
class TTCodegenVisitor(tir.StmtExprVisitor):
    """Base class for all TT codegen visitors"""

    def __init__(self, primfunc: tir.PrimFunc):
        self.func = primfunc
        self.code_lines = []
        self.indent_level = 0
        self.validate_ir()

    def validate_ir(self):
        """Ensure IR is complete - fail loudly if not"""
        if not self.func.body:
            raise ValueError(f"IR body is empty for {self.func.attrs.get('global_symbol')}")

    def emit(self, line: str):
        """Emit a line of C++ code"""
        indent = "    " * self.indent_level
        self.code_lines.append(indent + line)

    def generate(self) -> str:
        """Generate C++ code from IR"""
        self.visit(self.func.body)
        return "\n".join(self.code_lines)

    def visit_call(self, call: tir.Call):
        """Map IR intrinsic calls to C++ code"""
        intrinsic = INTRINSIC_REGISTRY.get(call.op.name)
        if not intrinsic:
            raise ValueError(f"Unknown intrinsic: {call.op.name}")
        self.emit(intrinsic.emit(call.args))
```

**Effort:** 4 hours

#### Task 2.2: Consolidate Intrinsic Mappings

**File:** `tilelang/tenstorrent/codegen/intrinsics.py`

**Goal:** Single source of truth for all IR�C++ mappings.

**Implementation:** See Section 2.2 above.

**Effort:** 2 hours

#### Task 2.3: Remove Template Fallback Code

**Files:**
- `kernel_generators.py` lines 221-249 (reader template)
- `kernel_generators.py` lines 399-419 (compute template)

**Action:** Delete all template fallback code.

**Effort:** 1 hour

#### Task 2.4: Refactor ReaderVisitor

**File:** `kernel_generators.py` � `tilelang/tenstorrent/codegen/reader_visitor.py`

**Goal:** Pure IR-driven reader generation.

**Implementation:**
```python
class ReaderVisitor(TTCodegenVisitor):
    """Generate reader.cpp from reader kernel IR"""

    def generate(self) -> str:
        self.emit("#include <cstdint>")
        self.emit('#include "dataflow_api.h"')
        self.emit("")
        self.emit("void MAIN {")
        self.indent_level += 1

        # IR traversal - no templates
        self.visit(self.func.body)

        self.indent_level -= 1
        self.emit("}")

        return "\n".join(self.code_lines)
```

**Effort:** 4 hours

#### Task 2.5: Refactor ComputeVisitor

**File:** `kernel_generators.py` � `tilelang/tenstorrent/codegen/compute_visitor.py`

**Goal:** Pure IR-driven compute generation.

**Effort:** 6 hours

#### Task 2.6: Refactor WriterVisitor

**File:** `kernel_generators.py` � `tilelang/tenstorrent/codegen/writer_visitor.py`

**Goal:** Pure IR-driven writer generation.

**Effort:** 4 hours

#### Task 2.7: Update Tests

**File:** `testing/python/tenstorrent/test_codegen_pipeline.py`

**Changes:**
- Test for IR-driven code markers
- Test that templates are never used
- Test that incomplete IR fails loudly

**Effort:** 2 hours

#### Task 2.8: Add Error Handling

**Goal:** All visitors fail loudly on incomplete IR.

**Effort:** 2 hours

---

### Phase 3: Unify C++ Visitors (Optional - Future)

**Goal:** Consider C++ visitor implementation for performance.

**Evaluation Criteria:**
- Is Python visitor performance acceptable?
- Would C++ implementation improve build times?
- Is the added complexity worth it?

**Decision:** Defer to future sprint after Phase 2 complete.

---

## 4. Task Breakdown

### 4.1 High-Priority Tasks (Week 1)

| Task | File | Description | Effort |
|------|------|-------------|--------|
| 1. **Add IR validation** | `tilelang/tenstorrent/passes/validate_split_kernels.py` | Create pass to validate 3 split kernels have proper IR | 4h |
| 2. **Fix empty IR bodies** | Debug v5 Stage D passes | Investigate why IR bodies are empty | 8h |
| 3. **Create visitor base** | `tilelang/tenstorrent/codegen/visitor_base.py` | Pure visitor base class | 4h |
| 4. **Consolidate intrinsics** | `tilelang/tenstorrent/codegen/intrinsics.py` | Single registry for all mappings | 2h |
| 5. **Remove template code** | `kernel_generators.py` lines 221-249, 399-419 | Delete fallback templates | 1h |

**Total Week 1:** 19 hours

### 4.2 Refactoring Tasks (Week 2)

| Task | File | Description | Effort |
|------|------|-------------|--------|
| 6. **Refactor ReaderVisitor** | `reader_visitor.py` | Pure IR-driven reader generation | 4h |
| 7. **Refactor ComputeVisitor** | `compute_visitor.py` | Pure IR-driven compute generation | 6h |
| 8. **Refactor WriterVisitor** | `writer_visitor.py` | Pure IR-driven writer generation | 4h |
| 9. **Update tests** | `test_codegen_pipeline.py` | Test for IR-driven markers | 2h |
| 10. **Add error handling** | All visitors | Fail loudly on incomplete IR | 2h |

**Total Week 2:** 18 hours

### 4.3 Documentation Tasks

| Task | File | Description | Effort |
|------|------|-------------|--------|
| 11. **Visitor guide** | `docs/tenstorrent/codegen/VISITOR_GUIDE.md` | How to extend visitors | 2h |
| 12. **Intrinsic reference** | `docs/tenstorrent/codegen/INTRINSICS.md` | Complete IR�C++ mapping | 2h |
| 13. **Migration notes** | `docs/tenstorrent/codegen/MIGRATION.md` | Record changes made | 1h |

**Total Documentation:** 5 hours

**GRAND TOTAL:** 42 hours (~1 sprint)

---

## 5. Testing Strategy

### 5.1 Unit Tests for Visitors

```python
def test_visitor_requires_complete_ir():
    """Visitor should fail on empty IR"""
    empty_func = create_empty_primfunc()
    visitor = TTCodegenVisitor(empty_func)

    with pytest.raises(ValueError, match="IR body is empty"):
        visitor.generate()

def test_visitor_maps_intrinsics():
    """Visitor should map IR intrinsics to C++ calls"""
    # Create IR with cb_reserve_back call
    func = create_func_with_cb_ops()
    visitor = ReaderVisitor(func)
    code = visitor.generate()

    assert "cb_reserve_back(cb_in0, 1)" in code
    assert "// Template" not in code  # No templates!

def test_no_template_fallback():
    """Codegen should never use templates"""
    kernel = create_matmul_kernel()
    mod = run_v5_pipeline(kernel)
    artifacts = codegen(mod)

    for name, code in artifacts.items():
        assert "// Template" not in code
        assert "// Fallback" not in code
```

### 5.2 Integration Tests

```python
def test_e2e_matmul_codegen():
    """Full pipeline from TileLang to C++ code"""
    # Start with TileLang kernel
    kernel = create_matmul_kernel()

    # Run v5 pipeline
    mod = run_v5_pipeline(kernel)

    # Verify IR is complete
    assert has_three_kernels(mod)
    assert reader_has_noc_ops(mod)
    assert compute_has_matmul(mod)
    assert writer_has_noc_ops(mod)

    # Generate code
    artifacts = codegen(mod)

    # Verify no templates
    for name, code in artifacts.items():
        assert "// Template" not in code
        assert "// IR-Driven" in code

def test_incomplete_ir_fails():
    """Incomplete IR should fail validation"""
    # Create IR with missing reader body
    mod = create_incomplete_ir()

    with pytest.raises(ValueError, match="Reader IR missing NOC"):
        run_v5_pipeline(mod)
```

### 5.3 Validation Checklist

- [ ] All generated code comes from IR (no hardcoded patterns)
- [ ] Empty IR causes explicit failure (no silent fallbacks)
- [ ] Intrinsic mapping is centralized (single source of truth)
- [ ] Metadata flows through IR attributes only
- [ ] C++ and Python visitors share same architecture
- [ ] Tests verify IR completeness before codegen
- [ ] 120 tests still passing (no regressions)

---

## 6. Risk Mitigation

### Risk 1: v5 Passes Not Producing Complete IR
**Likelihood:** HIGH
**Impact:** CRITICAL
**Mitigation:**
- Add comprehensive IR validation after each stage
- Create test fixtures with known-good IR
- Document expected IR structure for each kernel type
- Add debug utilities to dump IR after each pass

### Risk 2: Breaking Existing Tests
**Likelihood:** MEDIUM
**Impact:** HIGH
**Mitigation**:
- Run full test suite after each change
- Keep template code in separate branch initially
- Add feature flag to toggle visitor vs template
- Create rollback plan if refactor fails

### Risk 3: Performance Regression
**Likelihood:** LOW
**Impact:** MEDIUM
**Mitigation**:
- Profile visitor performance vs templates
- Consider C++ visitor implementation if needed
- Cache intrinsic lookups
- Benchmark code generation time

### Risk 4: Incomplete Intrinsic Coverage
**Likelihood:** MEDIUM
**Impact:** HIGH
**Mitigation**:
- Audit all IR intrinsics currently used
- Build comprehensive intrinsic registry
- Add fallback error messages for unknown intrinsics
- Document all IR�C++ mappings

---

## 7. Success Criteria

The refactoring is complete when:

1.  **No hardcoded templates** remain in codegen
2.  **All code generation** is driven by IR traversal
3.  **Failing loudly** on incomplete IR (no silent fallbacks)
4.  **Single intrinsic registry** (no duplication)
5.  **120 tests still passing** (no regressions)
6.  **Clear separation** between transforms and codegen
7.  **Documentation** updated to reflect new architecture
8.  **Code review** passes with no architectural concerns

---

## 8. References

### Architecture Documents
- [TT_ARCHITECTURE.md](../architecture/TT_ARCHITECTURE.md) - Backend architecture
- [v5_pipeline.md](../architecture/v5_pipeline.md) - v5 pipeline reference

### Implementation Files
- `tilelang/tenstorrent/codegen/kernel_generators.py` - Current codegen (to refactor)
- `tilelang/tenstorrent/passes/split_device_kernel.py` - Stage D1 pass
- `src/target/tenstorrent/codegen_tenstorrent.cc` - C++ codegen entry

### Test Files
- `testing/python/tenstorrent/test_codegen_pipeline.py` - Codegen tests
- `testing/python/tenstorrent/test_v5_passes_integration.py` - Pipeline tests

---

## 9. Next Steps

### Immediate Actions (Today)

1. **Create this document**:  Done
2. **Start Task 1**: Add IR validation pass
3. **Review v5 passes**: Understand IR completeness

### This Week

- Complete Phase 1 (Fix v5 Pipeline Integration)
- Begin Phase 2 (Refactor Python CodeGen)

### Next Week

- Complete Phase 2
- Evaluate Phase 3 (C++ unification)
- Update documentation

---

## 10. Changelog

### 2025-10-17 - Phase 1 Complete
- ✅ Completed Phase 1: Fix v5 Pipeline Integration
- ✅ Created IR validation pass (validate_split_kernels.py)
- ✅ Created IR dump utility (debug_ir.py)
- ✅ Integrated validation into v5 pipeline
- ✅ Identified root cause: Stage C passes not generating expected protocol operations
- See [PHASE_1_FINDINGS.md](../planning/PHASE_1_FINDINGS.md) for detailed findings

### 2025-10-17 - Initial Plan
- Initial plan created based on system-architect analysis
- Identified hybrid template/visitor architecture as root cause
- Proposed 3-phase refactoring approach
- Estimated 42 hours total effort

---

**Status:** Phase 1 Complete, Phase 2 Ready to Start
**Next Review:** After Stage C fixes
**Owner:** TileLang Tenstorrent Team
