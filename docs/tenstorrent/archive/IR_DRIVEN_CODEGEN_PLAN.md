# IR-Driven Codegen Migration: Implementation Plan

**Version**: 1.0
**Date**: 2025-10-07
**Status**: Implementation Plan
**Estimated Effort**: 10-13 weeks (compressed to tasks)

---

## Executive Summary

This document provides a detailed implementation plan for migrating from **template-based codegen** to **IR-driven codegen** for the Tenstorrent backend.

**Goal**: Enable the backend to support arbitrary TileLang kernels by walking the actual TIR body structure instead of emitting hardcoded patterns.

**Approach**: Implement `StmtExprVisitor` pattern to analyze IR and generate code dynamically.

---

## Current vs Target Architecture

### Current: Template-Based

```
EmitTTComputeKernel(func):
  1. Read func->attrs (grid_x, grid_y, etc.)
  2. Extract dimensions
  3. Emit HARDCODED matmul pattern:
     - Fixed loop structure
     - Hardcoded CB operations
     - Matmul-specific logic
  4. Return fixed string
```

**Limitations**:
- Only works for matmul
- Ignores `func->body` entirely
- Cannot handle arbitrary kernels

### Target: IR-Driven

```
EmitTTComputeKernel(func):
  1. Create TTCodegenVisitor
  2. Walk func->body using visitor pattern:
     - Visit ForNode → generate loop
     - Visit AttrStmtNode → detect operations (matmul, copy, etc.)
     - Visit AllocateNode → handle buffers
     - Visit other nodes → translate to TT code
  3. Return generated string
```

**Benefits**:
- Works for any TileLang kernel
- Uses actual IR structure
- Extensible to new operations
- Leverages WS3 transform annotations

---

## Task Breakdown

### Task 1: Create IR Visitor Base Class
**File**: `src/target/tt/codegen_tt_visitor.h`
**Estimated**: 1-2 days
**Dependencies**: None

**Specification**:

Create a base visitor class that extends `StmtExprVisitor` and provides:

1. **Base Infrastructure**:
   ```cpp
   class TTCodegenVisitor : public StmtExprVisitor {
   public:
     TTCodegenVisitor(const PrimFunc& func);
     std::string GetCode() const;

   protected:
     std::ostringstream code_;
     const PrimFunc& func_;
     int indent_level_;

     void Indent();
     void Emit(const std::string& line);
     void EmitLine(const std::string& line);
   };
   ```

2. **Visitor Methods to Implement**:
   - `VisitStmt_(const ForNode* op)` - Loop handling
   - `VisitStmt_(const AttrStmtNode* op)` - Operation detection
   - `VisitStmt_(const AllocateNode* op)` - Buffer allocation
   - `VisitStmt_(const DeclBufferNode* op)` - Buffer declarations
   - `VisitStmt_(const BufferStoreNode* op)` - Store operations
   - `VisitExpr_(const BufferLoadNode* op)` - Load operations
   - `VisitStmt_(const SeqStmtNode* op)` - Statement sequences
   - `VisitStmt_(const IfThenElseNode* op)` - Conditional handling

3. **Helper Methods**:
   ```cpp
   std::string GetVarName(const Var& var);
   std::string GetBufferName(const Buffer& buf);
   std::string EmitExpr(const PrimExpr& expr);
   bool IsCircularBuffer(const Buffer& buf);
   uint32_t GetCBId(const Buffer& buf);
   ```

**Test Plan**:
- Unit tests for base infrastructure
- Test indent/emit helpers
- Test variable/buffer name generation

**Success Criteria**:
- Base class compiles
- Helper methods work correctly
- Foundation ready for specialization

---

### Task 2: Implement Compute Kernel IR-Driven Visitor
**File**: `src/target/tt/codegen_tt_compute_visitor.cc`
**Estimated**: 3-4 days
**Dependencies**: Task 1

**Specification**:

Specialize `TTCodegenVisitor` for compute kernel generation.

1. **Class Definition**:
   ```cpp
   class TTComputeCodegenVisitor : public TTCodegenVisitor {
   public:
     TTComputeCodegenVisitor(const PrimFunc& func);

   protected:
     void VisitStmt_(const ForNode* op) override;
     void VisitStmt_(const AttrStmtNode* op) override;
     void VisitStmt_(const BufferStoreNode* op) override;

   private:
     void EmitMatmulIntrinsic(const AttrStmtNode* op);
     void EmitCBWait(uint32_t cb_id, uint32_t ntiles);
     void EmitCBPop(uint32_t cb_id, uint32_t ntiles);
     void EmitCBPush(uint32_t cb_id, uint32_t ntiles);
   };
   ```

2. **For Loop Handling**:
   ```cpp
   void VisitStmt_(const ForNode* op) override {
     std::string loop_var = GetVarName(op->loop_var);
     code_ << "for (uint32_t " << loop_var << " = ";
     code_ << EmitExpr(op->min) << "; ";
     code_ << loop_var << " < " << EmitExpr(op->extent) << "; ";
     code_ << "++" << loop_var << ") {\n";
     indent_level_++;
     VisitStmt(op->body);
     indent_level_--;
     Emit("}\n");
   }
   ```

3. **AttrStmt Handling** (detects operations):
   ```cpp
   void VisitStmt_(const AttrStmtNode* op) override {
     if (op->attr_key == "tt.matmul_intrinsic") {
       EmitMatmulIntrinsic(op);
     } else if (op->attr_key == "tt.copy") {
       EmitCopyOperation(op);
     } else {
       // Default: visit body
       VisitStmt(op->body);
     }
   }
   ```

4. **Matmul Intrinsic Emission**:
   ```cpp
   void EmitMatmulIntrinsic(const AttrStmtNode* op) {
     // Extract matmul ID from op->value
     int matmul_id = GetMatmulId(op->value);

     // Determine if this is init or accumulate
     bool accumulate = matmul_id > 0;

     if (!accumulate) {
       EmitLine("matmul_tiles_init(CB_A, CB_B, CB_C);");
     } else {
       EmitLine("matmul_tiles(CB_A, CB_B, CB_C, true);");
     }

     // Visit body for nested operations
     VisitStmt(op->body);
   }
   ```

5. **Integration Point**:
   ```cpp
   std::string EmitTTComputeKernelIRDriven(const PrimFunc& func) {
     TTComputeCodegenVisitor visitor(func);

     // Emit preamble (includes, CB defines)
     visitor.EmitPreamble();

     // Walk the IR body
     visitor(func->body);

     return visitor.GetCode();
   }
   ```

**Test Plan**:
- Test for loop generation
- Test matmul intrinsic detection
- Test nested loop handling
- Integration test with TensorizeTT output

**Success Criteria**:
- Generates valid compute kernel for matmul
- Handles K-loop correctly
- Detects matmul_intrinsic annotations
- Output matches template version functionally

---

### Task 3: Implement Reader Kernel IR-Driven Visitor
**File**: `src/target/tt/codegen_tt_reader_visitor.cc`
**Estimated**: 2-3 days
**Dependencies**: Task 1

**Specification**:

Specialize `TTCodegenVisitor` for reader kernel generation.

1. **Class Definition**:
   ```cpp
   class TTReaderCodegenVisitor : public TTCodegenVisitor {
   public:
     TTReaderCodegenVisitor(const PrimFunc& func);

   protected:
     void VisitStmt_(const ForNode* op) override;
     void VisitStmt_(const BufferLoadNode* op) override;

   private:
     void EmitNOCRead(const Buffer& src, uint32_t tile_idx, uint32_t cb_id);
     void EmitCBReserve(uint32_t cb_id, uint32_t ntiles);
     void EmitCBPushBack(uint32_t cb_id, uint32_t ntiles);
   };
   ```

2. **BufferLoad Detection**:
   ```cpp
   void VisitStmt_(const BufferLoadNode* op) override {
     // Detect loads from global buffers
     if (IsGlobalBuffer(op->buffer)) {
       uint32_t cb_id = GetTargetCB(op->buffer);

       // Emit NOC read sequence
       EmitLine("cb_reserve_back(CB_" + GetBufferName(op->buffer) + ", 1);");
       EmitLine("uint32_t l1_addr = get_write_ptr(CB_" + GetBufferName(op->buffer) + ");");

       // Calculate tile index from buffer indices
       std::string tile_idx = CalculateTileIndex(op->indices);
       EmitLine("noc_async_read_tile(" + tile_idx + ", dram_addr_" +
                GetBufferName(op->buffer) + ", l1_addr);");
       EmitLine("noc_async_read_barrier();");
       EmitLine("cb_push_back(CB_" + GetBufferName(op->buffer) + ", 1);");
     }
   }
   ```

3. **Tile Index Calculation**:
   ```cpp
   std::string CalculateTileIndex(const Array<PrimExpr>& indices) {
     // For 2D buffer A[m, k]:
     // tile_idx = m * num_tiles_k + k
     ICHECK_EQ(indices.size(), 2);

     std::ostringstream idx;
     idx << "(" << EmitExpr(indices[0]) << " * ";
     idx << "num_tiles_k + " << EmitExpr(indices[1]) << ")";
     return idx.str();
   }
   ```

**Test Plan**:
- Test buffer load detection
- Test tile index calculation
- Test NOC read emission
- Test with actual matmul IR

**Success Criteria**:
- Generates valid reader kernel
- Correctly calculates tile indices
- Handles A and B buffer loads

---

### Task 4: Implement Writer Kernel IR-Driven Visitor
**File**: `src/target/tt/codegen_tt_writer_visitor.cc`
**Estimated**: 2 days
**Dependencies**: Task 1

**Specification**:

Specialize `TTCodegenVisitor` for writer kernel generation.

1. **Class Definition**:
   ```cpp
   class TTWriterCodegenVisitor : public TTCodegenVisitor {
   public:
     TTWriterCodegenVisitor(const PrimFunc& func);

   protected:
     void VisitStmt_(const ForNode* op) override;
     void VisitStmt_(const BufferStoreNode* op) override;

   private:
     void EmitNOCWrite(const Buffer& dst, uint32_t tile_idx, uint32_t cb_id);
     void EmitCBWaitFront(uint32_t cb_id, uint32_t ntiles);
     void EmitCBPopFront(uint32_t cb_id, uint32_t ntiles);
   };
   ```

2. **BufferStore Detection**:
   ```cpp
   void VisitStmt_(const BufferStoreNode* op) override {
     // Detect stores to global buffers
     if (IsGlobalBuffer(op->buffer)) {
       uint32_t cb_id = GetSourceCB(op->buffer);

       // Emit NOC write sequence
       EmitLine("cb_wait_front(CB_C, 1);");
       EmitLine("uint32_t l1_addr = get_read_ptr(CB_C);");

       std::string tile_idx = CalculateTileIndex(op->indices);
       EmitLine("noc_async_write_tile(" + tile_idx + ", l1_addr, dram_addr_C);");
       EmitLine("noc_async_write_barrier();");
       EmitLine("cb_pop_front(CB_C, 1);");
     }
   }
   ```

**Test Plan**:
- Test buffer store detection
- Test NOC write emission
- Test with matmul output IR

**Success Criteria**:
- Generates valid writer kernel
- Correctly handles output buffer
- Synchronization is correct

---

### Task 5: Integrate IR-Driven Codegen
**File**: `src/target/tt/codegen_tt.cc` (updated)
**Estimated**: 2-3 days
**Dependencies**: Tasks 1-4

**Specification**:

Update main codegen entry points to use IR-driven visitors.

1. **Add Codegen Mode Selection**:
   ```cpp
   enum class CodegenMode {
     TEMPLATE,   // Old template-based
     IR_DRIVEN   // New IR-driven
   };

   CodegenMode GetCodegenMode(const PrimFunc& func) {
     // Check for flag in attrs
     auto mode = func->attrs.GetAttr<String>("tt_codegen_mode");
     if (mode.defined() && mode.value() == "ir_driven") {
       return CodegenMode::IR_DRIVEN;
     }

     // Default to IR-driven if WS3 transforms detected
     if (func->attrs.GetAttr<Integer>("tt_num_matmuls").defined()) {
       return CodegenMode::IR_DRIVEN;
     }

     return CodegenMode::TEMPLATE;  // Fallback
   }
   ```

2. **Update EmitTTComputeKernel**:
   ```cpp
   std::string EmitTTComputeKernel(const PrimFunc& func) {
     CodegenMode mode = GetCodegenMode(func);

     if (mode == CodegenMode::IR_DRIVEN) {
       return EmitTTComputeKernelIRDriven(func);
     } else {
       return EmitTTComputeKernelTemplate(func);  // Renamed old version
     }
   }
   ```

3. **Update EmitTTReaderKernel**:
   ```cpp
   std::string EmitTTReaderKernel(const PrimFunc& func) {
     CodegenMode mode = GetCodegenMode(func);

     if (mode == CodegenMode::IR_DRIVEN) {
       return EmitTTReaderKernelIRDriven(func);
     } else {
       return EmitTTReaderKernelTemplate(func);
     }
   }
   ```

4. **Update EmitTTWriterKernel**: Similar pattern

5. **Maintain Backward Compatibility**:
   - Keep template-based functions as `*Template()` variants
   - Default to IR-driven for new code
   - Allow explicit override via attribute

**Test Plan**:
- Test mode detection
- Test IR-driven path works
- Test template path still works (backward compat)
- Integration tests for full pipeline

**Success Criteria**:
- Both modes work correctly
- Default is IR-driven
- Backward compatibility maintained

---

### Task 6: Update Tests for IR-Driven Codegen
**Files**: `testing/python/tt/test_ws4_codegen.py`, `test_ws5_reader_writer.py`, `test_ws6_host_program.py`
**Estimated**: 2-3 days
**Dependencies**: Task 5

**Specification**:

Update existing tests to work with IR-driven codegen.

1. **Update WS4 Compute Tests**:
   ```python
   def test_ir_driven_compute_kernel():
       """Test IR-driven compute kernel generation."""
       # Create module with TensorizeTT annotations
       mod = create_module_with_matmul_ir()

       # Apply full WS1-3 pipeline
       mod = tt.apply_tt_defaults(mod)
       mod = tt.apply_ws2_passes(mod)
       mod = tt.apply_ws3_passes(mod)  # Includes TensorizeTT

       # Generate artifacts (should use IR-driven)
       artifacts = tt.emit_tt_artifacts(mod)

       compute_cpp = artifacts["compute.cpp"]

       # Verify IR-driven generation
       assert "// IR-driven codegen" in compute_cpp
       assert "matmul_tiles" in compute_cpp

       # Should NOT have template markers
       assert "// Template codegen" not in compute_cpp
   ```

2. **Add IR Structure Tests**:
   ```python
   def test_ir_driven_respects_loop_structure():
       """Test that IR-driven codegen respects actual loop structure."""
       # Create IR with specific loop pattern
       mod = create_custom_loop_structure()

       # Apply transforms
       mod = apply_full_pipeline(mod)

       artifacts = tt.emit_tt_artifacts(mod)
       compute_cpp = artifacts["compute.cpp"]

       # Verify loops match IR structure
       assert compute_cpp.count("for (") == expected_loop_count
   ```

3. **Add Visitor Coverage Tests**:
   ```python
   def test_visitor_handles_all_stmt_types():
       """Test that visitor handles all TIR statement types."""
       # Test ForNode handling
       # Test AttrStmtNode handling
       # Test AllocateNode handling
       # Test SeqStmtNode handling
       # etc.
   ```

4. **Update MVP Acceptance Tests**:
   - Ensure they work with IR-driven
   - Add flag to test both modes

**Test Plan**:
- All existing tests pass
- New IR-driven specific tests pass
- Template mode tests still pass
- Coverage for all visitor methods

**Success Criteria**:
- 77+ tests passing
- IR-driven mode tested
- Template mode still works
- New tests for IR-specific features

---

## Implementation Strategy

### Phase 1: Foundation (Tasks 1-2)
1. Implement base visitor class
2. Implement compute kernel visitor
3. Test with simple matmul

### Phase 2: Complete Kernels (Tasks 3-4)
1. Implement reader visitor
2. Implement writer visitor
3. Test full 3-kernel system

### Phase 3: Integration (Tasks 5-6)
1. Integrate with main codegen
2. Update tests
3. Verify backward compatibility

### Phased Rollout

**Option A: Big Bang** - Switch entirely to IR-driven
**Option B: Gradual** - Keep both, default to IR-driven
**Option C: Feature Flag** - User-selectable mode

**Recommendation**: Option B (Gradual) - Maintain both modes with IR-driven as default

---

## Testing Strategy

### Unit Tests
- Each visitor method individually
- Helper functions
- Edge cases

### Integration Tests
- Full WS1-6 pipeline with IR-driven
- Multiple kernel types
- Complex loop structures

### Regression Tests
- All existing tests must pass
- Template mode continues to work
- Performance comparisons

### Acceptance Tests
- MVP tests with IR-driven
- Arbitrary kernel tests (new)
- Extensibility demonstrations

---

## Success Criteria

### Functional
- ✅ IR-driven generates correct code for matmul
- ✅ IR-driven supports arbitrary kernels
- ✅ All visitor methods implemented
- ✅ Integration complete

### Quality
- ✅ 80+ tests passing
- ✅ Code coverage >80%
- ✅ Documentation complete
- ✅ Backward compatible

### Performance
- ✅ Generated code quality equivalent to template
- ✅ Codegen time acceptable (<1s for typical kernels)

---

## Risks and Mitigation

**Risk 1: IR Structure Complexity**
- **Mitigation**: Start with matmul, gradually add patterns

**Risk 2: Backward Compatibility**
- **Mitigation**: Keep template mode, default to IR-driven

**Risk 3: Performance Regression**
- **Mitigation**: Benchmark both modes, optimize visitors

**Risk 4: Incomplete Coverage**
- **Mitigation**: Comprehensive testing, visitor coverage analysis

---

## Future Extensions

After IR-driven codegen is complete:

1. **Support New Operations**
   - Flash attention kernels
   - Reduce operations
   - Custom intrinsics

2. **Optimization Passes**
   - Loop fusion
   - CB reuse optimization
   - NOC traffic optimization

3. **Multi-Kernel Support**
   - Pipeline multiple operations
   - Cross-kernel optimization

---

## References

**TVM Visitor Pattern**:
- `include/tvm/tir/stmt_functor.h` - `StmtExprVisitor` base class
- `src/target/source/codegen_c.cc` - C codegen visitor example

**Tenstorrent Backend**:
- `src/transform/tt/tensorize_tt.cc` - Matmul annotation visitor
- `src/target/tt/codegen_tt.cc` - Current template codegen

---

**Document Version**: 1.0
**Author**: Claude Code
**Date**: 2025-10-07
**Status**: Implementation Plan - Ready for Execution
