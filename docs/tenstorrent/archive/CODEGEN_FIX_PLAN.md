# Tenstorrent Codegen Fix Plan

**Date:** 2025-10-08
**Status:** ✅ Complete (2025-10-08)
**Related:** IR Lowering Tasks 1-8 (Complete), Codegen Issues

**Completion Summary:**
- ✅ Task 1: Fix Compute Visitor (PR #82)
- ✅ Task 2: Fix Host Program (PR #83)
- ✅ Task 3: Disable Template Codegen (PR #84)
- ✅ Task 4: Matmul Test Example (test_matmul_codegen.py)
- ✅ Task 5: Integration Testing (95/95 tests passing)

## Problem Statement

After completing IR lowering unification (Tasks 1-8, 95/95 tests passing), the code generation has several issues:

### Issues Identified

1. **Compute kernel incomplete**
   - Current output: `C[/* unsupported expr: tir.Ramp */] = ...`
   - Root cause: IR visitor doesn't handle vectorized load/store (tir.Ramp expressions)
   - Location: `src/target/tt/codegen_tt_compute_visitor.cc`

2. **Host program incomplete**
   - Missing: SetRuntimeArgs calls for all 3 kernels
   - Missing: Actual kernel creation from generated sources
   - Mock stubs: "TODO: Load actual kernel sources"
   - Location: `src/target/tt/codegen_tt.cc` (EmitTTHostProgram)

3. **Wrong example used**
   - Current: Element-wise add (simple_add)
   - Needed: Matmul with K-loop (256x256 gemm)
   - Reason: Matmul demonstrates full 3-kernel architecture with circular buffer flow

4. **Template-based codegen still active**
   - Feature flag: `USE_IR_DRIVEN_CODEGEN = true` (line 104)
   - But template code still exists and may be confusing
   - Need: Clean separation or removal of template-based flow

## Current Architecture Analysis

### Code Generation Flow

```
Python: emit_tt_artifacts(mod)
    ↓
C++: EmitTTArtifacts(mod, target)
    ↓
C++: CodegenTT(mod, target)
    ↓
    ├─ EmitTTReaderKernelIRDriven(func) → reader.cpp
    ├─ EmitTTComputeKernelIRDriven(func) → compute.cpp
    ├─ EmitTTWriterKernelIRDriven(func) → writer.cpp
    ├─ EmitTTHostProgram(func) → main.cpp
    └─ EmitTTPlanJSON(func) → tt.plan.json
```

### Visitor Classes

1. **TTCodegenVisitor** (base class)
   - File: `src/target/tt/codegen_tt_visitor_base.cc`
   - Provides: IR walking infrastructure, expression emission
   - Methods: `VisitStmt_()`, `VisitExpr_()`, `EmitExpr()`, `EmitLine()`

2. **TTComputeCodegenVisitor** (compute kernel)
   - File: `src/target/tt/codegen_tt_compute_visitor.cc`
   - Purpose: Generate MAIN() function for compute
   - Current issues:
     - ✅ DST double buffering working (Pattern 1 & 3)
     - ❌ Doesn't handle vectorized expressions (tir.Ramp, tir.Load with ramp)
     - ❌ Doesn't emit proper matmul_tiles() calls
     - ❌ Doesn't handle T.gemm intrinsic

3. **TTReaderCodegenVisitor** (reader kernel)
   - File: `src/target/tt/codegen_tt_reader_visitor.cc`
   - Purpose: Generate kernel_main() for reader
   - Status: ✅ Template-based, generates correct K-loop structure

4. **TTWriterCodegenVisitor** (writer kernel)
   - File: `src/target/tt/codegen_tt_writer_visitor.cc`
   - Purpose: Generate kernel_main() for writer
   - Status: ✅ Template-based, generates correct write loop

### Host Program Status

**EmitTTHostProgram()** (lines 350-650 in codegen_tt.cc):
- ✅ Includes (real/mock conditional)
- ✅ Device setup
- ✅ Tile configuration
- ✅ Circular buffer creation (real/mock)
- ❌ **Kernel creation incomplete** (line 476-480):
  ```cpp
  // TODO: Load actual kernel sources from generated reader.cpp, compute.cpp, writer.cpp
  // auto reader_kernel = CreateKernel(program, "reader.cpp", core, DataMovementConfig{...});
  ```
- ❌ **SetRuntimeArgs incomplete** (lines 547-650):
  - Only prints mock comments
  - Doesn't actually call SetRuntimeArgs with proper values
- ❌ **Kernel launch incomplete**
  - No actual program execution
  - Mock stubs only

## Fix Plan

### Phase 1: Fix Compute Kernel IR Visitor (Priority 1)

**Goal:** Make compute kernel generate proper matmul code from IR

**File:** `src/target/tt/codegen_tt_compute_visitor.cc`

**Changes needed:**

1. **Add matmul intrinsic detection**
   - Detect T.gemm calls in IR (likely appear as tir::Call nodes)
   - Map to `matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false)`
   - Track accumulation state (first K iteration vs. subsequent)

2. **Handle vectorized load/store**
   - Add `VisitExpr_(const RampNode* op)` to handle tir.Ramp
   - Add `VisitExpr_(const LoadNode* op)` to handle vectorized loads
   - Add `VisitStmt_(const StoreNode* op)` to handle vectorized stores
   - For now: emit tile-wise operations, not element-wise

3. **Proper K-loop detection**
   - Detect K-reduction loop (should have extent = Kt)
   - Emit proper DST lifecycle:
     ```cpp
     for (kt = 0; kt < Kt; kt++) {
         cb_wait_front(cb_in0, 1);
         cb_wait_front(cb_in1, 1);
         matmul_tiles(..., kt > 0 /* accumulate */);
         cb_pop_front(cb_in0, 1);
         cb_pop_front(cb_in1, 1);
     }
     ```

4. **Test with matmul IR**
   - Use proper matmul kernel (not element-wise)
   - Verify K-loop generates correctly
   - Check DST acquire/commit/release pattern

**Estimated effort:** 4-6 hours (complex IR walking logic)

### Phase 2: Fix Host Program (Priority 2)

**Goal:** Generate complete, runnable host program

**File:** `src/target/tt/codegen_tt.cc` (EmitTTHostProgram function)

**Changes needed:**

1. **Complete kernel creation (lines 476-480)**
   ```cpp
   // Real Metalium mode:
   auto reader_kernel = CreateKernel(
       program, "kernels/reader.cpp", core,
       DataMovementConfig{
           .processor = DataMovementProcessor::RISCV_0,
           .noc = NOC::RISCV_0_default
       }
   );
   auto compute_kernel = CreateKernel(
       program, "kernels/compute.cpp", core,
       ComputeConfig{
           .math_fidelity = MathFidelity::HiFi4,
           .fp32_dest_acc_en = false,
           .math_approx_mode = false
       }
   );
   auto writer_kernel = CreateKernel(
       program, "kernels/writer.cpp", core,
       DataMovementConfig{
           .processor = DataMovementProcessor::RISCV_1,
           .noc = NOC::RISCV_1_default
       }
   );
   ```

2. **Add SetRuntimeArgs calls (lines 550-600)**
   ```cpp
   // Reader kernel args
   SetRuntimeArgs(
       program, reader_kernel, core,
       {
           buffer_a->address(),           // dram_addr_a
           buffer_b->address(),           // dram_addr_b
           static_cast<uint32_t>(Mt),     // Mt
           static_cast<uint32_t>(Kt),     // Kt
           static_cast<uint32_t>(Nt),     // Nt
           out_tile_start_id,             // start_tile_id
           num_out_tiles_per_core         // num_tiles
       }
   );

   // Compute kernel args
   SetRuntimeArgs(
       program, compute_kernel, core,
       {
           out_tile_start_id,             // start_tile_id
           num_out_tiles_per_core,        // num_output_tiles
           static_cast<uint32_t>(Kt)      // Kt
       }
   );

   // Writer kernel args
   SetRuntimeArgs(
       program, writer_kernel, core,
       {
           buffer_c->address(),           // dram_addr_c
           out_tile_start_id,             // start_tile_id
           num_out_tiles_per_core,        // num_tiles
           static_cast<uint32_t>(Nt)      // Nt
       }
   );
   ```

3. **Add program launch (lines 600-620)**
   ```cpp
   #ifdef TL_USE_REAL_METALIUM
   // Real Metalium: EnqueueProgram
   EnqueueProgram(cq, program, true /* blocking */);
   Finish(cq);
   std::cout << "Program execution complete" << std::endl;

   // Read back results
   EnqueueReadBuffer(cq, buffer_c, host_c, true);
   std::cout << "Results read back to host" << std::endl;
   #else
   // Mock mode
   CommandQueue cq;
   cq.EnqueueProgram(&program, true);
   cq.Finish();
   std::cout << "Program execution complete (Mock)" << std::endl;
   #endif
   ```

4. **Add result verification (lines 620-650)**
   ```cpp
   // Verify first tile of output
   std::cout << "Result verification:" << std::endl;
   std::cout << "  C[0,0] (first element): " << host_c[0] << std::endl;
   std::cout << "  C[0,1]: " << host_c[1] << std::endl;
   // ... print first few elements
   ```

**Estimated effort:** 2-3 hours (straightforward codegen)

### Phase 3: Disable Template-Based Codegen (Priority 3)

**Goal:** Clean up old template-based flow to avoid confusion

**Option A: Remove template code entirely**
- Delete `EmitTTComputeKernel()` (lines 137-210)
- Delete `EmitTTReaderKernel()` (lines 215-300)
- Delete `EmitTTWriterKernel()` (lines 305-350)
- Remove `USE_IR_DRIVEN_CODEGEN` flag (always use IR-driven)
- Update CodegenTT to only call IR-driven functions

**Option B: Keep template code but clearly mark deprecated**
- Add comments: "// DEPRECATED: Template-based codegen"
- Wrap in `#if 0 ... #endif` blocks
- Update USE_IR_DRIVEN_CODEGEN to compile-time constant (not configurable)

**Recommended:** Option A (clean removal)

**Estimated effort:** 1 hour

### Phase 4: Create Proper Matmul Test Example (Priority 4)

**Goal:** Replace element-wise example with proper matmul

**File:** `test_matmul_codegen.py` (create new)

**Changes needed:**

```python
@T.prim_func
def matmul_256x256(
    A: T.Buffer((256, 256), "float16"),
    B: T.Buffer((256, 256), "float16"),
    C: T.Buffer((256, 256), "float16")
):
    """256x256 matmul using T.gemm intrinsic"""
    with T.Kernel(8, 8) as (bx, by):
        # Allocate tile storage
        A_tile = T.alloc_fragment((32, 32), "float16")
        B_tile = T.alloc_fragment((32, 32), "float16")
        C_tile = T.alloc_fragment((32, 32), "float16")

        # Initialize accumulator
        T.clear(C_tile)

        # K-loop over 8 tiles
        for k in range(8):
            # Load tiles
            T.copy(A[by*32:(by+1)*32, k*32:(k+1)*32], A_tile)
            T.copy(B[k*32:(k+1)*32, bx*32:(bx+1)*32], B_tile)

            # Matmul accumulate
            T.gemm(A_tile, B_tile, C_tile)

        # Store result
        T.copy(C_tile, C[by*32:(by+1)*32, bx*32:(bx+1)*32])
```

**Note:** May need to work around T.gemm layout inference issue. If T.gemm still fails, use manual nested loops:

```python
for k in range(8):
    T.copy(A[...], A_tile)
    T.copy(B[...], B_tile)

    # Manual matmul (instead of T.gemm)
    for i, j in T.Parallel(32, 32):
        for k_inner in range(32):
            C_tile[i, j] += A_tile[i, k_inner] * B_tile[k_inner, j]

    T.copy(C_tile, C[...])
```

**Estimated effort:** 1 hour

### Phase 5: Integration Testing (Priority 5)

**Test plan:**

1. **Build TileLang**
   ```bash
   bash maint/scripts/local_build_and_test_tt.sh --skip-deps --jobs 4
   ```

2. **Run matmul example**
   ```bash
   python test_matmul_codegen.py
   ```

3. **Verify generated files**
   - `compute.cpp`: Should have proper K-loop with matmul_tiles()
   - `reader.cpp`: Should load A[m,k] and B[k,n] tiles
   - `writer.cpp`: Should write C[m,n] tiles
   - `main.cpp`: Should have complete SetRuntimeArgs and launch
   - `tt.plan.json`: Should have correct grid metadata

4. **Check for regressions**
   ```bash
   pytest testing/python/tt/ -v
   ```
   - All 95 tests should still pass

**Estimated effort:** 2 hours (including debugging)

## Task Breakdown

### Task 1: Fix Compute Visitor ⏳
**PR:** `fix-compute-kernel-matmul`
- Add matmul intrinsic detection
- Handle vectorized expressions (Ramp, Load, Store)
- Proper K-loop DST lifecycle
- Test with matmul IR

**Files:**
- `src/target/tt/codegen_tt_compute_visitor.cc`
- `src/target/tt/codegen_tt_compute_visitor.h`

### Task 2: Fix Host Program ⏳
**PR:** `fix-host-program-runtime-args`
- Complete kernel creation
- Add SetRuntimeArgs for all 3 kernels
- Add program launch and synchronization
- Add result readback and verification

**Files:**
- `src/target/tt/codegen_tt.cc` (EmitTTHostProgram)

### Task 3: Disable Template Codegen ⏳
**PR:** `remove-template-codegen`
- Remove template-based functions
- Remove USE_IR_DRIVEN_CODEGEN flag
- Update CodegenTT function
- Clean up dead code

**Files:**
- `src/target/tt/codegen_tt.cc`

### Task 4: Matmul Test Example ⏳
**PR:** `add-matmul-codegen-test`
- Create proper matmul example
- Use T.gemm or manual nested loops
- Add verification logic
- Document usage

**Files:**
- `test_matmul_codegen.py` (new)
- `examples/tenstorrent/matmul_256x256.py` (new)

### Task 5: Integration Testing ⏳
- Build and test
- Verify all artifacts
- Regression testing
- Document results

## Success Criteria

### Must Have
- ✅ Compute kernel generates correct matmul K-loop
- ✅ Host program has complete SetRuntimeArgs
- ✅ Host program has kernel creation
- ✅ All 95 tests still pass
- ✅ Matmul example runs end-to-end

### Should Have
- ✅ Clean separation (no template codegen confusion)
- ✅ Proper DST lifecycle in compute kernel
- ✅ Reader/writer kernels correctly structured
- ✅ Host program has result verification

### Nice to Have
- Real Metalium mode compiles (if SDK available)
- Documentation updated
- Session summary document created

## Timeline

**Total estimated effort:** 10-13 hours

**Aggressive timeline (1 working day):**
- Morning (4 hours): Tasks 1-2 (compute visitor + host program)
- Afternoon (3 hours): Tasks 3-4 (cleanup + matmul example)
- Evening (2 hours): Task 5 (integration testing)

**Conservative timeline (2 working days):**
- Day 1: Tasks 1-3 (core fixes)
- Day 2: Tasks 4-5 (example + testing)

## Implementation Order

**Critical path:**
1. Task 1 (compute visitor) - BLOCKS everything else
2. Task 4 (matmul example) - NEEDED to test Task 1
3. Task 2 (host program) - INDEPENDENT, can parallel with Task 1
4. Task 3 (cleanup) - LOW PRIORITY, do last
5. Task 5 (testing) - FINAL validation

**Recommended execution:**
1. Start with Task 4 (matmul example) - quick win, defines test case
2. Move to Task 1 (compute visitor) - hardest, most critical
3. Do Task 2 (host program) - complete the pipeline
4. Do Task 5 (testing) - validate everything works
5. Do Task 3 (cleanup) - polish and cleanup

## Notes

- **IR lowering is complete** - These are codegen-only fixes
- **No IR transform changes needed** - All 95 tests passing
- **Focus on codegen quality** - Make generated code production-ready
- **Both mock and real modes** - Keep conditional compilation working

## References

- IR Lowering Tasks 1-8: Complete (PRs #72-79)
- IR Lowering Validation: `docs/tenstorrent/IR_LOWERING_VALIDATION.md`
- Unified Plan: `docs/tenstorrent/UNIFIED_MATMUL_MVP_PLAN.md`
- Backend Interface: `tilelang/engine/BACKEND_INTERFACE.md`

---

**Created:** 2025-10-08
**Status:** Ready for implementation
