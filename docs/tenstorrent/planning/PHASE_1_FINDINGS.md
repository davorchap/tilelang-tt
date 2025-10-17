# Phase 1 CodeGen Refactoring: Findings Report

## Date: 2025-10-17
## Status: Phase 1 Implementation Complete

## Executive Summary

Phase 1 of the CodeGen Visitor Refactoring Plan has been successfully implemented. The validation infrastructure is now in place and correctly identifies incomplete IR, preventing silent fallback to template-based code generation.

## Completed Tasks

### ✅ Task 1.1: Add IR Validation Pass
- Created `tilelang/tenstorrent/passes/validate_split_kernels.py`
- Validates that reader/compute/writer kernels have complete IR
- Checks for specific operations expected in each kernel role
- Fails loudly when IR is incomplete

### ✅ Task 1.2: Debug Empty IR Bodies
- Successfully integrated validation into the v5 pipeline after D1 (split_device_kernel)
- Validation correctly detects empty kernel bodies
- Root cause identified: Stage C passes are not generating expected protocol operations

### ✅ Task 1.3: Add IR Dump Utility
- Created `tilelang/tenstorrent/utils/debug_ir.py`
- Provides automatic IR dumping after each pass
- Generates analysis reports with kernel statistics
- Controlled via environment variables (TT_DUMP_IR=1)

### ✅ Task 1.4: Integration
- Validation pass integrated into pipeline.py after D1
- IR dump utility integrated with pipeline wrapper
- Tests confirm validation is working correctly

## Key Findings

### 1. Empty IR Bodies After Split

**Issue:** The split_device_kernel pass (D1) creates three kernels but all have empty bodies.

**Root Cause:** The Stage C passes (C1-C3) are not generating the expected protocol operations:
- Missing `read_to_cb` operations for reader kernels
- Missing `write_from_cb` operations for writer kernels
- Only generating simple operations like `tt.fpu.add`
- The tile dataflow graph (DFG) has 0 nodes and 0 edges

**Evidence:**
```python
# Expected operations (not found):
- tt.read_to_cb
- tt.write_from_cb
- tt.mm.mma
- tt.alloc_cb

# Actual operations found:
- tt.fpu.add (simple FPU operation)
```

### 2. Validation Pass Working Correctly

The validation pass successfully:
- Detects missing kernel bodies
- Identifies which operations are missing
- Provides actionable error messages
- Prevents codegen from using fallback templates

Example output:
```
Split kernel validation failed - IR is incomplete:
[reader] Kernel simple_add_reader has no body
[compute] Kernel simple_add_compute has no body
[writer] Kernel simple_add_writer has no body
```

### 3. Stage C Passes Need Work

The following passes need investigation:
- **C1 (lower_shared_to_cb_v5)**: Should generate CB allocation and data movement ops
- **C2 (lower_tt_tile_intrinsics_v5)**: Should lower high-level ops to TT intrinsics
- **C3 (build_tile_dfg_tt)**: Reports 0 nodes/edges, indicating incomplete analysis

## Impact on Pipeline

1. **Positive:** No more silent fallbacks to template code
2. **Negative:** Most kernels will fail validation until Stage C is fixed
3. **Workaround:** Validation can be disabled for testing other passes

## Phase 1 Fixes Applied (2025-10-17)

### Stage C Pass Fixes

#### C1: lower_shared_to_cb_v5 (FIXED ✅)
- **Issue:** Not detecting buffer copy patterns, only looking for explicit T.copy intrinsics
- **Fix:** Added `visit_block_realize` to detect copy blocks by analyzing T.reads/T.writes patterns
- **Result:** Now generates:
  - `tt.alloc_cb` for shared buffer allocations
  - `tt.read_to_cb` for copies from global to shared memory
  - `tt.write_from_cb` for copies from shared to global memory

#### C2: lower_tt_tile_intrinsics_v5 (FIXED ✅)
- **Issue:** Detecting binary operations but not transforming them
- **Fix:** Modified `_lower_binary_operation` to emit `tt.fpu.*` intrinsics and added `_get_cb_name_for_operand` helper
- **Result:** Now generates `tt.fpu.add`, `tt.fpu.multiply`, etc. for binary operations

#### C3: build_tile_dfg_tt (PARTIAL ⚠️)
- **Issue:** Building empty dataflow graph (0 nodes, 0 edges)
- **Root Cause:** C3 expects operations in `tir.Evaluate` nodes but C1/C2 structure them differently
- **Status:** C3 needs refactoring to traverse the modified IR structure from C1/C2
- **Workaround:** The operations are present in the IR, just not being detected by C3's visitor

### Results After Fixes

The Stage C pipeline now:
1. ✅ Generates CB allocations and data movement operations (C1)
2. ✅ Generates compute intrinsics for tile operations (C2)
3. ⚠️ Builds dataflow graph (C3 needs update to traverse new IR structure)

The validation pass can now detect proper IR operations instead of empty bodies, addressing the original issue identified in Phase 1.

## Recommended Next Steps (Phase 2)

### Priority 1: Complete Stage C3 Fix
1. Update build_tile_dfg_tt visitor to traverse the IR structure created by fixed C1/C2
2. Ensure dataflow graph captures all CB and compute operations
3. Verify kernel role assignment for split_device_kernel pass

### Priority 2: Enhance Split Logic
1. Add fallback patterns for simple operations
2. Improve operation classification
3. Support more diverse IR patterns

### Priority 3: Complete Codegen
1. Implement proper visitor-based codegen (Phase 2)
2. Remove template fallback code (Phase 3)
3. Add comprehensive test coverage

## Environment Variables

The following environment variables control debugging:
- `TT_DUMP_IR=1` - Enable IR dumping
- `TT_DUMP_IR_DIR=<path>` - Set dump directory (default: ir_dumps)

## Test Commands

```bash
# Run with validation and IR dumping
export TT_DUMP_IR=1
export TT_DUMP_IR_DIR=test_ir_dump
python testing/python/tenstorrent/test_v5_pipeline_e2e.py

# Check validation on simple kernel
python test_validation.py
```

## Files Modified

### New Files Created:
- `/tilelang/tenstorrent/passes/validate_split_kernels.py`
- `/tilelang/tenstorrent/utils/debug_ir.py`
- `/docs/tenstorrent/planning/PHASE_1_FINDINGS.md`

### Files Updated:
- `/tilelang/tenstorrent/passes/pipeline.py` - Added validation pass
- `/tilelang/tenstorrent/passes/__init__.py` - Exported validation pass

## Conclusion

Phase 1 successfully established the validation infrastructure and identified the root cause of empty IR bodies. The validation pass prevents silent failures and provides clear diagnostics. The next phase should focus on fixing the Stage C passes to generate complete protocol operations before attempting to refactor the codegen visitors.

The refactoring plan's phased approach has proven valuable - by adding validation first, we've identified critical issues that would have caused problems in later phases.