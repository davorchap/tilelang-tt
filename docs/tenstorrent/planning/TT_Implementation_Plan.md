# Tenstorrent Backend Implementation Plan (Python-First)

**Version:** 2.0
**Date:** 2025-10-15
**Status:** Rapid Prototyping Focus
**Based on:** TileLang TT TIR Lowering Guide v5

---

## Executive Summary

This updated plan prioritizes **rapid prototyping in Python** to achieve a working end-to-end GEMM compilation quickly. C++ migration is postponed to Phase 2, after all passes are stable and proven. This approach optimizes for development speed and iteration time.

### Key Changes from v1
- **Python-first development** - All new passes implemented in Python
- **C++ migration postponed** - Only after passes stabilize
- **Faster iteration** - No Python/C++ boundary complexity during prototyping
- **Focus on GEMM** - Get one pattern working end-to-end first

---

## Phase Overview

| Phase | Duration | Focus | Language | Goal |
|-------|----------|-------|----------|------|
| **Phase 1** | 3-4 weeks | Complete all passes | **Python** | Working GEMM end-to-end |
| **Phase 2** | 2-3 weeks | Optimization & Migration | Python → C++ | Performance & production quality |
| **Phase 3** | 1-2 weeks | Polish & Documentation | Mixed | Release ready |

---

## Phase 1: Python Implementation Sprint (Weeks 1-4)

### Goal: Get GEMM Working End-to-End

**Success Criteria:**
- ✅ All 17 passes implemented in Python
- ✅ GEMM compiles from DSL to C++
- ✅ Generated code matches expected structure
- ✅ Mock execution validates correctness

### Pass Implementation Order (Optimized for Dependencies)

#### Week 1: Foundation Passes

| Priority | Pass | Location | Dependencies | Complexity |
|----------|------|----------|--------------|------------|
| **P0** | C3: BuildTileDFGTT | `tilelang/tenstorrent/passes/build_tile_dfg_tt.py` | None | Low |
| **P0** | D1: SplitDeviceKernel | `tilelang/tenstorrent/passes/split_device_kernel_tt.py` | C3 | High |
| **P1** | A3: AttachTensorAccessorTT | `tilelang/tenstorrent/passes/attach_tensor_accessor_tt.py` | A1 (exists) | Low |

#### Week 2: Protocol Insertion

| Priority | Pass | Location | Dependencies | Complexity |
|----------|------|----------|--------------|------------|
| **P0** | D3: LowerCBIntrinsics | `tilelang/tenstorrent/passes/lower_cb_intrinsics_tt.py` | D1 | Medium |
| **P0** | D5: InsertDSTManagementTT | `tilelang/tenstorrent/passes/insert_dst_management_tt.py` | D1 | Medium |
| **P1** | D4: InsertComputeInitTT | `tilelang/tenstorrent/passes/insert_compute_init_tt.py` | D1 | Low |
| **P1** | D2: ConfigureTensorAccessorTT | `tilelang/tenstorrent/passes/configure_tensor_accessor_tt.py` | D1, A3 | Low |

#### Week 3: Integration & Refinement

| Priority | Task | Description | Deliverable |
|----------|------|-------------|-------------|
| **P0** | Update existing passes | Consume new metadata format | Updated passes |
| **P0** | E1: FinalizePersistentSignatureTT | Finalize runtime args | Complete pass |
| **P1** | Fix C1: LowerSharedToCB | Remove CB ID heuristics | Protocol-less version |
| **P1** | Fix C2: LowerTTTileIntrinsics | Remove "_tile" suffix heuristics | Clean version |

#### Week 4: Testing & Stabilization

| Priority | Task | Description | Deliverable |
|----------|------|-------------|-------------|
| **P0** | GEMM end-to-end test | Full compilation test | Working example |
| **P0** | Fix integration issues | Debug and fix | Stable pipeline |
| **P1** | Add more test patterns | Eltwise, Conv2D | Extended tests |
| **P2** | Performance measurement | Baseline metrics | Benchmark data |

---

## Python Implementation Templates

### Quick-Start Template for New Passes

```python
# tilelang/tenstorrent/passes/pass_name.py

import tvm
from tvm import tir
from tvm.tir import stmt_functor

@tvm.tir.transform.prim_func_pass(opt_level=0)
def PassName(func, mod, ctx):
    """
    Brief description of what this pass does.

    Input: Description of expected IR state
    Output: Description of IR after transformation
    """

    class PassNameMutator(stmt_functor.IRMutator):
        def __init__(self):
            super().__init__()
            # Initialize any state needed

        def visit_evaluate(self, op):
            # Main transformation logic
            if self._is_target_pattern(op):
                return self._transform_pattern(op)
            return super().visit_evaluate(op)

        def _is_target_pattern(self, op):
            # Pattern detection logic
            pass

        def _transform_pattern(self, op):
            # Transformation logic
            pass

    # Apply the mutator
    mutator = PassNameMutator()
    func = func.with_body(mutator.visit(func.body))

    # Update attributes if needed
    func = func.with_attr("tt.some_attr", some_value)

    return func
```

### D1: SplitDeviceKernel (Python Implementation)

```python
# tilelang/tenstorrent/passes/split_device_kernel_tt.py

import tvm
from tvm import tir
import copy

@tvm.tir.transform.module_pass(opt_level=0)
def SplitDeviceKernel(mod, ctx):
    """
    Split monolithic kernel into reader/compute/writer.

    This is a module pass because it creates multiple functions.
    """

    class DataflowAnalyzer(tir.stmt_functor.StmtVisitor):
        def __init__(self):
            self.read_stmts = []
            self.compute_stmts = []
            self.write_stmts = []

        def visit_evaluate(self, op):
            call = op.value
            if isinstance(call, tir.Call):
                if call.op.name == "tt.read_to_cb":
                    self.read_stmts.append(op)
                elif call.op.name in ["tt.mm.mma", "tt.fpu.add", "tt.sfpu.unary"]:
                    self.compute_stmts.append(op)
                elif call.op.name == "tt.write_from_cb":
                    self.write_stmts.append(op)
            super().visit_evaluate(op)

    # Process each function in the module
    new_funcs = {}

    for gvar, func in mod.functions.items():
        if not isinstance(func, tir.PrimFunc):
            new_funcs[gvar] = func
            continue

        # Analyze the function
        analyzer = DataflowAnalyzer()
        analyzer.visit(func.body)

        # Create three kernel variants
        base_name = gvar.name_hint

        # Reader kernel
        reader_func = create_reader_kernel(func, analyzer.read_stmts)
        reader_func = reader_func.with_attr("tt.kernel_role", "reader")
        reader_gvar = tvm.ir.GlobalVar(f"{base_name}_reader")
        new_funcs[reader_gvar] = reader_func

        # Compute kernel
        compute_func = create_compute_kernel(func, analyzer.compute_stmts)
        compute_func = compute_func.with_attr("tt.kernel_role", "compute")
        compute_gvar = tvm.ir.GlobalVar(f"{base_name}_compute")
        new_funcs[compute_gvar] = compute_func

        # Writer kernel
        writer_func = create_writer_kernel(func, analyzer.write_stmts)
        writer_func = writer_func.with_attr("tt.kernel_role", "writer")
        writer_gvar = tvm.ir.GlobalVar(f"{base_name}_writer")
        new_funcs[writer_gvar] = writer_func

    return tvm.ir.IRModule(new_funcs)

def create_reader_kernel(orig_func, read_stmts):
    """Create reader kernel with only read operations"""
    # Keep input buffers, remove output buffers
    params = [p for p in orig_func.params if is_input_buffer(p, orig_func)]
    buffer_map = {p: orig_func.buffer_map[p] for p in params}

    # Build new body with only read statements
    body = tir.SeqStmt(read_stmts) if len(read_stmts) > 1 else read_stmts[0]

    return tir.PrimFunc(params, body, buffer_map=buffer_map)
```

### D3: LowerCBIntrinsics (Python Implementation)

```python
# tilelang/tenstorrent/passes/lower_cb_intrinsics_tt.py

@tvm.tir.transform.prim_func_pass(opt_level=0)
def LowerCBIntrinsics(func, mod, ctx):
    """
    Replace abstract CB operations with NOC/CB protocol.
    Only applies to reader/writer kernels.
    """

    # Check kernel role
    if func.attrs.get("tt.kernel_role") not in ["reader", "writer"]:
        return func

    class CBProtocolInserter(tir.stmt_functor.IRMutator):
        def __init__(self, kernel_role):
            super().__init__()
            self.kernel_role = kernel_role

        def visit_evaluate(self, op):
            call = op.value
            if isinstance(call, tir.Call):
                if call.op.name == "tt.read_to_cb":
                    return self._lower_read_to_cb(call)
                elif call.op.name == "tt.write_from_cb":
                    return self._lower_write_from_cb(call)
            return super().visit_evaluate(op)

        def _lower_read_to_cb(self, call):
            """Insert NOC read protocol"""
            tensor_slice = call.args[0]
            cb_name = call.args[1]

            # Build protocol sequence
            stmts = []

            # 1. Reserve CB space
            stmts.append(tir.Evaluate(
                tir.call_extern("void", "cb_reserve_back", cb_name, 1)
            ))

            # 2. Get write pointer
            write_ptr = tir.call_extern("uint32_t", "get_write_ptr", cb_name)

            # 3. Calculate tile ID and accessor
            tile_id = self._calculate_tile_id(tensor_slice)
            accessor = self._get_tensor_accessor(tensor_slice)

            # 4. NOC async read
            stmts.append(tir.Evaluate(
                tir.call_extern("void", "noc_async_read_tile",
                              tile_id, accessor, write_ptr)
            ))

            # 5. Barrier
            stmts.append(tir.Evaluate(
                tir.call_extern("void", "noc_async_read_barrier")
            ))

            # 6. Push to CB
            stmts.append(tir.Evaluate(
                tir.call_extern("void", "cb_push_back", cb_name, 1)
            ))

            return tir.SeqStmt(stmts)

    mutator = CBProtocolInserter(func.attrs["tt.kernel_role"])
    func = func.with_body(mutator.visit(func.body))
    return func
```

### D5: InsertDSTManagementTT (Python Implementation)

```python
# tilelang/tenstorrent/passes/insert_dst_management_tt.py

@tvm.tir.transform.prim_func_pass(opt_level=0)
def InsertDSTManagementTT(func, mod, ctx):
    """
    Insert DST lifecycle management for compute kernels.
    """

    # Only apply to compute kernels
    if func.attrs.get("tt.kernel_role") != "compute":
        return func

    class DSTInserter(tir.stmt_functor.IRMutator):
        def visit_for(self, op):
            # Detect K-loop pattern
            if self._is_k_loop_pattern(op):
                return self._wrap_k_loop_with_dst(op)
            return super().visit_for(op)

        def _is_k_loop_pattern(self, loop):
            """Check if loop contains accumulating matmul"""
            class AccumDetector(tir.stmt_functor.StmtVisitor):
                def __init__(self):
                    self.has_accumulating_mma = False

                def visit_evaluate(self, op):
                    if isinstance(op.value, tir.Call):
                        if op.value.op.name == "tt.mm.mma":
                            # Check accumulate flag
                            if len(op.value.args) > 3 and op.value.args[3]:
                                self.has_accumulating_mma = True

            detector = AccumDetector()
            detector.visit(loop.body)
            return detector.has_accumulating_mma

        def _wrap_k_loop_with_dst(self, loop):
            stmts = []

            # 1. Acquire DST before loop
            stmts.append(tir.Evaluate(
                tir.call_extern("void", "tt.dst.acquire")
            ))

            # 2. Transform loop body to add CB sync
            new_body = self._add_cb_sync_to_compute(loop.body)
            new_loop = tir.For(loop.loop_var, loop.min, loop.extent,
                               loop.kind, new_body)
            stmts.append(new_loop)

            # 3. Pack and release after loop
            stmts.extend([
                tir.Evaluate(tir.call_extern("void", "cb_reserve_back", "cb_out0", 1)),
                tir.Evaluate(tir.call_extern("void", "tt.dst.commit")),
                tir.Evaluate(tir.call_extern("void", "tt.dst.wait")),
                tir.Evaluate(tir.call_extern("void", "pack_tile", 0, "cb_out0", 0)),
                tir.Evaluate(tir.call_extern("void", "tt.dst.release")),
                tir.Evaluate(tir.call_extern("void", "cb_push_back", "cb_out0", 1))
            ])

            return tir.SeqStmt(stmts)

    mutator = DSTInserter()
    func = func.with_body(mutator.visit(func.body))
    return func
```

---

## Updated Pipeline Configuration

```python
# tilelang/tenstorrent/lower.py

def get_tt_pass_pipeline_v5():
    """
    Progressive lowering pipeline with late protocol insertion.
    All passes in Python for rapid prototyping.
    """
    return [
        # Stage A: Metadata (existing + new)
        InferTTLayout(),           # A1 - Existing Python
        PropagateTTLayout(),       # A2 - Existing Python
        AttachTensorAccessorTT(),  # A3 - New Python

        # Stage B: Partitioning (existing)
        LayoutAwareWorkPartitionTT(),  # B1 - Existing Python
        GridToCoreGrid(),              # B2 - Existing (update)

        # Stage C: Protocol-less lowering (update existing)
        LowerSharedToCB(),         # C1 - Update to be protocol-less
        LowerTTTileIntrinsics(),   # C2 - Update to remove heuristics
        BuildTileDFGTT(),          # C3 - New Python

        # Stage D: Late split & protocol (all new)
        SplitDeviceKernel(),       # D1 - New Python
        ConfigureTensorAccessorTT(), # D2 - New Python
        LowerCBIntrinsics(),       # D3 - New Python
        InsertComputeInitTT(),     # D4 - New Python
        InsertDSTManagementTT(),   # D5 - New Python

        # Stage E: Finalization
        FinalizePersistentSignatureTT(),  # E1 - Update existing

        # Stage F: Verification
        VerifyTTIR(),              # F - Update for new schema
    ]
```

---

## Testing Strategy for Rapid Development

### Incremental Testing Approach

```python
# testing/python/tenstorrent/test_v5_pipeline.py

def test_pass_incrementally(kernel):
    """Test each pass in isolation and incrementally"""

    # Test individual passes
    for i, pass_func in enumerate(get_tt_pass_pipeline_v5()):
        print(f"Testing pass {i}: {pass_func.__name__}")

        # Apply pass
        result = pass_func(kernel)

        # Basic validation
        assert result is not None
        assert hasattr(result, 'body')

        # Pass-specific validation
        validate_pass_output(pass_func.__name__, result)

        # Use result for next pass
        kernel = result

    return kernel

def test_gemm_end_to_end():
    """Rapid validation of GEMM compilation"""

    # Create simple GEMM
    kernel = create_gemm_kernel(M=256, N=256, K=256)

    # Apply full pipeline
    pipeline = tvm.transform.Sequential(get_tt_pass_pipeline_v5())
    result = pipeline(kernel)

    # Generate code (even if mock)
    code = CodegenTT(result)

    # Validate structure
    assert "reader.cpp" in code
    assert "compute.cpp" in code
    assert "writer.cpp" in code
    assert "main.cpp" in code

    print("✅ GEMM compiles end-to-end!")
```

---

## Phase 2: Selective C++ Migration (Weeks 5-7)

### After Phase 1 Success, Consider Migration

**Migration Candidates (Performance-Critical):**
1. InferTTLayout - Heavy metadata processing
2. LayoutAwareWorkPartitionTT - Complex computation
3. SplitDeviceKernel - IR manipulation intensive

**Keep in Python (Rarely Changed):**
1. AttachTensorAccessorTT - Simple metadata
2. ConfigureTensorAccessorTT - Simple binding
3. BuildTileDFGTT - Graph building

**Migration Decision Matrix:**

| Pass | Complexity | Performance Impact | Change Frequency | Migrate? |
|------|------------|-------------------|------------------|----------|
| InferTTLayout | High | High | Low | Yes |
| SplitDeviceKernel | High | High | Medium | Yes |
| LowerCBIntrinsics | Medium | Medium | Low | Maybe |
| InsertDSTManagementTT | Medium | Low | Low | No |
| BuildTileDFGTT | Low | Low | Low | No |

---

## Advantages of Python-First Approach

### Development Speed
- ✅ **10x faster iteration** - No compilation required
- ✅ **Interactive debugging** - Use Python debugger
- ✅ **Easy prototyping** - Test ideas quickly
- ✅ **No build system complexity** - Just edit and run

### Flexibility
- ✅ **Easy to experiment** - Try different approaches
- ✅ **Quick fixes** - Patch issues immediately
- ✅ **Better error messages** - Python stack traces
- ✅ **Simpler testing** - pytest instead of gtest

### Risk Reduction
- ✅ **Validate design early** - Ensure approach works
- ✅ **Defer complexity** - C++ migration only when stable
- ✅ **Maintain momentum** - Keep making progress
- ✅ **Team efficiency** - More developers can contribute

---

## Success Metrics (Revised)

### Phase 1 (Python Implementation)
- [ ] All 17 passes implemented
- [ ] GEMM compiles end-to-end
- [ ] Generated code structure correct
- [ ] Basic test suite passing

### Phase 2 (Optimization - Optional)
- [ ] Critical passes migrated to C++
- [ ] 2-5x compilation speedup
- [ ] Memory usage optimized

### Phase 3 (Release)
- [ ] Documentation complete
- [ ] Examples working
- [ ] CI/CD integrated

---

## Risk Mitigation (Updated)

| Risk | Mitigation |
|------|------------|
| Python performance | Profile and migrate only bottlenecks |
| Code maintainability | Strong typing with annotations |
| Testing complexity | Incremental validation approach |
| Integration issues | Test after each pass |

---

## Conclusion

This Python-first approach will get us to a working GEMM implementation **2-3x faster** than the C++ approach. Once we have a stable, working pipeline, we can selectively migrate performance-critical passes to C++ in Phase 2.

**Key Benefits:**
- 🚀 Rapid prototyping and iteration
- 🔧 Easy debugging and testing
- 📈 Incremental progress visibility
- 🎯 Focus on correctness first, performance second

---

**Document Version:** 2.0
**Last Updated:** 2025-10-15
**Next Review:** End of Week 1

---

## Implementation Timeline (4-Week Sprint)

### Week 1: Foundation & Kernel Split

**Goal:** Implement the critical infrastructure for splitting kernels

#### Monday-Tuesday: Setup & Analysis
| Task | Duration | Output |
|------|----------|--------|
| Setup pass infrastructure | 4h | Pass templates, test framework |
| Implement BuildTileDFGTT | 4h | Dataflow graph builder |
| Design SplitDeviceKernel algorithm | 4h | Design doc & pseudocode |

#### Wednesday-Friday: Kernel Splitting
| Task | Duration | Output |
|------|----------|--------|
| Implement SplitDeviceKernel (core logic) | 8h | Basic splitting working |
| Add CB ID assignment | 4h | CB allocation logic |
| Implement AttachTensorAccessorTT | 4h | Abstract accessors |
| Test with simple GEMM | 4h | Unit tests passing |

**Week 1 Deliverables:**
- BuildTileDFGTT.py - Dataflow graph construction
- SplitDeviceKernel.py - Monolithic → 3 kernels
- AttachTensorAccessorTT.py - Tensor accessor metadata
- Basic test suite with simple patterns

---

### Week 2: Protocol Insertion

**Goal:** Add all protocol insertion passes

#### Monday-Tuesday: NOC/CB Protocol
| Task | Duration | Output |
|------|----------|--------|
| Implement LowerCBIntrinsics (reader) | 6h | Reader protocol insertion |
| Implement LowerCBIntrinsics (writer) | 4h | Writer protocol insertion |
| Test NOC protocol generation | 2h | Protocol tests |

#### Wednesday-Thursday: DST & Compute Init
| Task | Duration | Output |
|------|----------|--------|
| Implement InsertDSTManagementTT | 6h | DST lifecycle management |
| Implement InsertComputeInitTT | 4h | Engine initialization |
| Test compute protocol generation | 2h | Compute tests |

#### Friday: Integration
| Task | Duration | Output |
|------|----------|--------|
| Implement ConfigureTensorAccessorTT | 3h | Runtime binding |
| Wire up all protocol passes | 3h | Integrated pipeline |
| Debug integration issues | 2h | Fixes |

**Week 2 Deliverables:**
- LowerCBIntrinsics.py - NOC/CB protocol insertion
- InsertDSTManagementTT.py - DST lifecycle
- InsertComputeInitTT.py - Compute engine init
- ConfigureTensorAccessorTT.py - Accessor binding
- Protocol tests passing

---

### Week 3: Integration & Refinement

**Goal:** Fix existing passes and integrate everything

#### Monday-Tuesday: Fix Existing Passes
| Task | Duration | Output |
|------|----------|--------|
| Fix LowerSharedToCB (remove heuristics) | 6h | Protocol-less version |
| Fix LowerTTTileIntrinsics (remove "_tile") | 4h | Clean tensorization |
| Update GridToCoreGrid for new metadata | 2h | Updated pass |

#### Wednesday-Thursday: Runtime & Verification
| Task | Duration | Output |
|------|----------|--------|
| Update FinalizePersistentSignatureTT | 4h | Runtime args finalized |
| Update VerifyTTIR for new schema | 4h | Validation pass |
| Fix metadata flow issues | 4h | Clean metadata |

#### Friday: First End-to-End Test
| Task | Duration | Output |
|------|----------|--------|
| Run full pipeline on GEMM | 4h | First compilation |
| Debug and fix issues | 4h | Working pipeline |

**Week 3 Deliverables:**
- All existing passes updated for v5 design
- Complete pipeline assembled
- First successful GEMM compilation
- Generated C++ code structure correct

---

### Week 4: Testing & Stabilization

**Goal:** Comprehensive testing and bug fixes

#### Monday-Tuesday: Test Suite
| Task | Duration | Output |
|------|----------|--------|
| Add comprehensive GEMM tests | 4h | GEMM test suite |
| Add element-wise operation tests | 4h | Eltwise tests |
| Add edge case tests | 4h | Robustness tests |

#### Wednesday-Thursday: Bug Fixes
| Task | Duration | Output |
|------|----------|--------|
| Fix issues from testing | 8h | Stable passes |
| Add better error messages | 2h | Improved diagnostics |
| Add validation checks | 2h | Safety checks |

#### Friday: Documentation & Demo
| Task | Duration | Output |
|------|----------|--------|
| Update documentation | 3h | Current docs |
| Create demo notebook | 3h | Working examples |
| Team demo preparation | 2h | Demo ready |

**Week 4 Deliverables:**
- Comprehensive test suite (50+ tests)
- All major bugs fixed
- GEMM variants working (different sizes)
- Clear documentation
- Demo notebook with examples

---

## Daily Standup Template

```markdown
## Day X Standup

**Yesterday:**
- Completed: [Task] ✅
- Progress: [Task] (75%)

**Today:**
- [ ] Task 1 (2h)
- [ ] Task 2 (3h)
- [ ] Task 3 (2h)

**Blockers:**
- [Issue if any]

**Help Needed:**
- [Specific assistance required]
```

---

## Testing Milestones

### Week 1 Test
```python
def test_week1_milestone():
    """Kernel splitting works"""
    kernel = simple_matmul()
    result = SplitDeviceKernel()(kernel)
    assert len(result) == 3
```

### Week 2 Test
```python
def test_week2_milestone():
    """Protocol insertion works"""
    kernels = get_split_kernels()
    reader = LowerCBIntrinsics()(kernels.reader)
    compute = InsertDSTManagementTT()(kernels.compute)
    assert "noc_async_read" in str(reader)
    assert "dst.acquire" in str(compute)
```

### Week 3 Test
```python
def test_week3_milestone():
    """Full pipeline works"""
    result = run_full_pipeline(gemm_kernel)
    assert result.success
    assert len(result.kernels) == 3
```

### Week 4 Test
```python
def test_week4_milestone():
    """Multiple patterns work"""
    for pattern in [gemm_256, gemm_512, eltwise_add]:
        result = run_full_pipeline(pattern)
        assert result.success
```

---

## Quick Iteration Workflow

```bash
# Edit pass
vim tilelang/tenstorrent/passes/my_pass.py

# Test immediately (no compilation!)
pytest testing/python/tenstorrent/test_my_pass.py -v

# Interactive debugging
ipdb testing/python/tenstorrent/test_my_pass.py
```

---

## Communication Plan

### Daily Updates (Slack/Discord)
```
✅ Completed: SplitDeviceKernel basic structure
🚧 In Progress: CB ID assignment (75%)
🔴 Blocker: Need clarification on CB allocation strategy
📊 Tests: 5/8 passing
```

### Weekly Demo (Friday 4pm)
- Live compilation of current state
- Test results summary
- Next week priorities