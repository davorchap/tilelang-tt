# Tenstorrent Backend Implementation Timeline (Python-First)

**Version:** 2.0
**Date:** 2025-10-15
**Duration:** 4 weeks (Phase 1) + Optional Phase 2
**Status:** Rapid Prototyping Focus

---

## Executive Summary

This revised timeline focuses on **rapid Python prototyping** to achieve a working GEMM compilation in 4 weeks. C++ migration is deferred to an optional Phase 2 after the passes are stable and proven.

### Key Changes
- **Phase 1 (Weeks 1-4):** All passes in Python, focus on GEMM working end-to-end
- **Phase 2 (Optional):** Selective C++ migration for performance
- **Reduced complexity:** No Python/C++ boundary during development
- **Faster iteration:** ~3-5 day reduction in timeline

---

## Phase 1: Python Sprint (4 Weeks)

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
```python
✅ BuildTileDFGTT.py - Dataflow graph construction
✅ SplitDeviceKernel.py - Monolithic → 3 kernels
✅ AttachTensorAccessorTT.py - Tensor accessor metadata
✅ Basic test suite with simple patterns
```

**Validation Checkpoint:**
```python
# Should work by end of Week 1
kernel = create_simple_gemm()
dfg = BuildTileDFGTT()(kernel)
kernels = SplitDeviceKernel()(kernel)
assert len(kernels) == 3
assert kernels[0].attrs["tt.kernel_role"] == "reader"
```

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
```python
✅ LowerCBIntrinsics.py - NOC/CB protocol insertion
✅ InsertDSTManagementTT.py - DST lifecycle
✅ InsertComputeInitTT.py - Compute engine init
✅ ConfigureTensorAccessorTT.py - Accessor binding
✅ Protocol tests passing
```

**Validation Checkpoint:**
```python
# Should work by end of Week 2
pipeline = [
    SplitDeviceKernel(),
    LowerCBIntrinsics(),
    InsertDSTManagementTT(),
    InsertComputeInitTT(),
]
result = apply_pipeline(kernel, pipeline)
assert contains_noc_protocol(result.reader)
assert contains_dst_protocol(result.compute)
```

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
```python
✅ All existing passes updated for v5 design
✅ Complete pipeline assembled
✅ First successful GEMM compilation
✅ Generated C++ code structure correct
```

**Validation Checkpoint:**
```python
# Should work by end of Week 3
full_pipeline = get_tt_pass_pipeline_v5()
gemm = create_gemm_kernel(256, 256, 256)
result = apply_pipeline(gemm, full_pipeline)
code = CodegenTT()(result)
assert validate_generated_code(code)
print("🎉 First GEMM compiled successfully!")
```

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
```python
✅ Comprehensive test suite (50+ tests)
✅ All major bugs fixed
✅ GEMM variants working (different sizes)
✅ Clear documentation
✅ Demo notebook with examples
```

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

## Risk Management (Simplified)

| Risk | Probability | Impact | Mitigation |
|------|------------|---------|------------|
| Protocol ordering bugs | Medium | High | Test incrementally |
| Integration complexity | Medium | Medium | Daily integration tests |
| Missing edge cases | Low | Low | Add tests as found |

### Escalation Triggers
- **Day 3 blocker:** Escalate if any pass takes >2 days
- **Week 2 integration:** Escalate if protocols don't compose
- **Week 3 milestone:** Escalate if GEMM doesn't compile

---

## Phase 2: Optional C++ Migration (Weeks 5-7)

### Only If Needed for Performance

**Week 5-6: Selective Migration**
- Identify bottleneck passes through profiling
- Migrate only 2-3 critical passes
- Maintain Python versions for reference

**Week 7: Integration**
- Wire C++ passes into pipeline
- Performance validation
- Documentation update

**Decision Criteria for Migration:**
- Pass takes >100ms for typical kernel
- Pass is on critical path
- Pass is stable (no design changes expected)

---

## Success Criteria (Phase 1)

### Functional
- [x] All 17 passes implemented in Python
- [x] GEMM compiles end-to-end
- [x] Generated code matches expected structure
- [x] Mock execution validates correctness

### Quality
- [x] Test coverage >70%
- [x] Clear error messages
- [x] Documentation for each pass

### Performance (Relaxed for Python)
- [x] Compilation <5s for 256×256 GEMM
- [x] Memory usage <1GB
- [x] Iteration time <1min for changes

---

## Tools & Environment Setup

### Required Setup (Day 0)
```bash
# Install TVM with Python bindings
pip install -e .

# Install dev dependencies
pip install pytest ipdb notebook matplotlib

# Setup test environment
export TILELANG_TT_TEST_MODE=mock
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Quick Iteration Workflow
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

---

## Advantages Over C++ First Approach

| Aspect | Python-First | C++-First | Advantage |
|--------|--------------|-----------|-----------|
| **Iteration Speed** | <1 min | 5-10 min | **10x faster** |
| **Debugging** | Interactive (ipdb) | gdb/print | **Much easier** |
| **Testing** | pytest, immediate | cmake + gtest | **5x faster** |
| **Learning Curve** | Low | High | **More contributors** |
| **Refactoring** | Trivial | Time-consuming | **Agile changes** |

---

## Conclusion

This Python-first approach will deliver a **working GEMM in 4 weeks** vs 6-8 weeks for C++-first. The key is to:

1. **Focus on correctness** over performance
2. **Test incrementally** after each pass
3. **Defer optimization** until it works
4. **Migrate to C++ selectively** if needed

**Expected Outcome:** By end of Week 4, we'll have a fully functional TT backend that compiles GEMM correctly, with the option to optimize performance in Phase 2.

---

**Document Version:** 2.0
**Last Updated:** 2025-10-15
**Start Date:** Immediate
**Review:** Daily standups, Weekly demos