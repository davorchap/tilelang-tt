# Next Steps: Post IR-Driven Codegen Migration

**Date**: 2025-10-08
**Status**: Planning Document
**Context**: IR-driven codegen migration complete (Tasks 1-6), all 95 tests passing

---

## Current State

### ✅ Completed (2025-10-08)

**Phase 2 IR-Driven Codegen (Weeks 11-15)**:
- ✅ Task 1: IR Visitor Base Class (12 tests)
- ✅ Task 2: Compute Kernel Visitor (4 tests)
- ✅ Task 3: Reader Kernel Visitor (1 test)
- ✅ Task 4: Writer Kernel Visitor (1 test)
- ✅ Task 5: Integration + Bug Fixes
- ✅ Task 6: Test Updates (18 tests)
- ✅ WS2 Grid Calculation Bug Fix
- ✅ **All 95 tests passing (100% pass rate)**

**Previous Milestones**:
- ✅ Phase 1 MVP: Template-based dry-run (23 tests)
- ✅ Phase 2 WS3-Extended: 5 deferred transforms (TTShardToCoreMap, MemorySpaceLowerTT, TilePadTT, TensorizeTT, VerifyTTIR)
- ✅ WS1-3: Target registration, metadata inference, transforms (56 tests)

### Current Architecture

**Code Generation**: IR-Driven ✅
- Walks actual TIR body structure
- Generates code from IR nodes (ForNode, AttrStmt, etc.)
- Supports arbitrary kernels (not just matmul)
- Extensible visitor pattern

**Runtime**: Mock Metalium APIs ⚠️
- Dry-run only (no hardware execution)
- Template functions return void
- No actual NOC operations or CB management

**Test Coverage**: 95 tests
- Unit tests for all components
- Integration tests for pipeline
- MVP acceptance tests

---

## Roadmap: Weeks 16-26

### Phase: Real Metalium API Integration (Weeks 16-18)

**Goal**: Replace mock APIs with actual Metalium runtime calls for hardware execution.

#### Week 16-17: Metalium API Research & Planning

**Tasks**:
1. **Study Metalium API Documentation**
   - Review TT-Metalium reference docs (https://docs.tenstorrent.com)
   - Understand device API: CreateDevice, CloseDevice
   - Study program API: CreateProgram, EnqueueProgram
   - Learn kernel API: circular buffers, NOC operations
   - Review example code in TT-Metalium repo

2. **API Mapping Specification**
   - Map mock APIs to real Metalium equivalents:
     - `get_arg_val<T>()` → Metalium runtime args
     - `cb_reserve_back()` → Real CB API
     - `cb_push_back()` → Real CB API
     - `cb_wait_front()` → Real CB API
     - `cb_pop_front()` → Real CB API
     - `noc_async_read_tile()` → Real NOC read
     - `noc_async_write_tile()` → Real NOC write
     - `matmul_tiles()` → Real matmul intrinsic

3. **Create Metalium Integration Plan**
   - Document: `docs/tenstorrent/METALIUM_INTEGRATION_PLAN.md`
   - Define phases of integration
   - Identify dependencies (Metalium version, headers, libraries)
   - Plan test strategy (simulator first, then hardware)

**Deliverables**:
- Metalium API mapping document
- Integration plan with timeline
- Dependency list

#### Week 18: Host Program Integration

**Tasks**:
1. **Update Host Program Codegen** (`src/target/tt/codegen_tt.cc`)
   - Replace mock device setup with real `tt::tt_metal::CreateDevice()`
   - Implement real program creation with `tt::tt_metal::CreateProgram()`
   - Add real CB configuration with `CircularBufferConfig`
   - Implement runtime argument passing
   - Add program launch with `EnqueueProgram()`

2. **Create Build System Integration**
   - Add Metalium headers to CMakeLists.txt
   - Link against Metalium libraries
   - Handle version compatibility
   - Add feature flag: `USE_REAL_METALIUM=ON/OFF`

3. **Update Tests**
   - Create `test_metalium_integration.py`
   - Simulator-based tests (if available)
   - Skip hardware tests if device not available

**Deliverables**:
- Host program generates real Metalium code
- Build system supports both mock and real Metalium
- Basic integration tests passing

---

### Phase: Hardware Execution & Validation (Weeks 19-22)

#### Week 19-20: Kernel API Integration

**Tasks**:
1. **Update Compute Kernel Codegen**
   - Replace mock matmul with real SFPU calls
   - Use real CB reserve/push/pop/wait APIs
   - Handle L1 memory addressing correctly

2. **Update Reader/Writer Kernels**
   - Replace mock NOC APIs with real NOC read/write
   - Implement proper tile addressing
   - Handle DRAM-L1 data movement

3. **CB Configuration**
   - Generate proper CB config based on buffer sizes
   - Handle dataflow vs compute kernel differences
   - Implement L1 memory allocation

**Deliverables**:
- All three kernels use real Metalium APIs
- Code compiles against real Metalium
- Ready for hardware testing

#### Week 21: Hardware Testing Setup

**Tasks**:
1. **Access Hardware**
   - Acquire Grayskull or Wormhole device
   - Install drivers and runtime
   - Verify basic Metalium examples work

2. **Create Hardware Test Suite**
   - Port existing tests to hardware
   - Add device availability checks
   - Create CI infrastructure for hardware tests

3. **Debug Infrastructure**
   - Set up logging and tracing
   - Create debugging utilities
   - Implement error handling

**Deliverables**:
- Hardware test environment ready
- Basic matmul running on hardware
- Debug tools in place

#### Week 22: Validation & Performance

**Tasks**:
1. **Correctness Testing**
   - Validate output against reference implementation
   - Test various matrix sizes (64x64 to 4096x4096)
   - Test edge cases (non-square, odd dimensions)

2. **Performance Benchmarking**
   - Measure TFLOPS for matmul
   - Compare against theoretical peak
   - Profile bottlenecks

3. **Bug Fixes**
   - Address any hardware-specific issues
   - Fix synchronization problems
   - Optimize data movement

**Deliverables**:
- Correctness validated on hardware
- Performance benchmarks documented
- Known issues catalogued

---

### Phase: Optimization & Polish (Weeks 23-26)

#### Week 23-24: Optimization

**Tasks**:
1. **Performance Optimization**
   - Optimize CB sizes and layout
   - Improve NOC traffic patterns
   - Tune core utilization

2. **Code Quality**
   - Refactor codegen for clarity
   - Add comprehensive error handling
   - Improve code documentation

3. **Additional Kernels**
   - Implement flash attention kernel
   - Add reduce operations
   - Support element-wise ops

**Deliverables**:
- Optimized matmul performance
- Additional kernel types working
- Clean, maintainable code

#### Week 25: Documentation

**Tasks**:
1. **User Documentation**
   - Getting started guide
   - API reference
   - Example kernels

2. **Developer Documentation**
   - Architecture deep-dive
   - Codegen internals
   - Testing guide

3. **Performance Guide**
   - Optimization best practices
   - Profiling techniques
   - Troubleshooting

**Deliverables**:
- Complete documentation set
- Tutorial examples
- Performance guide

#### Week 26: Release Preparation

**Tasks**:
1. **Final Testing**
   - Full regression suite
   - Hardware validation
   - Performance verification

2. **Release Artifacts**
   - Version tagging
   - Release notes
   - Binary distributions

3. **Community Preparation**
   - Example repository
   - Blog post / announcement
   - Support infrastructure

**Deliverables**:
- v1.0 release ready
- Documentation published
- Community launch

---

## Immediate Next Steps (This Week)

### Priority 1: Metalium API Research
- [ ] Access TT-Metalium documentation
- [ ] Review reference implementations
- [ ] Understand device/program/kernel API hierarchy
- [ ] Document API surface we need

### Priority 2: Create Integration Plan
- [ ] Write `METALIUM_INTEGRATION_PLAN.md`
- [ ] Define mock→real API mappings
- [ ] Identify build dependencies
- [ ] Plan testing strategy

### Priority 3: Build System Prep
- [ ] Research Metalium build requirements
- [ ] Identify library dependencies
- [ ] Plan CMake integration
- [ ] Define feature flags (USE_REAL_METALIUM)

---

## Success Criteria

### Week 18 Milestone:
- [ ] Host program generates real Metalium device setup
- [ ] Build system supports real Metalium libraries
- [ ] Basic integration tests pass (simulator or mock device)

### Week 22 Milestone:
- [ ] Matmul executes correctly on Grayskull/Wormhole hardware
- [ ] Achieves >50% of theoretical peak TFLOPS
- [ ] All existing tests pass with real runtime

### Week 26 Milestone:
- [ ] v1.0 release with hardware support
- [ ] 100+ tests passing (hardware + unit)
- [ ] Documentation complete
- [ ] Ready for community use

---

## Risk Mitigation

**Risk 1: Metalium API Complexity**
- **Mitigation**: Incremental integration, start with simplest APIs
- **Fallback**: Keep mock mode for development

**Risk 2: Hardware Access**
- **Mitigation**: Use simulator if available, coordinate with TT team
- **Fallback**: Develop against mock device, validate later

**Risk 3: Performance Issues**
- **Mitigation**: Profile early and often, consult TT experts
- **Fallback**: Document known limitations, optimize in v1.1

**Risk 4: API Compatibility**
- **Mitigation**: Version pin Metalium, abstract API surface
- **Fallback**: Support multiple Metalium versions

---

## Resources Needed

**Documentation**:
- TT-Metalium API reference
- Device architecture guides
- Example kernel implementations

**Hardware**:
- Grayskull or Wormhole device (at least 1)
- Access to TT cloud if local device unavailable

**Support**:
- TT engineering contact for API questions
- Access to TT GitHub issues/discussions
- Community support channels

**Tools**:
- Metalium SDK with headers/libraries
- Device drivers and runtime
- Profiling and debugging tools

---

## Document History

- 2025-10-08: Initial planning document (post IR-driven codegen completion)
