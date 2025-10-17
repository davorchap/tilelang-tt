# Tenstorrent Backend Tasks

**Document Version:** 4.0
**Date:** 2025-10-16
**Status:** v5 Complete - Future Enhancements

---

## Overview

This document captures the task tracker for the TileLang Tenstorrent (TT) backend. **The v5 pipeline is complete and production-ready.** This document now focuses on future enhancements and remaining work items.

### v5 Pipeline Status (Complete)

✅ **v5 metadata-driven pipeline**: 14 Python passes in stages A-E provide canonical `tt.buffer.*`, `tt.cb.*`, and `tt.runtime_args` metadata
✅ **Shard-aware persistent lowering**: Grid-to-core transformation with canonical runtime schema
✅ **Python-only architecture**: All TT backend passes in Python (no C++ migration planned)
✅ **Mock-mode CI**: Primary validation path with 120 passing tests (85.1% pass rate)
✅ **Old pipeline removed**: Legacy 5-pass pipeline deleted (PR #135)

### Architecture Snapshot

- **v5 Pipeline**: 14 passes organized in stages A-E (Metadata → Partitioning → Protocol-less → Late Split → Finalization)
- **Python Implementation**: All passes in `tilelang/tenstorrent/passes/` for maintainability and rapid iteration
- **C++ Codegen**: IR-driven visitors in `src/target/tenstorrent/` generate 5 artifacts (reader.cpp, compute.cpp, writer.cpp, main.cpp, tt.plan.json)
- **Test Coverage**: 120 tests passing, 21 skipped (TVM bugs, hardware-specific features)

### Out of Scope (Current Implementation)

- **T.Parallel (threadIdx) support**: Requires LowerToSFPU pass (planned Python implementation)
- **Halo exchange**: Currently rejected, needs future design
- **Dynamic sharding**: Current implementation uses static shard configurations
- **Hardware validation**: Pending SDK access for real device testing

---

## Quick Status (v5 Pipeline)

| Component | Status | Location | Documentation |
|-----------|--------|----------|---------------|
| **Stage A: Metadata** | | | |
| infer_tt_layout_v5 | ✅ Complete | `tilelang/tenstorrent/passes/infer_tt_layout_v5.py` | [📄 v5_pipeline.md#stage-a](../architecture/v5_pipeline.md) |
| propagate_tt_layout_v5 | ✅ Complete | `tilelang/tenstorrent/passes/propagate_tt_layout_v5.py` | [📄 v5_pipeline.md#stage-a](../architecture/v5_pipeline.md) |
| attach_tensor_accessor_tt | ✅ Complete | `tilelang/tenstorrent/passes/attach_tensor_accessor_tt.py` | [📄 v5_pipeline.md#stage-a](../architecture/v5_pipeline.md) |
| **Stage B: Partitioning** | | | |
| layout_aware_work_partition_tt_v5 | ✅ Complete | `tilelang/tenstorrent/passes/layout_aware_work_partition_tt_v5.py` | [📄 v5_pipeline.md#stage-b](../architecture/v5_pipeline.md) |
| grid_to_core_grid_v5 | ✅ Complete | `tilelang/tenstorrent/passes/grid_to_core_grid_v5.py` | [📄 v5_pipeline.md#stage-b](../architecture/v5_pipeline.md) |
| **Stage C: Protocol-less** | | | |
| lower_shared_to_cb_v5 | ✅ Complete | `tilelang/tenstorrent/passes/lower_shared_to_cb_v5.py` | [📄 v5_pipeline.md#stage-c](../architecture/v5_pipeline.md) |
| lower_tt_tile_intrinsics_v5 | ✅ Complete | `tilelang/tenstorrent/passes/lower_tt_tile_intrinsics_v5.py` | [📄 v5_pipeline.md#stage-c](../architecture/v5_pipeline.md) |
| build_tile_dfg_tt | ✅ Complete | `tilelang/tenstorrent/passes/build_tile_dfg_tt.py` | [📄 v5_pipeline.md#stage-c](../architecture/v5_pipeline.md) |
| **Stage D: Late Split & Protocol** | | | |
| split_device_kernel | ✅ Complete | `tilelang/tenstorrent/passes/split_device_kernel.py` | [📄 v5_pipeline.md#stage-d](../architecture/v5_pipeline.md) |
| configure_tensor_accessor_tt | ✅ Complete | `tilelang/tenstorrent/passes/configure_tensor_accessor_tt.py` | [📄 v5_pipeline.md#stage-d](../architecture/v5_pipeline.md) |
| lower_cb_intrinsics | ✅ Complete | `tilelang/tenstorrent/passes/lower_cb_intrinsics.py` | [📄 v5_pipeline.md#stage-d](../architecture/v5_pipeline.md) |
| insert_compute_init_tt | ✅ Complete | `tilelang/tenstorrent/passes/insert_compute_init_tt.py` | [📄 v5_pipeline.md#stage-d](../architecture/v5_pipeline.md) |
| insert_dst_management_tt | ✅ Complete | `tilelang/tenstorrent/passes/insert_dst_management_tt.py` | [📄 v5_pipeline.md#stage-d](../architecture/v5_pipeline.md) |
| **Stage E: Finalization** | | | |
| finalize_persistent_signature_tt | ✅ Complete | `tilelang/tenstorrent/passes/finalize_persistent_signature_tt.py` | [📄 v5_pipeline.md#stage-e](../architecture/v5_pipeline.md) |
| **Verification** | | | |
| verify_tt_ir | ✅ Complete | `tilelang/tenstorrent/passes/verify_tt_ir.py` | [📄 passes/verify_tt_ir.md](../passes/verify_tt_ir.md) |

---

## Completed Work (Historical)

### Phase 0: v5 Pipeline Default (✅ Complete - PR #135)

**Goal**: Make v5 pipeline the default and remove old pipeline

**Completed Tasks:**
- ✅ Made v5 pipeline the default implementation
- ✅ Removed old 5-pass pipeline (1,245 lines of implementation)
- ✅ Deleted deprecated test files (1,882 lines)
- ✅ Deleted old documentation (superseded by v5 docs)
- ✅ Updated all documentation to reflect v5 as current reality
- ✅ Created comprehensive v5_pipeline.md reference (800+ lines)

**Artifacts:**
- PR #135: Old pipeline removal
- PR #136: Week 1 documentation cleanup
- Test suite: 120 passing, 21 skipped (85.1% pass rate)

### Phase 1: v5 Pass Implementation (✅ Complete)

**Goal**: Implement all 14 v5 passes

**Completed Tasks:**
- ✅ Stage A: Metadata (A1-A3 passes)
- ✅ Stage B: Partitioning (B1-B2 passes)
- ✅ Stage C: Protocol-less Lowering (C1-C3 passes)
- ✅ Stage D: Late Split & Protocol (D1-D5 passes)
- ✅ Stage E: Finalization (E1 pass)
- ✅ IR-driven codegen visitors (reader, compute, writer, host, plan)
- ✅ Per-core runtime metadata tables in host artifacts
- ✅ TensorAccessor metadata for deterministic addressing

**Artifacts:**
- All 14 passes implemented in Python
- Codegen visitors in C++ (`src/target/tenstorrent/`)
- Comprehensive test suite

### Phase 2: Documentation (🚧 In Progress)

**Goal**: Complete documentation cleanup and create comprehensive references

**Completed (Week 1):**
- ✅ Deleted old documentation (superseded by v5 docs)
- ✅ Main README.md updated to reflect v5
- ✅ TT_ARCHITECTURE.md updated with 14-pass v5 pipeline
- ✅ Created authoritative v5_pipeline.md reference (800+ lines)

**In Progress (Week 2):**
- 🚧 Planning docs cleanup (TT_Pass_Status.md, TT_BACKEND_TASKS.md)
- 🚧 Stage-based pass documentation
- 🚧 passes/README.md update with v5 navigation

**Remaining:**
- ⏸️ Individual pass documentation (11 of 14 passes need docs)
- ⏸️ Python implementation patterns guide
- ⏸️ Debugging and troubleshooting guide

---

## Future Enhancements

### High Priority

**1. LowerToSFPU Pass (Python Implementation)**
- **Goal**: Support T.Parallel (threadIdx) constructs for intra-tile SIMD operations
- **Status**: Design phase
- **Priority**: MEDIUM
- **Location**: To be created as `tilelang/tenstorrent/passes/lower_to_sfpu.py`
- **Documentation**: [📄 passes/lower_to_sfpu.md](../passes/lower_to_sfpu.md)
- **Note**: Python implementation following v5 architecture

**2. Hardware Validation**
- **Goal**: Validate generated code on real Tenstorrent devices
- **Status**: Blocked (awaiting hardware access)
- **Priority**: HIGH
- **Dependencies**: SDK access, device availability
- **Tasks**:
  - SDK-backed CI workflow
  - Performance profiling
  - API validation
  - End-to-end execution tests

### Medium Priority

**3. Optimization Passes**
- **Goal**: Improve generated code performance
- **Priority**: MEDIUM
- **Tasks**:
  - CB allocation sharing
  - Tile reuse optimization
  - Overlapped execution optimization
  - L1 memory layout optimization

**4. Enhanced Diagnostics**
- **Goal**: Better error messages and validation
- **Priority**: MEDIUM
- **Tasks**:
  - Improved halo hint error messages
  - L1 capacity check enhancements
  - Better N-D sharding validation
  - Pass-level diagnostics framework

### Low Priority

**5. Advanced Sharding Features**
- **Goal**: Support more sharding configurations
- **Priority**: LOW
- **Tasks**:
  - Block linear order support
  - Halo exchange support (currently rejected)
  - Dynamic sharding configurations
  - Multi-device sharding

**6. Example Updates**
- **Goal**: Showcase v5 pipeline capabilities
- **Priority**: LOW
- **Tasks**:
  - Update `examples/tenstorrent/` with v5 patterns
  - Add sharding examples
  - Add custom annotation examples
  - Performance tuning examples

---

## Success Criteria (v5 - Achieved ✅)

- ✅ v5 pipeline with 14 passes implemented
- ✅ All passes in Python (no C++ migration)
- ✅ Old pipeline removed
- ✅ Test suite: 120 passing (85.1% pass rate)
- ✅ Mock-mode CI validates `@tilelang.jit` entry point
- ✅ Comprehensive documentation (v5_pipeline.md, TT_ARCHITECTURE.md)
- ✅ Layout-aware metadata pipeline operational
- ✅ Shard-aware persistent lowering working
- ✅ IR-driven codegen generating 5 artifacts

---

## Success Criteria (Future Enhancements)

**Hardware Validation:**
- [ ] SDK-backed tests passing on real devices
- [ ] Performance benchmarks established
- [ ] API validation complete

**LowerToSFPU:**
- [ ] T.Parallel (threadIdx) constructs supported
- [ ] SFPU operations mapped correctly
- [ ] Test suite for intra-tile operations

**Optimization:**
- [ ] CB allocation sharing implemented
- [ ] Tile reuse optimization working
- [ ] Performance improvements measured

**Documentation:**
- [ ] All 14 passes documented individually
- [ ] Stage-based documentation complete
- [ ] Python implementation patterns guide
- [ ] Debugging guide

---

## Timeline & Milestones

| Phase | Status | Completion Date | Notes |
|-------|--------|-----------------|-------|
| Phase 0: v5 Default | ✅ Complete | 2025-10-16 | PR #135 merged |
| Phase 1: v5 Implementation | ✅ Complete | 2025-10-15 | All 14 passes working |
| Phase 2: Documentation (Week 1) | ✅ Complete | 2025-10-16 | PR #136 merged |
| Phase 2: Documentation (Week 2) | 🚧 In Progress | 2025-10-17 (est) | Reference updates |
| LowerToSFPU | ⏸️ Planned | TBD | Awaiting design approval |
| Hardware Validation | ⏸️ Blocked | TBD | Awaiting SDK access |

---

## Reference Documentation

### Architecture
- [v5_pipeline.md](../architecture/v5_pipeline.md) - Authoritative v5 pipeline reference (800+ lines)
- [TT_ARCHITECTURE.md](../architecture/TT_ARCHITECTURE.md) - Complete backend architecture
- [IR_LOWERING_ANALYSIS.md](../architecture/IR_LOWERING_ANALYSIS.md) - GPU vs TT comparison

### Pass Documentation
- [passes/README.md](../passes/README.md) - Pass documentation index
- Individual pass docs in `docs/tenstorrent/passes/stages/`

### Planning
- [METALIUM_SDK_VALIDATION_PLAN.md](./METALIUM_SDK_VALIDATION_PLAN.md) - SDK validation phases

### Setup
- [local_build_guide.md](../setup/local_build_guide.md) - Local build instructions
- [CI.md](../setup/CI.md) - CI/CD workflows
- [METALIUM_SETUP_GUIDE.md](../setup/METALIUM_SETUP_GUIDE.md) - SDK setup

---

## Changelog

### Version 4.0 (2025-10-16)
- Complete rewrite for v5 completion
- Reorganized into Completed Work and Future Enhancements
- Updated all pass references to v5
- Removed C++ migration references
- Added comprehensive status table

### Version 3.0 (2025-10-11)
- Added v5 pipeline status
- Updated Phase 2 with C++ migration note

### Version 2.0 (2025-10-08)
- Original consolidation plan

---

**Last Updated**: 2025-10-16
**Maintainer**: TileLang Tenstorrent Team
**Status**: ✅ v5 Complete - Future Enhancements Planned
