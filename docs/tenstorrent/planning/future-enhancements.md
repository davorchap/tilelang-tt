# TileLang Tenstorrent Future Enhancements

**Document Version:** 1.0
**Date:** 2025-10-17
**Status:** v5 Complete - Planning Future Work

---

## Overview

This document tracks planned enhancements for the TileLang Tenstorrent backend. The v5 pipeline (14 passes, Python-only) is complete and production-ready. This document focuses exclusively on future work.

---

## High Priority

### 1. LowerToSFPU Pass (Python Implementation)

- **Goal**: Support T.Parallel (threadIdx) constructs for intra-tile SIMD operations
- **Status**: Design phase
- **Priority**: MEDIUM
- **Location**: To be created as `tilelang/tenstorrent/passes/lower_to_sfpu.py`
- **Documentation**: [passes/lower_to_sfpu.md](../passes/lower_to_sfpu.md)
- **Note**: Python implementation following v5 architecture

### 2. Hardware Validation

- **Goal**: Validate generated code on real Tenstorrent devices
- **Status**: Blocked (awaiting hardware access)
- **Priority**: HIGH
- **Dependencies**: SDK access, device availability
- **Tasks**:
  - SDK-backed CI workflow
  - Performance profiling
  - API validation
  - End-to-end execution tests

---

## Medium Priority

### 3. Optimization Passes

- **Goal**: Improve generated code performance
- **Priority**: MEDIUM
- **Tasks**:
  - CB allocation sharing
  - Tile reuse optimization
  - Overlapped execution optimization
  - L1 memory layout optimization

### 4. Enhanced Diagnostics

- **Goal**: Better error messages and validation
- **Priority**: MEDIUM
- **Tasks**:
  - Improved halo hint error messages
  - L1 capacity check enhancements
  - Better N-D sharding validation
  - Pass-level diagnostics framework

---

## Low Priority

### 5. Advanced Sharding Features

- **Goal**: Support more sharding configurations
- **Priority**: LOW
- **Tasks**:
  - Block linear order support
  - Halo exchange support (currently rejected)
  - Dynamic sharding configurations
  - Multi-device sharding

### 6. Example Updates

- **Goal**: Showcase v5 pipeline capabilities
- **Priority**: LOW
- **Tasks**:
  - Update `examples/tenstorrent/` with v5 patterns
  - Add sharding examples
  - Add custom annotation examples
  - Performance tuning examples

---

## Success Criteria

### Hardware Validation

- [ ] SDK-backed tests passing on real devices
- [ ] Performance benchmarks established
- [ ] API validation complete

### LowerToSFPU

- [ ] T.Parallel (threadIdx) constructs supported
- [ ] SFPU operations mapped correctly
- [ ] Test suite for intra-tile operations

### Optimization

- [ ] CB allocation sharing implemented
- [ ] Tile reuse optimization working
- [ ] Performance improvements measured

### Documentation

- [ ] All 14 passes documented individually
- [ ] Stage-based documentation complete
- [ ] Python implementation patterns guide
- [ ] Debugging guide

---

## Reference Documentation

### Architecture
- [v5_pipeline.md](../architecture/v5_pipeline.md) - Authoritative v5 pipeline reference (800+ lines)
- [TT_ARCHITECTURE.md](../architecture/TT_ARCHITECTURE.md) - Complete backend architecture
- [GPU_vs_Tenstorrent_Analysis.md](../architecture/GPU_vs_Tenstorrent_Analysis.md) - GPU vs TT comparison

### Pass Documentation
- [passes/README.md](../passes/README.md) - Pass documentation index
- Individual pass docs in `docs/tenstorrent/passes/stages/`

### Planning
- [sdk-validation-plan.md](./sdk-validation-plan.md) - SDK validation phases

### Setup
- [local_build_guide.md](../setup/local_build_guide.md) - Local build instructions
- [CI.md](../setup/CI.md) - CI/CD workflows
- [METALIUM_SETUP_GUIDE.md](../setup/METALIUM_SETUP_GUIDE.md) - SDK setup

---

**Last Updated**: 2025-10-17
**Maintainer**: TileLang Tenstorrent Team
**Status**: Planning Future Enhancements
