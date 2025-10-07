# Workstream 2: Schedule & Sharding Metadata

This directory contains documentation and planning materials for **Workstream 2** of the Tenstorrent backend MVP.

## Overview

**Goal:** Inject TT schedule and sharding metadata that describes contiguous per-core tile ranges and DRAM interleaved tilization.

**Status:** 📝 Planning phase - documentation complete, implementation TODO

## What WS2 Delivers

Two TVM C++ passes that automatically compute and attach metadata:

1. **Schedule Inference** (`InferDefaultTTSchedule`)
   - Computes per-core tile assignments
   - Generates runtime argument schemas
   - Implements contiguous row-major partitioning

2. **Sharding Inference** (`InferDefaultTTShard`)
   - Generates DRAM interleaved layout descriptors
   - Identifies padding requirements for non-tile-aligned dimensions
   - Prepares metadata for TensorAccessor configuration

## Documents in This Directory

| Document | Purpose | Status |
|----------|---------|--------|
| [WS2_STATUS.md](WS2_STATUS.md) | Overall progress tracking | ✅ Complete |
| [ws2_schedule_inference.md](ws2_schedule_inference.md) | Schedule inference pass specification | ✅ Complete |
| [ws2_shard_inference.md](ws2_shard_inference.md) | Sharding inference pass specification | ✅ Complete |
| [ws2_python_integration.md](ws2_python_integration.md) | Python bindings and integration tests | ✅ Complete |

## Prerequisites

- ✅ **WS1 Complete** - Default annotations must be present
- Understanding of TVM pass infrastructure
- Familiarity with TT-metal core topology and memory architecture

## Implementation Roadmap

### Phase 1: C++ Infrastructure
1. Set up `src/tt/transform/` directory
2. Update CMakeLists.txt for TT transform sources
3. Investigate TVM pass registration and examples

### Phase 2: Schedule Inference
1. Implement `InferDefaultTTSchedule` pass
2. Add tile partitioning logic (contiguous row-major)
3. Create C++ unit tests
4. Verify on test grids (1x1, 8x8, 16x16)

### Phase 3: Sharding Inference
1. Implement `InferDefaultTTShard` pass
2. Add padding detection logic
3. Create C++ unit tests
4. Verify on aligned and non-aligned matrices

### Phase 4: Python Integration
1. Add Python bindings in `tilelang/tt/passes.py`
2. Create `test_inferred_metadata.py` integration tests
3. Test full WS1 + WS2 pipeline
4. Update CI to run WS2 tests

## Testing Strategy

### C++ Unit Tests
- **Location:** `tests/cpp/tt/`
- **Files:**
  - `test_infer_tt_schedule.cc`
  - `test_infer_tt_shard.cc`
- **Approach:** Synthetic PrimFuncs, assert metadata correctness

### Python Integration Tests
- **Location:** `testing/python/tt/`
- **File:** `test_inferred_metadata.py`
- **Approach:** Real TileLang examples, end-to-end pipeline validation

## Key Design Decisions

1. **Two separate passes** (not one combined) - clearer separation of concerns
2. **Contiguous schedule only** (MVP) - advanced schedules deferred
3. **DRAM interleaved layout** - leverages TT-metal TensorAccessor
4. **Metadata only** (WS2) - no code generation yet (deferred to WS4)

## Dependencies

### WS2 Depends On
- WS1: Default annotation helper (must run first)

### WS2 Enables
- WS3: TIR transform pipeline (needs metadata from WS2)
- WS4: Code generation (uses metadata for kernel emission)

## Success Criteria

WS2 is complete when:
- ✅ Both C++ passes implemented and tested
- ✅ Python bindings functional
- ✅ All unit and integration tests pass
- ✅ CI successfully runs WS2 test suite
- ✅ Documentation updated

## Quick Start (Future)

Once implemented, the WS2 pipeline will be:

```python
from tilelang.tt import apply_tt_defaults, infer_default_tt_schedule, infer_default_tt_shard

# Create TileLang GEMM
mod = create_gemm_module(M=256, N=256, K=256)

# WS1: Apply defaults
mod = apply_tt_defaults(mod)

# WS2: Infer metadata
mod = infer_default_tt_schedule(mod)  # Schedule
mod = infer_default_tt_shard(mod)     # Sharding

# Verify metadata
func = mod["main"]
assert "tt_num_tiles" in func.attrs
assert "tt_tiles_per_core" in func.attrs
# ... ready for WS3 transforms
```

## Related Documentation

- [WS1 Status](../workstream1/WS1_STATUS.md) - Prerequisite frontend integration
- [Unified MVP Plan](../UNIFIED_MATMUL_MVP_PLAN.md) - Overall TT backend MVP architecture
- [CLAUDE.md](../../CLAUDE.md) - Development workflow and build instructions

## Questions?

See [WS2_STATUS.md](WS2_STATUS.md) for detailed progress tracking and next steps.
