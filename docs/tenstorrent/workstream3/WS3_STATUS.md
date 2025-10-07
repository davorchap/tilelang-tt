# Workstream 3 Status - TIR Transform Pipeline

**Last Updated:** 2025-10-07

## Overview

Workstream 3 focuses on transforming annotated PrimFuncs into TT-ready IR with:
- Persistent loops wrapping grid-style kernels
- CoreRangeSet mappings for Tensix topology
- Circular buffer allocations and memory space lowering
- Tile padding for non-32-multiple dimensions
- Tensorized matmul operations
- IR verification before codegen

**Goal:** Convert WS2-annotated IRModules into execution-ready TT IR that can be passed to WS4 codegen.

## Progress Summary

| Task | Status | Priority | Blocker |
|------|--------|----------|---------|
| **WS3 Documentation Structure** | ✅ **COMPLETE** | High | None |
| **GridToPersistentTT Pass** | ✅ **COMPLETE** | **Critical** | None |
| **TTShardToCoreMap Pass** | ⏭️ **DEFERRED** | Medium | Post-MVP |
| **MemorySpaceLowerTT Pass** | ⏭️ **DEFERRED** | Medium | Post-MVP |
| **TilePadTT Pass** | ⏭️ **DEFERRED** | Medium | Post-MVP |
| **TensorizeTT Pass** | ⏭️ **DEFERRED** | Medium | Post-MVP |
| **VerifyTTIR Pass** | ⏭️ **DEFERRED** | Medium | Post-MVP |
| **Pipeline Integration** | ✅ **COMPLETE** | High | None |
| **C++ Unit Tests** | ⏭️ **DEFERRED** | Low | Post-MVP |
| **Python Integration Tests** | ✅ **COMPLETE** | High | None |

**Overall WS3 Progress:** ✅ **Foundation Complete** (35% - Critical path implemented, remaining passes deferred to post-MVP)

---

## Implementation Plan

### Phase 1: GridToPersistentTT (Critical Foundation)

**File:** `src/transform/tt/grid_to_persistent.cc`

**Objective:** Transform grid-style kernel into persistent per-core loop.

**Input IR Pattern:**
```python
# Grid-style kernel (GPU paradigm)
with T.Kernel(grid_x=8, grid_y=8) as (bx, by):
    # ... kernel body using bx, by ...
```

**Output IR Pattern:**
```cpp
// Persistent per-core kernel (TT paradigm)
for (i = 0; i < tt_count; ++i) {
    tile_id = tt_start_id + i
    bx = tile_id % grid_x
    by = tile_id / grid_x
    // ... original kernel body ...
}
```

**Implementation Steps:**
1. Read schedule metadata from WS2 (`tt_tiles_per_core`, `tt_grid_x/y`)
2. Wrap kernel body with persistent loop: `for (i = 0; i < count; ++i)`
3. Compute `tile_id = start_id + i`
4. Replace symbolic `bx`, `by` with computed values: `bx = tile_id % grid_x`, `by = tile_id / grid_x`
5. Annotate function with `tt_runtime_args` schema
6. Remove original grid binding attributes

**Key Metadata:**
- Reads: `tt_tiles_per_core`, `tt_grid_x`, `tt_grid_y`, `tt_num_cores`
- Writes: `tt_runtime_args` schema for host invocation

**Dependencies:** WS2 complete (schedule metadata available)

---

### Phase 2: TTShardToCoreMap

**File:** `src/transform/tt/shard_to_core_map.cc`

**Objective:** Translate tile assignments to CoreRangeSet topology.

**Implementation Steps:**
1. Read `tt_num_cores` from function attributes
2. For MVP: Map to rectangular grid (8×8 for 64 cores)
3. Generate `CoreRangeSet` specification: `{start: {x:0, y:0}, end: {x:7, y:7}}`
4. Attach per-core runtime args arrays: `[(start_id, count), ...]` for each core
5. Generate `tt_core_ranges` attribute

**Key Metadata:**
- Reads: `tt_num_cores`, `tt_tiles_per_core`
- Writes: `tt_core_ranges`, `tt_core_runtime_args`

**Dependencies:** GridToPersistentTT (needs persistent loop structure)

---

### Phase 3: MemorySpaceLowerTT

**File:** `src/transform/tt/memory_space_lower.cc`

**Objective:** Lower shared/fragment buffers to TT circular buffers.

**Implementation Steps:**
1. Identify `T.alloc_fragment()` and `T.alloc_shared()` allocations
2. Replace with `tt.alloc_circular_buffer(size, depth=2)`
3. Lower `T.copy()` operations:
   - DRAM→L1: `tt.cb_wait_front(cb_in, n_tiles)` + `tt.cb_push_back(cb_in, n_tiles)`
   - L1→Compute: `tt.cb_reserve_back(cb_compute, n_tiles)` + tile ops + `tt.cb_push_back(cb_compute, n_tiles)`
   - Compute→DRAM: `tt.cb_wait_front(cb_out, n_tiles)` + write + `tt.cb_pop_front(cb_out, n_tiles)`
4. Attach `tt_circular_buffer` attributes: buffer index, tile size, depth

**Key Metadata:**
- Reads: Buffer allocation sites, copy operations
- Writes: `tt_cb_index`, `tt_cb_depth`, `tt_cb_tile_size`

**Dependencies:** GridToPersistentTT (needs IR structure)

---

### Phase 4: TilePadTT

**File:** `src/transform/tt/tile_pad.cc`

**Objective:** Handle non-tile-multiple dimensions with padding.

**Implementation Steps:**
1. Read `tt_buffer_{name}_needs_padding` from WS2
2. For padded buffers:
   - Insert pad operation at reader: zero-fill to `tt_buffer_{name}_padded_shape`
   - Adjust compute indices to use padded dimensions
   - Insert unpad operation at writer: slice back to original shape
3. Update buffer metadata with actual runtime dimensions

**Key Metadata:**
- Reads: `tt_buffer_{name}_needs_padding`, `tt_buffer_{name}_padded_shape`
- Writes: Padding/unpadding operations in IR

**Dependencies:** None (can run early or late)

---

### Phase 5: TensorizeTT

**File:** `src/transform/tt/tensorize_matmul.cc`

**Objective:** Replace matmul loop nests with TT tile intrinsics.

**Implementation Steps:**
1. Pattern match matmul loop nests: `for m, for n, for k: C[m,n] += A[m,k] * B[k,n]`
2. Verify loops operate on tile granularity (32×32)
3. Replace with: `tt.matmul_tiles(cb_a, cb_b, cb_c, m_tiles, n_tiles, k_tiles)`
4. Attach metadata: tile dimensions, circular buffer indices

**Key Metadata:**
- Reads: Loop structure, buffer circular buffer indices
- Writes: `tt_matmul_intrinsic` call

**Dependencies:** MemorySpaceLowerTT (needs circular buffers allocated)

---

### Phase 6: VerifyTTIR

**File:** `src/transform/tt/verify.cc`

**Objective:** Validate IR before codegen.

**Verification Checks:**
1. All required attributes present:
   - `tt_runtime_args` schema
   - `tt_core_ranges`
   - Circular buffer metadata
2. Persistent loop structure correct:
   - Loop variable ranges match `tt_tiles_per_core`
   - Block index recovery correct
3. Circular buffer invariants:
   - All CBs have valid indices
   - Depth values are reasonable (typically 2)
   - Push/pop pairs balanced
4. Memory access patterns valid:
   - All buffer accesses use circular buffer APIs
   - No direct DRAM access in compute kernel

**Dependencies:** All other passes (runs last)

---

### Phase 7: Pipeline Integration

**File:** `tilelang/tt/pipeline.py` (new)

**Objective:** Orchestrate all WS2 + WS3 passes in correct order.

**Pipeline Sequence:**
```python
def apply_tt_transform_pipeline(mod):
    # WS2: Metadata inference
    mod = infer_default_tt_schedule(mod)
    mod = infer_default_tt_shard(mod)

    # WS3: Transform pipeline
    mod = grid_to_persistent_tt(mod)
    mod = tt_shard_to_core_map(mod)
    mod = memory_space_lower_tt(mod)
    mod = tile_pad_tt(mod)  # Can run early or late
    mod = tensorize_tt(mod)
    mod = verify_tt_ir(mod)

    return mod
```

---

## Testing Strategy

### C++ Unit Tests (Per Pass)

Each transform gets dedicated C++ tests in `tests/cpp/tt/`:

1. **`test_grid_to_persistent.cc`:**
   - Test 8×8 grid → 64-core persistent loop
   - Verify block index recovery: `bx = tile_id % grid_x`
   - Assert runtime args schema correct

2. **`test_shard_to_core_map.cc`:**
   - Test CoreRangeSet generation for 64 cores
   - Verify per-core runtime args arrays
   - Test edge cases (non-power-of-2 core counts)

3. **`test_memory_space_lower.cc`:**
   - Test `T.copy()` → circular buffer intrinsics
   - Verify CB depth and tile size metadata
   - Test DRAM→L1→Compute→DRAM pipeline

4. **`test_tile_pad.cc`:**
   - Test padding for 100×100 matrix → 128×128
   - Verify pad/unpad operations inserted correctly
   - Test no-op for tile-aligned dimensions

5. **`test_tensorize.cc`:**
   - Test matmul loop nest → `tt.matmul_tiles()` intrinsic
   - Verify circular buffer indices passed correctly
   - Test various tile dimensions

6. **`test_verify.cc`:**
   - Test verification passes for valid IR
   - Test verification fails for invalid IR (missing attrs, unbalanced CB ops)

### Python Integration Tests

**File:** `testing/python/tt/test_tir_pipeline.py`

**Objective:** Test full WS1 → WS2 → WS3 pipeline on realistic GEMM.

**Test Cases:**
1. **test_full_pipeline_256x256:**
   - Create TileLang GEMM (256×256)
   - Apply WS1 defaults
   - Apply WS2 metadata inference
   - Apply WS3 transform pipeline
   - Verify final IR has:
     - Persistent loop structure
     - Circular buffer allocations
     - Matmul intrinsic
     - All required attributes

2. **test_pipeline_non_tile_aligned:**
   - Test 100×100 GEMM (requires padding)
   - Verify padding operations present
   - Verify compute operates on padded dimensions

3. **test_pipeline_various_grid_sizes:**
   - Test 4×4, 8×8, 16×16 grids
   - Verify persistent loops adjust correctly
   - Verify tile assignments correct

---

## Build & Test Instructions

### Building WS3 Passes

```bash
# Incremental build after adding each pass
USE_LLVM=true pip install -e . --no-build-isolation
```

### Running Tests

```bash
# C++ unit tests (once implemented)
cd build
ctest -R tt -V

# Python integration tests
source .venv/bin/activate
export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH
pytest testing/python/tt/test_tir_pipeline.py -v
```

---

## Dependencies

### External Dependencies
- **TVM C++ infrastructure:** Pass framework, IRModule manipulation
- **WS1 complete:** Target registration, default annotations
- **WS2 complete:** Schedule and sharding metadata

### Internal Dependencies
```
WS1 ✓ → WS2 ✓ → WS3 (current)
                  ├─→ GridToPersistentTT (foundation)
                  │    ├─→ TTShardToCoreMap
                  │    └─→ MemorySpaceLowerTT
                  │         └─→ TensorizeTT
                  ├─→ TilePadTT (independent)
                  └─→ VerifyTTIR (final gate)
```

---

## Success Criteria

WS3 is complete when:
- [ ] All 6 transform passes implemented and tested (C++)
- [ ] Pipeline integration implemented (Python)
- [ ] C++ unit tests pass for each transform
- [ ] Python integration test passes on representative GEMM
- [ ] `VerifyTTIR` pass validates MVP matmul IR
- [ ] All existing tests still pass (no regressions)
- [ ] Documentation complete

---

## Implementation Sequence

**Week 1: Foundation**
1. Implement GridToPersistentTT
2. Write C++ unit test
3. Implement Python binding
4. Test on simple 8×8 grid

**Week 2: Core Transforms**
1. Implement TTShardToCoreMap
2. Implement MemorySpaceLowerTT
3. Write C++ unit tests for both
4. Test integration

**Week 3: Refinements**
1. Implement TilePadTT
2. Implement TensorizeTT
3. Write C++ unit tests
4. Test on various grid sizes

**Week 4: Validation & Integration**
1. Implement VerifyTTIR
2. Create pipeline.py orchestration
3. Write Python integration tests
4. Full end-to-end testing
5. Documentation

---

## Key Design Decisions

### 1. Pass Ordering
**Decision:** GridToPersistent first, VerifyTTIR last, others can vary.
**Rationale:** Persistent loop is foundation; verification is final gate.

### 2. Circular Buffer Depth
**Decision:** Fixed depth=2 for MVP.
**Rationale:** Simplifies implementation; sufficient for double buffering.

### 3. Core Topology
**Decision:** Fixed 8×8 grid (64 cores) for MVP.
**Rationale:** Matches Grayskull/Wormhole; defer multi-device to post-MVP.

### 4. Padding Strategy
**Decision:** Reader-side zero-fill.
**Rationale:** Keeps compute kernel simple; no conditional logic needed.

---

## Next Steps - Workstream 4 Preview

Once WS3 is complete, **Workstream 4: Code Generation** will:
- Emit Metalium-compatible C++ kernels (reader/compute/writer)
- Generate host program with TensorAccessor setup
- Produce `tt.plan.json` with scheduling metadata
- Enable dry-run execution without hardware

**Cannot start until:** WS3 transform pipeline is complete and verified.

---

## Related Documentation

- [WS1 Status](../workstream1/WS1_STATUS.md) - Target registration
- [WS2 Status](../workstream2/WS2_STATUS.md) - Metadata inference
- [Unified MVP Plan](../UNIFIED_MATMUL_MVP_PLAN.md) - Overall TT backend MVP plan

---
