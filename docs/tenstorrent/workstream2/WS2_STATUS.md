# Workstream 2 Status - Schedule & Sharding Metadata

**Last Updated:** 2025-10-07

## Overview

Workstream 2 focuses on injecting TT schedule and sharding metadata that describes:
- Contiguous per-core tile ranges (schedule)
- DRAM interleaved tilization (sharding)
- Runtime argument schemas for kernel execution

**Goal:** Enable the TileLang frontend to automatically compute and attach TT-specific metadata to IRModules, preparing them for the TIR transform pipeline (WS3).

## Progress Summary

| Task | Status | Priority | Blocker |
|------|--------|----------|---------|
| **WS2 Documentation Structure** | ⚙️ **IN PROGRESS** | **High** | None |
| Schedule Inference Pass (C++) | ❌ TODO | High | Documentation |
| Sharding Inference Pass (C++) | ❌ TODO | High | Schedule pass |
| Python Bindings | ❌ TODO | High | C++ passes |
| C++ Unit Tests | ❌ TODO | High | C++ passes |
| Python Integration Tests | ❌ TODO | Medium | Python bindings |

**Overall WS2 Progress:** ~0% complete (Planning phase)

---

## Workstream 2 Tasks

### Task 1: Schedule Inference Pass
**File:** `src/tt/transform/infer_tt_schedule.cc`
**Status:** TODO

**What it does:**
- Reads `T.Kernel(grid_x, grid_y)` metadata from PrimFunc
- Computes total tiles: `grid_x * grid_y`
- Partitions tiles across available Tensix cores
- Generates contiguous per-core ranges: `(start_id, count)` per core
- Attaches `tt.schedule` attributes with runtime arg schemas

**Key metadata produced:**
```python
# Example attributes attached to PrimFunc:
func.attrs["tt_schedule_policy"] = "contiguous"
func.attrs["tt_schedule_order"] = "row_major"
func.attrs["tt_num_tiles"] = grid_x * grid_y
func.attrs["tt_tiles_per_core"] = [...]  # Array of (start_id, count) per core
func.attrs["tt_runtime_args_schema"] = {...}  # Schema for kernel runtime args
```

**Inputs:**
- PrimFunc with `T.Kernel` grid dimensions
- Default annotations from WS1 (`tt_schedule_policy="contiguous"`, etc.)

**Outputs:**
- Enhanced PrimFunc with per-core scheduling metadata
- Runtime argument schemas for host-side kernel invocation

**Dependencies:**
- WS1 complete (default annotations present)
- TVM C++ pass infrastructure
- Understanding of TT-metal core topology

**Acceptance criteria:**
- C++ unit test passes: creates correct tile ranges for various grid sizes
- Metadata validates correctly for 1x1, 8x8, and 16x16 grids

---

### Task 2: Sharding Inference Pass
**File:** `src/tt/transform/infer_tt_shard.cc`
**Status:** TODO

**What it does:**
- Generates DRAM interleaved tensor descriptors
- References TensorAccessor stride rules for interleaved layout
- Marks tensors with non-32-multiple dimensions for padding
- Attaches `tt.shard` attributes describing memory layout

**Key metadata produced:**
```python
# Example attributes for buffer parameters:
buffer.attrs["tt_layout"] = "dram_interleaved"
buffer.attrs["tt_tile_shape"] = (32, 32)
buffer.attrs["tt_needs_padding"] = True/False  # If dims not tile-multiple
buffer.attrs["tt_tensor_accessor_config"] = {...}  # TensorAccessor metadata
```

**Inputs:**
- PrimFunc with buffer parameters (A, B, C for GEMM)
- Default layout annotations from WS1 (`tt_layout_type="dram_interleaved"`)

**Outputs:**
- Enhanced buffers with sharding/layout metadata
- Padding requirements for non-tile-multiple shapes

**Dependencies:**
- Schedule inference pass (needs tile counts)
- TT-metal TensorAccessor header understanding
- Knowledge of DRAM interleaving scheme

**Acceptance criteria:**
- C++ unit test passes: generates correct interleaved descriptors
- Correctly identifies padding needs for non-32-multiple matrices
- Metadata aligns with TT-metal TensorAccessor expectations

---

### Task 3: Python Bindings
**File:** `tilelang/tt/__init__.py` (update)
**Status:** TODO

**What it does:**
- Exposes C++ passes to Python via TVM FFI
- Allows passes to be invoked from Python test/pipeline code
- Registers passes with TVM's pass infrastructure

**Implementation:**
```python
# tilelang/tt/__init__.py additions:
from .target import apply_tt_defaults
from .passes import infer_default_tt_schedule, infer_default_tt_shard

__all__ = [
    "apply_tt_defaults",
    "infer_default_tt_schedule",
    "infer_default_tt_shard",
]
```

**Dependencies:**
- C++ passes implemented and registered with TVM

**Acceptance criteria:**
- Can import and call passes from Python
- Passes work on test IRModules

---

### Task 4: C++ Unit Tests
**Files:** `tests/cpp/tt/test_infer_tt_schedule.cc`, `tests/cpp/tt/test_infer_tt_shard.cc`
**Status:** TODO

**What they test:**
- Schedule inference on synthetic PrimFuncs with various grid sizes
- Shard inference on buffers with different shapes (tile-aligned, non-aligned)
- Correct metadata attachment
- Edge cases (1x1 grid, very large grids, prime-number dimensions)

**Test strategy:**
- Use `TVM_REGISTER_GLOBAL` to invoke passes
- Create minimal PrimFuncs with known parameters
- Assert generated attributes match expected values
- Use `tvm::support::AsText` for IR comparisons

**Dependencies:**
- C++ passes implemented
- TVM C++ test infrastructure

**Acceptance criteria:**
- All C++ unit tests pass
- Coverage of edge cases

---

### Task 5: Python Integration Tests
**File:** `tests/python/tt/test_inferred_metadata.py`
**Status:** TODO

**What it tests:**
- End-to-end: Apply WS1 defaults → Run WS2 passes → Verify metadata
- Use real TileLang GEMM example
- Assert schedule and shard metadata present and correct
- Verify metadata format matches expectations for WS3 passes

**Test flow:**
```python
# Pseudo-code
mod = create_tilelang_gemm(M=256, N=256, K=256)
mod = apply_tt_defaults(mod)  # WS1
mod = infer_default_tt_schedule(mod)  # WS2
mod = infer_default_tt_shard(mod)  # WS2

# Verify metadata
func = mod["main"]
assert "tt_num_tiles" in func.attrs
assert "tt_tiles_per_core" in func.attrs
# ... more assertions
```

**Dependencies:**
- Python bindings complete
- WS1 integration working

**Acceptance criteria:**
- Python integration test passes
- Metadata validates for representative GEMM sizes

---

## Testing Strategy

### C++ Tests (Unit Level)
- Test each pass in isolation with synthetic IR
- Focus on correctness of metadata generation
- Cover edge cases and boundary conditions

### Python Tests (Integration Level)
- Test pass pipeline on realistic TileLang examples
- Verify metadata format and content
- Ensure WS1 → WS2 integration works correctly

### CI Integration
- Add C++ tests to existing TVM test suite
- Add Python tests to `testing/python/tt/`
- Ensure tests run in Tenstorrent CI workflow

---

## Build & Test Instructions

### Building C++ Passes (Once Implemented)

**Via local build script:**
```bash
bash maint/scripts/local_build_and_test_tt.sh --skip-deps --jobs 4
```

**Manual build:**
```bash
mkdir -p build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build . -j 4
```

### Running C++ Tests

```bash
cd build
ctest -R tt -V  # Run TT-specific C++ tests
```

### Running Python Tests

```bash
source .venv/bin/activate
export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH
pytest testing/python/tt/test_inferred_metadata.py -v
```

---

## Dependencies

### External Dependencies
- **TVM C++ infrastructure:** Pass registration, IRModule manipulation
- **TT-metal headers (optional):** For TensorAccessor reference (may defer to WS4 codegen)
- **CMake/Ninja:** Build system

### Internal Dependencies
- **WS1 complete:** Default annotations must be present before inference
- **Understanding of TT architecture:** Core topology, tile size, interleaved layout

---

## Dependency Graph

```
WS1 Complete (Default Annotations)
    ↓
    └─→ WS2 Documentation (IN PROGRESS)
         ↓
         ├─→ C++ Build Setup
         │    ↓
         │    ├─→ Schedule Inference Pass
         │    │    ↓
         │    │    └─→ C++ Unit Tests (Schedule)
         │    │
         │    └─→ Sharding Inference Pass
         │         ↓
         │         └─→ C++ Unit Tests (Shard)
         │
         └─→ Python Bindings
              ↓
              └─→ Python Integration Tests
                   ↓
                   └─→ WS2 Complete ✅
```

---

## Key Design Decisions

### 1. Contiguous Schedule Only (MVP)
**Decision:** Only implement contiguous, row-major tile assignment for MVP.
**Rationale:** Simplifies implementation; advanced schedules (strided, rectangular) deferred to post-MVP.

### 2. DRAM Interleaved Layout
**Decision:** Use TT-metal TensorAccessor with interleaved layout.
**Rationale:** Avoids manual address swizzling; leverages TT runtime support.

### 3. Separate Passes for Schedule and Shard
**Decision:** Implement as two distinct TVM passes rather than one combined pass.
**Rationale:**
- Clearer separation of concerns
- Easier testing and debugging
- Allows future extension (e.g., different sharding strategies)

### 4. Pure Metadata (No Code Generation Yet)
**Decision:** WS2 passes only attach metadata; no code generation.
**Rationale:** Code generation happens in WS4; keeping WS2 focused on metadata inference.

---

## Success Criteria

WS2 is complete when:
- ✅ Schedule inference pass implemented and tested (C++)
- ✅ Sharding inference pass implemented and tested (C++)
- ✅ Python bindings expose both passes
- ✅ C++ unit tests pass for both passes
- ✅ Python integration test passes on representative GEMM
- ✅ CI runs all WS2 tests successfully
- ✅ Documentation updated to reflect WS2 completion

---

## Next Steps - Workstream 3 Preview

Once WS2 is complete, **Workstream 3: TIR Transform Pipeline** will:
- Transform annotated IR into TT-ready IR with persistent loops
- Implement GridToPersistentTT pass
- Add memory space lowering for circular buffers
- Implement tensorization for matmul operations

**Cannot start until:** WS2 metadata inference is complete and tested.

---

## Related Documentation

- [WS1 Status](../workstream1/WS1_STATUS.md) - Prerequisite frontend integration
- [Project Plan](../project_1.md) - Overall TT backend MVP plan
- [Local Build Guide](../local_build_guide.md) - Build and test instructions

---

## Questions or Issues?

- Check [CLAUDE.md](../../CLAUDE.md) for development workflow
- Review [project_1.md](../project_1.md) for overall architecture
- WS2 task breakdown documents in this directory (once created)
