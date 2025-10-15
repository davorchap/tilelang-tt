# Tenstorrent Backend Implementation Plan

**Version:** 1.0
**Date:** 2025-10-15
**Status:** Active Implementation Plan
**Based on:** TileLang TT TIR Lowering Guide v5

---

## Executive Summary

This plan defines the complete implementation roadmap for the TileLang Tenstorrent backend based on the progressive lowering approach with late protocol insertion. The backend transforms high-level TileLang DSL to Tenstorrent TT-Metalium API code through a series of well-defined passes that maintain analyzable TIR as long as possible before inserting device-specific protocols.

### Key Design Principles

1. **Progressive Lowering**: Early passes attach metadata and restructure loops; late passes insert device protocols
2. **Protocol-less Mid-level TIR**: Compute expressed over CB-backed buffers without CB handshakes or DST logic
3. **Late Protocolization**: After kernel split, inject NOC/CB sequences, DST management, and compute engine init
4. **Persistent Kernels**: Materialize per-core tile loops during codegen, not as TIR `for` loops
5. **First-class Layout/Sharding**: Rich buffer layout attributes with interleaved DRAM as default

---

## Pass Pipeline Overview

The complete pipeline consists of 17 passes organized into 7 stages:

| Stage | Passes | Purpose | Status |
|-------|--------|---------|--------|
| **A: Metadata** | A1-A3 | Layout inference & CB geometry | 🟡 Python impl |
| **B: Partitioning** | B1-B2 | Core assignment & grid transform | 🟡 Partial |
| **C: Protocol-less** | C1-C3 | CB allocation & compute lowering | 🟡 Partial |
| **D: Late Split** | D1-D5 | Kernel split & protocol insertion | 🔴 New |
| **E: Finalization** | E1 | Runtime args & signature | 🟡 Partial |
| **F: Verification** | F | Constraint checking | ✅ Basic |
| **G: Codegen** | G | Emit C++ kernels + host | ✅ Working |

---

## Stage A: Early Metadata Passes (A1-A3)

### A1: InferTTLayout
**Purpose**: Normalize buffer defaults and explicit annotations
**Status**: 🟡 Python implementation exists, C++ port needed

#### Specification
```python
# Input: Buffer declarations with optional annotations
@annotate_tt_layout(A, memory="DRAM", layout="sharded",
                    nd_shard={"axes": ["M","N"], "grid": [2,4]})
def func(A: T.Buffer((256, 256), "bf16"), ...):
    ...

# Output: Standardized tt.layout_desc attributes
"tt.layout_desc[A]": {
    "memory": "DRAM",           # Default: DRAM
    "layout": "sharded",         # Default: interleaved
    "tile_shape": [32, 32],      # Default: 32×32
    "dtype": "bf16",             # Default: bf16
    "nd_shard": {
        "axes": ["M", "N"],
        "grid": [2, 4],
        "shard_shape_elems": [128, 64],
        "order": "row_major",
        "align_tiles": true,
        "projected_grid": [2, 4],
        "projected_shard_tiles": [4, 2]
    }
}
```

#### Implementation Tasks
- [ ] **A1.1**: Port Python `infer_tt_layout.py` to C++ `infer_tt_layout.cc`
- [ ] **A1.2**: Add ND sharding validation and projection logic
- [ ] **A1.3**: Implement L1 capacity and alignment checks
- [ ] **A1.4**: Add halo metadata rejection with clear diagnostics

### A2: PropagateTTLayout
**Purpose**: Derive conceptual CB geometry from layout
**Status**: 🟡 Python implementation exists, C++ port needed

#### Specification
```python
# Input: tt.layout_desc from A1
# Output: tt.cb_desc attributes
"tt.cb_desc[cb_in0]": {
    "page_size": 2048,      # bf16 × 32×32 = 2KB
    "depth": 2,             # Default double-buffering
    "data_format": "Float16_b"
}
```

#### Implementation Tasks
- [ ] **A2.1**: Port Python `propagate_tt_layout.py` to C++
- [ ] **A2.2**: Implement page size calculation from dtype and tile shape
- [ ] **A2.3**: Add configurable depth with L1 memory validation

### A3: AttachTensorAccessorTT
**Purpose**: Attach abstract TensorAccessor descriptors
**Status**: 🔴 New pass needed

#### Specification
```python
# Output: tt.tensor_accessor attributes (compile-time only)
"tt.tensor_accessor[A]": {
    "type": "abstract",
    "layout_ref": "tt.layout_desc[A]",
    "stride_mode": "tiled",
    "base_offset": null,  # Filled during D2
    "runtime_arg_idx": null  # Filled during D2
}
```

#### Implementation Tasks
- [ ] **A3.1**: Create new pass `attach_tensor_accessor_tt.cc`
- [ ] **A3.2**: Link accessors to layout descriptors
- [ ] **A3.3**: Prepare accessor schema for late binding

---

## Stage B: Work Partitioning (B1-B2)

### B1: LayoutAwareWorkPartitionTT
**Purpose**: Choose core grid and work partition
**Status**: 🟡 Python driver exists, C++ port needed

#### Specification
```python
# Output: PrimFunc attributes
"tt.core_grid": [8, 8],
"tt.core_ranges": [[0,0,7,7]],
"tt.partition_mode": "global",  # or "local_shard"
"tt.grid_tiles": [8, 8],  # Mt, Nt
"tt.work_partition": {
    "core_0_0": [(0,0), (0,1), ...],  # Tile assignments
    ...
}
```

#### Implementation Tasks
- [ ] **B1.1**: Port to C++ with canonical attribute output
- [ ] **B1.2**: Implement global vs local_shard mode selection
- [ ] **B1.3**: Add balanced work distribution algorithm
- [ ] **B1.4**: Generate runtime_args template

### B2: GridToCoreGrid
**Purpose**: Convert grid loops to core launch
**Status**: ✅ Implemented as `grid_to_persistent_tt.cc`

#### Specification
```python
# Before
with T.Kernel(8, 8) as (bx, by):
    C[bx*32:(bx+1)*32, by*32:(by+1)*32] = ...

# After
cx = T.launch_core("coreIdx.x", 8)
cy = T.launch_core("coreIdx.y", 8)
# Body remains intact for now
```

#### Implementation Tasks
- [ ] **B2.1**: Update to consume new metadata format
- [ ] **B2.2**: Add axis mapping attributes (tt.core_map_i/j)

---

## Stage C: Protocol-less Lowering (C1-C3)

### C1: LowerSharedToCB
**Purpose**: Convert shared memory to conceptual CBs
**Status**: 🟡 Partial in `memory_space_lower_tt.cc`

#### Specification
```python
# Before
A_sh = T.alloc_shared((128, 32), "bf16")
T.copy(A[...], A_sh)

# After (protocol-less)
A_cb = tt.alloc_cb("cb_in0", (128, 32), "bf16")
T.evaluate(tt.read_to_cb(A[...], A_cb))  # Abstract, no NOC
```

#### Implementation Tasks
- [ ] **C1.1**: Refactor to emit abstract intrinsics only
- [ ] **C1.2**: Remove premature CB ID assignment
- [ ] **C1.3**: Keep CB allocations conceptual

### C2: LowerTTTileIntrinsics
**Purpose**: Tensorize compute to CB-based intrinsics
**Status**: 🟡 Exists as `lower_gemm_to_tt_intrinsics.cc`

#### Specification
```python
# Before
T.gemm(A_cb, B_cb, C_cb)

# After (protocol-less)
T.evaluate(tt.mm.mma(A_cb, B_cb, dst=0, accumulate=False))
```

#### Implementation Tasks
- [ ] **C2.1**: Remove CB ID heuristics ("_tile" suffix)
- [ ] **C2.2**: Emit protocol-less compute intrinsics
- [ ] **C2.3**: Add pattern tags for later passes

### C3: BuildTileDFGTT
**Purpose**: Build tile dataflow graph
**Status**: 🔴 New pass needed

#### Specification
```python
# Output: tt.tile_dfg attribute
"tt.tile_dfg": {
    "nodes": ["cb_in0", "cb_in1", "compute", "cb_out0"],
    "edges": [
        ("cb_in0", "compute"),
        ("cb_in1", "compute"),
        ("compute", "cb_out0")
    ],
    "cb_roles": {
        "cb_in0": "input_a",
        "cb_in1": "input_b",
        "cb_out0": "output"
    }
}
```

#### Implementation Tasks
- [ ] **C3.1**: Create new pass `build_tile_dfg_tt.cc`
- [ ] **C3.2**: Analyze CB producer-consumer relationships
- [ ] **C3.3**: Prepare metadata for kernel split

---

## Stage D: Late Protocol Insertion (D1-D5)

### D1: SplitDeviceKernel
**Purpose**: Split monolithic kernel into reader/compute/writer
**Status**: 🔴 New pass needed (critical)

#### Specification
```python
# Input: Monolithic kernel with conceptual CBs
# Output: 3 PrimFuncs with roles

@T.prim_func(attrs={"tt.kernel_role": "reader"})
def gemm_reader(A, B): ...

@T.prim_func(attrs={"tt.kernel_role": "compute"})
def gemm_compute(): ...

@T.prim_func(attrs={"tt.kernel_role": "writer"})
def gemm_writer(C): ...
```

#### Implementation Tasks
- [ ] **D1.1**: Create new pass `split_device_kernel_tt.cc`
- [ ] **D1.2**: Clone and slice TIR by dataflow role
- [ ] **D1.3**: Assign CB IDs (≤32 total)
- [ ] **D1.4**: Update kernel signatures

### D2: ConfigureTensorAccessorTT
**Purpose**: Bind TensorAccessorArgs slots
**Status**: 🔴 New pass needed

#### Specification
```python
# Update tt.tensor_accessor with runtime binding
"tt.tensor_accessor[A]": {
    "type": "bound",
    "runtime_arg_idx": 0,
    "tile_size_bytes": 2048,
    "base_offset": "get_arg_val<uint32_t>(0)"
}
```

#### Implementation Tasks
- [ ] **D2.1**: Create new pass `configure_tensor_accessor_tt.cc`
- [ ] **D2.2**: Assign runtime arg slots for reader/writer
- [ ] **D2.3**: Calculate tile addressing parameters

### D3: LowerCBIntrinsics
**Purpose**: Insert NOC/CB protocol for reader/writer
**Status**: 🔴 New pass needed (critical)

#### Specification
```python
# Before (abstract)
T.evaluate(tt.read_to_cb(A[...], "cb_in0"))

# After (protocol)
T.evaluate(cb_reserve_back("cb_in0", 1))
T.evaluate(noc_async_read_tile(tile_id, A_accessor, get_write_ptr("cb_in0")))
T.evaluate(noc_async_read_barrier())
T.evaluate(cb_push_back("cb_in0", 1))
```

#### Implementation Tasks
- [ ] **D3.1**: Create new pass `lower_cb_intrinsics_tt.cc`
- [ ] **D3.2**: Implement reader NOC protocol insertion
- [ ] **D3.3**: Implement writer NOC protocol insertion
- [ ] **D3.4**: Handle double-buffering patterns

### D4: InsertComputeInitTT
**Purpose**: Insert compute engine initialization
**Status**: 🔴 New pass needed

#### Specification
```python
# Insert before compute loops
T.evaluate(tt.engine.init_common("cb_in0", "cb_in1", "cb_out"))
T.evaluate(tt.fpu.matmul_init("cb_in0", "cb_in1", "cb_out"))
```

#### Implementation Tasks
- [ ] **D4.1**: Create new pass `insert_compute_init_tt.cc`
- [ ] **D4.2**: Detect compute patterns (matmul, binary, unary)
- [ ] **D4.3**: Insert appropriate init calls

### D5: InsertDSTManagementTT
**Purpose**: Wrap compute with DST lifecycle
**Status**: 🔴 New pass needed (critical)

#### Specification
```python
# K-loop pattern
T.evaluate(tt.dst.acquire())
for kt in T.serial(Kt):
    T.evaluate(cb_wait_front("cb_in0", 1))
    T.evaluate(tt.mm.mma("cb_in0", "cb_in1", dst=0, accumulate=(kt>0)))
    T.evaluate(cb_pop_front("cb_in0", 1))
T.evaluate(cb_reserve_back("cb_out", 1))
T.evaluate(tt.dst.commit())
T.evaluate(tt.dst.wait())
T.evaluate(pack_tile(dst=0, cb="cb_out", tile_index=0))
T.evaluate(tt.dst.release())
T.evaluate(cb_push_back("cb_out", 1))
```

#### Implementation Tasks
- [ ] **D5.1**: Create new pass `insert_dst_management_tt.cc`
- [ ] **D5.2**: Detect K-loop vs per-tile patterns
- [ ] **D5.3**: Insert DST acquire/commit/release
- [ ] **D5.4**: Add pack_tile calls

---

## Stage E: Finalization (E1)

### E1: FinalizePersistentSignatureTT
**Purpose**: Freeze runtime args and kernel signatures
**Status**: 🟡 Partial implementation

#### Specification
```python
# Global mode
"tt.runtime_args": ["start_id", "count", "Mt", "Kt", "Nt"]

# Local shard mode
"tt.runtime_args": ["start_id", "count", "Mt", "Kt", "Nt",
                    "Sm", "Sn", "Gy", "Gx", "sy", "sx"]
```

#### Implementation Tasks
- [ ] **E1.1**: Standardize runtime arg ordering
- [ ] **E1.2**: Add shard-specific parameters
- [ ] **E1.3**: Update codegen to consume canonical args

---

## Stage F: Verification (F)

### F: VerifyTTIR
**Purpose**: Validate constraints and metadata
**Status**: ✅ Basic implementation exists

#### Implementation Tasks
- [ ] **F.1**: Validate CB count (≤32)
- [ ] **F.2**: Check L1 memory capacity
- [ ] **F.3**: Verify dtype support
- [ ] **F.4**: Validate metadata completeness

---

## Stage G: Codegen (G)

### G: CodegenTT
**Purpose**: Emit C++ kernels and host
**Status**: ✅ Working with IR-driven visitors

#### Implementation Tasks
- [ ] **G.1**: Update to consume new metadata format
- [ ] **G.2**: Generate persistent loop in codegen (not TIR)
- [ ] **G.3**: Emit metadata summary in main.cpp
- [ ] **G.4**: Generate tt.plan.json

---

## Implementation Priority Matrix

### Phase 1: Critical Path (Weeks 1-2)
**Goal**: Enable end-to-end T.gemm compilation

| Priority | Task | Dependencies | Owner |
|----------|------|--------------|-------|
| P0 | D1: SplitDeviceKernel | C3 | TBD |
| P0 | D3: LowerCBIntrinsics | D1 | TBD |
| P0 | D5: InsertDSTManagementTT | D1 | TBD |
| P1 | C3: BuildTileDFGTT | C2 | TBD |
| P1 | D2: ConfigureTensorAccessorTT | D1 | TBD |
| P1 | D4: InsertComputeInitTT | D1 | TBD |

### Phase 2: C++ Migration (Weeks 3-4)
**Goal**: Port Python passes to C++

| Priority | Task | Dependencies | Owner |
|----------|------|--------------|-------|
| P0 | A1: Port InferTTLayout to C++ | - | TBD |
| P0 | A2: Port PropagateTTLayout to C++ | A1 | TBD |
| P0 | B1: Port LayoutAwareWorkPartitionTT | A1, A2 | TBD |
| P1 | A3: AttachTensorAccessorTT | A1 | TBD |
| P2 | Cleanup: Remove legacy passes | All above | TBD |

### Phase 3: Polish & Documentation (Week 5)
**Goal**: Finalize and document

| Priority | Task | Dependencies | Owner |
|----------|------|--------------|-------|
| P1 | Update all pass documentation | Phase 1-2 | TBD |
| P1 | Update example kernels | Phase 1-2 | TBD |
| P2 | Performance tuning | Phase 1-2 | TBD |

---

## Testing Strategy

### Unit Tests (Per Pass)
Each pass needs comprehensive unit tests:

```python
# Test structure for each pass
def test_pass_basic():
    """Test basic functionality"""

def test_pass_edge_cases():
    """Test boundary conditions"""

def test_pass_error_handling():
    """Test error cases and diagnostics"""

def test_pass_metadata_propagation():
    """Test metadata flow"""
```

### Integration Tests

#### Level 1: Pass Pipeline Tests
- Test pass combinations (A1→A2→A3, etc.)
- Verify metadata flow through pipeline
- Check IR transformations

#### Level 2: Kernel Pattern Tests
- GEMM patterns (various sizes)
- Element-wise operations
- Mixed compute patterns

#### Level 3: End-to-End Tests
- Full compilation from DSL to C++
- Artifact validation
- Mock execution verification

### Test Matrix

| Pattern | Sizes | Layouts | Modes | Priority |
|---------|-------|---------|-------|----------|
| GEMM | 256×256, 512×512, 1024×1024 | Interleaved, Sharded | Global, Local | P0 |
| Eltwise Add | 256×256, 512×512 | Interleaved | Global | P1 |
| Conv2D | 32×32×64 | Sharded | Local | P2 |
| Reduction | 256×1 | Interleaved | Global | P2 |

---

## Validation Checkpoints

### Checkpoint 1: Protocol-less TIR (After Stage C)
- [ ] CBs allocated conceptually
- [ ] Compute uses abstract intrinsics
- [ ] No protocol calls present
- [ ] Dataflow graph built

### Checkpoint 2: Split Kernels (After D1)
- [ ] 3 separate PrimFuncs created
- [ ] Kernel roles assigned
- [ ] CB IDs assigned (≤32)
- [ ] Signatures updated

### Checkpoint 3: Protocolized TIR (After Stage D)
- [ ] Reader has NOC/CB protocol
- [ ] Compute has engine init + DST
- [ ] Writer has NOC/CB protocol
- [ ] Runtime args finalized

### Checkpoint 4: Generated Code (After Stage G)
- [ ] reader.cpp compiles
- [ ] compute.cpp compiles
- [ ] writer.cpp compiles
- [ ] main.cpp has correct setup
- [ ] tt.plan.json valid

---

## Risk Mitigation

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| CB ID exhaustion (>32) | High | Early validation in C3 |
| L1 memory overflow | High | Capacity checks in A1 |
| Protocol ordering bugs | Medium | Strict templates in D3-D5 |
| Metadata inconsistency | Medium | Validation at each stage |

### Schedule Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| C++ port complexity | High | Start with minimal viable port |
| SDK API changes | Medium | Abstract behind interface layer |
| Test coverage gaps | Medium | Incremental testing per pass |

---

## Success Metrics

### Functional Metrics
- [ ] 100% of T.gemm kernels compile
- [ ] All 95 existing tests pass
- [ ] New protocol passes have >80% test coverage

### Quality Metrics
- [ ] Zero heuristics in compute path
- [ ] Deterministic CB assignment
- [ ] Clear error messages for violations

### Performance Metrics (Future)
- [ ] Generated code matches hand-written performance
- [ ] CB double-buffering utilized effectively
- [ ] Minimal L1 memory waste

---

## Appendix A: Intrinsic Reference

### Abstract Intrinsics (Protocol-less)
```python
tt.alloc_cb(name, shape, dtype)
tt.read_to_cb(tensor_slice, cb)
tt.write_from_cb(cb, tensor_slice)
tt.mm.mma(cb_a, cb_b, dst, accumulate)
tt.fpu.add(cb_x, cb_y, dst)
tt.sfpu.unary(op, dst)
```

### Protocol Intrinsics (Late insertion)
```python
# Engine init
tt.engine.init_common(cb_in0, cb_in1, cb_out)
tt.fpu.matmul_init(cb_a, cb_b, cb_out)
tt.sfpu.init(op, cb_in, cb_out)

# DST lifecycle
tt.dst.acquire()
tt.dst.commit()
tt.dst.wait()
tt.dst.release()
pack_tile(dst, cb, tile_index)

# NOC/CB protocol
cb_reserve_back(cb, num_tiles)
cb_push_back(cb, num_tiles)
cb_wait_front(cb, num_tiles)
cb_pop_front(cb, num_tiles)
get_write_ptr(cb)
get_read_ptr(cb)
noc_async_read_tile(tile_id, accessor, ptr)
noc_async_write_tile(tile_id, accessor, ptr)
noc_async_read_barrier()
noc_async_write_barrier()
```

---

## Appendix B: Metadata Schema

### PrimFunc Attributes
```json
{
  "tt.core_grid": [8, 8],
  "tt.core_ranges": [[0,0,7,7]],
  "tt.partition_mode": "global|local_shard",
  "tt.grid_tiles": [Mt, Nt],
  "tt.work_partition": {...},
  "tt.runtime_args": [...],
  "tt.tile_dfg": {...},
  "tt.cb_desc": {...},
  "tt.kernel_role": "monolithic|reader|compute|writer"
}
```

### Buffer Attributes
```json
{
  "tt.layout_desc": {
    "memory": "DRAM|L1",
    "layout": "interleaved|sharded",
    "tile_shape": [32, 32],
    "dtype": "bf16|fp16|fp32",
    "nd_shard": {...}
  },
  "tt.tensor_accessor": {
    "type": "abstract|bound",
    "layout_ref": "tt.layout_desc[name]",
    "runtime_arg_idx": null|int,
    "tile_size_bytes": int
  }
}
```

---

**Document Version:** 1.0
**Last Updated:** 2025-10-15
**Next Review:** After Phase 1 completion