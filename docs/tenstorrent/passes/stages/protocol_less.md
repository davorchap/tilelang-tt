# Stage C: Protocol-less Lowering

**Stage:** C (Protocol-less Lowering)
**Passes:** 3 (C1-C3)
**Purpose:** Lower to abstract tile operations without NOC/CB/DST protocol

---

## Overview

Stage C lowers high-level operations to TT tile intrinsics while remaining protocol-free. The actual NOC/CB/DST protocol is inserted later in Stage D. This enables clean separation between:
- **What to compute** (Stage C)
- **How to coordinate** (Stage D)

---

## Pass Pipeline

```
Stage B Output (Persistent loop with partition metadata)
    ↓
C1: lower_shared_to_cb_v5 (Shared mem → Abstract CBs)
    ↓
C2: lower_tt_tile_intrinsics_v5 (Tile ops → TT intrinsics)
    ↓
C3: build_tile_dfg_tt (Build dataflow graph)
    ↓
Stage D (Late Split & Protocol Insertion)
```

---

## C1: lower_shared_to_cb_v5

**Purpose:** Lower shared memory to circular buffers (protocol-free)

**Location:** `tilelang/tenstorrent/passes/lower_shared_to_cb_v5.py`

### Transform
```python
# Before
shared_a = T.alloc_buffer((32, 32), "float16", scope="shared")

# After (abstract CB, no protocol)
cb_in0 = T.alloc_buffer((2, 2048), "uint8", scope="cb")  # 2 pages × 2KB
```

Uses `tt.cb.*` metadata from Stage A to determine page size and depth.

---

## C2: lower_tt_tile_intrinsics_v5

**Purpose:** Lower tile operations to TT intrinsics

**Location:** `tilelang/tenstorrent/passes/lower_tt_tile_intrinsics_v5.py`

### Key Operations

**Matmul Detection:**
```python
# Before
C[bx*32:(bx+1)*32, by*32:(by+1)*32] += A[...] * B[...]

# After
T.call_extern("matmul_tiles", cb_a, cb_b, 0, 0, 0, accumulate=True)
```

**Element-wise Operations:**
```python
# Before
C[...] = A[...] + B[...]

# After
T.call_extern("add_tiles", cb_a, cb_b, 0, 0, 0)
```

**Supported Intrinsics:**
- `matmul_tiles` - Matrix multiplication
- `add_tiles` - Element-wise addition
- `mul_tiles` - Element-wise multiplication
- `sub_tiles` - Element-wise subtraction

---

## C3: build_tile_dfg_tt

**Purpose:** Build tile-level dataflow graph for CB assignment and kernel splitting

**Location:** `tilelang/tenstorrent/passes/build_tile_dfg_tt.py`

### Algorithm

1. **Traverse TIR to find all CB allocations:**
   - Identify abstract CB definitions from C1
   - Track CB names and properties (size, depth, format)

2. **Build producer-consumer relationships:**
   - `read_to_cb` operations produce data to CBs
   - CBs are consumed by compute operations
   - Compute operations produce data to output CBs
   - `write_from_cb` operations consume data from CBs

3. **Assign roles to CBs:**
   - `input_a`, `input_b` for matrix operands
   - `output` for results
   - `intermediate` for temporary storage

4. **Validate CB count:**
   - Ensure total CBs ≤ 32 (hardware limit)
   - Warn if approaching limit

5. **Store dataflow graph as `tt.tile_dfg` attribute**

### What It Does

Analyzes tile data dependencies and builds a dataflow graph showing:
- **Producer-consumer relationships** between tiles
- **CB usage patterns** (which tiles use which CBs)
- **Optimization opportunities** (reuse, sharing, overlap)
- **Split boundaries** for kernel separation (used by D1)

### Output Metadata

```json
"tt.tile_dfg": {
  "nodes": [
    {"id": "tile_load_a", "type": "load", "cb": "cb_in0", "role": "producer"},
    {"id": "tile_load_b", "type": "load", "cb": "cb_in1", "role": "producer"},
    {"id": "tile_compute", "type": "matmul", "inputs": ["cb_in0", "cb_in1"], "output": "cb_out0", "role": "compute"},
    {"id": "tile_store", "type": "store", "cb": "cb_out0", "role": "consumer"}
  ],
  "edges": [
    {"from": "tile_load_a", "to": "tile_compute", "cb": "cb_in0", "type": "data"},
    {"from": "tile_load_b", "to": "tile_compute", "cb": "cb_in1", "type": "data"},
    {"from": "tile_compute", "to": "tile_store", "cb": "cb_out0", "type": "data"}
  ],
  "cb_assignments": {
    "cb_in0": {"id": 0, "role": "input_a"},
    "cb_in1": {"id": 1, "role": "input_b"},
    "cb_out0": {"id": 16, "role": "output"}
  },
  "validation": {
    "total_cbs": 3,
    "max_allowed": 32,
    "status": "valid"
  }
}
```

This metadata is critical for D1 (split_device_kernel) to determine kernel boundaries and CB routing.

---

## Stage C Design Principles

### 1. Protocol-Free Mid-Level IR

**Why:** Separation of concerns
- Stage C: What operations to perform
- Stage D: How to coordinate them

**Benefit:** Easier to optimize and transform

### 2. Pattern-Based Lowering

**No Heuristics:** Detects patterns based on IR structure, not variable names

**Example:**
```python
# Detected as matmul because of reduction structure
for k in T.serial(8):  # K-loop
    C[...] += A[...] * B[...]  # Accumulation pattern
```

### 3. Preserve K-Loop Structure

**Important:** K-loop preserved for Stage D
- D4 needs to know loop boundaries for `acquire_dst` placement
- D5 needs K-loop for `commit_dst` after accumulation

---

## Testing

```python
def test_stage_c_matmul():
    """Test Stage C protocol-less lowering."""
    mod = create_matmul_with_metadata()

    # Run Stages A-B first
    mod = run_stage_a(mod)
    mod = run_stage_b(mod)

    # Run Stage C
    mod = lower_shared_to_cb_v5(mod)
    mod = lower_tt_tile_intrinsics_v5(mod)
    mod = build_tile_dfg_tt(mod)

    # Validate: Abstract CBs present
    assert "scope=\"cb\"" in str(mod)

    # Validate: Intrinsics present
    assert "matmul_tiles" in str(mod)

    # Validate: Dataflow graph present
    assert "tt.tile_dfg" in mod.attrs
```

---

## References

- [v5_pipeline.md](../../architecture/v5_pipeline.md)
- [metadata.md](./metadata.md), [partitioning.md](./partitioning.md)

---

**Last Updated:** 2025-10-16
**Stage:** C (Protocol-less Lowering)
**Passes:** 3 (C1-C3)
**Status:** Production
