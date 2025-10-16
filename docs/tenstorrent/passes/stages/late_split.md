# Stage D: Late Split & Protocol Insertion

**Stage:** D (Late Split & Protocol)
**Passes:** 5 (D1-D5)
**Purpose:** Split kernels and insert NOC/CB/DST protocol

---

## Overview

Stage D is the most complex stage, performing two critical operations:
1. **Kernel Splitting** (D1): Split monolithic kernel into reader/compute/writer
2. **Protocol Insertion** (D2-D5): Insert all NOC/CB/DST protocol operations

After Stage D, kernels are ready for codegen with complete protocol.

---

## Pass Pipeline

```
Stage C Output (Protocol-free tile intrinsics + dataflow graph)
    ↓
D1: split_device_kernel (1 kernel → 3 kernels)
    ↓
D2: configure_tensor_accessor_tt (Bind runtime args)
    ↓
D3: lower_cb_intrinsics (Insert NOC/CB protocol)
    ↓
D4: insert_compute_init_tt (Insert acquire_dst, mm_init)
    ↓
D5: insert_dst_management_tt (Insert commit, pack, release)
    ↓
Stage E (Finalization)
```

---

## D1: split_device_kernel

**Purpose:** Split single kernel into reader/compute/writer kernels

**Location:** `tilelang/tenstorrent/passes/split_device_kernel.py`

### Transformation

```python
# Before (monolithic kernel)
def device_kernel():
    tile_a = load(A)  # Load
    tile_b = load(B)
    tile_c = matmul(tile_a, tile_b)  # Compute
    store(C, tile_c)  # Store

# After (3 kernels)
def reader_kernel():
    tile_a = load(A)
    tile_b = load(B)
    push_cb(cb_in0, tile_a)
    push_cb(cb_in1, tile_b)

def compute_kernel():
    tile_a = pop_cb(cb_in0)
    tile_b = pop_cb(cb_in1)
    tile_c = matmul(tile_a, tile_b)
    push_cb(cb_out0, tile_c)

def writer_kernel():
    tile_c = pop_cb(cb_out0)
    store(C, tile_c)
```

Uses dataflow graph from C3 to determine split boundaries.

---

## D2: configure_tensor_accessor_tt

**Purpose:** Configure TensorAccessor metadata per kernel

**Location:** `tilelang/tenstorrent/passes/configure_tensor_accessor_tt.py`

### What It Does

Binds runtime arguments to tensor accessors:

```json
// Before (from A3)
"tt.tensor_accessor.A": {
  "runtime_binding": null  // Placeholder
}

// After D2
"tt.tensor_accessor.A": {
  "runtime_binding": {
    "start_arg": "tt_start_tile",
    "count_arg": "tt_count",
    "grid_args": ["Mt", "Kt", "Nt"]
  }
}
```

Each kernel gets appropriate TensorAccessor configuration:
- **Reader**: Source buffer accessors
- **Compute**: Uses runtime args for index calculation
- **Writer**: Destination buffer accessors

---

## D3: lower_cb_intrinsics

**Purpose:** Insert NOC/CB API calls

**Location:** `tilelang/tenstorrent/passes/lower_cb_intrinsics.py`

### Reader Kernel Protocol

```cpp
// Before
push_cb(cb_in0, tile_a)

// After
cb_reserve_back(cb_in0, 1);
uint32_t l1_addr = get_write_ptr(cb_in0);
noc_async_read_tile(tile, dram_addr_a, l1_addr);
noc_async_read_barrier();
cb_push_back(cb_in0, 1);
```

### Compute Kernel Protocol

```cpp
// Before
tile_a = pop_cb(cb_in0)
... matmul ...
push_cb(cb_out0, tile_c)

// After
cb_wait_front(cb_in0, 1);
... matmul ...
cb_pop_front(cb_in0, 1);
cb_reserve_back(cb_out0, 1);
... pack tile ...
cb_push_back(cb_out0, 1);
```

### Writer Kernel Protocol

```cpp
// Before
tile_c = pop_cb(cb_out0)

// After
cb_wait_front(cb_out0, 1);
uint32_t l1_addr = get_read_ptr(cb_out0);
noc_async_write_tile(tile, l1_addr, dram_addr_c);
noc_async_write_barrier();
cb_pop_front(cb_out0, 1);
```

---

## D4: insert_compute_init_tt

**Purpose:** Insert compute initialization (acquire_dst, mm_init)

**Location:** `tilelang/tenstorrent/passes/insert_compute_init_tt.py`

### Patterns

**Matmul (K-loop):**
```cpp
for (tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    acquire_dst();           // ← D4
    mm_init(cb_in0, cb_in1, cb_out0);  // ← D4

    for (k = 0; k < Kt; ++k) {
        matmul_tiles(...);
    }
    // D5 will add commit/pack/release
}
```

**Elementwise (per-tile):**
```cpp
for (i = 0; i < num_tiles; ++i) {
    acquire_dst();           // ← D4
    add_tiles(cb_a, cb_b, 0, 0, 0);
    // D5 will add commit/pack/release
}
```

---

## D5: insert_dst_management_tt

**Purpose:** Insert DST lifecycle (commit, pack, release)

**Location:** `tilelang/tenstorrent/passes/insert_dst_management_tt.py`

### Complete Matmul DST Lifecycle

```cpp
for (tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    acquire_dst();                    // D4
    mm_init(cb_in0, cb_in1, cb_out0); // D4

    for (k = 0; k < Kt; ++k) {
        cb_wait_front(cb_in0, 1);           // D3
        cb_wait_front(cb_in1, 1);           // D3
        matmul_tiles(cb_in0, cb_in1, 0, 0, 0, k > 0);  // C2
        cb_pop_front(cb_in0, 1);            // D3
        cb_pop_front(cb_in1, 1);            // D3
    }

    cb_reserve_back(cb_out0, 1);      // D3
    commit_dst();                      // ← D5
    pack_tile(0, cb_out0);            // ← D5
    cb_push_back(cb_out0, 1);         // D3
    release_dst();                     // ← D5
}
```

### DST Lifecycle Stages

1. **acquire_dst()** - FPU reserves DST half for computation
2. **Computation** - FPU writes results to DST
3. **commit_dst()** - FPU signals computation complete
4. **pack_tile()** - Packer writes DST to CB (internally waits)
5. **release_dst()** - FPU releases DST back to packer

---

## Stage D Output Summary

After Stage D, each kernel has:

### Reader Kernel
- NOC read operations
- CB reserve/push protocol
- TensorAccessor configured for source buffers

### Compute Kernel
- CB wait/pop for inputs
- Tile intrinsics (matmul_tiles, add_tiles, etc.)
- DST lifecycle (acquire → compute → commit → pack → release)
- CB reserve/push for outputs

### Writer Kernel
- CB wait/pop for inputs
- NOC write operations
- TensorAccessor configured for destination buffers

---

## Key Design Decisions

### Why Late Split?

**Alternative:** Split early (before Stage C)
**Problem:** Harder to optimize, need protocol-aware transforms

**Solution:** Split late (Stage D)
**Benefit:** Clean protocol-free mid-level IR in Stage C

### Why Separate Protocol Insertion?

D3, D4, D5 are separate passes for:
1. **D3**: NOC/CB protocol (all kernels)
2. **D4**: Compute initialization (compute kernel only)
3. **D5**: DST lifecycle (compute kernel only)

Clear separation enables independent testing and modification.

### Why DST Lifecycle Matters?

**Hardware Constraint:** DST shared between FPU and Packer

**Without Lifecycle:** Race conditions, incorrect results

**With Lifecycle:** Clean handshake, pipelined execution

---

## Testing

```python
def test_stage_d_protocol():
    """Test Stage D protocol insertion."""
    mod = create_matmul_with_metadata()

    # Run Stages A-C
    mod = run_stages_abc(mod)

    # Run Stage D
    mod = split_device_kernel(mod)
    assert len(mod.functions) == 3  # reader, compute, writer

    mod = configure_tensor_accessor_tt(mod)
    mod = lower_cb_intrinsics(mod)
    mod = insert_compute_init_tt(mod)
    mod = insert_dst_management_tt(mod)

    # Validate reader
    reader = get_kernel(mod, "reader")
    assert "noc_async_read" in str(reader)
    assert "cb_reserve_back" in str(reader)

    # Validate compute
    compute = get_kernel(mod, "compute")
    assert "acquire_dst" in str(compute)
    assert "commit_dst" in str(compute)
    assert "pack_tile" in str(compute)
    assert "release_dst" in str(compute)

    # Validate writer
    writer = get_kernel(mod, "writer")
    assert "noc_async_write" in str(writer)
    assert "cb_wait_front" in str(writer)
```

---

## Common Issues

### Issue 1: Missing DST Lifecycle
**Symptom:** Incorrect compute results or crashes
**Cause:** Missing acquire/commit/release
**Solution:** Verify D4 and D5 ran correctly

### Issue 2: CB Protocol Mismatch
**Symptom:** Hangs or deadlocks
**Cause:** Mismatched reserve/push or wait/pop
**Solution:** Check D3 output for balanced protocol

### Issue 3: Wrong Kernel Split
**Symptom:** Operations in wrong kernel
**Cause:** Dataflow graph incorrect
**Solution:** Check C3 (build_tile_dfg_tt) output

---

## References

- [v5_pipeline.md](../../architecture/v5_pipeline.md)
- [PASS_TABLE_TT.md](../../reference/PASS_TABLE_TT.md)
- [protocol_less.md](./protocol_less.md)

---

**Last Updated:** 2025-10-16
**Stage:** D (Late Split & Protocol)
**Passes:** 5 (D1-D5)
**Status:** Production
