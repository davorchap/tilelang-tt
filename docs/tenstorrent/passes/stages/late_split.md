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

### Algorithm

1. **Analyze TIR to identify:**
   - Data movement operations (reads/writes)
   - Compute operations (matmul, binary ops)
   - Buffer dependencies from dataflow graph (C3)

2. **Create three PrimFunc clones:**
   - Reader: Contains only read_to_cb operations
   - Compute: Contains only compute operations
   - Writer: Contains only write_from_cb operations

3. **Assign CB IDs based on dataflow graph:**
   - Input CBs: cb_in0, cb_in1, ... (up to cb_in7)
   - Output CBs: cb_out0, cb_out1, ... (up to cb_out7)
   - Intermediate CBs: cb_intermed0, ... (if needed)

4. **Update function signatures:**
   - Reader: Takes input tensors as arguments
   - Compute: No tensor arguments (uses CBs only)
   - Writer: Takes output tensors as arguments

5. **Stamp kernel role attributes:**
   - `{"tt.kernel_role": "reader"}` for reader
   - `{"tt.kernel_role": "compute"}` for compute
   - `{"tt.kernel_role": "writer"}` for writer

### Transformation Example

```python
# Before (monolithic kernel)
@T.prim_func
def gemm_monolithic(A: T.Buffer, B: T.Buffer, C: T.Buffer):
    A_cb = tt.alloc_cb("cb_a", (4, 1), "bf16")  # 4 tiles
    B_cb = tt.alloc_cb("cb_b", (1, 4), "bf16")
    C_cb = tt.alloc_cb("cb_c", (4, 4), "bf16")

    for kt in T.serial(8):
        T.evaluate(tt.read_to_cb(A[m, kt*32], A_cb))
        T.evaluate(tt.read_to_cb(B[kt*32, n], B_cb))
        T.evaluate(tt.mm.mma(A_cb, B_cb, dst=0, accumulate=(kt>0)))

    T.evaluate(tt.write_from_cb(C_cb, C[m, n]))

# After (3 kernels with concrete CB IDs)
@T.prim_func(attrs={"tt.kernel_role": "reader"})
def gemm_reader(A: T.Buffer, B: T.Buffer):
    for kt in T.serial(8):
        T.evaluate(tt.read_to_cb(A[m, kt*32], "cb_in0"))
        T.evaluate(tt.read_to_cb(B[kt*32, n], "cb_in1"))

@T.prim_func(attrs={"tt.kernel_role": "compute"})
def gemm_compute():
    for kt in T.serial(8):
        T.evaluate(tt.mm.mma("cb_in0", "cb_in1", dst=0, accumulate=(kt>0)))
    # Note: packing to cb_out0 will be added by D5

@T.prim_func(attrs={"tt.kernel_role": "writer"})
def gemm_writer(C: T.Buffer):
    T.evaluate(tt.write_from_cb("cb_out0", C[m, n]))
```

Uses dataflow graph from C3 to determine split boundaries and CB assignments.

---

## D2: configure_tensor_accessor_tt

**Purpose:** Configure TensorAccessor metadata per kernel and bind to runtime arguments

**Location:** `tilelang/tenstorrent/passes/configure_tensor_accessor_tt.py`

### Algorithm

1. **Process reader/writer kernels only:**
   - Skip compute kernel (no tensor arguments)
   - Get runtime arg order from `tt.runtime_args` attribute

2. **For each buffer in kernel signature:**
   - Find corresponding accessor from A3
   - Determine runtime argument index
   - Calculate tile_size_bytes from dtype

3. **Update accessor with binding info:**
   - Set `type` from "abstract" to "bound"
   - Add `runtime_arg_idx` for argument position
   - Add `base_offset` (initially 0, set at runtime)
   - Keep all layout information from A3

4. **Store updated accessors** back to kernel attributes

### What It Does

Binds runtime arguments to tensor accessors:

```json
// Before (from A3)
"tt.tensor_accessor.A": {
  "type": "abstract",
  "buffer_name": "A",
  "layout_ref": "tt.buffer.A",
  "runtime_binding": null  // Placeholder
}

// After D2
"tt.tensor_accessor.A": {
  "type": "bound",
  "buffer_name": "A",
  "layout_ref": "tt.buffer.A",
  "runtime_binding": {
    "runtime_arg_idx": 0,
    "base_offset": 0,
    "tile_size_bytes": 2048
  }
}
```

Each kernel gets appropriate TensorAccessor configuration:
- **Reader**: Source buffer accessors bound to input arguments
- **Compute**: No tensor accessors (uses CBs only)
- **Writer**: Destination buffer accessors bound to output arguments

---

## D3: lower_cb_intrinsics

**Purpose:** Replace abstract CB operations with NOC/CB protocol sequences

**Location:** `tilelang/tenstorrent/passes/lower_cb_intrinsics.py`

### Algorithm

1. **For each `tt.read_to_cb` operation:**
   - Insert `cb_reserve_back` to allocate CB space
   - Get write pointer with `get_write_ptr`
   - Insert `noc_async_read_tile` for DMA transfer
   - Insert `noc_async_read_barrier` to ensure completion
   - Insert `cb_push_back` to make data available

2. **For each `tt.write_from_cb` operation:**
   - Insert `cb_wait_front` to wait for data
   - Get read pointer with `get_read_ptr`
   - Insert `noc_async_write_tile` for DMA transfer
   - Insert `noc_async_write_barrier` to ensure completion
   - Insert `cb_pop_front` to free CB space

3. **Handle pipelining optimizations:**
   - Batch barriers for multiple tile transfers
   - Support double-buffering patterns
   - Optimize producer-consumer synchronization

### Reader Kernel Protocol

```python
# Before (abstract)
T.evaluate(tt.read_to_cb(A[tile_m, tile_k], "cb_in0"))

# After (protocolized)
T.evaluate(cb_reserve_back("cb_in0", 1))
write_ptr = get_write_ptr("cb_in0")
tile_id = tile_m * Kt + tile_k
T.evaluate(noc_async_read_tile(tile_id, A_accessor, write_ptr))
T.evaluate(noc_async_read_barrier())
T.evaluate(cb_push_back("cb_in0", 1))
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
... pack tile (handled by D5) ...
cb_push_back(cb_out0, 1);
```

### Writer Kernel Protocol

```python
# Before (abstract)
T.evaluate(tt.write_from_cb("cb_out0", C[tile_m, tile_n]))

# After (protocolized)
T.evaluate(cb_wait_front("cb_out0", 1))
read_ptr = get_read_ptr("cb_out0")
tile_id = tile_m * Nt + tile_n
T.evaluate(noc_async_write_tile(read_ptr, C_accessor, tile_id))
T.evaluate(noc_async_write_barrier())
T.evaluate(cb_pop_front("cb_out0", 1))
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

**Purpose:** Wrap compute operations with DST register lifecycle management

**Location:** `tilelang/tenstorrent/passes/insert_dst_management_tt.py`

### Algorithm

1. **Identify compute patterns:**
   - K-loop accumulation (matmul with reduction)
   - Per-tile operations (element-wise ops)

2. **For K-loop patterns:**
   - Insert `acquire_dst` BEFORE loop
   - Keep compute operations unchanged
   - Insert `commit/wait/pack/release` AFTER loop

3. **For per-tile patterns:**
   - Insert full `acquire/commit/wait/pack/release` PER tile

4. **Add CB synchronization:**
   - `cb_wait_front` before compute (ensure input ready)
   - `cb_pop_front` after compute (free input space)
   - `cb_reserve_back` before pack (allocate output space)
   - `cb_push_back` after pack (signal output ready)

### Pattern Examples

#### K-loop Pattern (Matmul with Reduction)

```python
# Input (K-loop pattern from C2)
for kt in T.serial(8):
    T.evaluate(tt.mm.mma("cb_in0", "cb_in1", dst=0, accumulate=(kt>0)))

# Output (With DST management)
T.evaluate(tt.dst.acquire())
for kt in T.serial(8):
    T.evaluate(cb_wait_front("cb_in0", 1))
    T.evaluate(cb_wait_front("cb_in1", 1))
    T.evaluate(tt.mm.mma("cb_in0", "cb_in1", dst=0, accumulate=(kt>0)))
    T.evaluate(cb_pop_front("cb_in0", 1))
    T.evaluate(cb_pop_front("cb_in1", 1))

T.evaluate(cb_reserve_back("cb_out0", 1))
T.evaluate(tt.dst.commit())
T.evaluate(tt.dst.wait())
T.evaluate(pack_tile(dst=0, cb="cb_out0", tile_index=0))
T.evaluate(tt.dst.release())
T.evaluate(cb_push_back("cb_out0", 1))
```

#### Per-Tile Pattern (Element-wise)

```python
# Input (per-tile pattern)
for i in T.serial(num_tiles):
    T.evaluate(add_tiles("cb_in0", "cb_in1", 0, 0, 0))

# Output (With DST management per tile)
for i in T.serial(num_tiles):
    T.evaluate(tt.dst.acquire())
    T.evaluate(cb_wait_front("cb_in0", 1))
    T.evaluate(cb_wait_front("cb_in1", 1))
    T.evaluate(add_tiles("cb_in0", "cb_in1", 0, 0, 0))
    T.evaluate(cb_pop_front("cb_in0", 1))
    T.evaluate(cb_pop_front("cb_in1", 1))
    T.evaluate(cb_reserve_back("cb_out0", 1))
    T.evaluate(tt.dst.commit())
    T.evaluate(tt.dst.wait())
    T.evaluate(pack_tile(0, "cb_out0"))
    T.evaluate(tt.dst.release())
    T.evaluate(cb_push_back("cb_out0", 1))
```

### DST Lifecycle Stages

1. **acquire_dst()** - FPU reserves DST half for computation
2. **Computation** - FPU writes results to DST registers
3. **commit_dst()** - FPU signals computation complete
4. **wait_dst()** - Wait for commit to finish (implicit in pack)
5. **pack_tile()** - Packer transfers DST → CB (includes wait)
6. **release_dst()** - FPU releases DST back to packer

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
- [protocol_less.md](./protocol_less.md)

---

**Last Updated:** 2025-10-16
**Stage:** D (Late Split & Protocol)
**Passes:** 5 (D1-D5)
**Status:** Production
