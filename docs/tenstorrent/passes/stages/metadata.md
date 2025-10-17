# Stage A: Metadata Inference

**Stage:** A (Metadata)
**Passes:** 3 (A1-A3)
**Purpose:** Infer and propagate buffer metadata, attach tensor accessors

---

## Overview

Stage A establishes the metadata foundation for the entire v5 pipeline. These three passes work together to:
1. **Infer buffer layouts** from user annotations
2. **Derive circular buffer descriptors** from buffer metadata
3. **Attach tensor accessor metadata** for deterministic addressing

All downstream passes (Stages B-E) rely on the canonical metadata produced by Stage A.

---

## Pass Pipeline

```
User Annotations (annotate_tt_layout)
    ↓
A1: infer_tt_layout_v5
    ↓ Produces: tt.buffer.<name> attributes
A2: propagate_tt_layout_v5
    ↓ Produces: tt.cb.<name> attributes
A3: attach_tensor_accessor_tt
    ↓ Produces: tt.tensor_accessor.<name> attributes
    ↓
Stage B (Partitioning)
```

---

## A1: infer_tt_layout_v5

**Purpose:** Canonicalize buffer layout schema and validate constraints

**Location:** `tilelang/tenstorrent/passes/infer_tt_layout_v5.py`

### What It Does

- Normalizes user layout annotations into canonical schema
- Validates L1 shard constraints (tile-aligned, capacity checks)
- Rejects unsupported features (halo hints)
- Emits comprehensive buffer metadata for all buffers

### Input

TIR with user annotations:
```python
A = T.Buffer((256, 256), "bf16")
annotate_tt_layout(A, memory="DRAM", layout="interleaved")

B = T.Buffer((256, 256), "bf16")
annotate_tt_layout(B, memory="L1", layout="sharded",
                   nd_shard={"axes": ["M", "N"], "grid": [8, 8]})
```

### Output

PrimFunc with `tt.buffer.<name>` attributes:
```json
"tt.buffer.A": {
  "memory": "DRAM",
  "layout": "interleaved",
  "tile_shape": [32, 32],
  "dtype": "bf16"
}

"tt.buffer.B": {
  "memory": "L1",
  "layout": "sharded",
  "tile_shape": [32, 32],
  "dtype": "bf16",
  "nd_shard": {
    "axes": ["M", "N"],
    "grid": [8, 8],
    "shard_shape_elems": [32, 32],
    "order": "row_major",
    "align_tiles": true,
    "projected_grid": [8, 8],
    "projected_shard_tiles": [1, 1]
  }
}
```

### Key Validations

1. **L1 Capacity Check**: Ensures L1 shards fit within core memory
2. **Tile Alignment**: Validates that shards are tile-aligned (multiples of 32)
3. **Halo Rejection**: Rejects halo hints (not supported yet)
4. **ND Shard Validation**: Validates grid dimensions and shard shapes

### Error Messages

```python
# Example: L1 shard too large
"L1 shard exceeds capacity: 2048 bytes required, 1024 bytes available"

# Example: Not tile-aligned
"L1 shard not tile-aligned: shard_shape [30, 30] must be multiples of tile_shape [32, 32]"

# Example: Halo not supported
"Halo hints not supported: buffer A has halo [1, 1]"
```

---

## A2: propagate_tt_layout_v5

**Purpose:** Derive circular buffer metadata from buffer layouts

**Location:** `tilelang/tenstorrent/passes/propagate_tt_layout_v5.py`

### What It Does

- Reads buffer metadata from A1
- Calculates circular buffer parameters (page_size, depth, data_format)
- Stamps `tt.cb.<name>` attributes for each DRAM↔L1 transfer
- Assigns unique CB IDs

### Input

PrimFunc with `tt.buffer.*` attributes (from A1)

### Output

PrimFunc with `tt.cb.<name>` attributes:
```json
"tt.cb.A": {
  "cb_id": 0,
  "page_size": 2048,      // 32 × 32 × 2 bytes (bf16)
  "depth": 2,             // double buffering
  "data_format": "bfloat16"
}

"tt.cb.B": {
  "cb_id": 1,
  "page_size": 2048,
  "depth": 2,
  "data_format": "bfloat16"
}
```

### Key Operations

1. **Page Size Calculation**: `page_size = tile_height × tile_width × dtype_size`
   - FP16: 32 × 32 × 2 = 2048 bytes
   - BF16: 32 × 32 × 2 = 2048 bytes
   - FP32: 32 × 32 × 4 = 4096 bytes

2. **Depth Selection**: Default is 2 (double buffering)
   - Enables overlapped execution (reader fills page 1 while compute uses page 0)

3. **Data Format Mapping**:
   - `bf16` → `"bfloat16"`
   - `fp16` → `"float16"`
   - `fp32` → `"float32"`

4. **CB ID Assignment**: Sequential assignment starting from 0
   - Input buffers: `cb_in0`, `cb_in1`, ...
   - Output buffers: `cb_out0`, ...

---

## A3: attach_tensor_accessor_tt

**Purpose:** Attach TensorAccessor metadata for buffer addressing

**Location:** `tilelang/tenstorrent/passes/attach_tensor_accessor_tt.py`

### What It Does

- Creates TensorAccessor metadata for each buffer
- Enables deterministic global index computation in device kernels
- Supports both DRAM and L1 buffer addressing
- Guards against default-constructed accessors

### Input

PrimFunc with buffer and CB metadata (from A1 and A2)

### Output

PrimFunc with `tt.tensor_accessor.<name>` attributes:
```json
"tt.tensor_accessor.A": {
  "layout_ref": "tt.buffer.A",
  "stride_mode": "tiled",
  "tile_params": {
    "tile_height": 32,
    "tile_width": 32,
    "tiles_per_row": 8,
    "total_tiles": 64
  },
  "runtime_binding": null  // Filled by D2 later
}

"tt.tensor_accessor.B": {
  "layout_ref": "tt.buffer.B",
  "stride_mode": "sharded",
  "tile_params": {
    "tile_height": 32,
    "tile_width": 32,
    "shard_tiles_m": 1,
    "shard_tiles_n": 1,
    "grid_m": 8,
    "grid_n": 8
  },
  "runtime_binding": null  // Filled by D2 later
}
```

### Key Concepts

1. **Stride Mode**: Determines address calculation pattern
   - `"tiled"`: DRAM interleaved, simple tile indexing
   - `"linear"`: DRAM linear layout (rare)
   - `"sharded"`: L1 sharded, shard-aware indexing

2. **Tile Parameters**: Metadata for address computation
   - Tile dimensions (height, width)
   - Grid dimensions for sharded buffers
   - Tiles per row/column

3. **Runtime Binding**: Placeholder for Stage D
   - Filled by D2: ConfigureTensorAccessorTT
   - Contains actual runtime argument mappings

### Accessor Summary

A3 also generates an accessor summary for validation:
```python
# Example summary
"tt.accessor_summary": {
  "A": {"stride_mode": "tiled", "memory": "DRAM"},
  "B": {"stride_mode": "sharded", "memory": "L1"},
  "C": {"stride_mode": "tiled", "memory": "DRAM"}
}
```

---

## Stage A Output Summary

At the end of Stage A, each buffer has three types of metadata:

### 1. Buffer Metadata (`tt.buffer.<name>`)
- Memory space (DRAM/L1)
- Layout (interleaved/sharded)
- Tile shape [32, 32]
- Data type (bf16, fp16, fp32)
- Optional ND shard configuration

### 2. Circular Buffer Metadata (`tt.cb.<name>`)
- CB ID (unique per buffer)
- Page size (in bytes)
- Depth (double buffering)
- Data format

### 3. Tensor Accessor Metadata (`tt.tensor_accessor.<name>`)
- Layout reference
- Stride mode (tiled/linear/sharded)
- Tile parameters
- Runtime binding placeholder

---

## Downstream Dependencies

### Stage B (Partitioning) Uses:
- `tt.buffer.<name>.memory` - Determines partition mode (global vs local_shard)
- `tt.buffer.<name>.nd_shard` - Calculates grid dimensions and shard sizes

### Stage C (Protocol-less) Uses:
- `tt.cb.<name>` - Abstract CB operations (no protocol yet)

### Stage D (Late Split & Protocol) Uses:
- `tt.cb.<name>` - Inserts NOC/CB protocol with correct CB IDs
- `tt.tensor_accessor.<name>` - Configures runtime argument binding

### Codegen Uses:
- All three metadata types to generate correct C++ code:
  - Buffer creation APIs
  - CB configuration
  - Runtime argument setup
  - Address computation

---

## Design Rationale

### Why Three Passes?

1. **Separation of Concerns**
   - A1: Layout schema (user intent)
   - A2: CB descriptors (hardware mapping)
   - A3: Accessor metadata (addressing)

2. **Incremental Validation**
   - Each pass validates its specific domain
   - Errors caught early in pipeline

3. **Clear Dependencies**
   - A2 depends on A1
   - A3 depends on A1 and A2
   - Linear dependency chain

### Why Python Implementation?

- **Rapid Iteration**: Easy to modify and extend
- **Rich Data Structures**: Python dicts for metadata
- **Maintainability**: Easier to debug and test
- **Integration**: Seamless with TVM Python API

### Why Canonical Metadata?

- **No Heuristics**: Downstream passes don't guess
- **Deterministic**: Same input → same metadata
- **Verifiable**: Can validate completeness
- **Composable**: Easy to add new metadata fields

---

## Testing

### Test Files
- `testing/python/tenstorrent/test_v5_metadata_passes.py`
- `testing/python/tenstorrent/test_v5_passes_integration.py`

### Test Coverage

**A1: infer_tt_layout_v5**
- DRAM interleaved buffers
- DRAM sharded buffers
- L1 sharded buffers
- ND shard projection
- L1 capacity validation
- Halo rejection

**A2: propagate_tt_layout_v5**
- Page size calculation for different dtypes
- Depth assignment
- Data format mapping
- CB ID uniqueness

**A3: attach_tensor_accessor_tt**
- Tiled stride mode
- Sharded stride mode
- Tile parameter calculation
- Accessor summary generation

### Example Test

```python
def test_stage_a_matmul():
    """Test Stage A metadata for matmul."""
    @T.prim_func
    def matmul(
        A: T.Buffer((256, 256), "bf16"),
        B: T.Buffer((256, 256), "bf16"),
        C: T.Buffer((256, 256), "bf16")
    ):
        annotate_tt_layout(A, memory="DRAM", layout="interleaved")
        annotate_tt_layout(B, memory="DRAM", layout="interleaved")
        annotate_tt_layout(C, memory="DRAM", layout="interleaved")

        with T.Kernel(8, 8) as (bx, by):
            # matmul logic
            pass

    # Run Stage A
    mod = infer_tt_layout_v5(matmul)
    mod = propagate_tt_layout_v5(mod)
    mod = attach_tensor_accessor_tt(mod)

    # Validate metadata
    assert "tt.buffer.A" in mod.attrs
    assert "tt.cb.A" in mod.attrs
    assert "tt.tensor_accessor.A" in mod.attrs

    # Check CB parameters
    assert mod.attrs["tt.cb.A"]["page_size"] == 2048  # 32×32×2
    assert mod.attrs["tt.cb.A"]["depth"] == 2
```

---

## Common Issues

### Issue 1: L1 Shard Too Large
**Symptom**: Error "L1 shard exceeds capacity"
**Cause**: Shard doesn't fit in L1 memory (1MB per core)
**Solution**: Reduce shard size or use DRAM

### Issue 2: Not Tile-Aligned
**Symptom**: Error "shard not tile-aligned"
**Cause**: Shard dimensions not multiples of 32
**Solution**: Pad buffer or adjust shard configuration

### Issue 3: Missing Annotations
**Symptom**: Default DRAM interleaved used
**Cause**: No `annotate_tt_layout` call
**Solution**: Add explicit annotations for all buffers

### Issue 4: Halo Hints
**Symptom**: Error "Halo hints not supported"
**Cause**: Buffer annotated with halo metadata
**Solution**: Remove halo hints (not supported yet)

---

## Future Enhancements

1. **Halo Exchange Support**: Enable halo hints for stencil operations
2. **Dynamic Sharding**: Support runtime-determined shard configurations
3. **Custom CB Depth**: Allow user-specified CB depth (beyond default 2)
4. **Multi-Format CBs**: Support multiple data formats per CB

---

## References

- [v5_pipeline.md](../../architecture/v5_pipeline.md) - Complete v5 reference
- [TT_ARCHITECTURE.md](../../architecture/TT_ARCHITECTURE.md) - Backend architecture

**Individual Pass Documentation:**
- [infer_tt_layout_v5 implementation](../../../../tilelang/tenstorrent/passes/infer_tt_layout_v5.py)
- [propagate_tt_layout_v5 implementation](../../../../tilelang/tenstorrent/passes/propagate_tt_layout_v5.py)
- [attach_tensor_accessor_tt implementation](../../../../tilelang/tenstorrent/passes/attach_tensor_accessor_tt.py)

---

**Last Updated:** 2025-10-16
**Stage:** A (Metadata)
**Passes:** 3 (A1-A3)
**Status:** Production
