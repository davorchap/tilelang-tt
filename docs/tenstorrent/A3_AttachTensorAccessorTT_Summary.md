# A3: AttachTensorAccessorTT Pass - Implementation Summary

**Date:** 2025-10-15
**Status:** ✅ Complete
**Location:** `/tilelang/tenstorrent/passes/attach_tensor_accessor_tt.py`

## Overview

The A3: AttachTensorAccessorTT pass has been successfully implemented following the v5 specification. This pass creates abstract tensor accessor descriptors that serve as a bridge between buffer layouts (from A1) and runtime argument binding (in D2).

## Key Features Implemented

### 1. Abstract Accessor Creation
- Creates `tt.tensor_accessor.*` attributes for each buffer
- Links accessors to buffer layouts via `layout_ref`
- Maintains "abstract" type until D2 binds runtime arguments

### 2. Stride Mode Determination
The pass intelligently determines stride modes based on buffer properties:
- **"tiled"**: Default for DRAM interleaved and L1 buffers
- **"linear"**: Traditional row/column-major (available but not default)
- **"sharded"**: For L1 sharded buffers with ND sharding

### 3. Access Pattern Detection
Automatically infers access patterns from buffer names:
- **"input"**: For buffers named input/a/x
- **"output"**: For buffers named output/c/result/z
- **"weight"**: For buffers named weight/w/b/kernel
- **"inout"**: Default for unrecognized patterns

### 4. Tile Parameter Calculation
- Computes tile dimensions from buffer layouts
- Calculates tiles per dimension for the entire buffer
- Determines tile size in bytes based on dtype

### 5. Sharding Information
For sharded buffers, extracts and preserves:
- Grid dimensions
- Shard tiles per core
- Axis mapping
- Ordering (row_major/col_major)

## Accessor Schema

```python
{
    "type": "abstract",                    # Will become "bound" in D2
    "buffer_name": str,                    # Original buffer name
    "layout_ref": f"tt.buffer.{name}",    # Link to layout descriptor
    "stride_mode": str,                    # "tiled", "linear", or "sharded"
    "access_pattern": str,                 # "input", "output", "weight", "inout"
    "tile_dims": [int, int],              # Tile shape [M, N]
    "tiles_per_dim": [int, int],          # Total tiles in each dimension
    "memory": str,                         # "DRAM" or "L1"
    "layout_type": str,                    # "interleaved" or "sharded"
    "base_offset": None,                   # Filled by D2
    "runtime_arg_idx": None,              # Filled by D2
    "tile_size_bytes": int,               # Bytes per tile
    "sharding": {                          # Sharding metadata
        "enabled": bool,
        "axes": [str, ...],
        "grid": [int, int],
        "shard_tiles": [int, int],
        "order": str
    }
}
```

## Integration with Pipeline

### Inputs (from previous passes)
- **From A1:** `tt.buffer.*` layout descriptors with memory, layout type, tile shape, dtype
- **From A2:** CB descriptors (passed through, not modified)

### Outputs (for subsequent passes)
- **`tt.tensor_accessor.*`:** Abstract accessor for each buffer
- **`tt.accessor_summary`:** Statistics about all accessors
- **For D2:** Abstract accessors ready for runtime binding
- **For D3/D5:** Access patterns to guide protocol insertion

## Example Usage

```python
from attach_tensor_accessor_tt import AttachTensorAccessorTT

# Apply after A1 and A2
pass_a3 = AttachTensorAccessorTT()
result = pass_a3(ir_module)

# Access the generated accessors
func = result["gemm"]
accessor_a = func.attrs["tt.tensor_accessor.A"]
print(f"Buffer A stride mode: {accessor_a['stride_mode']}")
print(f"Buffer A access pattern: {accessor_a['access_pattern']}")
```

## Test Coverage

Created comprehensive test suite in:
- `/testing/python/tenstorrent/test_a3_attach_tensor_accessor.py` - Full pytest suite
- `/testing/python/tenstorrent/test_a3_simple.py` - Standalone test script

Tests cover:
1. Basic accessor creation for all buffers
2. Accessor structure validation
3. Stride mode determination logic
4. Access pattern detection
5. Tile parameter calculation
6. Sharding information extraction
7. Pipeline integration (A1→A2→A3)
8. Integration with B1 partition metadata

## Design Decisions

1. **Abstract vs Bound**: Accessors start as "abstract" with null runtime fields, becoming "bound" only after D2 processes them. This separation keeps concerns clean.

2. **Pattern-based Detection**: Access patterns are inferred from buffer names rather than requiring explicit annotations, simplifying the user API.

3. **Sharding-aware**: Full support for ND sharding metadata, preparing for distributed execution on Tenstorrent's grid architecture.

4. **Tile-centric**: All calculations are tile-based, aligning with Tenstorrent's tile-level execution model.

## Next Steps

With A3 complete, the metadata stage (A1-A3) is fully implemented. The next priorities are:

1. **C3: BuildTileDFGTT** - Build dataflow graph for kernel splitting
2. **D1: SplitDeviceKernel** - Split into reader/compute/writer kernels
3. **D2: ConfigureTensorAccessorTT** - Bind accessors to runtime arguments

The accessor metadata from A3 will be critical for D2's runtime binding and for guiding the protocol insertion in D3/D5.

## Summary

The A3 pass successfully bridges the gap between static buffer layouts and runtime tensor access. It provides a clean abstraction layer that:
- Preserves all layout information from A1
- Adds access semantics needed for protocol insertion
- Prepares for runtime argument binding in D2
- Maintains the progressive lowering philosophy of v5

This completes the metadata attachment stage of the pipeline, setting a solid foundation for the protocol-less and late-split stages to follow.