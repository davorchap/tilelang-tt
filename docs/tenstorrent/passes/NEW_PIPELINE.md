# Tenstorrent Pass Pipeline Documentation

## Overview

The new Tenstorrent lowering pipeline consists of five main passes that transform high-level TIR to device-specific persistent kernels. This document describes each pass in detail.

## Pass Order and Dependencies

```
InferTTLayout
    ↓
PropagateTTLayout
    ↓
TTTilesToCoreMap
    ↓
LowerTTTileIntrinsics
    ↓
GridToPersistentTT
```

## Pass Descriptions

### 1. InferTTLayout

**Purpose:** Stamps initial layout/shard metadata onto PrimFuncs when absent and normalizes user hints.

**Inputs:**
- IRModule with PrimFuncs (may or may not have `tt.layout_desc`)
- Optional default layout settings

**Outputs:**
- Same module with `tt.layout_desc` attached to all PrimFuncs
- No structural changes to TIR

**Attributes Set:**
- `tt.layout_desc`: Dict mapping buffer names to layout info
  - `shard`: "DRAM" or "L1"
  - `interleave`: bool
  - `stride`: [int, int]
  - `tile_id_order`: "row_major", "column_major", "match_shard", "z_order"

**Example:**
```python
pass_inst = InferTTLayout(defaults={"shard": "DRAM", "interleave": False})
mod = pass_inst(mod)
```

### 2. PropagateTTLayout

**Purpose:** Normalizes and propagates layout descriptors throughout the IR to ensure consistency.

**Inputs:**
- IRModule with `tt.layout_desc` present at PrimFunc level

**Outputs:**
- Same module with normalized and validated layout descriptors

**Invariants:**
- Does not mutate compute semantics
- Does not allocate buffers

**Normalization Rules:**
- "DDR" → "DRAM"
- "SRAM" → "L1"
- Missing `tile_id_order` defaults to "row_major"
- Invalid values trigger warnings and fallback to defaults

### 3. TTTilesToCoreMap

**Purpose:** Computes CoreRange(s) and builds a layout-aware work partition assigning tiles to cores.

**Inputs:**
- IR with `tt.core_grid` (or fallback grid)
- `tt.layout_desc` present
- Tiled compute regions

**Outputs:**
- Attaches:
  - `tt.core_ranges`: List of rectangular core regions
  - `tt.work_partition`: Mapping "(cx,cy)" → List[WorkItem]

**Partitioning Strategies:**
- `"row_major"`: Distribute tiles row-by-row across cores
- `"column_major"`: Distribute tiles column-by-column
- `"block"`: Each core gets a rectangular block of tiles

**Example Work Partition:**
```json
{
  "(0,0)": [{"io": 0, "jo": 0, "len_k": 128, "tile_order": "row_major"}],
  "(0,1)": [{"io": 0, "jo": 1, "len_k": 128, "tile_order": "row_major"}],
  "(1,0)": [{"io": 1, "jo": 0, "len_k": 128, "tile_order": "row_major"}],
  "(1,1)": [{"io": 1, "jo": 1, "len_k": 128, "tile_order": "row_major"}]
}
```

### 4. LowerTTTileIntrinsics

**Purpose:** Converts high-level tile intrinsics into TT-specific device calls.

**Intrinsic Mappings:**
- `T.gemm` → TT matrix engine operations
- `tile_load` → TT NOC DMA reads
- `tile_store` → TT NOC DMA writes
- Epilogue ops → TT SFPU operations

**Device Configurations:**
- Grayskull: 32×32 tiles, 12 L1 banks
- Wormhole: 32×32 tiles, enhanced NOC
- Blackhole: 32×32 tiles, 16 L1 banks

**Attributes Set:**
- `tt.tile_intrinsics_lowered`: True
- `tt.target_device`: Device name
- `tt.device_config`: Device-specific parameters

### 5. GridToPersistentTT

**Purpose:** Final lowering from grid-style IR to persistent kernels with staging and barriers.

**Inputs:**
- IR with all metadata attributes present
- Tile intrinsics already lowered

**Outputs:**
- IR with persistent loops injected
- `tt.plan.json` file for host/device coordination

**Transformations:**
1. Build per-core worklists from metadata
2. Insert persistent outer loop per core
3. Add double-buffering for tile staging
4. Place barriers at producer/consumer boundaries
5. Emit runtime plan

**Persistent Kernel Structure:**
```python
tid = T.get_core_id()
worklist = T.tt_load_worklist(tid)

for task in worklist:  # Persistent loop
    io, jo = task.io, task.jo
    
    for ko in range(0, K, TK):
        # Double-buffered tile loads
        a_buf.next = T.dram_to_l1(A, io, ko)
        b_buf.next = T.dram_to_l1(B, ko, jo)
        T.barrier()
        
        # Compute on current tiles
        T.tt_mma(a_buf.curr, b_buf.curr, accum)
        
        # Swap buffers
        a_buf.swap()
        b_buf.swap()
    
    # Write back result
    T.l1_to_dram(C, io, jo, accum)
```

## Pipeline Configuration

The pipeline can be configured through `build_tt_pipeline()`:

```python
pipeline = build_tt_pipeline(
    plan_path="my_plan.json",
    target_device="wormhole",
    partition_strategy="block",
    enable_double_buffer=True,
    enable_prefetch=True
)
```

Or use the convenience function:

```python
mod = run_pipeline(
    mod,
    plan_path="my_plan.json",
    target_device="grayskull",
    partition_strategy="row_major"
)
```

## Validation

Each pass includes validation to ensure correctness:

- **InferTTLayout**: Validates buffer names exist
- **PropagateTTLayout**: Checks layout consistency
- **TTTilesToCoreMap**: Ensures all tiles are assigned exactly once
- **LowerTTTileIntrinsics**: Verifies intrinsic support
- **GridToPersistentTT**: Validates complete metadata presence

## Error Handling

The pipeline provides clear error messages:

```python
errors = validate_module_for_tt(mod)
if errors:
    print("Module validation failed:")
    for error in errors:
        print(f"  - {error}")
```

## Testing Strategy

Each pass can be tested independently:

```python
# Test individual pass
pass_inst = InferTTLayout()
result = pass_inst(input_mod)
assert "tt.layout_desc" in result["main"].attrs

# Test full pipeline
result = run_pipeline(input_mod)
assert Path("tt.plan.json").exists()
```

## Performance Considerations

- **Double-buffering**: Enabled by default for latency hiding
- **Prefetching**: Overlaps computation with data movement
- **Barrier placement**: Minimized to avoid stalls
- **Work distribution**: Balanced across cores

## Future Extensions

1. **Dynamic work stealing**: Runtime load balancing
2. **Multi-chip support**: Cross-chip communication
3. **Autotuning integration**: Automatic parameter selection
4. **Memory optimization**: Automatic buffer sizing

## References

- [Architecture Overview](../NEW_LOWERING_ARCHITECTURE.md)
- [Runtime Plan Specification](../RUNTIME_PLAN.md)
- [TIR Basics](../TIR_BASICS.md)
