# Stage B: Partitioning

**Stage:** B (Partitioning)
**Passes:** 2 (B1-B2)
**Purpose:** Determine per-core work assignments and map to physical cores

---

## Overview

Stage B determines how work is distributed across Tenstorrent cores. These two passes work together to:
1. **Choose partition mode** based on buffer residency (DRAM vs L1)
2. **Calculate core assignments** and runtime arguments
3. **Transform GPU-style grid** to persistent loop model

The output of Stage B is a persistent kernel with per-core tile assignments.

---

## Pass Pipeline

```
Stage A Output (Buffer + CB + Accessor metadata)
    ↓
B1: layout_aware_work_partition_tt_v5
    ↓ Produces: tt.partition_mode, tt.grid_tiles, tt.core_ranges, tt.runtime_args
B2: grid_to_core_grid_v5
    ↓ Transforms: GPU grid → persistent loop
    ↓
Stage C (Protocol-less Lowering)
```

---

## B1: layout_aware_work_partition_tt_v5

**Purpose:** Choose per-core work assignments based on buffer residency

**Location:** `tilelang/tenstorrent/passes/layout_aware_work_partition_tt_v5.py`

### What It Does

- Analyzes buffer residency from Stage A metadata
- Chooses partition mode: `"global"` or `"local_shard"`
- Calculates grid dimensions and per-core tile assignments
- Emits canonical runtime arguments
- Generates CoreRangeSet for kernel launches

### Partition Modes

#### Global Mode (DRAM Buffers)
**When**: All buffers in DRAM (interleaved or sharded)
**Behavior**: Each core processes tiles from global pool
**Runtime Args**: `start_id`, `count`, `Mt`, `Kt`, `Nt`

```python
# Example: 8×8 matmul with DRAM buffers
"tt.partition_mode": "global"
"tt.grid_tiles": [8, 8]  # Mt=8, Nt=8
"tt.core_ranges": [[0,0], [7,7]]  # 8x8 grid
"tt.runtime_args": ["start_id", "count", "Mt", "Kt", "Nt"]
```

#### Local Shard Mode (L1 Buffers)
**When**: At least one buffer in L1 with sharding
**Behavior**: Each core owns local shard tiles
**Runtime Args**: Above + `Sm`, `Sn`, `Gy`, `Gx`, `sy`, `sx`

```python
# Example: 8×8 matmul with L1 sharded output
"tt.partition_mode": "local_shard"
"tt.grid_tiles": [8, 8]
"tt.shard_grid": [8, 8]  # Gy=8, Gx=8
"tt.local_shape_tiles": [1, 1]  # Sm=1, Sn=1
"tt.core_ranges": [[0,0], [7,7]]
"tt.runtime_args": ["start_id", "count", "Mt", "Kt", "Nt", "Sm", "Sn", "Gy", "Gx", "sy", "sx"]
```

### Input

PrimFunc with:
- `tt.buffer.<name>` (from A1) - Determines partition mode
- GPU-style grid kernel: `with T.Kernel(8, 8) as (bx, by)`

### Output

PrimFunc with partition metadata:
```json
{
  "tt.partition_mode": "global" | "local_shard",
  "tt.grid_tiles": [Mt, Nt],
  "tt.shard_grid": [Gy, Gx],              // local_shard only
  "tt.local_shape_tiles": [Sm, Sn],       // local_shard only
  "tt.core_ranges": [[y0,x0], [y1,x1], ...],
  "tt.runtime_args": ["start_id", "count", ...]
}
```

### Key Operations

1. **Buffer Residency Analysis**
```python
has_l1_shard = any(buf["memory"] == "L1" and buf["layout"] == "sharded"
                   for buf in buffer_metadata)
partition_mode = "local_shard" if has_l1_shard else "global"
```

2. **Grid Calculation**
```python
# From T.Kernel(8, 8)
Mt = 8  # Grid height in tiles
Nt = 8  # Grid width in tiles
grid_tiles = [Mt, Nt]
```

3. **Core Range Generation**
```python
# For 8×8 grid
core_ranges = [[y, x] for y in range(8) for x in range(8)]
# Result: [[0,0], [0,1], ..., [7,7]]
```

4. **Runtime Args Selection**
```python
if partition_mode == "global":
    runtime_args = ["start_id", "count", "Mt", "Kt", "Nt"]
else:  # local_shard
    runtime_args = ["start_id", "count", "Mt", "Kt", "Nt",
                    "Sm", "Sn", "Gy", "Gx", "sy", "sx"]
```

### Example Transform

```python
# Before B1
@T.prim_func
def matmul(A, B, C):
    # A, B: DRAM interleaved
    # C: DRAM interleaved
    with T.Kernel(8, 8) as (bx, by):
        # matmul logic
        pass

# After B1
# PrimFunc attrs now contain:
{
  "tt.partition_mode": "global",
  "tt.grid_tiles": [8, 8],
  "tt.core_ranges": [[0,0], [0,1], ..., [7,7]],
  "tt.runtime_args": ["start_id", "count", "Mt", "Kt", "Nt"]
}
```

---

## B2: grid_to_core_grid_v5

**Purpose:** Transform GPU-style grid kernel to persistent loop

**Location:** `tilelang/tenstorrent/passes/grid_to_core_grid_v5.py`

### What It Does

- Transforms `T.Kernel(M, N)` grid to persistent loop
- Maps `blockIdx.x/y` to persistent core iteration
- Calculates per-core tile assignments using partition metadata
- Inserts core ID retrieval and tile iteration logic

### Input

PrimFunc with:
- Partition metadata from B1
- GPU-style grid kernel

```python
with T.Kernel(8, 8) as (bx, by):
    C[bx*32:(bx+1)*32, by*32:(by+1)*32] = A[...] * B[...]
```

### Output

Persistent loop:

```python
# Get core ID and tile assignment
core_id = get_core_id()
start_tile, count = get_tile_assignment(core_id, tt_start_tile, tt_count)

# Persistent loop over assigned tiles
for tile_id in range(start_tile, start_tile + count):
    # Recover (bx, by) from tile_id
    bx = tile_id // 8  # Nt = 8
    by = tile_id % 8

    # Original kernel body (same indexing)
    C[bx*32:(bx+1)*32, by*32:(by+1)*32] = A[...] * B[...]
```

### Key Operations

1. **Core ID Retrieval**
```python
# Runtime intrinsic (different per kernel type)
core_id = T.call_extern("get_core_id")  # Reader/Writer
# or
core_id = T.call_extern("get_core_id_compute")  # Compute
```

2. **Tile Assignment Calculation**
```python
# Using runtime args from B1
start_tile = T.call_extern("get_arg", "tt_start_tile")
count = T.call_extern("get_arg", "tt_count")
```

3. **Index Recovery**
```python
# For row-major tile order (default)
bx = tile_id // Nt
by = tile_id % Nt

# For other orders (e.g., match_shard)
# Different calculation based on shard order
```

### Transformation Examples

#### Example 1: Simple 2D Grid
```python
# Before
with T.Kernel(4, 4) as (bx, by):
    C[bx*32:(bx+1)*32, by*32:(by+1)*32] = A[...] + B[...]

# After
core_id = get_core_id()
start, count = get_tile_assignment(core_id, tt_start_tile, tt_count)
for tile_id in range(start, start + count):
    bx = tile_id // 4
    by = tile_id % 4
    C[bx*32:(bx+1)*32, by*32:(by+1)*32] = A[...] + B[...]
```

#### Example 2: 3D Grid (Batch Dimension)
```python
# Before
with T.Kernel(2, 8, 8) as (b, bx, by):  # Batch=2, M=8, N=8
    C[b, bx*32:(bx+1)*32, by*32:(by+1)*32] = ...

# After
core_id = get_core_id()
start, count = get_tile_assignment(core_id, tt_start_tile, tt_count)
for tile_id in range(start, start + count):
    # 3D indexing
    b = tile_id // (8 * 8)
    rem = tile_id % (8 * 8)
    bx = rem // 8
    by = rem % 8
    C[b, bx*32:(bx+1)*32, by*32:(by+1)*32] = ...
```

---

## Stage B Output Summary

At the end of Stage B:

### Metadata Added
1. **Partition Mode**: `global` or `local_shard`
2. **Grid Tiles**: `[Mt, Nt]` - Global tile dimensions
3. **Core Ranges**: Physical core coordinates
4. **Runtime Args**: Canonical argument names

### IR Transformation
- GPU grid → Persistent loop
- Static `(bx, by)` → Dynamic tile iteration
- Added core ID retrieval
- Added tile assignment logic

---

## Tile Assignment Strategies

### Static Assignment (Current)
Each core gets equal number of tiles:
```python
tiles_per_core = (Mt * Nt) // num_cores
start = core_id * tiles_per_core
count = tiles_per_core
```

### Dynamic Assignment (Future)
Work-stealing or load balancing:
```python
# Not yet implemented
start, count = dynamic_work_queue(core_id)
```

---

## Design Rationale

### Why Two Passes?

1. **B1: Logical Partitioning**
   - High-level decision: global vs local_shard
   - Grid calculation and core assignment
   - Runtime argument selection

2. **B2: Physical Transformation**
   - IR transformation (grid → loop)
   - Core coordination code
   - Index recovery

### Why Persistent Model?

**GPU**: Launch N blocks, each processes 1 tile
- Pros: Simple, parallel
- Cons: Launch overhead, poor data reuse

**TT**: Launch N cores, each iterates over M tiles
- Pros: Better data reuse, reduced overhead
- Cons: Requires tile assignment logic

### Why Layout-Aware?

Different partition modes for different memory patterns:
- **Global**: Efficient for DRAM (any core can access any tile)
- **Local Shard**: Efficient for L1 (each core owns local data)

---

## Testing

### Test Files
- `testing/python/tenstorrent/test_v5_metadata_passes.py`
- `testing/python/tenstorrent/test_v5_passes_integration.py`

### Test Coverage

**B1: layout_aware_work_partition_tt_v5**
- Global mode (DRAM buffers)
- Local shard mode (L1 buffers)
- Grid dimension calculation
- Core range generation
- Runtime arg selection

**B2: grid_to_core_grid_v5**
- 2D grid transformation
- 3D grid transformation
- Index recovery correctness
- Core ID retrieval
- Tile assignment logic

### Example Test

```python
def test_stage_b_partitioning():
    """Test Stage B partitioning for matmul."""
    # Create matmul with metadata
    mod = create_matmul_with_metadata()

    # Run Stage B
    mod = layout_aware_work_partition_tt_v5(mod)
    mod = grid_to_core_grid_v5(mod)

    # Validate partition metadata
    assert mod.attrs["tt.partition_mode"] == "global"
    assert mod.attrs["tt.grid_tiles"] == [8, 8]
    assert len(mod.attrs["tt.core_ranges"]) == 64  # 8×8

    # Validate persistent loop transformation
    # Check for get_core_id and tile iteration
    assert "get_core_id" in str(mod)
    assert "for tile_id" in str(mod)
```

---

## Common Issues

### Issue 1: Wrong Partition Mode
**Symptom**: Local shard mode for DRAM buffers
**Cause**: Buffer metadata incorrectly marked as L1
**Solution**: Check A1 output, ensure DRAM buffers marked correctly

### Issue 2: Incorrect Grid Dimensions
**Symptom**: Grid tiles don't match buffer shape
**Cause**: Buffer not padded to tile boundaries
**Solution**: Ensure buffers padded (or use tile_pad_tt pass)

### Issue 3: Core Range Mismatch
**Symptom**: Not enough cores for tiles
**Cause**: Grid larger than available cores
**Solution**: Adjust grid size or use tiling strategy

---

## Future Enhancements

1. **Dynamic Work Assignment**: Work-stealing or load balancing
2. **Hierarchical Partitioning**: Multi-level tile assignment
3. **Custom Tile Orders**: Support arbitrary iteration orders
4. **Multi-Device Partitioning**: Distribute across multiple devices

---

## References

- [v5_pipeline.md](../../architecture/v5_pipeline.md) - Complete v5 reference
- [metadata.md](./metadata.md) - Stage A (prerequisite)

**Individual Pass Documentation:**
- [layout_aware_work_partition_tt_v5 implementation](../../../../tilelang/tenstorrent/passes/layout_aware_work_partition_tt_v5.py)
- [grid_to_core_grid_v5 implementation](../../../../tilelang/tenstorrent/passes/grid_to_core_grid_v5.py)

---

**Last Updated:** 2025-10-16
**Stage:** B (Partitioning)
**Passes:** 2 (B1-B2)
**Status:** Production
