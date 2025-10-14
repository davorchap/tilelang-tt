# New TT Lowering Architecture: Grid → Persistent

This document describes the new metadata-driven lowering architecture for the Tenstorrent backend, which replaces the previous ad-hoc approach with a cleaner, more maintainable pipeline.

## Overview

The new architecture introduces:
1. **Mid-level IR representation** with `T.launch_core` abstractions
2. **Centralized metadata** via attributes on PrimFuncs
3. **Clear pass pipeline** with well-defined responsibilities
4. **Single source of truth** via `tt.plan.json` for host/device coordination

## Key Design Principles

### 1. Metadata-Driven Approach
Instead of hardcoding transformations, all core assignments, work partitioning, and layout information is captured as metadata attributes on the IR. This allows passes to be more modular and easier to test.

### 2. Late Lowering to Persistent Kernels
The transformation to persistent kernels happens as the final pass (`GridToPersistentTT`), keeping the IR analyzable and schedulable for as long as possible.

### 3. Single Source of Truth
The `tt.plan.json` file serves as the canonical representation of the runtime execution plan, consumed by both host and device code.

## Pass Pipeline

```python
# 1. InferTTLayout
#    - Stamps initial layout/shard metadata onto buffers
#    - Applies sensible defaults (DRAM for inputs, L1 for outputs)

# 2. PropagateTTLayout  
#    - Normalizes and propagates layout descriptors
#    - Ensures consistency across the IR

# 3. TTTilesToCoreMap
#    - Computes CoreRange(s) based on grid size
#    - Builds layout-aware work partition
#    - Assigns tiles to cores

# 4. LowerTTTileIntrinsics
#    - Maps high-level tile ops to TT device intrinsics
#    - T.gemm → TT matrix engine ops
#    - tile_load/store → TT DMA operations

# 5. GridToPersistentTT
#    - Injects persistent outer loops
#    - Adds staging buffers and double-buffering
#    - Emits tt.plan.json
```

## Metadata Attributes

All attributes are centralized in `tilelang.tenstorrent.attrs`:

- `tt.core_grid`: (gx, gy) - dimensions of the core grid
- `tt.core_ranges`: List of CoreRange objects defining active regions
- `tt.work_partition`: Mapping of core coordinates to WorkItem lists
- `tt.layout_desc`: Buffer layout descriptors (shard, interleave, etc.)

## Mid-Level IR Example

```python
@T.prim_func
def mm_core_tiles(A: T.Buffer((M, K), "float16"),
                  B: T.Buffer((K, N), "float16"),
                  C: T.Buffer((M, N), "float32")):
    # Mid-level representation with core domains
    cx = T.launch_core("coreIdx.x", extent=4)
    cy = T.launch_core("coreIdx.y", extent=4)
    
    with T.block("C.tile"):
        io = T.axis.spatial(M_tiles, T.attr("tt.core_map_i", cx, cy))
        jo = T.axis.spatial(N_tiles, T.attr("tt.core_map_j", cx, cy))
        T.reads(A[io*TM:(io+1)*TM, 0:K], B[0:K, jo*TN:(jo+1)*TN])
        T.writes(C[io*TM:(io+1)*TM, jo*TN:(jo+1)*TN])
        
        for ko in range(0, K, TK):
            a_tile = T.tile_load(A, io*TM, ko, shape=(TM, TK))
            b_tile = T.tile_load(B, ko, jo*TN, shape=(TK, TN))
            T.gemm(a_tile, b_tile)
```

## Runtime Plan Format (tt.plan.json)

```json
{
  "core_grid": [4, 4],
  "core_ranges": [
    {"start": [0, 0], "extent": [4, 4]}
  ],
  "work_partition": {
    "(0,0)": [
      {"io": 0, "jo": 0, "len_k": 128, "tile_order": "row_major"},
      {"io": 0, "jo": 1, "len_k": 128, "tile_order": "row_major"}
    ],
    "(0,1)": [
      {"io": 1, "jo": 0, "len_k": 128, "tile_order": "row_major"}
    ]
  },
  "layouts": {
    "A": {"shard": "DRAM", "interleave": true},
    "B": {"shard": "DRAM", "interleave": true},
    "C": {"shard": "L1", "tile_id_order": "row_major"}
  }
}
```

## Usage Example

```python
from tilelang.tenstorrent import *

# Build your PrimFunc
@T.prim_func
def matmul(...):
    # ... compute definition ...

# Attach metadata
f = with_core_grid(matmul, 4, 4)
f = with_core_ranges(f, [CoreRange((0,0), (4,4))])
f = with_layout_desc(f, {
    "A": {"shard": "DRAM"}, 
    "B": {"shard": "DRAM"},
    "C": {"shard": "L1"}
})

# Run the pipeline
mod = tvm.IRModule({"main": f})
mod = run_pipeline(mod, plan_path="my_plan.json")
```

## Migration Guide

### For Existing Code

The new architecture maintains backward compatibility through the legacy pass names. However, we recommend migrating to the new metadata-driven approach:

**Old approach:**
```python
# Scattered metadata across multiple passes
func = apply_tt_defaults(func)
func = infer_tt_layout(func)
# ... many individual passes
```

**New approach:**
```python
# Centralized metadata + single pipeline
func = with_core_grid(func, 4, 4)
func = with_layout_desc(func, layouts)
mod = run_pipeline(tvm.IRModule({"main": func}))
```

## Benefits

1. **Maintainability**: Clear separation of concerns between passes
2. **Testability**: Each pass can be tested in isolation
3. **Debuggability**: Metadata is visible and inspectable at each stage
4. **Extensibility**: New passes can be easily inserted into the pipeline
5. **Performance**: Late lowering preserves optimization opportunities

## Future Work

- Support for multi-chip configurations
- Advanced work partitioning strategies (2D block-cyclic, etc.)
- Automatic layout inference from memory patterns
- Integration with autotuning framework

## References

- [TIR Basics](TIR_BASICS.md)
- [Pass Pipeline Details](passes/pipeline.md)
- [Runtime Plan Specification](RUNTIME_PLAN.md)
