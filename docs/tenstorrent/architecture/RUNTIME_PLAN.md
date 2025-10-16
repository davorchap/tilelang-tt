# TT Runtime Plan Specification

## Overview

The `tt.plan.json` file is the single source of truth for coordinating host and device execution in the Tenstorrent backend. It contains all necessary information for launching kernels, configuring data movement, and managing core resources.

## File Format

The runtime plan is a JSON file with the following structure:

```json
{
  "core_grid": [gx, gy],
  "core_ranges": [...],
  "work_partition": {...},
  "layouts": {...}
}
```

## Field Descriptions

### core_grid

**Type:** `[int, int]`

Specifies the dimensions of the core grid.

**Example:**
```json
"core_grid": [4, 4]
```

This defines a 4×4 grid of cores (16 cores total).

### core_ranges

**Type:** `Array<CoreRange>`

Defines the active rectangular regions of cores. Multiple disjoint ranges are supported.

**CoreRange Structure:**
```json
{
  "start": [sx, sy],  // Starting core coordinates
  "extent": [ex, ey]  // Width and height of the range
}
```

**Examples:**

Single range covering entire grid:
```json
"core_ranges": [
  {"start": [0, 0], "extent": [4, 4]}
]
```

Multiple disjoint ranges:
```json
"core_ranges": [
  {"start": [0, 0], "extent": [2, 2]},
  {"start": [2, 2], "extent": [2, 2]}
]
```

### work_partition

**Type:** `Map<string, Array<WorkItem>>`

Maps core coordinates to lists of work items. Keys are stringified coordinates `"(cx,cy)"`.

**WorkItem Structure:**
```json
{
  "io": int,              // M-dimension tile index
  "jo": int,              // N-dimension tile index
  "len_k": int,           // Optional: K-dimension extent
  "tile_order": string    // Optional: traversal order
}
```

**Tile Order Values:**
- `"row_major"`: Process tiles left-to-right, top-to-bottom
- `"column_major"`: Process tiles top-to-bottom, left-to-right
- `"match_shard"`: Match the buffer's shard layout order
- `"z_order"`: Z-order (Morton) curve traversal

**Example:**
```json
"work_partition": {
  "(0,0)": [
    {"io": 0, "jo": 0, "len_k": 128, "tile_order": "row_major"},
    {"io": 0, "jo": 1, "len_k": 128, "tile_order": "row_major"}
  ],
  "(0,1)": [
    {"io": 1, "jo": 0, "len_k": 128, "tile_order": "row_major"}
  ]
}
```

### layouts

**Type:** `Map<string, LayoutDescriptor>`

Describes memory layout and sharding for each buffer.

**LayoutDescriptor Structure:**
```json
{
  "shard": string,           // Memory location: "DRAM" or "L1"
  "interleave": bool,        // Whether to interleave across banks
  "stride": [int, int],      // Optional: memory strides
  "tile_id_order": string    // Tile addressing order
}
```

**Example:**
```json
"layouts": {
  "A": {
    "shard": "DRAM",
    "interleave": true,
    "stride": [1024, 32]
  },
  "B": {
    "shard": "DRAM",
    "interleave": false
  },
  "C": {
    "shard": "L1",
    "tile_id_order": "row_major"
  }
}
```

## Complete Example

Here's a complete runtime plan for a 128×128 matrix multiplication on a 2×2 core grid:

```json
{
  "core_grid": [2, 2],
  "core_ranges": [
    {"start": [0, 0], "extent": [2, 2]}
  ],
  "work_partition": {
    "(0,0)": [
      {"io": 0, "jo": 0, "len_k": 128, "tile_order": "row_major"},
      {"io": 0, "jo": 1, "len_k": 128, "tile_order": "row_major"}
    ],
    "(0,1)": [
      {"io": 0, "jo": 2, "len_k": 128, "tile_order": "row_major"},
      {"io": 0, "jo": 3, "len_k": 128, "tile_order": "row_major"}
    ],
    "(1,0)": [
      {"io": 1, "jo": 0, "len_k": 128, "tile_order": "row_major"},
      {"io": 1, "jo": 1, "len_k": 128, "tile_order": "row_major"}
    ],
    "(1,1)": [
      {"io": 1, "jo": 2, "len_k": 128, "tile_order": "row_major"},
      {"io": 1, "jo": 3, "len_k": 128, "tile_order": "row_major"}
    ]
  },
  "layouts": {
    "A": {
      "shard": "DRAM",
      "interleave": true
    },
    "B": {
      "shard": "DRAM",
      "interleave": true
    },
    "C": {
      "shard": "L1",
      "tile_id_order": "row_major"
    }
  }
}
```

## Python API

### Creating a Plan

```python
from tilelang.tenstorrent import (
    CoreRange, WorkItem, plan_dict
)

# Define components
core_grid = (4, 4)
core_ranges = [CoreRange((0, 0), (4, 4))]
work_partition = {
    "(0,0)": [WorkItem(io=0, jo=0, len_k=128)]
}
layouts = {
    "A": {"shard": "DRAM"},
    "B": {"shard": "DRAM"},
    "C": {"shard": "L1"}
}

# Create plan dictionary
plan = plan_dict(core_grid, core_ranges, work_partition, layouts)
```

### Emitting from IR

```python
from tilelang.tenstorrent import emit_tt_plan

# Emit plan from a PrimFunc with metadata
emit_tt_plan(func, out_path="my_plan.json")
```

### Loading and Validation

```python
from tilelang.tenstorrent import load_tt_plan, validate_plan

# Load plan from file
plan = load_tt_plan("my_plan.json")

# Validate plan structure
errors = validate_plan(plan)
if errors:
    print("Plan validation failed:")
    for error in errors:
        print(f"  - {error}")
```

## Host Runtime Usage

The host runtime reads the plan to:
1. Configure core activation
2. Set up memory allocations
3. Program DMA engines
4. Launch device kernels

```cpp
// Pseudo-code for host runtime
TTRuntimePlan plan = LoadPlan("tt.plan.json");

// Activate cores based on core_ranges
for (auto& range : plan.core_ranges) {
    ActivateCores(range.start, range.extent);
}

// Allocate buffers based on layouts
for (auto& [name, layout] : plan.layouts) {
    if (layout.shard == "DRAM") {
        AllocateDRAM(name, layout);
    } else {
        AllocateL1(name, layout);
    }
}

// Launch kernels with work assignments
for (auto& [core, work_items] : plan.work_partition) {
    LaunchKernel(core, work_items);
}
```

## Device Kernel Usage

Device kernels use the plan to determine their work assignment:

```cpp
// Pseudo-code for device kernel
uint32_t core_id = GetCoreID();
WorkList my_work = GetWorkForCore(core_id);

for (auto& work_item : my_work) {
    ProcessTile(work_item.io, work_item.jo, work_item.len_k);
}
```

## Validation Rules

The runtime plan must satisfy:

1. **Core ranges within grid bounds**: All core ranges must fit within the specified grid dimensions
2. **Non-overlapping ranges**: Core ranges should not overlap (unless intentionally replicated)
3. **Complete tile coverage**: All required output tiles must be assigned to at least one core
4. **Valid layout shards**: Shard values must be "DRAM" or "L1"
5. **Consistent buffer references**: All buffers in layouts must correspond to actual function parameters

## Performance Considerations

- **Work balance**: Distribute work items evenly across cores
- **Memory locality**: Place frequently accessed buffers in L1
- **Interleaving**: Enable for better bank utilization
- **Tile order**: Match computation pattern for cache efficiency

## Extensions

Future versions may support:
- Multi-chip configurations
- Hierarchical work partitions
- Dynamic work stealing
- Heterogeneous core types
- Conditional execution plans

## References

- [Pass Pipeline](passes/NEW_PIPELINE.md)
- [Architecture Overview](NEW_LOWERING_ARCHITECTURE.md)
- [Python API Reference](API_REFERENCE.md)
