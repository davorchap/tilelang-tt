# Task: Schedule Inference Pass

## Goal
Implement `InferDefaultTTSchedule` - a TVM pass that computes contiguous per-core tile ranges and runtime argument schemas for Tenstorrent kernel execution.

## Context
- **Workstream:** WS2 - Schedule & Sharding Metadata
- **Dependencies:** WS1 complete (default annotations present)
- **File:** `src/tt/transform/infer_tt_schedule.cc`
- **Priority:** High (blocks sharding inference)

## What This Pass Does

### Inputs
- PrimFunc with `T.Kernel(grid_x, grid_y)` metadata
- Default annotations from WS1:
  - `tt_schedule_policy = "contiguous"`
  - `tt_schedule_order = "row_major"`

### Processing
1. **Read grid dimensions** from `T.Kernel` attributes
2. **Compute total tiles:** `num_tiles = grid_x * grid_y`
3. **Query available cores** (MVP: assume fixed number, e.g., 64 Tensix cores)
4. **Partition tiles contiguously:**
   - For each core, assign a contiguous range of tiles
   - Use row-major ordering: `tile_id = by * grid_x + bx`
   - Compute `(start_id, count)` per core
5. **Generate runtime args schema:**
   - Schema for kernel invocation: `[start_id, count, grid_x, grid_y, kt_tiles]`
   - Attach to function metadata

### Outputs
Enhanced PrimFunc with scheduling metadata:

```cpp
// Attributes attached to PrimFunc:
func->attrs.Set("tt_num_tiles", IntImm(num_tiles));
func->attrs.Set("tt_grid_x", IntImm(grid_x));
func->attrs.Set("tt_grid_y", IntImm(grid_y));
func->attrs.Set("tt_num_cores", IntImm(num_cores));
func->attrs.Set("tt_tiles_per_core", Array<Array<IntImm>>(...));  // Per-core (start, count)
func->attrs.Set("tt_runtime_args_schema", Map<String, ObjectRef>(...));
```

## Implementation Plan

### Step 1: Pass Registration
```cpp
// src/tt/transform/infer_tt_schedule.cc
namespace tvm {
namespace tir {
namespace transform {

Pass InferDefaultTTSchedule() {
  auto pass_func = [](PrimFunc func, IRModule mod, PassContext ctx) {
    return InferDefaultTTScheduleImpl(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InferDefaultTTSchedule", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InferDefaultTTSchedule")
.set_body_typed(InferDefaultTTSchedule);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
```

### Step 2: Core Logic
```cpp
PrimFunc InferDefaultTTScheduleImpl(PrimFunc func) {
  // 1. Extract grid dimensions from T.Kernel metadata
  //    Look for func->attrs["tl_grid_x"] and func->attrs["tl_grid_y"]

  // 2. Read tt_schedule_policy (should be "contiguous" from WS1 defaults)

  // 3. Compute num_tiles = grid_x * grid_y

  // 4. Query num_cores (hardcode 64 for MVP, make configurable later)

  // 5. Partition tiles:
  //    tiles_per_core_base = num_tiles / num_cores
  //    remainder = num_tiles % num_cores
  //    For each core:
  //      start_id = core_id * tiles_per_core_base + min(core_id, remainder)
  //      count = tiles_per_core_base + (core_id < remainder ? 1 : 0)

  // 6. Build per-core array: [[start_0, count_0], [start_1, count_1], ...]

  // 7. Create runtime args schema

  // 8. Attach all metadata to func->attrs

  return func;
}
```

### Step 3: Tile Partitioning Algorithm
```cpp
// Contiguous row-major partitioning
struct TileRange {
  int start_id;
  int count;
};

std::vector<TileRange> PartitionTilesContiguous(int num_tiles, int num_cores) {
  std::vector<TileRange> ranges;
  int tiles_per_core_base = num_tiles / num_cores;
  int remainder = num_tiles % num_cores;

  int current_start = 0;
  for (int core_id = 0; core_id < num_cores; ++core_id) {
    int count = tiles_per_core_base + (core_id < remainder ? 1 : 0);
    if (count > 0) {
      ranges.push_back({current_start, count});
      current_start += count;
    } else {
      ranges.push_back({0, 0});  // Inactive core
    }
  }

  return ranges;
}
```

## Testing Strategy

### C++ Unit Test
**File:** `tests/cpp/tt/test_infer_tt_schedule.cc`

```cpp
TEST(InferTTSchedule, BasicGrid8x8) {
  // Create PrimFunc with 8x8 grid
  // Apply InferDefaultTTSchedule pass
  // Verify:
  //   - tt_num_tiles = 64
  //   - tt_grid_x = 8, tt_grid_y = 8
  //   - All 64 cores get 1 tile each (if 64 cores)
  //   - start_ids are contiguous [0, 1, 2, ..., 63]
}

TEST(InferTTSchedule, UnevenDistribution) {
  // Create PrimFunc with 100 tiles, 64 cores
  // Verify:
  //   - First 36 cores get 2 tiles each
  //   - Remaining 28 cores get 1 tile each
  //   - Total = 36*2 + 28*1 = 100 ✓
}

TEST(InferTTSchedule, SingleTile) {
  // Create PrimFunc with 1x1 grid
  // Verify:
  //   - Only core 0 active (start=0, count=1)
  //   - Other cores inactive (count=0)
}
```

## TVM Integration Points

### 1. Reading Kernel Metadata
TileLang likely stores grid dimensions in PrimFunc attributes. Check:
- `func->attrs["tl_grid_x"]`
- `func->attrs["tl_grid_y"]`
- Or look at how `T.Kernel` is lowered in existing TileLang code

### 2. Attribute Storage
Use TVM's attribute system:
```cpp
func = func.CopyOnWrite();
func->attrs.Set("tt_num_tiles", IntImm(DataType::Int(32), num_tiles));
func->attrs.Set("tt_tiles_per_core", MakeTilesArray(...));
```

### 3. Pass Ordering
This pass should run:
- **After:** WS1 default annotation (in Python)
- **Before:** WS2 shard inference (depends on tile counts)
- **Before:** WS3 TIR transforms

## Open Questions

1. **Core count:** Hardcode 64 for MVP, or query from target config?
   - **Recommendation:** Hardcode 64, add TODO for target query

2. **Grid dimensions source:** Where does TileLang store `T.Kernel(grid_x, grid_y)`?
   - **Action:** Investigate existing TileLang lowering to find attribute names

3. **Inactive cores:** How to represent cores with no tiles?
   - **Recommendation:** Store `(start_id=0, count=0)` for inactive cores

4. **Error handling:** What if grid dimensions not found?
   - **Recommendation:** Raise TVM error with clear message

## Acceptance Criteria

- ✅ C++ pass implemented and registered with TVM
- ✅ Pass can be invoked via `TVM_REGISTER_GLOBAL`
- ✅ C++ unit tests pass (basic grid, uneven distribution, single tile)
- ✅ Attributes correctly attached to PrimFunc
- ✅ Tile partitioning math verified for edge cases

## Next Steps

After schedule inference complete:
1. Implement shard inference pass (depends on tile counts)
2. Add Python bindings to expose pass
3. Add Python integration test using real TileLang GEMM

## References

- [Unified MVP Plan](../UNIFIED_MATMUL_MVP_PLAN.md) - WS2 specification
- [WS2 Status](WS2_STATUS.md) - Overall WS2 progress
- TVM Pass Infrastructure: `tvm/src/tir/transform/` (examples)
