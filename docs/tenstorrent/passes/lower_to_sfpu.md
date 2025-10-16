# LowerToSFPU Pass

**Status**: 🔴 Not Implemented - Future Python implementation needed
**Priority**: MEDIUM (required for T.Parallel support)
**File**: To be created as `tilelang/tenstorrent/passes/lower_to_sfpu.py`

---

## Purpose

Lower intra-tile parallelism (`threadIdx.x/y/z` from `T.Parallel`) to Tenstorrent SFPU (SIMD Floating Point Unit) operations. This pass bridges the gap between TileLang's parallel loop constructs and Tenstorrent's SIMD execution model.

---

## Why Needed

**TileLang Model**: `T.Parallel(32, 32)` creates threadIdx-based intra-tile parallelism
**TT Model**: SFPU operations provide SIMD execution within a tile

This pass is required to support element-wise operations and other SIMD patterns on Tenstorrent hardware.

---

## Implementation Note

**IMPORTANT**: This pass should be implemented in **Python** following the v5 pipeline architecture, not as a C++ pass. All TT backend passes are Python-based for maintainability and rapid iteration.

The previous C++ placeholder has been removed. When implemented, this pass should:
- Be written in Python in `tilelang/tenstorrent/passes/lower_to_sfpu.py`
- Follow the v5 pass function signature: `def lower_to_sfpu(mod: IRModule) -> IRModule`
- Use TVM's Python IR mutators/visitors for transformations
- Integrate into the v5 pipeline in Stage B (after grid transformation)

---

## Planned Transformation

**Before** (TileLang with T.Parallel):
```python
with T.Kernel(4, 4) as (bx, by):
    for i, j in T.Parallel(32, 32):
        C[bx*32 + i, by*32 + j] = A[bx*32 + i, by*32 + j] + B[bx*32 + i, by*32 + j]
```

**After** (TT SFPU operations - future):
```cpp
// Planned: Map to TT SFPU intrinsics
tt::sfpu::add_tiles(dst_cb, src0_cb, src1_cb, tile_size);
```

---

## Implementation Plan

1. **Detect threadIdx constructs** ✅ (current placeholder)
2. **Analyze access patterns** to determine SFPU operation type
3. **Map to TT-Metalium SFPU APIs**:
   - Element-wise ops → `sfpu::*` intrinsics
   - Reductions → SFPU reduction kernels
   - Custom patterns → fallback to scalar loops
4. **Generate SFPU kernel code** in compute visitor
5. **Update runtime metadata** with SFPU configuration

---

## Dependencies

**Depends On**:
- `grid_to_core_grid_v5` → transforms grid to persistent cores, leaving threadIdx for SFPU

**Depended On By**:
- Compute codegen visitor → needs SFPU IR to generate proper kernel code

---

## Related Files

- `src/transform/tenstorrent/lower_to_sfpu.cc` - Implementation (placeholder)
- `tilelang/tenstorrent/passes/lower_to_sfpu.py` - Python binding
- Future: SFPU codegen in `src/target/tenstorrent/codegen_tt_compute_visitor.cc`

---

## Success Criteria

- [ ] Detects threadIdx constructs (placeholder complete)
- [ ] Maps T.Parallel to SFPU operations
- [ ] Generates correct SFPU kernel code
- [ ] Handles common SIMD patterns (add, mul, reduce, etc.)
- [ ] Tests passing for SFPU operations

---

## Pipeline Position

LowerToSFPU would run in **Stage B (Partitioning)** after grid transformation:

**Stage A: Metadata**
1. infer_tt_layout_v5
2. propagate_tt_layout_v5
3. attach_tensor_accessor_tt

**Stage B: Partitioning**
1. layout_aware_work_partition_tt_v5
2. grid_to_core_grid_v5 (transforms blockIdx → persistent cores)
3. **LowerToSFPU** (would handle threadIdx → SFPU ops) ← Future insertion point

**Stage C-E**: ... rest of v5 pipeline

**Note**: This pass is not yet integrated into the v5 pipeline.

---

**Last Updated**: 2025-10-16
