# MemorySpaceLowerTT Pass

**Status**: ✅ Complete  
**Priority**: CRITICAL  
**File**: `src/transform/tt/memory_space_lower_tt.cc`

---

## Purpose

Annotate tile-local buffers with circular-buffer (CB) metadata required by the Tenstorrent code generators. The pass currently records configuration data; actual CB allocations are materialised during codegen.

---

## Why Needed

Tenstorrent kernels stage data through L1 circular buffers instead of accessing DRAM directly. Codegen needs to know:
- Which buffers should become CBs
- Expected tile size in bytes
- How many pages (double buffering vs single)
- Assigned CB identifiers

This pass extracts that information from the TIR and stores it on the `PrimFunc`.

---

## Implementation

1. Visit `DeclBuffer` statements produced by `alloc_fragment` / shared-memory tiles.
2. Heuristically identify tile-local buffers (2-D, <= 64 × 64).
3. Assign CB IDs sequentially (`cb_id = 0, 1, 2, ...`).
4. Compute tile size in bytes and choose `num_pages` (1 for accumulators, 2 for inputs).
5. Emit metadata:
   ```python
   "tt_circular_buffers" = [
     {"cb_id": 0, "num_pages": 2, "tile_size": 2048, "name": "A_tile"},
     {"cb_id": 1, "num_pages": 2, "tile_size": 2048, "name": "B_tile"},
     {"cb_id": 2, "num_pages": 1, "tile_size": 2048, "name": "C_accum"},
   ]
   "tt_num_cbs" = 3
   ```

The underlying `DeclBuffer` nodes remain unchanged; codegen replays the metadata to emit CB declarations in C++.

---

## Tests

**File**: Pending (metadata inspected indirectly through WS3 tests)

---

## Dependencies

**Depends On**:
- `infer_default_tt_shard.cc` (tile shape / padding hints)
- `grid_to_persistent_tt.cc` (ensures persistent loop form before CB planning)

**Depended On By**:
- `codegen_tt_reader_visitor.cc`, `codegen_tt_compute_visitor.cc`, `codegen_tt_writer_visitor.cc`

---

## Success Criteria

- [x] Identifies tile-local buffers that should map to CBs
- [x] Records CB IDs, tile sizes, and page counts
- [x] Leaves IR untouched apart from metadata so subsequent TVM passes remain valid

---

**Last Updated**: 2026-02-20
