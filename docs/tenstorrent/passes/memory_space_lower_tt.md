# MemorySpaceLowerTT Pass

**Status**: ✅ Complete
**Priority**: CRITICAL
**File**: `src/transform/tt/memory_space_lower_tt.cc`

---

## Purpose

Lower DRAM buffer allocations to L1 circular buffers (CBs), the fundamental data structure for Tensix cores.

---

## Why Needed

Tenstorrent cores don't access DRAM directly. All data flows through **L1 circular buffers**:
- Reader kernel: DRAM → CB
- Compute kernel: CB → Compute → CB
- Writer kernel: CB → DRAM

This pass creates CB allocations and management code.

---

## Transformation

**Before** (DRAM buffers):
```python
A = T.alloc_buffer((256, 256), "float16", scope="global")  # DRAM
B = T.alloc_buffer((256, 256), "float16", scope="global")
C = T.alloc_buffer((256, 256), "float16", scope="global")
```

**After** (L1 circular buffers):
```cpp
// L1 circular buffer allocations
CircularBufferConfig cb_in0_config(
    num_pages * page_size,  // total size
    {{CB::c_in0, DataFormat::Float16_b}}  // CB ID and format
);
cb_in0_config.set_page_size(CB::c_in0, 2048);  // 32×32 FP16 tile = 2KB

CircularBufferConfig cb_in1_config(...);  // For B
CircularBufferConfig cb_out0_config(...);  // For C
```

**Mapping**:
- Input `A` → `cb_in0` (CB index 0)
- Input `B` → `cb_in1` (CB index 1)
- Output `C` → `cb_out0` (CB index 2)

---

## Implementation Details

**Steps**:
1. Identify DRAM buffers in IR
2. Allocate CB indices (cb_in0, cb_in1, cb_out0, ...)
3. Calculate CB sizes:
   - **Page size**: 32×32 × sizeof(dtype) bytes
   - **Num pages**: Typically 2 (double buffering)
4. Replace buffer accesses with CB operations
5. Insert CB management code (reserve, push, wait, pop)

**CB Double Buffering**:
- 2 pages per CB enables overlap:
  - Reader fills page 1 while compute uses page 0
  - Improves throughput

---

## Tests

**File**: `testing/python/tt/test_memory_space_lower_tt.py`
**Status**: ✅ 8 tests passing

Tests cover:
- CB allocation correctness
- Page size calculation
- Buffer-to-CB mapping
- Double buffering configuration
- Multiple buffers

---

## Dependencies

**Depends On**:
- `infer_default_tt_shard.cc` - Uses shard info for sizing
- `tile_pad_tt.cc` - Requires tile-aligned buffers

**Depended On By**:
- `codegen_tt_reader_visitor.cc` - Emits CB push operations
- `codegen_tt_compute_visitor.cc` - Emits CB wait/pop operations
- `codegen_tt_writer_visitor.cc` - Emits CB pop operations

---

## Related Files

- `src/transform/tt/memory_space_lower_tt.cc` - Implementation
- `tilelang/tt/passes.py` - Python binding
- `testing/python/tt/test_memory_space_lower_tt.py` - Tests

---

## Success Criteria

- [x] Creates CB allocations for all DRAM buffers
- [x] Correct page sizes (2KB for FP16 32×32 tiles)
- [x] Enables double buffering (2 pages per CB)
- [x] All tests passing (8/8)

---

**Last Updated**: 2025-10-09
