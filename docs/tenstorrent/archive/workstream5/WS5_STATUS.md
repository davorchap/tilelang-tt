# Workstream 5 Status - Reader/Writer Kernel Emission

**Last Updated:** 2025-10-07

## Overview

Workstream 5 focuses on completing the kernel emission infrastructure:
- **Reader kernel**: DRAM → L1 circular buffer data movement
- **Writer kernel**: L1 circular buffer → DRAM data writeback
- **Circular buffer management**: CB allocation, push/pop operations
- **Integration with compute kernel**: Complete 3-kernel pipeline

**Goal:** Generate complete reader/compute/writer kernel trio for TT execution.

**Status:** ⏳ **NOT STARTED** - Next priority after WS4

## Progress Summary

| Task | Status | Priority | Blocker |
|------|--------|----------|---------|
| **WS5 Documentation Structure** | ✅ **COMPLETE** | High | None |
| **Reader Kernel Architecture** | ❌ TODO | **Critical** | None |
| **Writer Kernel Architecture** | ❌ TODO | **Critical** | None |
| **Circular Buffer Abstraction** | ❌ TODO | **Critical** | None |
| **EmitTTReaderKernel Implementation** | ❌ TODO | **Critical** | Architecture |
| **EmitTTWriterKernel Implementation** | ❌ TODO | **Critical** | Architecture |
| **CB API Integration** | ❌ TODO | High | Kernels |
| **3-Kernel Coordination** | ❌ TODO | High | All kernels |
| **Integration Tests** | ❌ TODO | High | Implementation |

**Overall WS5 Progress:** 10% (Planning only)

---

## Implementation Plan

### Phase 1: Reader Kernel Architecture

**File:** `src/target/tt/codegen_tt.cc` (extend existing)

**Objective:** Generate reader kernel that loads tiles from DRAM to L1 circular buffers.

**Reader Kernel Template:**
```cpp
// Reader Kernel (Data Mover: DRAM → L1)
void reader_kernel(
    uint32_t cb_id_in,
    uint32_t tile_bytes,
    uint32_t num_tiles,
    volatile tt_l1_ptr uint32_t* dram_addr
) {
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_reserve_back(cb_id_in, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in);

        noc_async_read(dram_addr, l1_write_addr, tile_bytes);
        noc_async_read_barrier();

        cb_push_back(cb_id_in, 1);
        dram_addr += tile_bytes / sizeof(uint32_t);
    }
}
```

**Key Design Decisions:**
- **Circular Buffer IDs**: Fixed mapping (CB 0=A, CB 1=B, CB 2=C)
- **Tile Size**: 32×32 × sizeof(dtype) bytes
- **NOC Operations**: Use `noc_async_read` for DMA
- **Barrier Sync**: `noc_async_read_barrier()` ensures completion

**Implementation Steps:**
1. Add `EmitTTReaderKernel()` function
2. Read buffer metadata from WS2 (buffer shapes, tile counts)
3. Generate CB reservation/push pairs
4. Emit NOC async read operations
5. Handle multiple input buffers (A, B for GEMM)

---

### Phase 2: Writer Kernel Architecture

**File:** `src/target/tt/codegen_tt.cc` (extend existing)

**Objective:** Generate writer kernel that writes tiles from L1 circular buffers to DRAM.

**Writer Kernel Template:**
```cpp
// Writer Kernel (Data Mover: L1 → DRAM)
void writer_kernel(
    uint32_t cb_id_out,
    uint32_t tile_bytes,
    uint32_t num_tiles,
    volatile tt_l1_ptr uint32_t* dram_addr
) {
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_id_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);

        noc_async_write(l1_read_addr, dram_addr, tile_bytes);
        noc_async_write_barrier();

        cb_pop_front(cb_id_out, 1);
        dram_addr += tile_bytes / sizeof(uint32_t);
    }
}
```

**Key Design Decisions:**
- **Single Output Buffer**: Most kernels write to 1 output (C for GEMM)
- **Wait-Write-Pop Pattern**: Standard producer-consumer flow
- **NOC Writeback**: Use `noc_async_write` for DMA

**Implementation Steps:**
1. Add `EmitTTWriterKernel()` function
2. Read output buffer metadata
3. Generate CB wait/pop pairs
4. Emit NOC async write operations

---

### Phase 3: Circular Buffer Management

**Objective:** Generate CB allocation and configuration code.

**CB Configuration:**
```cpp
// Circular Buffer Setup (in host or config)
constexpr uint32_t CB_A = 0;
constexpr uint32_t CB_B = 1;
constexpr uint32_t CB_C = 2;

CircularBufferConfig cb_config_a = {
    .buffer_index = CB_A,
    .tile_size = 32 * 32 * sizeof(float16),
    .num_pages = 2,  // Double buffering
    .data_format = DataFormat::Float16_b
};
```

**Implementation Steps:**
1. Add `EmitTTCBConfig()` helper
2. Determine CB depth (default: 2 for double buffering)
3. Calculate tile sizes based on dtype
4. Emit CB index constants

---

### Phase 4: 3-Kernel Coordination

**Objective:** Ensure reader → compute → writer pipeline works correctly.

**Synchronization Pattern:**
- **Reader ↔ Compute**: CB acts as producer-consumer queue
  - Reader: `cb_push_back(CB_A, n)` → Compute: `cb_wait_front(CB_A, n)`
- **Compute ↔ Writer**: Similar pattern for output
  - Compute: `cb_push_back(CB_C, n)` → Writer: `cb_wait_front(CB_C, n)`

**Implementation Steps:**
1. Update `EmitTTComputeKernel()` to add CB wait/push
2. Verify CB indices match across all 3 kernels
3. Add synchronization comments to generated code

---

### Phase 5: Integration with Codegen

**File:** `src/target/tt/codegen_tt.cc`

**Update `CodegenTT()` to emit all 3 kernels:**
```cpp
std::unordered_map<std::string, std::string> CodegenTT(const IRModule& mod, const std::string& target) {
    std::unordered_map<std::string, std::string> artifacts;

    PrimFunc main_func = /* ... */;

    // Generate all 3 kernels
    artifacts["reader.cpp"] = EmitTTReaderKernel(main_func);
    artifacts["compute.cpp"] = EmitTTComputeKernel(main_func);
    artifacts["writer.cpp"] = EmitTTWriterKernel(main_func);

    // Generate metadata
    artifacts["tt.plan.json"] = EmitTTPlanJSON(main_func);

    return artifacts;
}
```

---

## Testing Strategy

### Unit Tests

**File:** `testing/python/tt/test_ws5_reader_writer.py`

**Test Cases:**
1. **test_emit_reader_kernel_basic()**
   - Verify reader kernel generated
   - Check CB API calls present
   - Validate NOC operations

2. **test_emit_writer_kernel_basic()**
   - Verify writer kernel generated
   - Check CB API calls present
   - Validate NOC operations

3. **test_3_kernel_coordination()**
   - Verify all 3 kernels generated
   - Check CB indices match
   - Validate synchronization pattern

4. **test_multiple_buffers()**
   - Test GEMM with 2 inputs (A, B)
   - Verify separate CB handling

---

## Build & Test Instructions

```bash
# Rebuild with new kernels
USE_LLVM=true pip install -e . --no-build-isolation

# Run WS5 tests
pytest testing/python/tt/test_ws5_reader_writer.py -v
```

---

## Dependencies

- **WS1-4 complete** ✓
- **TVM C++ infrastructure** (codegen framework)
- **Metalium CB API knowledge** (cb_reserve_back, cb_push_back, etc.)

---

## Success Criteria

WS5 is complete when:
- [ ] Reader kernel generation working
- [ ] Writer kernel generation working
- [ ] All 3 kernels coordinate correctly
- [ ] CB API calls correct
- [ ] 5+ integration tests passing
- [ ] No regressions in existing 23 tests

---

## Key Design Principles

1. **Keep It Simple**: Use fixed CB indices (0, 1, 2) for MVP
2. **Standard Patterns**: Follow Metalium best practices (double buffering)
3. **Dry-Run Focus**: Generate compilable C++ without hardware dependencies
4. **Extensibility**: Design for future addition of multi-core variants

---

## Timeline

**Estimated Effort:** 4-6 hours

- **Hour 1-2**: Implement EmitTTReaderKernel()
- **Hour 3-4**: Implement EmitTTWriterKernel()
- **Hour 5**: CB coordination and integration
- **Hour 6**: Testing and validation

---

## Related Documentation

- [WS4 Status](../workstream4/WS4_STATUS.md) - Compute kernel (foundation)
- [WS6 Status](../workstream6/WS6_STATUS.md) - Host program (next)
- [Metalium Guide](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md)

---
