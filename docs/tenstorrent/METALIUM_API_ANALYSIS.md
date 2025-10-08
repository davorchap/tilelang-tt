# Metalium API Analysis: Real vs Current Codegen

**Date**: 2025-10-08
**Purpose**: Compare real Metalium examples with current TileLang compiler output to identify and fix discrepancies

---

## Executive Summary

**Critical Issues Found:**
1. ❌ **Wrong API names** for tile register management
2. ❌ **Incorrect API flow** for both element-wise and matmul operations
3. ❌ **Wrong CB index format**
4. ❌ **Missing initialization calls**
5. ❌ **Incorrect matmul_tiles signature**
6. ❌ **Wrong placement of tile_regs_acquire/release** in K-loop

**Impact**: Current codegen generates **non-functional** Metalium code that will not compile or run on hardware.

---

## 1. Tile Register Management APIs

### Real Metalium APIs (✅ Correct)

```cpp
tile_regs_acquire();   // Acquire 8 destination registers for use
tile_regs_commit();    // Commit computation results to registers
tile_regs_wait();      // Wait for computation to complete
tile_regs_release();   // Release registers for next operation
```

### Current TileLang Codegen (❌ Wrong)

```cpp
acquire_dst();    // WRONG NAME
commit_dst();     // WRONG NAME
wait_for_tile();  // WRONG NAME
release_dst();    // WRONG NAME
```

**Fix Required**: Rename all DST APIs to use `tile_regs_*` naming.

**Files to Fix**:
- `src/target/tt/codegen_tt_compute_visitor.cc` (lines 118-121, 455, 463)
- `src/target/tt/codegen_tt_compute_visitor.h` (EmitDSTAcquire, EmitDSTRelease methods)

---

## 2. Element-wise Add Pattern

### Real Metalium Code (tiles_add.cpp)

```cpp
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t dst_reg = 0;

    // INITIALIZATION (ONCE before loop)
    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    add_tiles_init(cb_in0, cb_in1);

    // LOOP over tiles
    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        tile_regs_acquire();
        add_tiles(cb_in0, cb_in1, 0, 0, dst_reg);
        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out0, 1);
        pack_tile(dst_reg, cb_out0);
        cb_push_back(cb_out0, 1);

        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);

        tile_regs_release();
    }
}
```

### Current TileLang Codegen (❌ Wrong)

```cpp
void MAIN() {
    // ...
    acquire_dst();  // WRONG: Called OUTSIDE loop (should be inside)

    for (uint32_t i = 0; i < tt_count; ++i) {
        cb_wait_front(CB_A, 1);  // WRONG: Should be tt::CBIndex::c_0
        cb_wait_front(CB_B, 1);

        // Compute C = A + B (element-wise)
        add_tiles_init();        // WRONG: Should be called BEFORE loop
        add_tiles(CB_A, CB_B, 0, 0, 0);

        cb_pop_front(CB_A, 1);
        cb_pop_front(CB_B, 1);
    }

    cb_reserve_back(CB_C, 1);
    commit_dst();   // WRONG: Should be inside loop
    pack_tile(0, CB_C);
    cb_push_back(CB_C, 1);
    release_dst();  // WRONG: Should be inside loop
}
```

**Issues**:
1. ❌ `tile_regs_acquire/commit/wait/release` should be **INSIDE** the loop
2. ❌ `add_tiles_init()` should be **BEFORE** the loop (initialization)
3. ❌ Missing `binary_op_init_common()` call
4. ❌ `cb_reserve_back/pack_tile/cb_push_back` should be **INSIDE** the loop
5. ❌ Wrong CB index format (`CB_A` instead of `tt::CBIndex::c_0`)

---

## 3. Matmul Pattern

### Real Metalium Code (mm.cpp)

```cpp
void MAIN {
    uint32_t num_output_tiles = get_arg_val<uint32_t>(0);
    uint32_t Kt = get_arg_val<uint32_t>(1);

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    // INITIALIZATION (ONCE before loops)
    mm_init(cb_in0, cb_in1, cb_out);

    // OUTER LOOP: For each output tile
    for (uint32_t i = 0; i < num_output_tiles; ++i) {
        tile_regs_acquire();  // ONCE per output tile, BEFORE K-loop

        // INNER LOOP: K dimension (reduction)
        for (uint32_t kt = 0; kt < Kt; kt++) {
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);

            matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);

            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
        }

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        tile_regs_release();
    }
}
```

### Current TileLang Codegen (❌ Wrong)

```cpp
void MAIN() {
    // ...
    acquire_dst();  // WRONG: Called OUTSIDE outer loop

    for (uint32_t i = 0; i < tt_count; ++i) {
        // K-loop: C[m,n] += sum(A[m,k] * B[k,n] for k in Kt)
        matmul_tiles_init(CB_A, CB_B, CB_C);
        for (uint32_t k = 0; k < 8; ++k) {
            cb_wait_front(CB_A, 1);
            cb_wait_front(CB_B, 1);

            bool accumulate = (k > 0);
            matmul_tiles(CB_A, CB_B, CB_C, accumulate);  // WRONG signature

            cb_pop_front(CB_A, 1);
            cb_pop_front(CB_B, 1);
        }

        cb_reserve_back(CB_C, 1);
    }

    commit_dst();
    pack_tile(0, CB_C);
    cb_push_back(CB_C, 1);
    release_dst();
}
```

**Issues**:
1. ❌ `tile_regs_acquire()` should be called **INSIDE outer loop**, **BEFORE K-loop**
2. ❌ `tile_regs_commit/wait/release()` should be called **INSIDE outer loop**, **AFTER K-loop**
3. ❌ `matmul_tiles` has wrong signature:
   - Real: `matmul_tiles(cb_in0, cb_in1, tile_idx_in0, tile_idx_in1, dst_tile_idx, transpose/accumulate)`
   - Ours: `matmul_tiles(CB_A, CB_B, CB_C, accumulate)`
4. ❌ Missing `mm_init()` call before loops
5. ❌ `matmul_tiles_init()` doesn't exist in real API
6. ❌ Wrong CB index format

---

## 4. Circular Buffer Index Format

### Real Metalium (✅ Correct)

```cpp
constexpr auto cb_in0 = tt::CBIndex::c_0;
constexpr auto cb_in1 = tt::CBIndex::c_1;
constexpr auto cb_out0 = tt::CBIndex::c_16;
```

### Current TileLang (❌ Wrong)

```cpp
constexpr uint32_t CB_A = 0;
constexpr uint32_t CB_B = 1;
constexpr uint32_t CB_C = 2;
```

**Fix Required**: Use `tt::CBIndex::c_N` format throughout.

**Files to Fix**:
- `src/target/tt/codegen_tt_compute_visitor.cc` (CB index declarations)
- `src/target/tt/codegen_tt_reader_visitor.cc`
- `src/target/tt/codegen_tt_writer_visitor.cc`

---

## 5. Initialization Calls

### Real Metalium Requirements

**Element-wise operations:**
```cpp
binary_op_init_common(cb_in0, cb_in1, cb_out0);
add_tiles_init(cb_in0, cb_in1);
```

**Matmul operations:**
```cpp
mm_init(cb_in0, cb_in1, cb_out);
```

**SFPU operations:**
```cpp
init_sfpu(cb_in, cb_out);
exp_tile_init();  // or other SFPU operation
```

### Current TileLang (❌ Wrong)

- ✅ We generate `add_tiles_init()` but **inside the loop** (should be before)
- ❌ Missing `binary_op_init_common()`
- ✅ We generate `matmul_tiles_init()` but this API **doesn't exist**
- ❌ Should use `mm_init()` instead

**Fix Required**:
1. Add proper initialization calls before loops
2. Remove non-existent `matmul_tiles_init()`
3. Use correct initialization for each operation type

---

## 6. API Signatures

### matmul_tiles

**Real API:**
```cpp
matmul_tiles(
    tt::CBIndex cb_in0,        // Input CB for A
    tt::CBIndex cb_in1,        // Input CB for B
    uint32_t tile_idx_in0,     // Tile index in CB for A (usually 0)
    uint32_t tile_idx_in1,     // Tile index in CB for B (usually 0)
    uint32_t dst_tile_idx,     // Destination tile index (usually 0)
    bool transpose             // Transpose flag (false for accumulation)
);
```

**Current:**
```cpp
matmul_tiles(CB_A, CB_B, CB_C, accumulate);  // WRONG - only 4 params
```

**Fix**: Use 6-parameter signature.

### add_tiles

**Real API:**
```cpp
add_tiles(
    tt::CBIndex cb_in0,        // Input CB for A
    tt::CBIndex cb_in1,        // Input CB for B
    uint32_t tile_idx_in0,     // Tile index in CB for A
    uint32_t tile_idx_in1,     // Tile index in CB for B
    uint32_t dst_reg           // Destination register index
);
```

**Current:**
```cpp
add_tiles(CB_A, CB_B, 0, 0, 0);  // Correct signature!
```

**Status**: ✅ This one is correct!

---

## 7. Control Flow Structure

### Element-wise Pattern (Correct)

```cpp
// INITIALIZATION
binary_op_init_common(...);
add_tiles_init(...);

// MAIN LOOP
for (tile in tiles) {
    cb_wait_front(...);
    tile_regs_acquire();
    add_tiles(...);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(...);
    pack_tile(...);
    cb_push_back(...);
    cb_pop_front(...);
    tile_regs_release();
}
```

### Matmul Pattern (Correct)

```cpp
// INITIALIZATION
mm_init(...);

// OUTER LOOP (output tiles)
for (out_tile in output_tiles) {
    tile_regs_acquire();  // ONCE per output tile

    // INNER LOOP (K dimension)
    for (kt in Kt) {
        cb_wait_front(...);
        matmul_tiles(..., false);  // Accumulates automatically
        cb_pop_front(...);
    }

    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(...);
    pack_tile(...);
    cb_push_back(...);
    tile_regs_release();
}
```

---

## 8. Required Fixes Summary

### High Priority (Breaks Compilation)

1. **Rename all tile register APIs**
   - `acquire_dst()` → `tile_regs_acquire()`
   - `commit_dst()` → `tile_regs_commit()`
   - `wait_for_tile()` → `tile_regs_wait()`
   - `release_dst()` → `tile_regs_release()`

2. **Fix CB index format**
   - `CB_A` → `tt::CBIndex::c_0`
   - `CB_B` → `tt::CBIndex::c_1`
   - `CB_C` → `tt::CBIndex::c_16`

3. **Fix matmul_tiles signature**
   - Change from 4 params to 6 params
   - Remove `matmul_tiles_init()` (doesn't exist)

### High Priority (Incorrect Behavior)

4. **Fix element-wise control flow**
   - Move initialization calls before loop
   - Move tile_regs_acquire/commit/wait/release inside loop
   - Move pack_tile inside loop

5. **Fix matmul control flow**
   - Move tile_regs_acquire inside outer loop, before K-loop
   - Move tile_regs_commit/wait/release inside outer loop, after K-loop
   - Add mm_init() before loops

### Medium Priority (Missing Features)

6. **Add proper initialization calls**
   - `binary_op_init_common()` for element-wise
   - `mm_init()` for matmul
   - Remove `add_tiles_init()` from inside loop

7. **Fix includes**
   - Add proper Metalium header includes
   - Remove mock API declarations for real build

---

## 9. Files to Modify

### C++ Codegen

1. **`src/target/tt/codegen_tt_compute_visitor.cc`**
   - Fix mock API names (lines 118-121)
   - Fix EmitDSTAcquire/Release calls
   - Fix CB index format
   - Fix control flow for element-wise and matmul
   - Add initialization calls

2. **`src/target/tt/codegen_tt_compute_visitor.h`**
   - Rename EmitDSTAcquire/EmitDSTRelease methods
   - Add EmitInitialization method

3. **`src/target/tt/codegen_tt_reader_visitor.cc`**
   - Fix CB index format

4. **`src/target/tt/codegen_tt_writer_visitor.cc`**
   - Fix CB index format

### Python Examples

5. **Update all example validation scripts** to check for:
   - Correct API names (`tile_regs_*`)
   - Correct CB index format (`tt::CBIndex::c_*`)
   - Correct control flow structure
   - Presence of initialization calls

---

## 10. Testing Strategy

### Unit Tests

1. Test CB index generation
2. Test tile_regs_* API calls placement
3. Test initialization call generation
4. Test matmul_tiles signature

### Integration Tests

1. Compare generated element-wise add against real `tiles_add.cpp`
2. Compare generated matmul against real `mm.cpp`
3. Verify compilation with real Metalium headers (when available)

### Validation Checks

Update validation in example files:
- Check for `tile_regs_acquire` (not `acquire_dst`)
- Check for `tt::CBIndex::c_0` (not `CB_A`)
- Check for `mm_init` or `binary_op_init_common`
- Check control flow structure (placement of acquire/release)

---

## 11. Implementation Plan

See `METALIUM_FIX_PLAN.md` for detailed task breakdown and timeline.
