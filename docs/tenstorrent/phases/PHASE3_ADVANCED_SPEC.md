# Phase 3: Advanced Patterns - GEMV, Convolution, Split-K

**Timeline**: Weeks 23-24 (2 weeks)
**Priority**: MEDIUM - Advanced compute patterns
**Status**: ⏳ Not Started (Blocked by Phase 2)

---

## Overview

Phase 3 introduces advanced compute patterns beyond simple element-wise and GEMM:
- **GEMV**: Matrix-vector multiplication with broadcast patterns
- **Convolution**: Im2col or direct convolution with weight multicast
- **Split-K GEMM**: K-dimension partitioning across cores with reduction

---

## Examples Covered

### 3.1 GEMV (Matrix-Vector Multiplication)
**File**: `examples/gemv/*.py`

**What's New**:
- Non-square tile patterns (vector tiles: 1×32, 32×1)
- Vector broadcast patterns
- Different CB depths for matrix vs vector

**Required Transforms**:
- 🆕 **NEW**: Non-square tile handling
  - Adjust tile padding for vector shapes
  - Handle 1×32 and 32×1 tile sizes

**Codegen Additions**:
- Vector broadcast in compute kernel
- Non-square DST tile handling
- Proper CB sizing for vectors

**Expected Pattern**:
```cpp
// y = A @ x (M×K matrix, K vector → M vector)
for (uint32_t m = 0; m < Mt; ++m) {
    acquire_dst();
    matmul_tiles_init(CB_A, CB_x, CB_y);

    for (uint32_t k = 0; k < Kt; ++k) {
        cb_wait_front(CB_A, 1);  // Matrix tile
        cb_wait_front(CB_x, 1);  // Vector tile (broadcast)

        matmul_tiles(CB_A, CB_x, 0, 0, 0, false);

        cb_pop_front(CB_A, 1);
        // Don't pop vector - reuse for all M iterations
    }

    cb_reserve_back(CB_y, 1);
    commit_dst();
    pack_tile(0, CB_y);
    cb_push_back(CB_y, 1);
    release_dst();
}
// Pop vector once at end
cb_pop_front(CB_x, Kt);
```

---

### 3.2 Convolution
**File**: `examples/convolution/*.py`

**What's New**:
- Im2col transformation OR direct convolution
- Weight reuse via multicast
- Complex indexing for sliding windows

**Required Transforms**:
- 🆕 **NEW**: Convolution pattern recognition
  - Im2col transform (convert to GEMM) OR
  - Direct convolution with window indexing
- 🆕 **NEW**: Weight multicast planning
  - Identify which weights can be broadcast
  - Plan NoC multicast groups

**Codegen Additions**:
- Multicast for weight tiles (broadcast to multiple cores)
- Window indexing for input tiles
- Handle padding (im2col or direct)

**Expected Pattern** (Im2col approach):
```cpp
// Conv2D as GEMM after im2col
// Input: im2col(I, kernel_size) → IM2COL matrix
// Weight: W (flattened filters)
// Output: O (output feature map)

// Same as GEMM with multicast weights
for (uint32_t out_tile = 0; out_tile < num_output_tiles; ++out_tile) {
    acquire_dst();
    matmul_tiles_init(CB_IM2COL, CB_WEIGHT, CB_OUT);

    for (uint32_t k = 0; k < Kt; ++k) {
        cb_wait_front(CB_IM2COL, 1);
        // Weights multicast - all cores get same weight tile
        noc_multicast_wait(CB_WEIGHT, 1);

        matmul_tiles(CB_IM2COL, CB_WEIGHT, 0, 0, 0, false);

        cb_pop_front(CB_IM2COL, 1);
    }

    cb_reserve_back(CB_OUT, 1);
    commit_dst();
    pack_tile(0, CB_OUT);
    cb_push_back(CB_OUT, 1);
    release_dst();
}
```

---

### 3.3 Split-K GEMM
**File**: `examples/gemm_splitk/*.py`

**What's New**:
- K-dimension split across cores
- Partial results written to temp buffer
- Second kernel to reduce partial sums

**Required Transforms**:
- 🆕 **NEW**: Split-K planning
  - Divide K-dimension among cores
  - Plan two-phase execution:
    - Phase 1: Each core computes partial sum
    - Phase 2: Reduce partial sums

**Codegen Additions**:
- Partial accumulation with separate output buffer
- Reduction kernel (sum partial results)
- Synchronization between phases

**Expected Pattern** (Two kernels):

**Kernel 1 (Partial GEMM)**:
```cpp
// Each core processes subset of K
void MAIN() {
    uint32_t k_start = get_arg_val<uint32_t>(3);
    uint32_t k_count = get_arg_val<uint32_t>(4);

    for (uint32_t tile_idx = 0; tile_idx < num_output_tiles; ++tile_idx) {
        acquire_dst();
        matmul_tiles_init(CB_A, CB_B, CB_PARTIAL);

        for (uint32_t k = k_start; k < k_start + k_count; ++k) {
            cb_wait_front(CB_A, 1);
            cb_wait_front(CB_B, 1);
            matmul_tiles(CB_A, CB_B, 0, 0, 0, false);
            cb_pop_front(CB_A, 1);
            cb_pop_front(CB_B, 1);
        }

        // Write partial result
        cb_reserve_back(CB_PARTIAL, 1);
        commit_dst();
        pack_tile(0, CB_PARTIAL);
        cb_push_back(CB_PARTIAL, 1);
        release_dst();
    }
}
```

**Kernel 2 (Reduce Partials)**:
```cpp
// One core reduces all partials
void MAIN() {
    uint32_t num_partials = get_arg_val<uint32_t>(0);

    for (uint32_t tile_idx = 0; tile_idx < num_output_tiles; ++tile_idx) {
        acquire_dst();
        reduce_init_tiles(CB_OUT);

        for (uint32_t p = 0; p < num_partials; ++p) {
            cb_wait_front(CB_PARTIAL, 1);
            reduce_sum_tiles(CB_PARTIAL, 0, 0);
            cb_pop_front(CB_PARTIAL, 1);
        }

        cb_reserve_back(CB_OUT, 1);
        commit_dst();
        pack_tile(0, CB_OUT);
        cb_push_back(CB_OUT, 1);
        release_dst();
    }
}
```

---

## Implementation Checklist

### 3.1 GEMV
- [ ] Non-square tile handling transform
- [ ] Vector broadcast codegen
- [ ] Test matrix-vector multiplication
- [ ] PR: "Implement GEMV with Vector Broadcast"

### 3.2 Convolution
- [ ] Convolution pattern recognition
- [ ] Im2col transform (or direct conv)
- [ ] Weight multicast planning
- [ ] Multicast codegen
- [ ] Test conv2d patterns
- [ ] PR: "Implement Convolution with Multicast"

### 3.3 Split-K GEMM
- [ ] Split-K planning transform
- [ ] Two-phase kernel generation
- [ ] Inter-kernel synchronization
- [ ] Test split-K on larger matrices
- [ ] PR: "Implement Split-K GEMM"

---

## Timeline Estimate

| Task | Duration | Dependencies |
|------|----------|--------------|
| 3.1 GEMV | 5-6 hours | Phase 1, 2 complete |
| 3.2 Convolution | 8-10 hours | Phase 1, 2 complete |
| 3.3 Split-K GEMM | 6-8 hours | Phase 2.3 (reduction) complete |
| **Total Phase 3** | **19-24 hours** | **4-5 days** |

---

**Status**: ⏳ Not Started
**Blocked By**: Phase 2 completion
**Last Updated**: 2025-10-08
