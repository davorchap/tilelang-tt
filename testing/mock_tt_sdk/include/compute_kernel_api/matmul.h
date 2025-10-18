/**
 * Mock TT-Metalium SDK Headers - Matrix Multiplication Operations
 *
 * Purpose: Validate C++ syntax of generated kernels without requiring real SDK.
 */

#pragma once

#include "common.h"

// Matrix multiplication operations
// Flexible overloads to handle mixed CBIndex/uint32_t arguments

// All CBIndex, 6 args
inline void matmul_tiles(tt::CBIndex cb_in0, tt::CBIndex cb_in1, tt::CBIndex start0, tt::CBIndex start1, tt::CBIndex dst_idx, bool accumulate = false) {}

// All uint32_t, 6 args
inline void matmul_tiles(uint32_t cb_in0, uint32_t cb_in1, uint32_t start0, uint32_t start1, uint32_t dst_idx, bool accumulate = false) {}

// Mixed: CBIndex CBs, uint32_t indices, 6 args
inline void matmul_tiles(tt::CBIndex cb_in0, tt::CBIndex cb_in1, uint32_t start0, uint32_t start1, uint32_t dst_idx, bool accumulate = false) {}

// Mixed: CBIndex CBs, uint32_t indices, 5 args (accumulate defaults to false)
inline void matmul_tiles(tt::CBIndex cb_in0, tt::CBIndex cb_in1, uint32_t start0, uint32_t start1, uint32_t dst_idx) {
    matmul_tiles(cb_in0, cb_in1, start0, start1, dst_idx, false);
}

// All uint32_t, 5 args
inline void matmul_tiles(uint32_t cb_in0, uint32_t cb_in1, uint32_t start0, uint32_t start1, uint32_t dst_idx) {
    matmul_tiles(cb_in0, cb_in1, start0, start1, dst_idx, false);
}

// All CBIndex, 5 args
inline void matmul_tiles(tt::CBIndex cb_in0, tt::CBIndex cb_in1, tt::CBIndex start0, tt::CBIndex start1, tt::CBIndex dst_idx) {
    matmul_tiles(cb_in0, cb_in1, start0, start1, dst_idx, false);
}

// Specific overload for buggy codegen: 3 CBIndex + 2 uint32_t
inline void matmul_tiles(tt::CBIndex cb_in0, tt::CBIndex cb_in1, tt::CBIndex cb_in2, uint32_t arg4, uint32_t arg5) {
    // Just a stub to make buggy code compile
}

// Matrix multiplication initialization
inline void mm_init(tt::CBIndex cb_in0, tt::CBIndex cb_in1, tt::CBIndex cb_out) {}
inline void mm_init(uint32_t cb_in0, uint32_t cb_in1, uint32_t cb_out) {}

// DST management for matmul
inline void acquire_dst(tt::DstMode mode = tt::DstMode::Half) {}
inline void release_dst(tt::DstMode mode = tt::DstMode::Half) {}
inline void pack_tile(uint32_t dst_idx, tt::CBIndex cb_out) {}
inline void pack_tile(uint32_t dst_idx, uint32_t cb_out) {}
