/**
 * Mock TT-Metalium SDK Headers - Binary Element-wise Operations
 *
 * Purpose: Validate C++ syntax of generated kernels without requiring real SDK.
 */

#pragma once

#include "common.h"

// Binary element-wise operations
inline void add_tiles(tt::CBIndex cb_in0, tt::CBIndex cb_in1, uint32_t start0, uint32_t start1, uint32_t dst_idx) {}
inline void sub_tiles(tt::CBIndex cb_in0, tt::CBIndex cb_in1, uint32_t start0, uint32_t start1, uint32_t dst_idx) {}
inline void mul_tiles(tt::CBIndex cb_in0, tt::CBIndex cb_in1, uint32_t start0, uint32_t start1, uint32_t dst_idx) {}

// Overloads for uint32_t
inline void add_tiles(uint32_t cb_in0, uint32_t cb_in1, uint32_t start0, uint32_t start1, uint32_t dst_idx) {}
inline void sub_tiles(uint32_t cb_in0, uint32_t cb_in1, uint32_t start0, uint32_t start1, uint32_t dst_idx) {}
inline void mul_tiles(uint32_t cb_in0, uint32_t cb_in1, uint32_t start0, uint32_t start1, uint32_t dst_idx) {}

// Binary operation initialization
inline void binary_op_init_common(tt::CBIndex cb_in0, tt::CBIndex cb_in1, tt::CBIndex cb_out) {}
inline void binary_op_init_common(uint32_t cb_in0, uint32_t cb_in1, uint32_t cb_out) {}
