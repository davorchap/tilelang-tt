/**
 * Mock TT-Metalium SDK Headers for Compilation Testing
 *
 * Purpose: Validate C++ syntax of generated kernels without requiring real SDK.
 *
 * IMPORTANT: These headers are READ-ONLY for compilation tests.
 * Do NOT modify these headers to make generated code compile.
 * If compilation fails, fix the code generator, not these headers.
 *
 * Based on TT-Metal SDK common APIs (simplified for testing).
 */

#pragma once

#include <cstdint>

namespace tt {

// Forward declarations for operations
namespace fill {
inline void zero(void *, int) {} // Dummy implementation
} // namespace fill

// Circular Buffer Index enum (matches real SDK)
enum class CBIndex : uint8_t {
  c_0 = 0,
  c_1 = 1,
  c_2 = 2,
  c_3 = 3,
  c_4 = 4,
  c_5 = 5,
  c_6 = 6,
  c_7 = 7,
  c_8 = 8,
  c_9 = 9,
  c_10 = 10,
  c_11 = 11,
  c_12 = 12,
  c_13 = 13,
  c_14 = 14,
  c_15 = 15,
  c_16 = 16,
  c_17 = 17,
  c_18 = 18,
  c_19 = 19,
  c_20 = 20,
  c_21 = 21,
  c_22 = 22,
  c_23 = 23,
  c_24 = 24,
  c_25 = 25,
  c_26 = 26,
  c_27 = 27,
  c_28 = 28,
  c_29 = 29,
  c_30 = 30,
  c_31 = 31,
};

// DST Mode enum
enum class DstMode { Half, Full };

} // namespace tt

// Runtime argument access (templated for type safety)
template <typename T> inline T get_arg_val(uint32_t arg_index) {
  // Mock implementation - just returns zero
  return T{};
}

// Circular buffer operations
inline void cb_reserve_back(uint32_t cb_id, uint32_t num_tiles) {}
inline void cb_push_back(uint32_t cb_id, uint32_t num_tiles) {}
inline void cb_wait_front(uint32_t cb_id, uint32_t num_tiles) {}
inline void cb_pop_front(uint32_t cb_id, uint32_t num_tiles) {}

// Overloads for CBIndex enum
inline void cb_reserve_back(tt::CBIndex cb_id, uint32_t num_tiles) {}
inline void cb_push_back(tt::CBIndex cb_id, uint32_t num_tiles) {}
inline void cb_wait_front(tt::CBIndex cb_id, uint32_t num_tiles) {}
inline void cb_pop_front(tt::CBIndex cb_id, uint32_t num_tiles) {}

// CB pointer access
inline uint32_t get_write_ptr(uint32_t cb_id) { return 0; }
inline uint32_t get_read_ptr(uint32_t cb_id) { return 0; }

// Overloads for CBIndex enum
inline uint32_t get_write_ptr(tt::CBIndex cb_id) { return 0; }
inline uint32_t get_read_ptr(tt::CBIndex cb_id) { return 0; }

// Include all other APIs
#include "eltwise_binary.h"
#include "matmul.h"
