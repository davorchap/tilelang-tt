/**
 * Mock TT-Metalium SDK Headers - Tile Movement and NOC Operations
 *
 * Purpose: Validate C++ syntax of generated kernels without requiring real SDK.
 *
 * IMPORTANT: These headers are READ-ONLY for compilation tests.
 * Do NOT modify these headers to make generated code compile.
 * If compilation fails, fix the code generator, not these headers.
 */

#pragma once

#include "common.h"

// NOC (Network-on-Chip) operations for tile-based data movement

// Asynchronous NOC read operations
inline void noc_async_read_tile(uint32_t tile_id, uint32_t accessor_id,
                                uint32_t l1_addr) {}
inline void noc_async_read(uint64_t src_addr, uint32_t dst_addr,
                           uint32_t size) {}
inline void noc_async_read_barrier() {}

// Asynchronous NOC write operations
inline void noc_async_write_tile(uint32_t tile_id, uint32_t accessor_id,
                                 uint32_t l1_addr) {}
inline void noc_async_write(uint32_t src_addr, uint64_t dst_addr,
                            uint32_t size) {}
inline void noc_async_write_barrier() {}

// Tile copy operations
inline void copy_tile(tt::CBIndex src_cb, uint32_t tile_idx,
                      tt::CBIndex dst_cb) {}
inline void copy_tile(uint32_t src_cb, uint32_t tile_idx, uint32_t dst_cb) {}
