// Generated TT Compute Kernel (IR-Driven)
// Grid: 8x8
// Cores: 64

#include <cstdint>

// Mock TT intrinsics for dry-run
template<typename T>
inline T get_arg_val(uint32_t idx) { return T(); }

// Mock TT circular buffer APIs for dry-run
inline void cb_wait_front(uint32_t cb_id, uint32_t n_tiles) {}
inline void cb_reserve_back(uint32_t cb_id, uint32_t n_tiles) {}
inline void cb_pop_front(uint32_t cb_id, uint32_t n_tiles) {}
inline void cb_push_back(uint32_t cb_id, uint32_t n_tiles) {}

// Mock TT matmul compute APIs for dry-run
inline void mm_init(uint32_t cb_in0, uint32_t cb_in1, uint32_t cb_out = 16) {}
inline void matmul_tiles(uint32_t cb_in0, uint32_t cb_in1, uint32_t tile_idx_in0, uint32_t tile_idx_in1, uint32_t dst_tile_idx, bool transpose) {}

// Mock TT tile register APIs for dry-run
inline void tile_regs_acquire() {}
inline void tile_regs_commit() {}
inline void tile_regs_wait() {}
inline void tile_regs_release() {}

// Mock TT element-wise compute APIs for dry-run
inline void binary_op_init_common(uint32_t cb_in0, uint32_t cb_in1, uint32_t cb_out = 16) {}
inline void add_tiles_init(uint32_t cb_in0 = 0, uint32_t cb_in1 = 1) {}
inline void add_tiles(uint32_t cb_a, uint32_t cb_b, uint32_t idx_a, uint32_t idx_b, uint32_t idx_dst) {}
inline void pack_tile(uint32_t idx_dst, uint32_t cb_out) {}

// Circular Buffer Indices
constexpr auto cb_in0 = tt::CBIndex::c_0;
constexpr auto cb_in1 = tt::CBIndex::c_1;
constexpr auto cb_out0 = tt::CBIndex::c_16;

void MAIN() {
    // Runtime arguments
    uint32_t tt_start_tile = get_arg_val<uint32_t>(0);
    uint32_t tt_tile_count = get_arg_val<uint32_t>(1);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t out_tile_start_id = tt_start_tile;
    uint32_t num_output_tiles = tt_tile_count;
    
    for (uint32_t tt_tile_iter = 0; tt_tile_iter < 0 + tt_tile_count; ++tt_tile_iter) {
        tl.fill(tir.tvm_access_ptr(tir.type_annotation(), C_tile, 0, 16384, 2), 0);
        for (uint32_t k = 0; k < 0 + 32; ++k) {
            tl.copy(tl.region(A[(((tt_tile_id / 8) % 8) * 128)][(k * 32)], 1, 128, 32), tl.region(A_tile[0][0], 2, 128, 32), -1, 0, 0);
            tl.copy(tl.region(B[(k * 32)][((tt_tile_id % 8) * 128)], 1, 32, 128), tl.region(B_tile[0][0], 2, 32, 128), -1, 0, 0);
            tl.gemm(tir.tvm_access_ptr(tir.type_annotation(), A_tile, 0, 4096, 1), tir.tvm_access_ptr(tir.type_annotation(), B_tile, 0, 4096, 1), tir.tvm_access_ptr(tir.type_annotation(), C_tile, 0, 16384, 3), 0, 0, 128, 128, 32, 0, 0, 32, 128, 0, 0, 1, 0, 0, 0, 0);
        }
        tl.copy(tl.region(C_tile[0][0], 1, 128, 128), tl.region(C[(((tt_tile_id / 8) % 8) * 128)][((tt_tile_id % 8) * 128)], 2, 128, 128), -1, 0, 0);
    }
}
