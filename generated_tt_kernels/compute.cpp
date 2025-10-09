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
    uint32_t out_tile_start_id = get_arg_val<uint32_t>(0);
    uint32_t num_output_tiles = get_arg_val<uint32_t>(1);
    uint32_t Kt = get_arg_val<uint32_t>(2);
    
    for (uint32_t i = 0; i < tt_count; ++i) {
        // Acquire tile registers for matmul accumulation
        // Acquire tile registers for computation
        tile_regs_acquire();
        
        // K-loop: C[m,n] += sum(A[m,k] * B[k,n] for k in Kt)
        // Initialize matmul (once before all loops)
        mm_init(cb_in0, cb_in1, cb_out0);
        
        for (uint32_t kt = 0; kt < 1; ++kt) {
            C[((((((((int64)tt_start_id) + ((int64)i)) % 8) * 8192) + ((((int64)tx) / 4) * 256)) + (((((int64)tt_start_id) + ((int64)i)) / 8) * 32)) + ((((int64)tx) % 4) * 8))] = (A[((((((((int64)tt_start_id) + ((int64)i)) % 8) * 8192) + ((((int64)tx) / 4) * 256)) + (((((int64)tt_start_id) + ((int64)i)) / 8) * 32)) + ((((int64)tx) % 4) * 8))] + B[((((((((int64)tt_start_id) + ((int64)i)) % 8) * 8192) + ((((int64)tx) / 4) * 256)) + (((((int64)tt_start_id) + ((int64)i)) / 8) * 32)) + ((((int64)tx) % 4) * 8))]);
        }
        
        // After K-loop: pack result
        // Commit tile register computation
        tile_regs_commit();
        // Wait for tile register computation to complete
        tile_regs_wait();
        cb_reserve_back(cb_out0, 1);
        pack_tile(0, cb_out0);
        cb_push_back(cb_out0, 1);
        // Release tile registers
        tile_regs_release();
        
    }
}
