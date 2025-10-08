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
inline void matmul_tiles_init(uint32_t cb_a, uint32_t cb_b, uint32_t cb_c) {}
inline void matmul_tiles(uint32_t cb_a, uint32_t cb_b, uint32_t cb_c, bool accumulate) {}

// Mock TT DST register double buffering APIs for dry-run
inline void acquire_dst() {}
inline void commit_dst() {}
inline void wait_for_tile() {}
inline void release_dst() {}

// Mock TT element-wise compute APIs for dry-run
inline void add_tiles_init() {}
inline void add_tiles(uint32_t cb_a, uint32_t cb_b, uint32_t idx_a, uint32_t idx_b, uint32_t idx_dst) {}
inline void pack_tile(uint32_t idx_dst, uint32_t cb_out) {}

// Circular Buffer Indices
constexpr uint32_t CB_A = 0;
constexpr uint32_t CB_B = 1;
constexpr uint32_t CB_C = 2;

void MAIN() {
    // Runtime arguments
    uint32_t out_tile_start_id = get_arg_val<uint32_t>(0);
    uint32_t num_output_tiles = get_arg_val<uint32_t>(1);
    uint32_t Kt = get_arg_val<uint32_t>(2);
    
    // Outer loop: process output tile
    // DST: Acquire registers for computation
    acquire_dst();
    
    for (uint32_t i = 0; i < tt_count; ++i) {
        /* unsupported call */;
        // K-loop: C[m,n] += sum(A[m,k] * B[k,n] for k in Kt)
        for (uint32_t k = 0; k < 8; ++k) {
            // T.copy - handled by reader/writer kernels
            // T.copy - handled by reader/writer kernels
            // Initialize matmul
            matmul_tiles_init(CB_A, CB_B, CB_C);
            // Wait for input tiles from reader
            cb_wait_front(CB_A, 1);
            cb_wait_front(CB_B, 1);
            
            // Matmul: first K iteration
            matmul_tiles(CB_A, CB_B, CB_C, false);
            
            // Release input tiles
            cb_pop_front(CB_A, 1);
            cb_pop_front(CB_B, 1);
            
        }
        
        // After K-loop: pack result
        cb_reserve_back(CB_C, 1);
        // DST: Commit computation complete
        commit_dst();
        pack_tile(0, CB_C);
        cb_push_back(CB_C, 1);
        // DST: Release registers
        release_dst();
        
        // T.copy - handled by reader/writer kernels
    }
}
