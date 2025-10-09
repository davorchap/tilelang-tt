// Generated TT Reader Kernel (IR-Driven)
// Matmul Reader: Loads A[m,k] and B[k,n] tiles

// Grid: 8x8

#include <cstdint>

// Mock TT intrinsics for dry-run
template<typename T>
inline T get_arg_val(uint32_t idx) { return T(); }

// Mock TT circular buffer APIs for dry-run
inline void cb_reserve_back(uint32_t cb_id, uint32_t n_tiles) {}
inline uint32_t get_write_ptr(uint32_t cb_id) { return 0; }
inline void noc_async_read_tile(uint32_t tile_idx, uint32_t base_addr, uint32_t l1_addr) {}
inline void noc_async_read_barrier() {}
inline void cb_push_back(uint32_t cb_id, uint32_t n_tiles) {}

// Circular Buffer Indices
constexpr auto cb_in0 = tt::CBIndex::c_0;
constexpr auto cb_in1 = tt::CBIndex::c_1;

constexpr uint32_t TILE_SIZE_BYTES = 32 * 32 * sizeof(uint16_t);  // fp16

void kernel_main() {
    // Runtime arguments
    uint32_t dram_addr_a = get_arg_val<uint32_t>(0);
    uint32_t dram_addr_b = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t out_tile_start_id = get_arg_val<uint32_t>(5);
    uint32_t num_out_tiles = get_arg_val<uint32_t>(6);
    
    // Process output tiles
    for (uint32_t out_tile = 0; out_tile < num_out_tiles; ++out_tile) {
        uint32_t current_tile_id = out_tile_start_id + out_tile;
        uint32_t out_m = current_tile_id / Nt;
        uint32_t out_n = current_tile_id % Nt;
        
        // Load tiles for this output: A[out_m,:] and B[:,out_n]
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            // Read A[out_m, kt]
            uint32_t tile_a_idx = out_m * Kt + kt;
            cb_reserve_back(cb_in0, 1);
            uint32_t l1_write_addr_a = get_write_ptr(cb_in0);
            noc_async_read_tile(tile_a_idx, dram_addr_a, l1_write_addr_a);
            noc_async_read_barrier();
            cb_push_back(cb_in0, 1);
            
            // Read B[kt, out_n]
            uint32_t tile_b_idx = kt * Nt + out_n;
            cb_reserve_back(cb_in1, 1);
            uint32_t l1_write_addr_b = get_write_ptr(cb_in1);
            noc_async_read_tile(tile_b_idx, dram_addr_b, l1_write_addr_b);
            noc_async_read_barrier();
            cb_push_back(cb_in1, 1);
        }
    }
}
