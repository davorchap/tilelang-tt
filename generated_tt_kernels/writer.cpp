// Generated TT Writer Kernel (IR-Driven)
// Matmul Writer: Writes C[m,n] output tiles

// Grid: 8x8

#include <cstdint>

// Mock TT intrinsics for dry-run
template<typename T>
inline T get_arg_val(uint32_t idx) { return T(); }

// Mock TT circular buffer APIs for dry-run
inline void cb_wait_front(uint32_t cb_id, uint32_t n_tiles) {}
inline uint32_t get_read_ptr(uint32_t cb_id) { return 0; }
inline void noc_async_write_tile(uint32_t tile_idx, uint32_t l1_addr, uint32_t base_addr) {}
inline void noc_async_write_barrier() {}
inline void cb_pop_front(uint32_t cb_id, uint32_t n_tiles) {}

// Circular Buffer Index
constexpr auto cb_out0 = tt::CBIndex::c_16;

constexpr uint32_t TILE_SIZE_BYTES = 32 * 32 * sizeof(uint16_t);  // fp16

void kernel_main() {
    // Runtime arguments
    uint32_t dram_addr_c = get_arg_val<uint32_t>(0);
    uint32_t out_tile_start_id = get_arg_val<uint32_t>(1);
    uint32_t num_out_tiles = get_arg_val<uint32_t>(2);
    uint32_t Nt = get_arg_val<uint32_t>(3);
    
    // Write output tiles
    for (uint32_t out_tile = 0; out_tile < num_out_tiles; ++out_tile) {
        uint32_t tile_idx = out_tile_start_id + out_tile;
        
        cb_wait_front(cb_out0, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_out0);
        noc_async_write_tile(tile_idx, l1_read_addr, dram_addr_c);
        noc_async_write_barrier();
        cb_pop_front(cb_out0, 1);
    }
}
