# Pass Implementation Documentation

This directory contains detailed documentation for individual pass implementations.

## Current Documentation

- **attach_tensor_accessor_tt_summary.md**: AttachTensorAccessorTT pass implementation details
- **grid_to_persistent_tt.md**: Grid to persistent loop transformation (deprecated, see v5 passes)
- **infer_tt_layout.md**: Layout inference for TT buffers (deprecated, see infer_tt_layout_v5)
- **lower_to_sfpu.md**: SFPU (Special Function Processing Unit) lowering (future Python implementation needed)
- **memory_space_lower_tt.md**: Memory space lowering for TT (deprecated)
- **propagate_tt_layout.md**: Layout propagation across IR (deprecated, see propagate_tt_layout_v5)
- **tile_pad_tt.md**: Tile padding for alignment (deprecated)
- **verify_tt_ir.md**: IR verification for TT constraints

## Adding New Pass Documentation

When implementing a new pass:

1. Create a markdown file named after the pass (snake_case)
2. Include:
   - Purpose and motivation
   - Input/output IR examples
   - Algorithm description
   - Implementation notes
   - Test cases
   - Known limitations

3. Follow the existing format in the documentation files

## Pass Categories

### V5 Active Passes (Python)
See `tilelang/tenstorrent/passes/` for current v5 implementation:
- infer_tt_layout_v5, propagate_tt_layout_v5, attach_tensor_accessor_tt (Stage A: Metadata)
- layout_aware_work_partition_tt_v5, grid_to_core_grid_v5 (Stage B: Partitioning)
- lower_shared_to_cb_v5, lower_tt_tile_intrinsics_v5, build_tile_dfg_tt (Stage C: Lowering)
- split_device_kernel, configure_tensor_accessor_tt, lower_cb_intrinsics, insert_compute_init_tt, insert_dst_management_tt (Stage D: Late Split & Protocol)
- finalize_persistent_signature_tt (Stage E: Finalization)

### Deprecated/Old Passes
- infer_tt_layout (replaced by infer_tt_layout_v5)
- propagate_tt_layout (replaced by propagate_tt_layout_v5)
- grid_to_persistent_tt (replaced by grid_to_core_grid_v5)
- memory_space_lower_tt (replaced by lower_shared_to_cb_v5)
- tile_pad_tt (deprecated)

### Future Passes (Not Yet Implemented)
- **lower_to_sfpu**: SFPU lowering for T.Parallel support (needs Python implementation)

### Verification Passes
- verify_tt_ir (IR validation and constraint checking)