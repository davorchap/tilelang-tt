# Pass Implementation Documentation

This directory contains detailed documentation for individual pass implementations.

## Current Documentation

- **attach_tensor_accessor_tt_summary.md**: AttachTensorAccessorTT pass implementation details
- **grid_to_persistent_tt.md**: Grid to persistent loop transformation
- **infer_tt_layout.md**: Layout inference for TT buffers
- **lower_to_sfpu.md**: SFPU (Special Function Processing Unit) lowering
- **memory_space_lower_tt.md**: Memory space lowering for TT
- **propagate_tt_layout.md**: Layout propagation across IR
- **tile_pad_tt.md**: Tile padding for alignment
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

### Metadata Passes
- infer_tt_layout
- propagate_tt_layout
- attach_tensor_accessor_tt

### Transformation Passes
- grid_to_persistent_tt
- memory_space_lower_tt
- tile_pad_tt
- lower_to_sfpu

### Verification Passes
- verify_tt_ir