# Reference Documentation

This directory contains reference materials for TT backend development.

## Documents

- **PASS_TABLE_SHARED.md**: Reference table of shared lowering and optimization passes used by both GPU and TT backends
- **PASS_TABLE_GPU.md**: CUDA/ROCm-specific passes for comparison and understanding
- **TT_Pass_Specifications.md**: Detailed specifications for implementing new TT passes with algorithms and examples

## Usage

### Finding a Pass
For Tenstorrent passes, see [passes/README.md](../passes/README.md) for the complete pass index and navigation.

For shared passes (used by both GPU and TT), see **PASS_TABLE_SHARED.md**.

For GPU-specific passes, see **PASS_TABLE_GPU.md**.

### Implementing a Pass
Refer to **TT_Pass_Specifications.md** for detailed implementation guidance including:
- Input/output examples
- Algorithm descriptions
- Testing strategies

### Understanding Pass Categories
- **Shared passes** (PASS_TABLE_SHARED.md): Common infrastructure
- **GPU passes** (PASS_TABLE_GPU.md): GPU-specific optimizations for comparison
- **TT passes** ([passes/README.md](../passes/README.md)): Tenstorrent-specific transformations organized by stage