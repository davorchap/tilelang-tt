# Reference Documentation

This directory contains reference materials for TT backend development.

## Documents

- **PASS_TABLE_SHARED.md**: Reference table of shared lowering and optimization passes used by both GPU and TT backends
- **PASS_TABLE_GPU.md**: CUDA/ROCm-specific passes for comparison and understanding
- **PASS_TABLE_TT.md**: Complete reference of Tenstorrent-specific passes with descriptions and status
- **TT_Pass_Specifications.md**: Detailed specifications for implementing new TT passes with algorithms and examples

## Usage

### Finding a Pass
Use the PASS_TABLE documents to look up passes by name, understand their purpose, and see their position in the pipeline.

### Implementing a Pass
Refer to **TT_Pass_Specifications.md** for detailed implementation guidance including:
- Input/output examples
- Algorithm descriptions
- C++ implementation templates
- Testing strategies

### Understanding Pass Categories
- **Shared passes** (PASS_TABLE_SHARED.md): Common infrastructure
- **GPU passes** (PASS_TABLE_GPU.md): GPU-specific optimizations for comparison
- **TT passes** (PASS_TABLE_TT.md): Tenstorrent-specific transformations