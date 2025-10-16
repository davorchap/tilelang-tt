# Development Guides

This directory contains practical guides for developing the Tenstorrent backend.

## Documents

- **TIR_BASICS.md**: TensorIR primer covering fundamental concepts and lowering approaches
- **TT_Python_Implementation_Quickstart.md**: Quick-start guide for implementing passes in Python with templates and examples
- **TT_Python_to_CPP_Migration.md**: Step-by-step guide for migrating Python passes to C++ for production
- **kernel_authoring_comparison.md**: Comparison between TileLang DSL and direct SDK kernel authoring

## Getting Started

For new contributors:
1. Start with **TIR_BASICS.md** to understand TensorIR concepts
2. Use **TT_Python_Implementation_Quickstart.md** to implement your first pass
3. Refer to **kernel_authoring_comparison.md** to understand the compilation target

For performance optimization:
- Follow **TT_Python_to_CPP_Migration.md** when a Python pass becomes a bottleneck