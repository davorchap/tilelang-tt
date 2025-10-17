# Architecture Documentation

This directory contains architectural documentation for the Tenstorrent backend.

## Documents

### Core Architecture
- **[TT_ARCHITECTURE.md](TT_ARCHITECTURE.md)**: Complete TT backend architecture overview, including compilation pipeline, code generation, 3-kernel architecture, and runtime integration. Includes runtime plan specification in Appendix A.

### Pipeline Reference
- **[v5_pipeline.md](v5_pipeline.md)**: Authoritative v5 pipeline reference with all 14 passes in 5 stages (A-E). Includes transformation examples, intrinsic quick reference, and complete dependency graphs.

### Comparative Analysis
- **[GPU_vs_Tenstorrent_Analysis.md](GPU_vs_Tenstorrent_Analysis.md)**: Comprehensive comparison between GPU and Tenstorrent architectures, including execution models, memory hierarchies, compiler pipelines, and code generation strategies.

## Key Concepts

### Pass Pipeline Architecture
The TT backend uses a metadata-driven approach with layout-aware passes that preserve high-level information throughout the compilation pipeline.

### Execution Model
Unlike GPU's transient threadblocks, TT uses persistent per-core loops where each Tensix core iterates over multiple tiles.

### Memory Hierarchy
Explicit L1 circular buffer management with NOC-based data movement, rather than implicit caching.

### Code Generation
IR-driven visitors generate three separate kernels (reader/compute/writer) plus host metadata and runtime plans.