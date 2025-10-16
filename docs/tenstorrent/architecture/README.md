# Architecture Documentation

This directory contains architectural documentation for the Tenstorrent backend.

## Documents

- **TT_ARCHITECTURE.md**: Complete TT backend architecture overview, including pass pipeline, code generation, and runtime integration
- **IR_LOWERING_ANALYSIS.md**: Comprehensive comparison between GPU and Tenstorrent lowering pipelines, execution models, and memory hierarchies
- **TileLang_TT_TIR_Lowering_Guide_v5.md**: V5 pass pipeline specification with late protocol insertion design
- **RUNTIME_PLAN.md**: Runtime plan specification defining the tt.plan.json format and host-device coordination

## Key Concepts

### Pass Pipeline Architecture
The TT backend uses a metadata-driven approach with layout-aware passes that preserve high-level information throughout the compilation pipeline.

### Execution Model
Unlike GPU's transient threadblocks, TT uses persistent per-core loops where each Tensix core iterates over multiple tiles.

### Memory Hierarchy
Explicit L1 circular buffer management with NOC-based data movement, rather than implicit caching.

### Code Generation
IR-driven visitors generate three separate kernels (reader/compute/writer) plus host metadata and runtime plans.