# Archived Old Pass Documentation

This directory contains documentation for the **deprecated old pipeline** that was removed in PR #135.

## Old Pipeline (Removed)

The old pipeline consisted of 5 Python passes:
1. **InferTTLayout** - Infer buffer layouts and metadata
2. **PropagateTTLayout** - Propagate and normalize layout info
3. **TTTilesToCoreMap** - Compute core mapping and work partition
4. **LowerTTTileIntrinsics** - Lower tile ops to device intrinsics
5. **GridToPersistentTT** - Final lowering to persistent kernels

Additionally:
- **MemorySpaceLowerTT** - Planned C++ pass for CB allocation (never implemented)

## Replacement: V5 Pipeline

The v5 pipeline (14 passes in stages A-E) replaced all old passes. See:
- `docs/tenstorrent/architecture/TT_ARCHITECTURE.md` - Complete v5 architecture
- `docs/tenstorrent/architecture/TileLang_TT_TIR_Lowering_Guide_v5.md` - V5 pass specifications
- `docs/tenstorrent/README.md` - Current pipeline overview

## Historical Context

These files are preserved for historical reference and to understand the evolution
from the old class-based pass API to the new function-based v5 pipeline.

**Last Updated**: 2025-10-16
