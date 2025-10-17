# Archived Documentation

This directory contains documentation that has been superseded or merged into other documents. These files are preserved for historical reference.

## Archived Files

### IR_LOWERING_ANALYSIS.md
- **Date Archived:** 2025-10-17
- **Original Location:** `docs/tenstorrent/architecture/`
- **Replaced By:** [GPU_vs_Tenstorrent_Analysis.md](../architecture/GPU_vs_Tenstorrent_Analysis.md)
- **Reason:** Content merged with historical GPU_vs_Tenstorrent.md to create a comprehensive comparison document. The new document eliminates redundancy, updates v5 pipeline references, and provides better organization.

### GPU_vs_Tenstorrent.md (Historical)
- **Date Removed:** 2025-10-16
- **Original Location:** `docs/tenstorrent/`
- **Replaced By:** [GPU_vs_Tenstorrent_Analysis.md](../architecture/GPU_vs_Tenstorrent_Analysis.md)
- **Reason:** Content preserved and enhanced in the new combined document. Original was removed during documentation consolidation but valuable content was resurrected and integrated.

## Why Archive?

These documents contained valuable information but had issues:
- **Redundancy:** Both documents contained identical sections on execution models, memory hierarchy, data movement, and compute operations
- **Outdated References:** IR_LOWERING_ANALYSIS.md referenced C++ implementations that are no longer planned (Python-only now)
- **Poor Organization:** Information was scattered between two documents that should have been unified

The new [GPU_vs_Tenstorrent_Analysis.md](../architecture/GPU_vs_Tenstorrent_Analysis.md) combines the best of both documents while:
- Eliminating redundancy
- Updating all v5 pipeline references
- Removing C++ implementation details
- Adding clear cross-references to v5_pipeline.md
- Providing better organization and flow