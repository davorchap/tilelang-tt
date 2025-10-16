# Pre-v5 Archive

This directory contains documentation from the development of the Tenstorrent backend before the v5 pipeline became the default and only implementation.

## Archive Date
**October 2025** - After PR #135 removed the old pipeline and established v5 as default

## Contents

### planning/
Historical implementation plans, timelines, and migration strategies that guided the development of the v5 pipeline:

- `TT_Backend_Implementation_Plan.md` - Original backend implementation plan
- `TT_Backend_Implementation_Plan_v2.md` - Revised implementation plan
- `TT_Codegen_Update_Plan.md` - Plan for codegen updates
- `TT_Implementation_Timeline.md` - Original project timeline
- `TT_Implementation_Timeline_v2.md` - Revised project timeline
- `V5_PASS_FIX_PLAN.md` - v5 pass bug fix and improvement plan

**Status**: SUPERSEDED - v5 pipeline is now complete and default

### progress-reports/
Historical progress reports documenting the development of individual v5 passes:

- `A3_AttachTensorAccessorTT_Summary.md` - Stage A pass implementation
- `Progress_Report_C3_D1_D2.md` - Stage C and D passes (C3, D1, D2)
- `Progress_Report_D3_D4_D5.md` - Stage D finalization passes (D3, D4, D5)

**Status**: COMPLETED - All passes implemented and working

### old-pipeline/
Documentation for the original 5-pass pipeline that was replaced by v5:

- `TT_Pass_Pipeline_Status.md` - Original pipeline status tracker
- `TT_Pass_Pipeline_Status_Updated.md` - Updated status before v5
- `TT_Pass_Specifications.md` - Original pass specifications

**Status**: REMOVED - Old pipeline deleted in PR #135

## Current Documentation

For current v5 pipeline documentation, see:
- `docs/tenstorrent/architecture/v5_pipeline.md` - Authoritative v5 reference
- `docs/tenstorrent/planning/TT_Pass_Status.md` - Current v5 pass status
- `docs/tenstorrent/architecture/TT_ARCHITECTURE.md` - Current architecture

## Why Archived

These documents were archived because:
1. **v5 Pipeline Complete**: The 14-pass v5 pipeline is now the only implementation
2. **Old Pipeline Removed**: The original 5-pass pipeline was deleted (PR #135)
3. **Python-Only Decision**: No C++ migration planned - all passes remain in Python
4. **Historical Value**: Preserved for institutional knowledge and development history

## Version History

- **Pre-v5 (2024-2025)**: Development of v5 pipeline alongside old pipeline
- **Phase 0 (Oct 2025)**: v5 made default, old pipeline deprecated
- **PR #135 (Oct 2025)**: Old pipeline completely removed, documentation cleanup
- **Current**: v5 is the only pipeline, Python-only architecture established
