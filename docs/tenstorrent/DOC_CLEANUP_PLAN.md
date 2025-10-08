# Tenstorrent Documentation Cleanup Plan

**Date:** 2025-10-08
**Purpose:** Remove obsolete/deprecated documentation and consolidate current docs
**Status:** Ready for execution

---

## Executive Summary

The `docs/tenstorrent/` directory has accumulated **48 markdown files** over the development of the Tenstorrent backend. Many are now obsolete planning documents, superseded by completed implementation, or duplicates of newer consolidated docs.

**Cleanup scope:**
- **Remove:** 21 obsolete/deprecated docs (44% of total)
- **Keep:** 27 current docs (56% of total)
- **Update:** CLAUDE.md and README.md references

**Benefits:**
- ✅ Clear documentation structure
- ✅ No confusion about current vs. historical status
- ✅ Easier onboarding for new contributors
- ✅ Reduced maintenance burden

---

## Current State Analysis

### Total Files: 48 markdown documents

**Categories:**
1. **Planning docs** (now obsolete): Analysis and task breakdowns completed
2. **Investigation docs** (now obsolete): Research that led to decisions
3. **Status docs** (keep): Active tracking of implementation state
4. **Implementation guides** (keep): How-to and setup documentation
5. **Architecture docs** (keep): Design and specifications
6. **Session summaries** (historical): Keep for reference

---

## Cleanup Actions

### 🗑️ Category 1: Obsolete Planning Documents (DELETE - 7 files)

These were planning/analysis docs that are now superseded by completed implementation (Tasks 1-8 complete, 95/95 tests passing).

| File | Reason | Superseded By |
|------|--------|---------------|
| `IR_LOWERING_ANALYSIS.md` | Analysis doc - IR lowering now complete | `IR_LOWERING_VALIDATION.md` |
| `IR_LOWERING_TASKS.md` | Task list - all tasks complete (PRs #72-79) | `IR_LOWERING_VALIDATION.md` |
| `METALIUM_FIX_PLAN.md` | Old fix plan - superseded by integration plan | `METALIUM_INTEGRATION_PLAN.md` |
| `METALIUM_INTEGRATION_APPROACHES.md` | Investigation - decision made | `EXTERNAL_SDK_IMPLEMENTATION_PLAN.md` |
| `METALIUM_INVESTIGATION_SUMMARY.md` | Investigation summary - approach decided | `EXTERNAL_SDK_IMPLEMENTATION_PLAN.md` |
| `PATTERN_DETECTION_REFACTOR_PLAN.md` | Old refactor plan - pattern detection in Phase 1 | `phases/PHASES_STATUS.md` |
| `TILELANG_TO_TT_EXAMPLES_PLAN.md` | Old examples plan - superseded by 6-phase plan | `phases/PHASES_STATUS.md` |

**Action:** Delete these files - content is historical, decisions already implemented.

---

### 🗑️ Category 2: Obsolete/Duplicate Status Documents (DELETE - 2 files)

| File | Reason | Superseded By |
|------|--------|---------------|
| `EXTERNAL_SDK_STATUS.md` | Old status - merged into unified plan | `EXTERNAL_SDK_IMPLEMENTATION_PLAN.md` |
| `PHASE_STATUS_SUMMARY.md` | Old summary - replaced by detailed status | `phases/PHASES_STATUS.md` |

**Action:** Delete - content is outdated or moved to newer docs.

---

### 🗑️ Category 3: Obsolete CI Documents (CONSOLIDATE - 2 files → 1)

| File | Reason | Action |
|------|--------|--------|
| `CI_AND_SCAFFOLDING.md` | Old comprehensive CI doc | Keep core content, merge into CI.md |
| `CI_FLOW_DIAGRAM.md` | Duplicate CI flow info | Merge diagrams into CI.md |

**Action:** Consolidate into single `CI.md` document, delete originals.

---

### 🗑️ Category 4: Detailed Workstream Docs (ARCHIVE - 10 files)

These are detailed design docs for individual workstreams. Now that implementation is complete, we can archive the detailed specs and keep only STATUS files.

**Workstream 1 (4 files) - Keep STATUS only:**
- ❌ DELETE: `workstream1/ws1_default_annotation_helper.md`
- ❌ DELETE: `workstream1/ws1_engine_adapter.md`
- ❌ DELETE: `workstream1/ws1_lower_hook.md`
- ❌ DELETE: `workstream1/ws1_target_registration.md`
- ❌ DELETE: `workstream1/ws1_target_registration_test.md`
- ✅ KEEP: `workstream1/WS1_STATUS.md`

**Workstream 2 (4 files) - Keep STATUS only:**
- ❌ DELETE: `workstream2/README.md`
- ❌ DELETE: `workstream2/ws2_python_integration.md`
- ❌ DELETE: `workstream2/ws2_schedule_inference.md`
- ❌ DELETE: `workstream2/ws2_shard_inference.md`
- ✅ KEEP: `workstream2/WS2_STATUS.md`

**Rationale:** The WS*_STATUS.md files contain the final implementation status and test results. The detailed design docs were planning documents that are no longer needed for reference.

**Action:** Delete detailed design docs (10 files), keep only 6 STATUS files.

---

### ✅ Category 5: Keep - Current Documentation (27 files)

#### **Core Reference Documents (7 files)**

| File | Purpose | Status |
|------|---------|--------|
| `README.md` | Main entry point for TT docs | ✅ Current |
| `UNIFIED_MATMUL_MVP_PLAN.md` | **Authoritative plan** - master reference | ✅ Current |
| `GPU_vs_Tenstorrent.md` | Architecture comparison | ✅ Current |
| `kernel_authoring_comparison.md` | Authoring guide | ✅ Current |
| `TIR_SPECIFICATIONS.md` | TIR extensions for TT | ✅ Current |
| `local_build_guide.md` | Build instructions | ✅ Current |
| `CI.md` (consolidated) | CI/CD documentation | ✅ Current |

#### **Implementation Plans (4 files)**

| File | Purpose | Status |
|------|---------|--------|
| `IR_DRIVEN_CODEGEN_PLAN.md` | Codegen implementation plan (Tasks 1-6) | ✅ Complete (reference) |
| `EXTERNAL_SDK_IMPLEMENTATION_PLAN.md` | External SDK integration plan | ✅ Complete (reference) |
| `METALIUM_INTEGRATION_PLAN.md` | Metalium API integration (Weeks 16-18) | ✅ Complete (reference) |
| `METALIUM_SDK_VALIDATION_PLAN.md` | SDK validation plan (Weeks 19-22) | 🚧 Active (next phase) |

#### **Setup & Validation (3 files)**

| File | Purpose | Status |
|------|---------|--------|
| `METALIUM_SETUP_GUIDE.md` | SDK installation guide | ✅ Current |
| `METALIUM_API_ANALYSIS.md` | API reference/gaps | ✅ Current |
| `IR_LOWERING_VALIDATION.md` | IR lowering validation results | ✅ Complete (reference) |

#### **Phase Implementation (7 files)**

| File | Purpose | Status |
|------|---------|--------|
| `phases/PHASES_STATUS.md` | Master phase tracking | 🚧 Active |
| `phases/PHASE1_FOUNDATION_SPEC.md` | Phase 1 specification | 🚧 Active |
| `phases/PHASE2_OPTIMIZATIONS_SPEC.md` | Phase 2 specification | 🚧 Active |
| `phases/PHASE3_ADVANCED_SPEC.md` | Phase 3 specification | 🚧 Active |
| `phases/PHASE4_ATTENTION_SPEC.md` | Phase 4 specification | 🚧 Active |
| `phases/PHASE5_SPECIALIZED_SPEC.md` | Phase 5 specification | 🚧 Active |
| `phases/PHASE6_COMPLEX_SPEC.md` | Phase 6 specification | 🚧 Active |

#### **Workstream Status (6 files)**

| File | Purpose | Status |
|------|---------|--------|
| `workstream1/WS1_STATUS.md` | WS1 implementation status | ✅ Complete |
| `workstream2/WS2_STATUS.md` | WS2 implementation status | ✅ Complete |
| `workstream3/WS3_STATUS.md` | WS3 implementation status | ✅ Complete |
| `workstream4/WS4_STATUS.md` | WS4 implementation status | ✅ Complete |
| `workstream5/WS5_STATUS.md` | WS5 implementation status | ✅ Complete |
| `workstream6/WS6_STATUS.md` | WS6 implementation status | ✅ Complete |

#### **Codegen Specs & Current Work (3 files)**

| File | Purpose | Status |
|------|---------|--------|
| `codegen/DST_DOUBLE_BUFFERING_SPEC.md` | DST lifecycle specification | ✅ Current |
| `CODEGEN_FIX_PLAN.md` | **NEW** - Codegen fix plan | 🚧 Active |
| `SESSION_SUMMARY_2025-10-08.md` | Session summary (today) | ✅ Current |

---

## File Deletion List

### Total to delete: 21 files

```bash
# Planning docs (7 files)
docs/tenstorrent/IR_LOWERING_ANALYSIS.md
docs/tenstorrent/IR_LOWERING_TASKS.md
docs/tenstorrent/METALIUM_FIX_PLAN.md
docs/tenstorrent/METALIUM_INTEGRATION_APPROACHES.md
docs/tenstorrent/METALIUM_INVESTIGATION_SUMMARY.md
docs/tenstorrent/PATTERN_DETECTION_REFACTOR_PLAN.md
docs/tenstorrent/TILELANG_TO_TT_EXAMPLES_PLAN.md

# Obsolete status (2 files)
docs/tenstorrent/EXTERNAL_SDK_STATUS.md
docs/tenstorrent/PHASE_STATUS_SUMMARY.md

# CI duplicates (2 files - after consolidation)
docs/tenstorrent/CI_AND_SCAFFOLDING.md
docs/tenstorrent/CI_FLOW_DIAGRAM.md

# Workstream detailed design docs (10 files)
docs/tenstorrent/workstream1/ws1_default_annotation_helper.md
docs/tenstorrent/workstream1/ws1_engine_adapter.md
docs/tenstorrent/workstream1/ws1_lower_hook.md
docs/tenstorrent/workstream1/ws1_target_registration.md
docs/tenstorrent/workstream1/ws1_target_registration_test.md
docs/tenstorrent/workstream2/README.md
docs/tenstorrent/workstream2/ws2_python_integration.md
docs/tenstorrent/workstream2/ws2_schedule_inference.md
docs/tenstorrent/workstream2/ws2_shard_inference.md
```

---

## Consolidation Actions

### 1. Consolidate CI Documentation

**Create new consolidated `CI.md`:**
```markdown
# Tenstorrent Backend CI/CD

[Keep current CI.md content]

## CI Flow Diagram
[Merge content from CI_FLOW_DIAGRAM.md]

## Scaffolding and Setup
[Merge relevant content from CI_AND_SCAFFOLDING.md]
```

**Then delete:**
- `CI_AND_SCAFFOLDING.md`
- `CI_FLOW_DIAGRAM.md`

---

## Documentation Structure After Cleanup

```
docs/tenstorrent/
├── README.md                                  # Main entry point
├── UNIFIED_MATMUL_MVP_PLAN.md                # ⭐ Authoritative plan
│
├── Architecture & Design
│   ├── GPU_vs_Tenstorrent.md                # Architecture comparison
│   ├── kernel_authoring_comparison.md        # Authoring guide
│   └── TIR_SPECIFICATIONS.md                 # TIR extensions
│
├── Setup & Validation
│   ├── local_build_guide.md                  # Build instructions
│   ├── CI.md                                 # CI/CD documentation
│   ├── METALIUM_SETUP_GUIDE.md              # SDK installation
│   ├── METALIUM_API_ANALYSIS.md             # API reference
│   └── IR_LOWERING_VALIDATION.md            # Validation results
│
├── Implementation Plans (Reference)
│   ├── IR_DRIVEN_CODEGEN_PLAN.md            # Codegen plan (complete)
│   ├── EXTERNAL_SDK_IMPLEMENTATION_PLAN.md  # SDK integration (complete)
│   ├── METALIUM_INTEGRATION_PLAN.md         # Metalium API (complete)
│   ├── METALIUM_SDK_VALIDATION_PLAN.md      # SDK validation (next)
│   └── CODEGEN_FIX_PLAN.md                  # Current work (active)
│
├── Workstream Status
│   ├── workstream1/WS1_STATUS.md            # Target registration ✅
│   ├── workstream2/WS2_STATUS.md            # Metadata inference ✅
│   ├── workstream3/WS3_STATUS.md            # TIR transforms ✅
│   ├── workstream4/WS4_STATUS.md            # Compute codegen ✅
│   ├── workstream5/WS5_STATUS.md            # Reader/Writer codegen ✅
│   └── workstream6/WS6_STATUS.md            # Host program ✅
│
├── Phase Implementation (Active)
│   ├── phases/PHASES_STATUS.md              # Master tracking 🚧
│   ├── phases/PHASE1_FOUNDATION_SPEC.md     # Phase 1 (37% complete)
│   ├── phases/PHASE2_OPTIMIZATIONS_SPEC.md  # Phase 2 (planned)
│   ├── phases/PHASE3_ADVANCED_SPEC.md       # Phase 3 (planned)
│   ├── phases/PHASE4_ATTENTION_SPEC.md      # Phase 4 (planned)
│   ├── phases/PHASE5_SPECIALIZED_SPEC.md    # Phase 5 (planned)
│   └── phases/PHASE6_COMPLEX_SPEC.md        # Phase 6 (planned)
│
├── Codegen Specifications
│   └── codegen/DST_DOUBLE_BUFFERING_SPEC.md # DST lifecycle
│
└── Session Summaries
    └── SESSION_SUMMARY_2025-10-08.md        # Today's work
```

**Total after cleanup: 27 files** (down from 48)

---

## Updates Required

### 1. Update CLAUDE.md

Remove references to deleted docs:
- ❌ Remove: IR_LOWERING_TASKS.md reference
- ❌ Remove: Detailed workstream doc references
- ✅ Keep: All current doc references

### 2. Update README.md

Update documentation index:
- Remove deleted file references
- Add new consolidated structure
- Update "Where to Start" section

### 3. Update Workstream STATUS Files

Remove references to detailed design docs:
- `workstream1/WS1_STATUS.md` - remove links to ws1_*.md files
- `workstream2/WS2_STATUS.md` - remove links to ws2_*.md files

---

## Implementation Plan

### Phase 1: Consolidate CI Docs (30 minutes)

1. Read current `CI.md`
2. Extract relevant content from `CI_AND_SCAFFOLDING.md`
3. Extract diagrams from `CI_FLOW_DIAGRAM.md`
4. Create new consolidated `CI.md`
5. Verify no information loss

### Phase 2: Delete Obsolete Files (10 minutes)

Execute deletion commands:
```bash
cd /home/ubuntu/code/tilelang-tt

# Remove planning docs
rm docs/tenstorrent/IR_LOWERING_ANALYSIS.md
rm docs/tenstorrent/IR_LOWERING_TASKS.md
rm docs/tenstorrent/METALIUM_FIX_PLAN.md
rm docs/tenstorrent/METALIUM_INTEGRATION_APPROACHES.md
rm docs/tenstorrent/METALIUM_INVESTIGATION_SUMMARY.md
rm docs/tenstorrent/PATTERN_DETECTION_REFACTOR_PLAN.md
rm docs/tenstorrent/TILELANG_TO_TT_EXAMPLES_PLAN.md

# Remove obsolete status
rm docs/tenstorrent/EXTERNAL_SDK_STATUS.md
rm docs/tenstorrent/PHASE_STATUS_SUMMARY.md

# Remove CI duplicates
rm docs/tenstorrent/CI_AND_SCAFFOLDING.md
rm docs/tenstorrent/CI_FLOW_DIAGRAM.md

# Remove workstream design docs
rm docs/tenstorrent/workstream1/ws1_*.md
rm docs/tenstorrent/workstream2/ws2_*.md
rm docs/tenstorrent/workstream2/README.md
```

### Phase 3: Update References (30 minutes)

1. Update CLAUDE.md (remove obsolete refs)
2. Update README.md (new doc structure)
3. Update WS1_STATUS.md (remove design doc links)
4. Update WS2_STATUS.md (remove design doc links)

### Phase 4: Verification (15 minutes)

1. Check no broken links in remaining docs
2. Verify all references in CLAUDE.md are valid
3. Test documentation builds (if applicable)
4. Review final structure

### Phase 5: Create PR (15 minutes)

1. Commit all changes
2. Create comprehensive PR description
3. List all deleted files
4. Explain rationale

**Total time: ~2 hours**

---

## PR Description Template

```markdown
## Documentation Cleanup: Remove 21 Obsolete Files

**Summary:** Clean up `docs/tenstorrent/` directory by removing obsolete planning documents, consolidating CI docs, and streamlining workstream documentation.

**Changes:**
- 🗑️ Deleted 21 obsolete/deprecated files (44% reduction)
- 📝 Consolidated 3 CI documents into 1
- ✅ Kept 27 current/active documents
- 🔗 Updated references in CLAUDE.md and README.md

**Files Deleted (21 total):**

Planning docs (obsolete):
- IR_LOWERING_ANALYSIS.md - Superseded by IR_LOWERING_VALIDATION.md
- IR_LOWERING_TASKS.md - All tasks complete (PRs #72-79)
- METALIUM_FIX_PLAN.md - Superseded by METALIUM_INTEGRATION_PLAN.md
- METALIUM_INTEGRATION_APPROACHES.md - Decision made and implemented
- METALIUM_INVESTIGATION_SUMMARY.md - Investigation complete
- PATTERN_DETECTION_REFACTOR_PLAN.md - Merged into phases
- TILELANG_TO_TT_EXAMPLES_PLAN.md - Superseded by 6-phase plan

Status docs (obsolete):
- EXTERNAL_SDK_STATUS.md - Merged into EXTERNAL_SDK_IMPLEMENTATION_PLAN.md
- PHASE_STATUS_SUMMARY.md - Superseded by phases/PHASES_STATUS.md

CI docs (consolidated):
- CI_AND_SCAFFOLDING.md - Merged into CI.md
- CI_FLOW_DIAGRAM.md - Merged into CI.md

Workstream design docs (10 files):
- workstream1/ws1_*.md (5 files) - Keep WS1_STATUS.md only
- workstream2/ws2_*.md + README.md (5 files) - Keep WS2_STATUS.md only

**Rationale:**
- IR lowering is complete (95/95 tests passing)
- Planning documents served their purpose
- Detailed design docs archived (keep STATUS only)
- Cleaner structure for new contributors

**Verification:**
- ✅ No broken links in remaining docs
- ✅ CLAUDE.md references updated
- ✅ README.md structure updated
- ✅ All tests still pass (no code changes)

**Documentation structure after cleanup:**
- 7 core reference docs
- 4 implementation plans
- 3 setup/validation guides
- 7 phase specifications (active)
- 6 workstream status files
- 1 codegen spec
- 1 session summary

**Before:** 48 files
**After:** 27 files (44% reduction)
```

---

## Success Criteria

- ✅ 21 obsolete files deleted
- ✅ CI docs consolidated into single file
- ✅ No broken links in remaining documentation
- ✅ CLAUDE.md references all valid
- ✅ Clear documentation structure
- ✅ All tests still pass (95/95)
- ✅ PR merged successfully

---

## Timeline

**Total estimated time:** 2 hours

**Breakdown:**
- Phase 1 (Consolidate CI): 30 min
- Phase 2 (Delete files): 10 min
- Phase 3 (Update refs): 30 min
- Phase 4 (Verification): 15 min
- Phase 5 (Create PR): 15 min
- Buffer: 20 min

**Can be completed in single session.**

---

**Status:** Ready for execution
**Next step:** Begin Phase 1 (CI consolidation)
