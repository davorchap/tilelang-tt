# Obsolete Documentation Archive

**Date Archived:** 2025-10-16
**Reason:** These documents are obsolete and no longer relevant to the project

---

## Archived Documents

### TT_Pass_Specifications.md
- **Original Location:** `docs/tenstorrent/reference/`
- **Date Created:** 2025-10-15
- **Why Obsolete:**
  - Contains C++ implementation templates which are no longer relevant (project is Python-only)
  - Mixed v1-v4 approaches that have been superseded by v5 pipeline
  - Valuable algorithm descriptions have been extracted to `extracted_valuable_content_from_specifications.md`
  - Pass specifications are now maintained in `v5_pipeline.md` and individual pass implementations

### TT_Python_to_CPP_Migration.md
- **Original Location:** `docs/tenstorrent/archive/`
- **Date Created:** 2025-10-15
- **Why Obsolete:**
  - Project has committed to Python-only implementation
  - No C++ migration is planned or needed
  - All passes are implemented in Python for maintainability and rapid iteration
  - The v5 pipeline (fully Python) is complete and production-ready

---

## Important Note

These documents are preserved for historical reference only. They should NOT be used for:
- Implementation guidance
- Architecture decisions
- Pass specifications
- Testing patterns

For current documentation, please refer to:
- `docs/tenstorrent/architecture/v5_pipeline.md` - Complete v5 pipeline documentation
- `docs/tenstorrent/reference/extracted_valuable_content_from_specifications.md` - Valuable content extracted from archived docs
- Individual pass implementations in `tilelang/tenstorrent/passes/`

---

## Why Python-Only?

The project made a strategic decision to remain Python-only because:
1. **Maintainability**: Python passes are easier to understand, modify, and debug
2. **Rapid Iteration**: Python allows for quick prototyping and experimentation
3. **Performance**: Compilation time is not a bottleneck for the use cases
4. **Consistency**: All passes in the same language simplifies the codebase
5. **Success**: The v5 pipeline is complete, tested, and working entirely in Python

---

**Document Created:** 2025-10-16