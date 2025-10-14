# Migration Plan: Legacy to New Metadata-Driven Architecture

**Created**: 2025-10-14
**PR Reference**: #127 (refactor-tt-metadata-driven-lowering)
**Status**: Ready for implementation

---

## Executive Summary

This document outlines the migration plan from the legacy Tenstorrent backend architecture to the new metadata-driven lowering pipeline introduced in PR #127. The new architecture provides cleaner abstractions, centralized attribute management, and a more maintainable pass pipeline.

---

## Architecture Comparison

### Legacy Architecture
- **Scattered metadata**: Attributes spread across multiple passes
- **Ad-hoc pass ordering**: No clear pipeline structure
- **Mixed responsibilities**: Passes doing multiple unrelated tasks
- **No runtime plan**: Missing coordination between IR and runtime

### New Architecture (PR #127)
- **Centralized metadata**: `CoreRange`, `WorkItem` dataclasses in `attrs.py`
- **Clear pipeline**: 5 well-defined passes with specific responsibilities
- **IR sugar helpers**: `with_core_grid()`, `with_layout_desc()` for clean metadata attachment
- **Runtime plan**: Emits `tt.plan.json` for host-device coordination
- **Compatibility layer**: Backward compatibility through `compat.py`

---

## Pass Mapping: Legacy → New

| Legacy Pass | New Pass | Notes |
|-------------|----------|-------|
| `infer_default_tt_schedule` | `InferTTLayout` | Schedule → Layout inference |
| `infer_default_tt_shard` | `PropagateTTLayout` | Shard → Layout propagation |
| `layout_aware_work_partition_tt` | `TTTilesToCoreMap` | Work partition → Core mapping |
| `lower_gemm_to_tt_intrinsics` | `LowerTTTileIntrinsics` | Pattern detection → Intrinsic lowering |
| `grid_to_persistent_tt` | `GridToPersistentTT` | Same name, refactored implementation |
| `memory_space_lower_tt` | Part of `GridToPersistentTT` | Integrated into persistent lowering |
| `tile_pad_tt` | Part of `TTTilesToCoreMap` | Padding integrated with core mapping |
| `tt_tiles_to_core_map` | `TTTilesToCoreMap` | Renamed and expanded |
| `verify_tt_ir` | Built into each pass | Verification distributed |

### New Pipeline Order
```python
1. InferTTLayout()        # Infer buffer layouts and metadata
2. PropagateTTLayout()    # Propagate and normalize layout info
3. TTTilesToCoreMap()     # Compute core mapping and partitioning
4. LowerTTTileIntrinsics() # Lower tile ops to device intrinsics
5. GridToPersistentTT()   # Final lowering to persistent kernels
```

---

## Migration Phases

### Phase 1: Parallel Compatibility (Week 1)
**Goal**: Run both architectures side-by-side without breaking existing code

1. **Keep compatibility layer active** (`compat.py`)
   - Legacy passes redirect to new implementation
   - Attribute conversion automatic
   - Tests continue to pass

2. **Update engine/tenstorrent/lower.py**
   ```python
   # Add feature flag
   USE_NEW_PIPELINE = os.environ.get("TT_USE_NEW_PIPELINE", "false").lower() == "true"

   if USE_NEW_PIPELINE:
       from tilelang.tenstorrent.passes import run_pipeline
       mod = run_pipeline(mod, plan_path="tt.plan.json")
   else:
       # Legacy path (current code)
       mod = infer_default_tt_schedule(mod)
       mod = infer_default_tt_shard(mod)
       # ...
   ```

3. **Validate both paths**
   - Run CI with both `TT_USE_NEW_PIPELINE=false` and `true`
   - Ensure identical outputs

### Phase 2: Test Migration (Week 2)
**Goal**: Update tests to use new API directly

#### Test Categories and Migration Strategy

1. **Metadata Tests** (`test_metadata_inference.py`)
   - Update to use `InferTTLayout`, `PropagateTTLayout`
   - Check new attribute names: `tt.core_grid`, `tt.layout_desc`
   - Remove legacy attribute checks

2. **Pipeline Tests** (`test_persistent_lowering.py`, `test_transform_pipeline_logging.py`)
   - Use `build_tt_pipeline()` and `run_pipeline()`
   - Update attribute inspection

3. **Pass-specific Tests** (`test_lower_gemm_to_tt_intrinsics.py`, etc.)
   - Update to use new pass classes
   - Verify new metadata format

4. **Integration Tests** (`test_mvp_acceptance.py`, `test_codegen_pipeline.py`)
   - Switch to `run_pipeline()`
   - Validate `tt.plan.json` generation

#### Example Test Migration:
```python
# Before (legacy)
from tilelang.tenstorrent import (
    infer_default_tt_schedule,
    infer_default_tt_shard,
    apply_tt_metadata_passes
)

mod = apply_tt_defaults(mod)
mod = infer_default_tt_schedule(mod)
mod = infer_default_tt_shard(mod)

# After (new)
from tilelang.tenstorrent.passes import (
    InferTTLayout,
    PropagateTTLayout,
    run_pipeline
)

# Option 1: Individual passes
mod = InferTTLayout()(mod)
mod = PropagateTTLayout()(mod)

# Option 2: Full pipeline
mod = run_pipeline(mod, plan_path="tt.plan.json")
```

### Phase 3: Remove Legacy Code (Week 3)
**Goal**: Clean up codebase by removing legacy implementations

1. **Remove C++ legacy passes**
   - Delete: `infer_tt_schedule.cc`, `infer_tt_shard.cc`
   - Keep: Passes that are still used or refactored

2. **Remove Python legacy wrappers**
   - Remove legacy imports from `__init__.py`
   - Keep `compat.py` for documentation/reference

3. **Update documentation**
   - Archive legacy docs to `docs/tenstorrent/legacy/`
   - Update all references in active docs

4. **Clean up attributes**
   - Remove support for legacy attribute names
   - Standardize on new names everywhere

### Phase 4: Default Switch (Week 4)
**Goal**: Make new architecture the default

1. **Remove feature flag**
   ```python
   # engine/tenstorrent/lower.py
   # Always use new pipeline
   from tilelang.tenstorrent.passes import run_pipeline
   mod = run_pipeline(mod, plan_path=plan_path)
   ```

2. **Update examples**
   - `examples/tenstorrent/example_gemm.py`
   - Use new API in all examples

3. **Final validation**
   - Full CI run
   - Performance benchmarks
   - Hardware validation (when available)

---

## File Changes Required

### Core Changes

| File | Action | Details |
|------|--------|---------|
| `tilelang/engine/tenstorrent/lower.py` | Modify | Replace legacy passes with `run_pipeline()` |
| `tilelang/tenstorrent/__init__.py` | Modify | Remove legacy exports, keep new ones |
| `src/transform/tenstorrent/infer_tt_schedule.cc` | Delete | Replaced by InferTTLayout |
| `src/transform/tenstorrent/infer_tt_shard.cc` | Delete | Replaced by PropagateTTLayout |
| `tilelang/tenstorrent/passes/*.py` | Add | New pass implementations (already in PR #127) |
| `tilelang/tenstorrent/compat.py` | Keep→Archive | Move to legacy/ after migration |

### Test Updates

| Test File | Changes Required |
|-----------|------------------|
| `test_metadata_inference.py` | Full rewrite - use new passes |
| `test_persistent_lowering.py` | Update imports and attribute names |
| `test_transform_pipeline_logging.py` | Use `run_pipeline()` |
| `test_layout_aware_metadata.py` | Update to new metadata format |
| `test_mvp_acceptance.py` | Switch to new pipeline |
| `test_codegen_pipeline.py` | Validate `tt.plan.json` |
| All other tests | Update imports and attribute checks |

### Documentation Updates

| Document | Action |
|----------|--------|
| `PASS_TABLE_TT.md` | Update pass status (remove legacy) |
| `TT_ARCHITECTURE.md` | Update pipeline description |
| `NEW_LOWERING_ARCHITECTURE.md` | Mark as current (remove "NEW") |
| `README.md` | Update examples to use new API |
| Create `LEGACY_MIGRATION.md` | Document what changed for users |

---

## Validation Checklist

### Pre-migration
- [ ] PR #127 merged to main
- [ ] Backup branch created
- [ ] All tests passing on main

### Phase 1: Compatibility
- [ ] Feature flag implemented
- [ ] Both paths tested in CI
- [ ] No regression in test results

### Phase 2: Test Migration
- [ ] All tests updated to new API
- [ ] Tests pass with new pipeline
- [ ] Coverage maintained

### Phase 3: Legacy Removal
- [ ] Legacy C++ passes deleted
- [ ] Legacy Python code removed
- [ ] Documentation updated
- [ ] No broken imports

### Phase 4: Default Switch
- [ ] New pipeline is default
- [ ] All examples updated
- [ ] CI fully green
- [ ] Performance validated

---

## Risk Mitigation

### Risks and Mitigations

1. **Risk**: Breaking existing user code
   - **Mitigation**: Keep compatibility layer for 1-2 releases
   - **Mitigation**: Provide migration guide with examples

2. **Risk**: Performance regression
   - **Mitigation**: Benchmark both pipelines before switch
   - **Mitigation**: Profile and optimize new passes

3. **Risk**: Missing functionality
   - **Mitigation**: Comprehensive test coverage
   - **Mitigation**: Phased rollout with feature flag

4. **Risk**: Hardware validation issues
   - **Mitigation**: Keep legacy path available via environment variable
   - **Mitigation**: Test on hardware before full switch

---

## Success Metrics

- **Test Coverage**: 100% of legacy tests migrated and passing
- **Performance**: No regression in compilation time or runtime
- **Code Quality**: Reduced lines of code, improved maintainability
- **Documentation**: All docs updated and accurate
- **User Impact**: Zero breaking changes for external users

---

## Timeline

| Week | Phase | Deliverables |
|------|-------|-------------|
| 1 | Parallel Compatibility | Feature flag, dual-path CI |
| 2 | Test Migration | All tests using new API |
| 3 | Legacy Removal | Clean codebase, archived docs |
| 4 | Default Switch | New architecture as default |

---

## Implementation Commands

```bash
# Phase 1: Test with feature flag
export TT_USE_NEW_PIPELINE=true
bash maint/scripts/local_build_and_test_tt.sh

# Phase 2: Run migrated tests
pytest testing/python/tenstorrent/ -v

# Phase 3: Find and remove legacy code
grep -r "infer_default_tt_schedule" --include="*.py" --include="*.cc"
grep -r "infer_default_tt_shard" --include="*.py" --include="*.cc"

# Phase 4: Full validation
unset TT_USE_NEW_PIPELINE  # Should use new by default
bash maint/scripts/local_build_and_test_tt.sh
```

---

## Appendix: Attribute Migration

### Legacy → New Attribute Names

| Legacy Attribute | New Attribute | Location |
|-----------------|---------------|----------|
| `tt.grid` | `tt.core_grid` | CoreRange |
| `tt.block_shape` | Part of `tt.work_partition` | WorkItem |
| `tt.start_tile` | Part of `tt.work_partition` | WorkItem |
| `tt.runtime_args` | `tt.work_partition` | Dict[str, List[WorkItem]] |
| `tt_num_tiles` | Computed from `tt.work_partition` | - |
| `tt_tiles_per_core` | Part of `tt.work_partition` | WorkItem.len_k |
| `tt_buffer_*_layout` | `tt.layout_desc[buffer_name]` | Dict |

---

## Questions/Concerns

For questions about this migration plan, please refer to:
- PR #127 for implementation details
- `docs/tenstorrent/NEW_LOWERING_ARCHITECTURE.md` for architecture overview
- `tilelang/tenstorrent/compat.py` for compatibility layer implementation

---

**Document Status**: Ready for review and implementation
**Next Steps**: Begin Phase 1 implementation after PR #127 is merged