# Migration Implementation Checklist

**Branch**: refactor-tt-metadata-driven-lowering (PR #127)
**Target**: main
**Created**: 2025-10-14

---

## Phase 1: Parallel Compatibility Implementation

### 1.1 Update engine/tenstorrent/lower.py

```python
# Add at top of file
import os

# In OptimizeForTargetTT function, replace lines 81-96 with:
def OptimizeForTargetTT(mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    """TT-specific optimization phase."""

    # Feature flag for new pipeline
    USE_NEW_PIPELINE = os.environ.get("TT_USE_NEW_PIPELINE", "false").lower() == "true"

    if USE_NEW_PIPELINE:
        # New metadata-driven pipeline
        from tilelang.tenstorrent.passes import run_pipeline

        # Apply compatibility transforms first
        from tilelang.tenstorrent.compat import apply_compatibility_transforms
        mod = apply_compatibility_transforms(mod)

        # Run new pipeline
        mod = run_pipeline(
            mod,
            plan_path="tt.plan.json",
            target_device="grayskull",  # TODO: Extract from target
            partition_strategy="row_major",
            enable_double_buffer=True,
            enable_prefetch=True
        )

        # Continue with common optimizations (line 113 onwards)
    else:
        # Legacy pipeline (current code)
        mod = infer_default_tt_schedule(mod)
        mod = infer_default_tt_shard(mod)
        mod = apply_layout_aware_metadata_passes(mod)
        mod = grid_to_persistent_tt(mod)
        mod = tt_tiles_to_core_map(mod)
        mod = memory_space_lower_tt(mod)
        mod = tile_pad_tt(mod)
        mod = lower_gemm_to_tt_intrinsics(mod)

    # Common optimizations continue...
    mod = tilelang.transform.FlattenBuffer()(mod)
    # ... rest of function
```

### 1.2 Add CI testing for both paths

Create `.github/workflows/test-dual-pipeline.yml`:
```yaml
name: Test Dual Pipeline
on: [push, pull_request]

jobs:
  test-legacy:
    runs-on: ubuntu-latest
    env:
      TT_USE_NEW_PIPELINE: "false"
    steps:
      - uses: actions/checkout@v2
      - name: Test Legacy Pipeline
        run: |
          bash maint/scripts/local_build_and_test_tt.sh

  test-new:
    runs-on: ubuntu-latest
    env:
      TT_USE_NEW_PIPELINE: "true"
    steps:
      - uses: actions/checkout@v2
      - name: Test New Pipeline
        run: |
          bash maint/scripts/local_build_and_test_tt.sh
```

---

## Phase 2: Test Migration

### 2.1 Update test_metadata_inference.py

```python
# Replace imports
# OLD:
from tilelang.tenstorrent import (
    apply_tt_defaults,
    infer_default_tt_schedule,
    infer_default_tt_shard,
    apply_tt_metadata_passes
)

# NEW:
from tilelang.tenstorrent import apply_tt_defaults
from tilelang.tenstorrent.passes import (
    InferTTLayout,
    PropagateTTLayout,
    TTTilesToCoreMap,
    run_pipeline
)
from tilelang.tenstorrent.attrs import TT_CORE_GRID, TT_LAYOUT_DESC, TT_WORK_PARTITION

# Update test methods to use new passes
def test_schedule_inference_8x8_grid(self):
    # ... create mod ...
    mod = apply_tt_defaults(mod)

    # NEW: Use new passes
    mod = InferTTLayout()(mod)
    mod = PropagateTTLayout()(mod)
    mod = TTTilesToCoreMap()(mod)

    # NEW: Check new attributes
    func = mod["main"]
    assert TT_CORE_GRID in func.attrs
    assert TT_WORK_PARTITION in func.attrs
    assert TT_LAYOUT_DESC in func.attrs

    # Validate core grid
    core_grid = func.attrs[TT_CORE_GRID]
    assert core_grid == (8, 8)

    # Validate work partition
    work_partition = func.attrs[TT_WORK_PARTITION]
    assert len(work_partition) == 64  # 8x8 cores
```

### 2.2 Update test_persistent_lowering.py

```python
# Replace apply_tt_transform_passes with run_pipeline
# OLD:
from tilelang.tenstorrent.passes import (
    apply_tt_transform_passes,
    grid_to_persistent_tt
)

# NEW:
from tilelang.tenstorrent.passes import (
    run_pipeline,
    GridToPersistentTT
)

# In test methods:
# OLD:
mod = apply_tt_transform_passes(mod)

# NEW:
mod = run_pipeline(mod, plan_path="test.plan.json")

# Check for plan file generation
assert os.path.exists("test.plan.json")
with open("test.plan.json") as f:
    plan = json.load(f)
    assert "cores" in plan
    assert "work_items" in plan
```

### 2.3 Update remaining tests

For each test file:
1. Replace legacy imports with new pass imports
2. Update attribute checking to use new names
3. Add plan.json validation where applicable

Test files to update:
- [ ] test_transform_pipeline_logging.py
- [ ] test_layout_aware_metadata.py
- [ ] test_lower_gemm_to_tt_intrinsics.py
- [ ] test_memory_space_lower_tt.py
- [ ] test_tile_pad_tt.py
- [ ] test_tt_tiles_to_core_map.py
- [ ] test_verify_tt_ir.py
- [ ] test_mvp_acceptance.py
- [ ] test_codegen_pipeline.py

---

## Phase 3: Remove Legacy Code

### 3.1 Remove C++ legacy passes

Delete files:
```bash
rm src/transform/tenstorrent/infer_tt_schedule.cc
rm src/transform/tenstorrent/infer_tt_shard.cc
```

Update `src/transform/CMakeLists.txt`:
```cmake
# Remove lines:
# transform/tenstorrent/infer_tt_schedule.cc
# transform/tenstorrent/infer_tt_shard.cc
```

### 3.2 Clean up Python imports

Update `tilelang/tenstorrent/__init__.py`:
```python
# Remove legacy imports
# DELETE:
from .passes import (
    infer_default_tt_schedule,
    infer_default_tt_shard,
    apply_tt_metadata_passes,
    apply_layout_aware_metadata_passes,
)

# Keep only new imports
from .passes import (
    InferTTLayout,
    PropagateTTLayout,
    TTTilesToCoreMap,
    LowerTTTileIntrinsics,
    GridToPersistentTT,
    build_tt_pipeline,
    run_pipeline,
)
```

### 3.3 Archive legacy documentation

```bash
# Create legacy directory
mkdir -p docs/tenstorrent/legacy

# Move legacy docs
mv docs/tenstorrent/PASS_TABLE_LEGACY.md docs/tenstorrent/legacy/
```

### 3.4 Remove legacy Python implementations

```bash
# Remove if they exist as separate files
rm tilelang/tenstorrent/passes/infer_default_tt_schedule.py
rm tilelang/tenstorrent/passes/infer_default_tt_shard.py
```

---

## Phase 4: Make New Architecture Default

### 4.1 Remove feature flag from engine/tenstorrent/lower.py

```python
def OptimizeForTargetTT(mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    """TT-specific optimization phase."""

    # Always use new pipeline
    from tilelang.tenstorrent.passes import run_pipeline
    from tilelang.tenstorrent.compat import apply_compatibility_transforms

    # Apply compatibility for any legacy attributes
    mod = apply_compatibility_transforms(mod)

    # Run new pipeline
    mod = run_pipeline(
        mod,
        plan_path="tt.plan.json",
        target_device=_extract_device_from_target(target),
        partition_strategy="row_major",
        enable_double_buffer=True,
        enable_prefetch=True
    )

    # Common optimizations...
    mod = tilelang.transform.FlattenBuffer()(mod)
    # ... rest
```

### 4.2 Update examples

Update `examples/tenstorrent/example_gemm.py`:
```python
# Use new API
from tilelang.tenstorrent.passes import run_pipeline

# In the example
mod = run_pipeline(mod, plan_path="gemm.plan.json")
```

### 4.3 Update main documentation

Update `docs/tenstorrent/README.md`:
- Remove references to legacy passes
- Update architecture diagram
- Add migration notes for users

Update `docs/tenstorrent/TT_ARCHITECTURE.md`:
- Update pass pipeline section
- Remove legacy pass descriptions

---

## Validation Steps

### After Each Phase

```bash
# Run tests
pytest testing/python/tenstorrent/ -v

# Check for legacy references
grep -r "infer_default_tt_schedule" . --include="*.py" --include="*.cc"
grep -r "infer_default_tt_shard" . --include="*.py" --include="*.cc"

# Build and test
bash maint/scripts/local_build_and_test_tt.sh

# Check generated artifacts
ls -la tt.plan.json
cat tt.plan.json | jq .
```

### Final Validation

```bash
# Full clean build
rm -rf build/
cmake -B build -DUSE_LLVM=true
cmake --build build -j$(nproc)

# Run all tests
LD_LIBRARY_PATH=build/tvm:$LD_LIBRARY_PATH pytest testing/python/tenstorrent/ -v

# Check no legacy code remains
find . -name "*.py" -o -name "*.cc" | xargs grep -l "infer_default_tt"

# Verify examples work
python examples/tenstorrent/example_gemm.py
```

---

## Rollback Plan

If issues are discovered:

1. **Quick rollback** (Phase 1-2):
   ```bash
   export TT_USE_NEW_PIPELINE=false
   ```

2. **Code rollback** (Phase 3-4):
   ```bash
   git revert <migration-commit>
   git push
   ```

3. **Keep compatibility layer** for 2 releases to allow gradual migration

---

## Sign-off Checklist

- [ ] All tests passing with new pipeline
- [ ] Performance benchmarks show no regression
- [ ] Documentation updated
- [ ] Examples working
- [ ] Legacy code removed
- [ ] CI green on main
- [ ] Migration guide published

---

**Status**: Ready to implement after PR #127 is merged