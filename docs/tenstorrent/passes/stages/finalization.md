# Stage E: Finalization

**Stage:** E (Finalization)
**Passes:** 1 (E1)
**Purpose:** Finalize runtime signature and validate metadata completeness

---

## Overview

Stage E performs final validation and metadata finalization before codegen. This is the last transform stage - after E1, the IR is ready for code generation.

---

## Pass Pipeline

```
Stage D Output (3 kernels with complete protocol)
    ↓
E1: finalize_persistent_signature_tt
    ↓ Validates: All metadata complete
    ↓ Finalizes: Runtime argument schema
    ↓
Verification (verify_tt_ir)
    ↓
Codegen (5 artifacts)
```

---

## E1: finalize_persistent_signature_tt

**Purpose:** Finalize runtime args and validate metadata completeness

**Location:** `tilelang/tenstorrent/passes/finalize_persistent_signature_tt.py`

### What It Does

1. **Validates Metadata Completeness**
   - All buffers have `tt.buffer.*`, `tt.cb.*`, `tt.tensor_accessor.*`
   - Partition metadata present (`tt.partition_mode`, `tt.grid_tiles`, etc.)
   - All runtime args defined

2. **Finalizes Runtime Signature**
   - Orders runtime arguments consistently
   - Generates per-core argument tables
   - Emits host metadata for codegen

3. **Verifies Guardrails**
   - No default-constructed TensorAccessors
   - All shard metadata present for local_shard mode
   - CB IDs unique and valid (0-31)
   - Core ranges match partition mode

### Validation Checks

```python
# Required metadata for each kernel
assert "tt.partition_mode" in func.attrs
assert "tt.grid_tiles" in func.attrs
assert "tt.core_ranges" in func.attrs
assert "tt.runtime_args" in func.attrs

# Per-buffer validation
for buf in buffers:
    assert f"tt.buffer.{buf.name}" in func.attrs
    assert f"tt.cb.{buf.name}" in func.attrs
    assert f"tt.tensor_accessor.{buf.name}" in func.attrs

# Partition-specific validation
if func.attrs["tt.partition_mode"] == "local_shard":
    assert "tt.shard_grid" in func.attrs
    assert "tt.local_shape_tiles" in func.attrs
```

### Runtime Argument Finalization

**Global Mode:**
```json
{
  "tt.runtime_args": ["start_id", "count", "Mt", "Kt", "Nt"],
  "tt.runtime_arg_types": ["uint32", "uint32", "uint32", "uint32", "uint32"],
  "tt.runtime_arg_offsets": [0, 4, 8, 12, 16]
}
```

**Local Shard Mode:**
```json
{
  "tt.runtime_args": ["start_id", "count", "Mt", "Kt", "Nt", "Sm", "Sn", "Gy", "Gx", "sy", "sx"],
  "tt.runtime_arg_types": ["uint32", ...],
  "tt.runtime_arg_offsets": [0, 4, 8, ...]
}
```

### Host Metadata Emission

E1 generates metadata for host codegen:

```json
{
  "tt.host_metadata": {
    "buffers": {
      "A": {
        "memory": "DRAM",
        "layout": "interleaved",
        "size_bytes": 131072,
        "cb_config": {
          "cb_id": 0,
          "page_size": 2048,
          "depth": 2
        }
      }
    },
    "kernels": {
      "reader": {
        "type": "DataMovement",
        "core_ranges": [[0,0], [7,7]]
      },
      "compute": {
        "type": "Compute",
        "core_ranges": [[0,0], [7,7]]
      },
      "writer": {
        "type": "DataMovement",
        "core_ranges": [[0,0], [7,7]]
      }
    },
    "runtime_args_per_core": {
      "global": ["start_id", "count", "Mt", "Kt", "Nt"]
    }
  }
}
```

### Error Examples

**Missing Buffer Metadata:**
```
Error: Buffer 'A' missing tt.buffer.A metadata
Fix: Ensure Stage A (infer_tt_layout_v5) ran correctly
```

**Default TensorAccessor:**
```
Error: Buffer 'A' has default-constructed TensorAccessor
Fix: Ensure Stage A3 (attach_tensor_accessor_tt) ran correctly
```

**CB ID Conflict:**
```
Error: Duplicate CB ID 0 for buffers 'A' and 'B'
Fix: Ensure Stage A2 (propagate_tt_layout_v5) assigns unique CB IDs
```

**Partition Metadata Missing:**
```
Error: Local shard mode but missing tt.shard_grid
Fix: Ensure Stage B1 (layout_aware_work_partition_tt_v5) ran correctly
```

---

## Output for Codegen

After E1, codegen has everything needed:

### For Reader/Writer Kernels
- NOC operation parameters
- Buffer addressing (TensorAccessor metadata)
- CB configuration
- Core ranges for kernel launch

### For Compute Kernel
- Tile intrinsics with CB IDs
- DST lifecycle operations
- Runtime argument access
- Core ranges for kernel launch

### For Host Program
- Buffer creation API calls
- CB configuration tables
- Kernel creation parameters
- Runtime argument setup
- Core range sets

---

## Design Rationale

### Why Final Validation?

**Catch Errors Early:** Before codegen attempts to generate invalid code

**Example:**
```python
# Without E1: Codegen crashes on missing metadata
# With E1: Clear error message pointing to exact issue
```

### Why Separate from Codegen?

**Clean Separation:** Transform passes vs code generation
**Reusable:** Same validation for all codegen backends

### Why Host Metadata?

**Abstraction:** Codegen doesn't parse all metadata
**Efficiency:** Pre-computed values ready for emission

---

## Testing

```python
def test_stage_e_finalization():
    """Test Stage E finalization."""
    mod = create_matmul_with_metadata()

    # Run Stages A-D
    mod = run_stages_abcd(mod)

    # Run Stage E
    mod = finalize_persistent_signature_tt(mod)

    # Validate runtime args finalized
    assert "tt.runtime_arg_types" in mod.attrs
    assert "tt.runtime_arg_offsets" in mod.attrs

    # Validate host metadata present
    assert "tt.host_metadata" in mod.attrs

    # Validate completeness
    host_meta = mod.attrs["tt.host_metadata"]
    assert "buffers" in host_meta
    assert "kernels" in host_meta
    assert "runtime_args_per_core" in host_meta
```

---

## Common Issues

### Issue 1: Incomplete Pipeline
**Symptom:** E1 validation fails
**Cause:** Earlier stage skipped or failed
**Solution:** Ensure all stages A-D ran successfully

### Issue 2: Metadata Inconsistency
**Symptom:** E1 detects conflicting metadata
**Cause:** Manual IR modification between stages
**Solution:** Don't manually modify IR, let passes handle it

---

## After Stage E

After E1 completes:
1. **IR is immutable** - No more transforms
2. **Verification runs** - `verify_tt_ir` checks TT constraints
3. **Codegen proceeds** - 5 artifacts generated

---

## References

- [v5_pipeline.md](../../architecture/v5_pipeline.md)
- [late_split.md](./late_split.md)

---

**Last Updated:** 2025-10-16
**Stage:** E (Finalization)
**Passes:** 1 (E1)
**Status:** Production
