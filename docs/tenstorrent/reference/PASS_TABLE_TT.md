# TileLang Tenstorrent Pass Reference

**Document Version:** 2.0
**Date:** 2025-10-14
**Status:** Active

## Overview

This document describes the Tenstorrent-specific transformation pipeline. Shared passes are documented in `PASS_TABLE_SHARED.md`, and GPU-only passes in `PASS_TABLE_GPU.md`.

```
             ┌──────────────────────────────┐
             │   Lower & Legalize (shared) │
             │   PASS_TABLE_SHARED.md      │
             └─────────────┬────────────────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
┌──────────────────────┐      ┌────────────────────────────┐
│ GPU / HIP Pipeline   │      │ Tenstorrent Pipeline       │
│ (PASS_TABLE_GPU.md)  │      │ (PASS_TABLE_TT.md)         │
├──────────────────────┤      ├────────────────────────────┤
│ • InferFragment      │      │ • InferTTLayout            │
│ • Warp/thread passes │      │ • PropagateTTLayout        │
│ • SplitHostDevice    │      │ • TTTilesToCoreMap         │
│ • CUDA/HIP codegen   │      │ • LowerTTTileIntrinsics    │
└──────────────────────┘      │ • GridToPersistentTT       │
                               └────────────────────────────┘
```

## Phase 2B: Tenstorrent-Specific Optimization

Applied only for Tenstorrent target via `OptimizeForTargetTT()`.

### Metadata Inference Pipeline

| Pass | Category | Purpose | Metadata Added |
|------|----------|---------|----------------|
| **InferTTLayout** | Memory | Infer buffer layouts and metadata | `tt.layout_desc` with buffer layouts |
| **PropagateTTLayout** | Memory | Propagate and normalize layout info | Normalized layout descriptors |
| **TTTilesToCoreMap** | Device | Compute core mapping and partitioning | `tt.core_grid`, `tt.work_partition` |

### Transform Pipeline

| Pass | Category | Purpose | Transformation |
|------|----------|---------|----------------|
| **LowerTTTileIntrinsics** | Device | Lower tile ops to device intrinsics | `tl.gemm` → `tt.*` intrinsic sequences |
| **GridToPersistentTT** | Device | Final lowering to persistent kernels | Grid iteration → Persistent loops |

**Example Transform (GridToPersistentTT):**

```python
# Before
@T.prim_func
def kernel(...):
  with T.Kernel(8, 8) as (bx, by):  # 64 blocks
    # Compute for block (bx, by)
    C[bx*32:(bx+1)*32, by*32:(by+1)*32] = ...

# After
@T.prim_func
def kernel(...):
  core_id = get_core_id()
  # Persistent loop over assigned tiles
  for tile_id in assigned_tiles(core_id):
    compute_tile(tile_id)
```

**Example Transform (LowerTTTileIntrinsics):**

```python
# Before
T.evaluate(tl.tl_gemm(...))

# After (intrinsics injected)
tt.tile_regs_acquire()
tt.mm_init(cb_in0, cb_in1, cb_out)
tt.cb_wait_front(cb_in0, 1)
tt.cb_wait_front(cb_in1, 1)
tt.matmul_tiles(cb_in0, cb_in1, 0, 0, 0, 0)
tt.cb_pop_front(cb_in0, 1)
tt.cb_pop_front(cb_in1, 1)
tt.tile_regs_commit()
tt.tile_regs_wait()
tt.cb_reserve_back(cb_out, 1)
tt.pack_tile(0, cb_out)
tt.cb_push_back(cb_out, 1)
tt.tile_regs_release()
```

### Common Optimizations (Shared with GPU)

See [PASS_TABLE_SHARED.md](./PASS_TABLE_SHARED.md) for optimization passes that run identically on GPU and Tenstorrent targets.

---

## Phase 3: Tenstorrent Codegen

### TT Device Codegen

| Component | Output | Purpose |
|-----------|--------|---------|
| **TTReaderCodegenVisitor** | `reader.cpp` | Generate reader kernel |
| **TTComputeCodegenVisitor** | `compute.cpp` | Generate compute kernel |
| **TTWriterCodegenVisitor** | `writer.cpp` | Generate writer kernel |
| **EmitTTHostProgram** | `main.cpp` | Generate host program |
| **EmitTTPlanJSON** | `tt.plan.json` | Generate execution plan |

**Output:** 5 TT artifacts (reader, compute, writer, host, plan)

**Note:** TT codegen is **IR-driven** - visitors walk the annotated TIR and emit code based on metadata attributes.

---

## Pipeline Dependencies

```
Apply TT Defaults
  ↓
Lower & Legalize (shared passes)
  ↓
InferTTLayout
  ↓
PropagateTTLayout
  ↓
TTTilesToCoreMap
  ↓
LowerTTTileIntrinsics
  ↓
GridToPersistentTT
  ↓
Common optimizations (shared)
  ↓
Codegen (5 artifacts)
```

---

## Key Files

**Python Pipeline:**
- `tilelang/engine/tenstorrent/lower.py` - Orchestration
- `tilelang/tenstorrent/passes/*.py` - Pass implementations
- `tilelang/tenstorrent/attrs.py` - Metadata dataclasses

**Codegen:**
- `src/target/tenstorrent/codegen_tt*.cc` - IR-driven visitors

---

## References

- [Lowering Architecture](./LOWERING_ARCHITECTURE.md) - Metadata-driven architecture
- [TT Architecture](./TT_ARCHITECTURE.md) - Complete backend architecture
- [Runtime Plan](./RUNTIME_PLAN.md) - tt.plan.json specification