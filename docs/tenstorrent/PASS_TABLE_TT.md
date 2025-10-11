# TileLang Tenstorrent Pass Reference

**Document Version:** 1.1
**Date:** 2025-10-11
**Status:** Active (Consolidation Plan tracking)

## Overview

This document captures the Tenstorrent-specific transformation pipeline, metadata inference, and code generation passes. Shared passes are documented in `PASS_TABLE_SHARED.md`, and GPU-only passes in `PASS_TABLE_GPU.md`.
It now mirrors the 2025-10-11 consolidation roadmap in `TT_BACKEND_CONSOLIDATION_PLAN.md`, updating pass statuses to match the phased rollout.

## Phase 2B: Tenstorrent-Specific Optimization

Applied only for Tenstorrent target via `OptimizeForTargetTT()`.

### Metadata Inference: Layout-Aware (New Pipeline)

> The canonical ordering and responsibilities are described in
> [TT_ARCHITECTURE.md](TT_ARCHITECTURE.md#layout-aware-metadata). This table tracks status at the pass level
> to avoid duplicating the full pipeline documentation.

| Pass | Status | Category | Input IR | Output IR | Purpose | Documentation |
|------|--------|----------|----------|-----------|---------|---------------|
| **InferTTLayout** | 🟡 Python impl; C++ port pending (Phase 2) | Memory | PrimFunc (`tt.user_layout`) | PrimFunc + `tt.buffer.*` | Stamp buffer layout metadata (alignments + N-D shard projection) | [📄 Doc](./passes/infer_layout_tt.md) |
| **PropagateTTLayout** | 🟡 Python impl; C++ port pending (Phase 2) | Memory | PrimFunc + `tt.buffer.*` | PrimFunc + `tt.cb.*` | Derive circular buffer metadata consumed by codegen | [📄 Doc](./passes/propagate_layout_tt.md) |
| **LayoutAwareWorkPartitionTT** | 🟡 Python driver; C++ port pending (Phase 2) | Device | PrimFunc + buffer metadata | PrimFunc + partition attrs | Emit `tt.partition_mode`, runtime arg schema, core ranges | [📄 Doc](./passes/layout_aware_partition_tt.md) |

**Annotations Added:**
_Current implementation remains in Python helpers; Phase 2 tracks the C++ port._
```json
"tt.buffer.A": {
  "memory": "DRAM",
  "layout": "sharded",
  "tile_shape": [32, 32],
  "dtype": "bf16",
  "nd_shard": {
    "axes": ["B", "H", "M", "N"],
    "grid": [2, 4, 1, 1],
    "shard_shape_elems": [B//2, H//4, M, N],
    "projected_grid": [Gy, Gx],
    "projected_shard_tiles": [Sm, Sn]
  }
},
"tt.cb.A": {
  "page_size": 2048,
  "depth": 2,
  "data_format": "BFloat16_b"
},
"tt.partition_mode": "local_shard",
"tt.grid_tiles": [Mt, Nt],
"tt.local_shape_tiles": [Sm, Sn],
"tt.shard_grid": [Gy, Gx],
"tt.runtime_arg_names": [
  "tt_start_tile",
  "tt_tile_count",
  "Mt",
  "Kt",
  "Nt",
  "Sm",
  "Sn",
  "Gy",
  "Gx",
  "tt_shard_coord_y",
  "tt_shard_coord_x"
],
"tt.runtime_constants": {
  "Mt": Mt,
  "Nt": Nt,
  "Kt": 1,
  "Sm": Sm,
  "Sn": Sn,
  "Gy": Gy,
  "Gx": Gx
},
"tt_runtime_args": {
  "start_tile": {"name": "tt_start_tile", "dtype": "int32", "semantic": "tile_start"},
  "tile_count": {"name": "tt_tile_count", "dtype": "int32", "semantic": "tile_count"},
  "grid_shape": [grid_y, grid_x, grid_z],
  "grid_tiles": [Mt, Nt],
  "local_shape_tiles": [Sm, Sn],
  "shard_grid": [Gy, Gx],
  "runtime_constants": {"Mt": Mt, "Nt": Nt, "Kt": 1, "Sm": Sm, "Sn": Sn, "Gy": Gy, "Gx": Gx},
  "param_order": ["tt_start_tile", "tt_tile_count", "Mt", "Kt", "Nt", "Sm", "Sn", "Gy", "Gx", "tt_shard_coord_y", "tt_shard_coord_x"],
  "arg_names": ["tt_start_tile", "tt_tile_count", "Mt", "Kt", "Nt", "Sm", "Sn", "Gy", "Gx", "tt_shard_coord_y", "tt_shard_coord_x"],
  "partition_mode": "local_shard"
},
"tt.core_ranges": [[x, y, x, y, start, count], "..."],
"tt_core_runtime_args": [[start, count, Mt, 1, Nt, Sm, Sn, Gy, Gx, shard_sy, shard_sx], "..."]
```

Legacy schedule/shard passes remain for compatibility:

| Pass | Status | Category | Purpose |
|------|--------|----------|---------|
| **infer_default_tt_schedule** | 🟡 Legacy (removal tracked in Phase 2) | Device | Seed default per-core ranges when no annotations are provided. |
| **infer_default_tt_shard** | 🟡 Legacy (removal tracked in Phase 2) | Memory | Provide DRAM layout descriptors until layout-aware pipeline lands. |

### Transform Pipeline: TIR Transformations

| Pass | Status | Category | Input IR | Output IR | Purpose | Documentation |
|------|--------|----------|----------|-----------|---------|---------------|
| **grid_to_persistent_tt** | 🟡 Consumes new runtime args; diagnostics refresh queued | Device | Persistent kernel metadata | Persistent loop + runtime metadata | Consumes layout-aware attributes (global + local shard); additional halo/L1 diagnostics tracked separately | [📄 Doc](./passes/grid_to_persistent_tt.md) |
| **tt_tiles_to_core_map** | 🟡 Legacy (removal tracked in Phase 2) | Device | Tile assignments | Core (x, y) coords | Compatibility path when layout-aware metadata is unavailable | [📄 Doc](./passes/tt_tiles_to_core_map.md) |
| **memory_space_lower_tt** | 🟡 Heuristic CB sizing; Phase 2 rework | Memory | DRAM buffers | L1 circular buffers | Lower DRAM → L1 CB (consumes `tt.cb.*`) | [📄 Doc](./passes/memory_space_lower_tt.md) |
| **tile_pad_tt** | ✅ Complete | Memory | Arbitrary shapes | Tile-aligned shapes | Pad to 32×32 tiles | [📄 Doc](./passes/tile_pad_tt.md) |
| **tensorize_tt** | 🟡 Phase 1 focus: `T.gemm` path | Device | Loops | Loops + TT intrinsic evaluate nodes | Detect matmul regions, rewrite to `tt.*` intrinsics, attach metadata | [📄 Doc](./passes/tensorize_tt.md) |
| **rasterization_tt** | ⚠️ Planned | Optimization | Tile iteration | Optimized tile order | Remap tile iteration order | [📄 Spec](#rasterization_tt-specification) |
| **tt_multicast_reuse** | ⚠️ Planned | Optimization | NOC ops | NOC + multicast | Insert multicast for reuse | [📄 Spec](#tt_multicast_reuse-specification) |
| **verify_tt_ir** | 🟡 Needs `T.gemm` schema update | Verification | TT IR | Verified TT IR | Verify TT constraints | [📄 Doc](./passes/verify_tt_ir.md) |

**Example Transform (grid_to_persistent_tt):**

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
  start, count, Mt, Nt = get_runtime_args(core_id)[:4]
  mode = get_partition_mode()

  if mode == "global":
    for i in range(count):
      tid = start + i
      m = tid // Nt
      n = tid % Nt
      compute_tile(m, n)
  else:
    Sm, Sn, Gy, Gx, sy, sx = get_runtime_args(core_id)[4:]
    for i in range(count):
      tid = start + i
      m_local = tid // Sn
      n_local = tid % Sn
      m = sy * Sm + m_local
      n = sx * Sn + n_local
      compute_tile(m, n)
```

**Example Transform (tensorize_tt):**

```python
# Before
for kt in T.serial(Kt):
  for i, j in T.Parallel(32, 32):
    C[m, n] += A[m, kt] * B[kt, n]

# After (intrinsics injected by TensorizeTT)
tt.tile_regs_acquire()
tt.mm_init(cb_in0, cb_in1, cb_out)
for kt in T.serial(Kt):
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

See [PASS_TABLE_SHARED.md](./PASS_TABLE_SHARED.md) for the optimization passes that run identically on GPU and Tenstorrent targets. They execute unchanged before and after the Tenstorrent-specific passes described above.

**Total:** 7 active TT-specific (+3 legacy compatibility) + 11 shared passes

**Input:** Legalized TIR from Phase 1 + TT defaults

**Output:** TT-ready IR with persistent loops, CB allocations, intrinsic annotations

**Key Transforms:**
- Layout annotations → Canonical `tt.buffer.*` schema (memory, layout, ND sharding).
- DRAM↔L1 copies → `tt.cb.*` circular buffer metadata.
- Buffer residency → Partition mode (`global` vs `local_shard`) and runtime args.
- Grid kernel → Persistent kernel with shard-aware `(m, n)` recovery.
- DRAM buffers → L1 circular buffers.
- Manual matmul loops → Rewritten to `tt.*` intrinsic sequences and tracked via `tt_matmul_patterns`.
- Buffers padded to 32×32 tile boundaries.

---

## Phase 3: Tenstorrent Codegen

### TT Device Codegen

| Component | Input IR | Output Artifacts | Purpose | File |
|-----------|----------|------------------|---------|------|
| **TTReaderCodegenVisitor** | Annotated TIR | `reader.cpp` | Generate reader kernel | `src/target/tt/codegen_tt_reader_visitor.cc` |
| **TTComputeCodegenVisitor** | Annotated TIR | `compute.cpp` | Generate compute kernel | `src/target/tt/codegen_tt_compute_visitor.cc` |
| **TTWriterCodegenVisitor** | Annotated TIR | `writer.cpp` | Generate writer kernel | `src/target/tt/codegen_tt_writer_visitor.cc` |
| **EmitTTHostProgram** | Annotated TIR | `main.cpp` | Generate host program | `src/target/tt/codegen_tt.cc` |
| **EmitTTPlanJSON** | Schedule metadata | `tt.plan.json` | Generate execution plan | `src/target/tt/codegen_tt.cc` |

**Output:** 5 TT artifacts (reader, compute, writer, host, plan)

**Note:** TT codegen is **IR-driven** - visitors walk the annotated TIR and emit code based on annotations (not templates).

---

## Tenstorrent Pipeline Dependencies

```
Phase 0 (TT Defaults)
  ↓
apply_tt_defaults (Target Registration)
  ↓
Phase 1 (Shared)
  ↓
LowerAndLegalize (12 passes)
  ↓
Phase 2B (TT-Specific)
  ↓
Metadata Inference: Schedule/Shard Inference (2 passes)
  ↓
Transform Pipeline: TIR Transformations (4 passes)
  ↓
Transform Pipeline: Tensorization (1 pass) ⭐
  ↓
Common optimizations (11 passes)
  ↓
Verification (1 pass)
  ↓
Phase 3 (Codegen)
  ↓
3 Kernel Visitors + Host + Plan
  ↓
5 TT artifacts (.cpp + .json)
```

---

## Current TT Gaps

| Gap | Current Behavior | Expected Behavior | Fix |
|-----|------------------|-------------------|-----|
| **K-loop detection** | Codegen heuristics (variable name) | Transform pass annotation | Extend `tensorize_tt.cc` |
| **Intrinsic emission** | Raw array operations emitted | Metalium intrinsics emitted | Update compute visitor |
| **Element-wise ops** | Manual pattern in codegen | Transform pass annotation | Extend `tensorize_tt.cc` |
| **T.gemm() support** | Layout inference fails | Full T.gemm() support | Implement layout inference for TT |

See [IR Lowering Tasks](./IR_LOWERING_TASKS.md) for implementation plan.

---

## Quick Reference

**TT-Specific Passes:** 7 active (plus 3 legacy compatibility passes)
- Layout-aware metadata: `infer_layout_tt`, `propagate_layout_tt`, `layout_aware_partition_tt`
- Transform pipeline: `grid_to_persistent_tt`, `memory_space_lower_tt`, `tile_pad_tt`, `tensorize_tt`
- Verification: `verify_tt_ir`

**Key TT Files:**
- `tilelang/engine/tt/lower.py` (Tenstorrent orchestration)
- `src/transform/tt/*.cc` (Tenstorrent transforms)
- `src/target/tt/codegen_tt*.cc` (reader/compute/writer/host visitors)

## Planned Pass Specifications

### rasterization_tt Specification

**Status:** ⚠️ Planned (P1 - Performance Optimization)

**Purpose:** Optimize tile iteration order for better cache locality, NOC traffic reduction, and multicast efficiency.

**Input IR:**
```python
# Row-major tile iteration (default)
for tile_id in range(start_id, start_id + count):
    by = tile_id // Nt  # Row index
    bx = tile_id % Nt   # Column index
    # Process tile (by, bx)
```

**Output IR:**
```python
# Optimized iteration order (e.g., block-linear, Z-order)
for tile_id in range(start_id, start_id + count):
    # Block-linear rasterization for better locality
    block_y = tile_id // (BLOCK_H * Nt / BLOCK_W)
    block_x = (tile_id % (BLOCK_H * Nt / BLOCK_W)) // BLOCK_H
    local_y = (tile_id % BLOCK_H)
    by = block_y * BLOCK_H + local_y
    bx = block_x * BLOCK_W + (tile_id % BLOCK_W)
    # Process tile (by, bx)
```

**Supported Rasterization Policies:**
- `row_major`: Sequential tiles in row-major order (default)
- `column_major`: Sequential tiles in column-major order
- `block_linear`: Tiles grouped in rectangular blocks for locality
- `z_order`: Morton/Z-order curve for 2D locality
- `hilbert`: Hilbert curve (better locality than Z-order)

**Metadata Required:**
- `tt.schedule.order`: Current iteration order
- `tt.schedule.rect`: Rectangular block dimensions for block-linear
- `tt.grid_x`, `tt.grid_y`: Grid dimensions

**Benefits:**
- **Locality:** Reduce DRAM/L1 traffic by reusing nearby tiles
- **NOC Efficiency:** Group cores with similar access patterns
- **Multicast Setup:** Enable multicast by creating core groups with shared data

**Implementation Location:** `src/transform/tt/rasterization_tt.cc` (to be created)

**Related Passes:**
- Runs after `grid_to_persistent_tt` (modifies tile ID → (bx, by) mapping)
- Before `tt_multicast_reuse` (sets up core groups for multicast)

---

### tt_multicast_reuse Specification

**Status:** ⚠️ Planned (P1 - Performance Optimization)

**Purpose:** Insert NOC multicast operations to reduce DRAM bandwidth when multiple cores need the same data.

**Use Cases:**
1. **GEMM:** Row tiles of A reused across columns, column tiles of B reused across rows
2. **FlashAttention:** Q tiles broadcast to multiple cores processing KV chunks
3. **Convolution:** Filter weights broadcast to multiple spatial positions

**Input IR:**
```cpp
// Reader kernel (all cores read same A tile independently)
for (uint32_t out_tile = 0; out_tile < count; ++out_tile) {
    uint32_t tile_a_idx = ...;
    noc_async_read(tile_a_idx, dram_addr_a, get_write_ptr(cb::c_in0));
    // Each core issues separate DRAM read
}
```

**Output IR:**
```cpp
// Reader kernel (sender core multicasts to receivers)
CoreRange sender_range = get_sender_range();
CoreRangeSet receiver_ranges = get_receiver_ranges();

if (is_sender_core()) {
    // Sender: Read once, multicast to multiple cores
    noc_async_read(tile_a_idx, dram_addr_a, get_write_ptr(cb::c_in0));
    noc_async_write_multicast(get_write_ptr(cb::c_in0),
                               receiver_ranges,  // Multiple destinations
                               cb_addr_remote,
                               tile_size);
} else {
    // Receiver: Wait for multicast data
    cb_wait_front(cb::c_in0, 1);  // Data arrives via NOC
}
```

**Analysis Required:**
1. **Reuse Detection:**
   - Analyze buffer access patterns across cores
   - Identify tiles accessed by multiple cores in same iteration
   - Compute reuse factor (how many cores share the tile)

2. **Core Grouping:**
   - Partition cores into sender/receiver groups
   - Minimize DRAM reads (one read per sender group)
   - Balance NOC traffic (avoid hotspots)

3. **Synchronization:**
   - Insert barriers for sender/receiver coordination
   - Ensure CB availability before multicast
   - Handle dependencies (sender must read before multicast)

**Metadata Required:**
- `tt.core_ranges`: CoreRangeSet for kernel execution
- `tt.schedule.assignments`: Per-core tile assignments
- `tt.shard`: Buffer sharding/layout metadata

**Metadata Generated:**
- `tt.multicast_groups`: List of (sender_core, receiver_cores, tile_indices)
- `tt.sync_barriers`: Barrier points for sender/receiver coordination

**Benefits:**
- **Bandwidth Reduction:** 1 DRAM read instead of N (reuse factor N)
- **Latency Hiding:** Multicast overlaps with compute on other cores
- **Scalability:** Enables scaling to more cores without bandwidth saturation

**Example Savings (GEMM 8×8 grid, 64 cores):**
- Without multicast: 64 cores × 8 A tiles = 512 DRAM reads for row
- With multicast: 8 senders × 1 read = 8 DRAM reads (64× reduction)

**Implementation Location:** `src/transform/tt/tt_multicast_reuse.cc` (to be created)

**Related Passes:**
- Runs after `rasterization_tt` (needs optimized core grouping)
- Before `verify_tt_ir` (verification checks multicast validity)

**References:**
- [TT-Metalium Multicast APIs](https://docs.tenstorrent.com/tt-metalium/latest/tt_metal/programming_guide/data_movement.html#multicast)
- `noc_async_write_multicast()`: Core → CoreRangeSet broadcast
- `noc_semaphore_*()`: Synchronization for sender/receiver coordination

---

**References:**
- [IR Lowering Analysis](./IR_LOWERING_ANALYSIS.md) - GPU vs TT architecture comparison
- [IR Lowering Tasks](./IR_LOWERING_TASKS.md) - Implementation tasks for TT gaps
- [TT Architecture](./TT_ARCHITECTURE.md) - Complete TT backend architecture
