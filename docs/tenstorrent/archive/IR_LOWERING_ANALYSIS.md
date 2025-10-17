# IR Lowering Pipeline Analysis: GPU vs Tenstorrent

**Document Version:** 1.0
**Date:** 2025-10-08
**Status:** Complete

## Overview

This document provides an in-depth analysis of how TileLang lowers IR for GPU (CUDA/ROCm) targets versus Tenstorrent targets, with particular focus on **where pattern detection and tensorization occur**.

## Key Architectural Question

**Question:** Should matmul pattern detection and intrinsic emission happen in:
- **Transform passes** (earlier in pipeline), OR
- **Codegen** (later in pipeline)?

**Answer:** Transform passes ✅

**Rationale:** GPU backends (CUDA) use transform passes for pattern detection and annotation, keeping codegen "dumb" (just emit based on annotations). This separation of concerns makes the compiler more maintainable and allows IR optimizations to work on annotated patterns.

## GPU (CUDA/ROCm) Lowering Pipeline

### Phase 1: Frontend Lowering (Shared with TT)

**Location:** `tilelang/engine/phase.py` → `LowerAndLegalize()`

This phase is **backend-agnostic** and shared between CUDA and TT:

```python
def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
    mod = tir.transform.BindTarget(target)(mod)

    # Inline let expressions
    mod = tilelang.transform.LetInline()(mod)

    # Add wrapper for single buf store
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)

    # Inject assumes to speedup TVM prover
    mod = tilelang.transform.InjectAssumes()(mod)

    # Simplify IR expressions
    mod = tir.transform.Simplify()(mod)

    # Set layouts for reducers
    mod = tilelang.transform.LayoutReducer()(mod)

    # ⭐ CRITICAL: Layout inference for fragments and shared memory
    # This detects tensor operations and infers memory layouts
    mod = tilelang.transform.LayoutInference()(mod)

    # Lower high-level tile operations to low-level operations
    mod = tilelang.transform.LowerTileOp()(mod)

    # Lower L2 persistent map
    mod = tilelang.transform.LowerL2Persistent()(mod)

    # Legalize vectorized loops
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)

    # Add safety checks for memory accesses
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)

    # Simplify again
    mod = tilelang.transform.Simplify()(mod)

    # Try to vectorize loop with dynamic shape
    mod = tilelang.transform.LoopVectorizeDynamic()(mod)

    return mod
```

**Key Passes:**
- **LayoutInference**: Infers memory layouts for fragments (TensorCore tiles) and shared memory
- **LowerTileOp**: Lowers high-level tile operations to loops and buffer operations
- **LayoutReducer**: Configures layouts for reduction operations

### Phase 2: Target-Specific Optimization (CUDA)

**Location:** `tilelang/engine/phase.py` → `OptimizeForTarget()`

This phase has **CUDA-specific branches**:

```python
def OptimizeForTarget(mod: IRModule, target: Target) -> IRModule:
    # Lower shared memory barriers
    mod = tilelang.transform.LowerSharedBarrier()(mod)

    # Lower shared.tmem
    mod = tilelang.transform.LowerSharedTmem()(mod)

    # Hopper (H100) path with TMA and warp specialization
    if allow_tma_and_warp_specialized(pass_ctx, target):
        mod = tilelang.transform.IfStmtBinding()(mod)
        mod = tilelang.transform.MultiVersionBuffer()(mod)

        # ⭐ Warp specialization for async operations
        mod = tilelang.transform.WarpSpecialized()(mod)

        mod = tilelang.transform.InjectTmaBarrier()(mod)
        mod = tilelang.transform.AnnotateWarpGroupRegAlloc()(mod)
        mod = tilelang.transform.PipelinePlanning()(mod)
        mod = tilelang.transform.InjectSoftwarePipeline()(mod)
        mod = tilelang.transform.LowerOpaqueBlock()(mod)
        mod = tilelang.transform.MergeIfStmt()(mod)

        if is_hopper(target):
            # ⭐ Rewrite wgmma operations for Hopper architecture
            mod = tilelang.transform.RewriteWgmmaSync()(mod)

        mod = tilelang.transform.InjectFenceProxy()(mod)
    else:
        # Non-Hopper path
        mod = tilelang.transform.IfStmtBinding()(mod)
        mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        mod = tilelang.transform.PipelinePlanning()(mod)
        mod = tilelang.transform.InjectSoftwarePipeline()(mod)
        mod = tilelang.transform.MergeIfStmt()(mod)

        if allow_fence_proxy(target):
            mod = tilelang.transform.InjectFenceProxy()(mod)

    # Common optimizations
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=True)(mod)
    mod = tilelang.transform.StorageRewrite()(mod)
    mod = tir.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.RemoveNoOp()(mod)
    mod = tir.transform.RewriteUnsafeSelect()(mod)
    mod = tir.transform.HoistIfThenElse()(mod)

    mod = tir.transform.VerifyMemory()(mod)
    mod = tir.transform.AnnotateEntryFunc()(mod)

    # ⭐ CRITICAL: TensorCore fragment inference
    # This is where GPU detects TensorCore operations and annotates fragments
    mod = tir.transform.InferFragment()(mod)

    mod = tilelang.transform.LowerThreadAllreduce()(mod)
    mod = tilelang.transform.LowerHopperIntrin()(mod)

    # Thread synchronization
    if allow_global_thread_synchronization():
        mod = tilelang.transform.ThreadSync("global")(mod)

    mod = tilelang.transform.AnnotateDeviceRegions()(mod)

    # ⭐ Split into host and device functions
    mod = tir.transform.SplitHostDevice()(mod)

    # Merge shared memory allocations
    mod = tilelang.transform.MergeSharedMemoryAllocations(...)(mod)

    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    mod = tilelang.transform.InjectPTXAsyncCopy()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    mod = tilelang.transform.PersistThreadblock()(mod)

    return mod
```

**Key GPU-Specific Passes:**
- **InferFragment**: Detects TensorCore operations (wmma, mma) and annotates fragment metadata
- **WarpSpecialized**: Enables async warp specialization for Hopper
- **RewriteWgmmaSync**: Rewrites wgmma operations for Hopper architecture
- **SplitHostDevice**: Splits IR into host and device functions

### Phase 3: TensorCore Pattern Detection (CUDA)

**Location:** `3rdparty/tvm/src/tir/transforms/tensorcore_infer_fragment.cc`

**How GPU Detects TensorCore Operations:**

```cpp
// GPU approach: Look for intrinsic calls in the IR
class FragmentGetter : public StmtExprVisitor {
  void VisitExpr_(const CallNode* op) final {
    // Detect TensorCore load/store intrinsics
    if (op->op.same_as(builtin::tvm_load_matrix_sync()) ||
        op->op.same_as(builtin::tvm_store_matrix_sync())) {
      // Extract shape: m, n, k
      const IntImmNode* m = op->args[1].as<IntImmNode>();
      const IntImmNode* n = op->args[2].as<IntImmNode>();
      const IntImmNode* k = op->args[3].as<IntImmNode>();
      const StringImmNode* layout = op->args[7].as<StringImmNode>();

      // Get memory scope (wmma.matrix_a, wmma.matrix_b, wmma.accumulator)
      std::string scope = GetPtrStorageScope(buffer_var);

      // Store fragment metadata
      FragmentInfo info(m, n, k, layout, scope);
      fragments[buffer_var] = info;
    }

    // Detect fill intrinsic
    else if (op->op.same_as(builtin::tvm_fill_fragment())) {
      // Similar extraction and storage
    }
  }
};

// Annotate allocations with fragment metadata
class InferFragmenter : public StmtMutator {
  Stmt VisitStmt_(const AllocateNode* op) final {
    const VarNode* buffer = op->buffer_var.get();
    if (fragment_getter.fragments.count(buffer)) {
      FragmentInfo info = fragment_getter.fragments.at(buffer);

      // Add attribute: fragment_shape = "16, 16, 16"
      PrimExpr shape_expr = StringImm(
        std::to_string(info.m) + ", " +
        std::to_string(info.n) + ", " +
        std::to_string(info.k)
      );
      Stmt shape_attr = AttrStmt(
        op->buffer_var,
        attr::fragment_shape,
        shape_expr,
        stmt
      );

      // Add attribute: fragment_layout = "row_major"
      if (info.layout != "") {
        Stmt layout_attr = AttrStmt(
          op->buffer_var,
          attr::fragment_layout,
          StringImm(info.layout),
          shape_attr
        );
        return layout_attr;
      }
      return shape_attr;
    }
    return stmt;
  }
};
```

**Key Insight:** GPU uses **intrinsic calls** inserted during frontend lowering (`T.gemm()` → `tvm_load_matrix_sync`, `tvm_mma_sync`, etc.) to detect patterns. The transform pass annotates these with metadata.

### Phase 4: Device Codegen (CUDA)

**Location:** `tilelang/engine/lower.py` → `device_codegen()`

```python
def device_codegen(device_mod: tvm.IRModule, target: Target):
    # Lower device storage access info
    device_mod = tilelang.transform.LowerDeviceStorageAccessInfo()(device_mod)

    # ⭐ Lower intrinsics to target-specific code
    device_mod = tir.transform.LowerIntrin()(device_mod)

    device_mod = tir.transform.Simplify()(device_mod)

    # Call target-specific codegen backend
    if target.kind.name == "cuda":
        device_mod = tvm.ffi.get_global_func("target.build.tilelang_cuda")(
            device_mod, target
        )
    elif target.kind.name == "hip":
        device_mod = tvm.ffi.get_global_func("target.build.tilelang_hip")(
            device_mod, target
        )

    return device_mod
```

**Key Pass:**
- **LowerIntrin**: Lowers TVM intrinsics (like `tvm_mma_sync`) to target-specific code (PTX asm for CUDA)

**CUDA Codegen:** The actual code generation reads the fragment annotations and emits PTX assembly:

```cpp
// Pseudo-code for CUDA codegen
if (call->op == "tvm_mma_sync") {
  // Read fragment_shape attribute from annotated buffers
  auto shape_attr = GetAttr(buffer_d, "fragment_shape");
  auto [m, n, k] = ParseShape(shape_attr);

  // Read fragment_layout attribute
  auto layout = GetAttr(buffer_a, "fragment_layout");

  // Emit PTX assembly for TensorCore
  stream << "mma.sync.aligned.m" << m << "n" << n << "k" << k
         << ".row.col.f16.f16.f16.f16 "
         << "{" << d_regs << "}, "
         << "{" << a_regs << "}, "
         << "{" << b_regs << "}, "
         << "{" << c_regs << "};\n";
}
```

## Tenstorrent Lowering Pipeline

### Phase 1: Apply TT Defaults (TT-Specific)

**Location:** `tilelang/engine/tenstorrent/lower.py` → `apply_tt_defaults()`

```python
# Apply default TT annotations (Target Registration)
# - Schedule: policy="contiguous", order="row_major"
# - Layout: DRAM interleaved, 32×32 tiles
mod = apply_tt_defaults(mod)
```

### Phase 2: Frontend Lowering (Shared with CUDA)

**Location:** `tilelang/engine/tenstorrent/lower.py` → `LowerAndLegalizeTT()`

```python
# Calls the same LowerAndLegalize() as CUDA
with target:
    mod = LowerAndLegalize(mod, target)
```

**Same 15 passes as CUDA** (see Phase 1 above).

### Phase 3: TT-Specific Optimizations

**Location:** `tilelang/engine/tenstorrent/lower.py` → `OptimizeForTargetTT()`

```python
def OptimizeForTargetTT(mod: IRModule, target: Target) -> IRModule:
    # === Metadata Inference ===
    mod = infer_default_tt_schedule(mod)          # Compatibility defaults
    mod = infer_default_tt_shard(mod)             # Compatibility defaults
    mod = apply_layout_aware_metadata_passes(mod) # Authoritative metadata

    # === Transform Pipeline (TT specific) ===
    mod = grid_to_persistent_tt(mod)
    mod = tt_tiles_to_core_map(mod)
    mod = memory_space_lower_tt(mod)
    mod = tile_pad_tt(mod)

    # ⭐ CRITICAL: Tensorization (pattern detection)
    mod = lower_gemm_to_tt_intrinsics(mod)

    # === Common Optimizations (Shared with CUDA) ===
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(...)(mod)
    mod = tilelang.transform.StorageRewrite()(mod)
    mod = tvm.tir.transform.UnrollLoop()(mod)
    mod = tvm.tir.transform.RenormalizeSplitPattern()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    mod = tvm.tir.transform.RemoveNoOp()(mod)
    mod = tvm.tir.transform.RewriteUnsafeSelect()(mod)
    mod = tvm.tir.transform.HoistIfThenElse()(mod)
    mod = tvm.tir.transform.VerifyMemory()(mod)

    # TT-specific verification
    mod = verify_tt_ir(mod)

    return mod
```

**TT-Specific Passes:**
- **infer_default_tt_schedule** / **infer_default_tt_shard** *(legacy defaults)*: Seed metadata when annotations are missing.
- **apply_layout_aware_metadata_passes**: Runs `InferTTLayout`, `PropagateTTLayout`, and `LayoutAwareWorkPartitionTT` (see [TT_ARCHITECTURE.md](TT_ARCHITECTURE.md#layout-aware-metadata) for details).
- **grid_to_persistent_tt**: Transform GPU grid to TT persistent loops using the emitted partition mode.
- **tt_tiles_to_core_map** *(legacy fallback)*: Provide core assignments when layout-aware metadata is absent.
- **memory_space_lower_tt**: Lower DRAM allocations to L1 circular buffers.
- **tile_pad_tt**: Pad buffers to 32×32 tile boundaries.
- **LowerGemmToTTIntrinsics**: ⭐ Map frontend `tl.gemm` intrinsics to TT intrinsic sequences (manual loop matcher removed; element-wise TBD).
- **verify_tt_ir**: Verify TT constraints (grid size, CB counts, runtime args).

### Phase 4: LowerGemmToTTIntrinsics Pass

**Location:** `src/transform/tenstorrent/lower_gemm_to_tt_intrinsics.cc`

**Current State:**
- Consumes frontend-issued `tl.gemm` intrinsics (mirroring the CUDA pipeline)
  and drops the bespoke loop-pattern matcher.
- Resolves circular-buffer IDs from `tt_circular_buffers` metadata and records
  the basic bookkeeping (`cb_in0`, `cb_in1`, `cb_out`, signature string) in
  `tt_matmul_patterns`, falling back to the canonical `c0/c1/c16` mapping when
  metadata is absent.
- Expands each `tl.gemm` evaluate node into the TT intrinsic sequence
  (`tt.tile_regs_acquire`, `tt.mm_init`, `tt.cb_wait_front`, `tt.matmul_tiles`,
  `tt.cb_pop_front`, `tt.tile_regs_commit`, `tt.cb_reserve_back`,
  `tt.pack_tile`, `tt.cb_push_back`, `tt.tile_regs_release`).
- Still TODO: propagate richer metadata (e.g., loop/reduction context), handle
  element-wise tensorization, and tighten diagnostics when CB attribution
  fails.

### Phase 5: Device Splitting (TT 3-Kernel Architecture)

**Location:** `tilelang/engine/tenstorrent/lower.py` → `SplitTTKernels()`

```python
def SplitTTKernels(mod: IRModule) -> Tuple[IRModule, IRModule]:
    # Annotate device regions
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)

    # ⭐ Do NOT call SplitHostDevice!
    # TT's 3-kernel split happens during codegen, not IR transformation

    return mod, mod  # Return same module for both
```

**Key Difference:** Unlike CUDA which splits IR into host/device functions, TT keeps IR intact and splits during codegen.

### Phase 6: Codegen (IR-Driven Visitors)

**Location:** `src/target/tenstorrent/codegen_tt_*.cc`

**Current Implementation:**

- `LowerGemmToTTIntrinsics` lowers frontend `tl.gemm` calls into explicit TT intrinsic sequences before codegen, emitting `tt.tile_regs_acquire`, `tt.mm_init`, `tt.matmul_tiles`, `tt.cb_wait_front`, `tt.cb_pop_front`, `tt.cb_reserve_back`, `tt.pack_tile`, and `tt.cb_push_back` sequences (`src/transform/tenstorrent/lower_gemm_to_tt_intrinsics.cc`).
- `TTComputeCodegenVisitor` walks the existing IR without heuristics. It prints the surrounding loop structure and the `Evaluate(tt.*)` nodes left behind by tensorization (`src/target/tenstorrent/codegen_tt_compute_visitor.cc`).

```cpp
void TTComputeCodegenVisitor::VisitStmt_(const ForNode* op) {
  std::string loop_var = GetVarName(op->loop_var);
  std::string min_expr = EmitExpr(op->min);
  std::string extent_expr = EmitExpr(op->extent);
  EmitLine("for (uint32_t " + loop_var + " = " + min_expr + "; " +
           loop_var + " < " + min_expr + " + " + extent_expr + "; ++" +
           loop_var + ") {");
  IncIndent();
  VisitStmt(op->body);
  DecIndent();
  EmitLine("}");
}

void TTComputeCodegenVisitor::VisitStmt_(const AttrStmtNode* op) {
  std::string attr_key = std::string(op->attr_key);
  if (attr_key.rfind("tt.", 0) == 0) {
    VisitStmt(op->body);
    return;
  }
  TTCodegenVisitor::VisitStmt_(op);
}
```

- Evaluate nodes fall through to `TTCodegenVisitor::VisitStmt_(const EvaluateNode* op)`, which renders the intrinsic call after stripping the `tt.` prefix inside `EmitExpr`. The generated C++ therefore already contains `mm_init`, `matmul_tiles`, `cb_wait_front`, etc.
- Remaining TODOs: `LowerGemmToTTIntrinsics` still guards against multiple matmuls in the same reduction loop, and element-wise tensorization plus tilize/untilize lowering remain unimplemented.

## Comparison Summary

| Aspect | GPU (CUDA) | Tenstorrent (Current) |
|--------|------------|-----------------------|
| **Pattern Detection** | Transform pass (`InferFragment`) finds TensorCore regions | Transform pass (`LowerGemmToTTIntrinsics`) rewrites `tl.gemm` calls into TT intrinsics |
| **Annotation Method** | AttrStmt metadata on fragment buffers (`fragment_shape`, `fragment_layout`) | Evaluate nodes with `tt.*` intrinsics + `tt_matmul_patterns`/`tt_num_matmuls` metadata |
| **Intrinsic Insertion** | Frontend lowering inserts `tvm_load_matrix_sync`/`tvm_mma_sync` | `LowerGemmToTTIntrinsics` emits `tt.tile_regs_*`, `tt.mm_init`, `tt.matmul_tiles`, wait/pop intrinsics |
| **Codegen Role** | Emit CUDA/PTX from annotated IR (no pattern detection) | Emit Metalium intrinsics directly from intrinsic-bearing IR (no heuristics) |
| **IR Representation** | Intrinsic calls guarded by fragment annotations | Intrinsic calls (`Evaluate(tt.*)`) surrounded by TT runtime metadata |
| **Device Splitting** | `SplitHostDevice` separates host/device functions | 3-visitor split happens during codegen (`TTReader/Compute/WriterCodegenVisitor`) |

## Recommended Architecture Changes

- Extend `LowerGemmToTTIntrinsics` to support multiple matmuls per reduction loop, element-wise tensorization, and richer diagnostics when CB metadata is missing or ambiguous.
- Add regression coverage ensuring the TT codegen visitors continue to treat `Evaluate(tt.*)` nodes as intrinsics as new operations are added.

---

## Execution Model Comparison

### GPU (CUDA/HIP): Massive Parallelism with Transient Kernels

**Grid-based model:**
```python
# GPU kernel launch
grid_dim = (Mt, Nt)  # Grid of thread blocks
block_dim = (32, 32)  # Threads per block

# Each block processes one output tile independently
@cuda.kernel
def matmul_gpu(A, B, C):
    bx, by = blockIdx.x, blockIdx.y  # Block coordinates
    tx, ty = threadIdx.x, threadIdx.y  # Thread coordinates

    # Each block computes C[by, bx] = A[by, :] @ B[:, bx]
    # Block exits after computing its tile
```

**Key characteristics:**
- **Transient execution**: Each block processes its assigned tile and exits
- **Hardware scheduling**: GPU hardware schedules blocks to SMs dynamically
- **No persistent state**: Blocks don't communicate or persist across kernel launches
- **Occupancy-driven**: Performance depends on maximizing SM occupancy

### Tenstorrent: Persistent Per-Core Loops

**Persistent kernel model:**
```cpp
// Tenstorrent persistent kernel
void MAIN() {
    // Runtime args provided by host
    uint32_t start_id = get_arg_val<uint32_t>(0);
    uint32_t count    = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Nt = get_arg_val<uint32_t>(3);

    // Core runs a persistent loop over assigned tiles
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t tile_id = start_id + i;
        uint32_t by = tile_id / Nt;  // Row index (row-major)
        uint32_t bx = tile_id % Nt;  // Column index

        // Process tile (by, bx)
        // Core remains active, processing multiple tiles
    }
}
```

**Key characteristics:**
- **Persistent execution**: Each Tensix core runs a long-lived kernel loop
- **Static partitioning**: Host assigns tile ranges to cores before launch
- **Per-core state**: Cores can maintain state across tile iterations
- **Explicit scheduling**: Developer controls tile-to-core mapping

---

## Memory Hierarchy Comparison

### GPU Memory Hierarchy

```
Global Memory (DRAM)   →  Gigabytes, ~500 GB/s, high latency
    ↕
L2 Cache               →  Megabytes, shared across SMs
    ↕
L1 Cache/Shared Memory →  Kilobytes per SM, low latency
    ↕
Registers              →  Thousands per thread, ultra-low latency
```

**Access patterns:**
- **Coalescing**: Threads in a warp access consecutive addresses
- **Shared memory**: Explicitly managed scratch space for block collaboration
- **Caching**: L1/L2 caches handle some locality automatically

### Tenstorrent Memory Hierarchy

```
DRAM                   →  External DDR, accessed via NOC
    ↕ (NOC async read/write)
L1 Circular Buffers    →  1 MB per Tensix core, explicit staging
    ↕ (cb_wait_front/cb_pop_front)
Destination Registers  →  Tile-sized (32×32 elements), compute accumulation
```

**Access patterns:**
- **Circular buffers (CBs)**: Explicit producer-consumer staging areas in L1
- **NOC operations**: Async reads/writes move tiles between DRAM and L1
- **Tile-oriented**: All operations work on 32×32 tiles (2KB for FP16)
- **No implicit caching**: Developer must explicitly manage L1 staging

---

## Data Movement Comparison

### GPU: Implicit + Explicit Transfers

**Global → Shared:**
```cuda
__shared__ float tile_A[32][32];
tile_A[ty][tx] = A[by * 32 + ty, k * 32 + tx];  // Coalesced load
__syncthreads();  // Synchronize before use
```

**Characteristics:**
- **Implicit caching**: L1/L2 caches reduce DRAM traffic
- **Explicit shared memory**: Developer copies global→shared for reuse
- **Thread-level granularity**: Each thread loads individual elements

### Tenstorrent: Explicit NOC Operations

**DRAM → L1 (Reader kernel):**
```cpp
// Compute tile address in DRAM (row-major tiled layout)
uint32_t tile_idx = by * Nt + bx;
uint64_t dram_addr = base_addr + tile_idx * TILE_SIZE;

// Asynchronous NOC read
cb_reserve_back(cb_in, 1);
noc_async_read_tile(dram_addr, get_write_ptr(cb_in), TILE_SIZE);
noc_async_read_barrier();  // Ensure transfer completes
cb_push_back(cb_in, 1);  // Signal compute kernel
```

**L1 → DRAM (Writer kernel):**
```cpp
cb_wait_front(cb_out, 1);  // Wait for compute result
uint64_t dram_addr = base_addr + tile_idx * TILE_SIZE;
noc_async_write_tile(get_read_ptr(cb_out), dram_addr, TILE_SIZE);
noc_async_write_barrier();  // Ensure write completes
cb_pop_front(cb_out, 1);  // Release CB space
```

**Characteristics:**
- **Three-kernel pattern**: Separate reader, compute, writer kernels
- **Tile granularity**: Move entire 32×32 tiles (2KB for FP16)
- **Asynchronous NOC**: Network-on-chip handles inter-core/DRAM transfers
- **CB synchronization**: Producer-consumer protocol via circular buffers

---

## Compute Operations Comparison

### GPU: Tensor Cores + CUDA Cores

**Matrix multiplication (Tensor Cores):**
```cuda
// WMMA API (Warp Matrix Multiply-Accumulate)
wmma::fragment<...> a_frag, b_frag, c_frag;
wmma::load_matrix_sync(a_frag, tile_A, 32);
wmma::load_matrix_sync(b_frag, tile_B, 32);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // C += A @ B
wmma::store_matrix_sync(tile_C, c_frag, 32);
```

**Characteristics:**
- **Warp-level operations**: 32 threads cooperate on matrix fragments
- **Mixed precision**: FP16/BF16 inputs, FP32 accumulation
- **Implicit accumulation**: Tensor cores accumulate in registers

### Tenstorrent: Tile-Based Compute Intrinsics

**Matrix multiplication (matmul_tiles):**
```cpp
// Initialize matmul for CB indices
mm_init(cb_in0, cb_in1, cb_out);

// Acquire destination tile registers
tile_regs_acquire();

// K-loop: Accumulate partial products
for (uint32_t k = 0; k < Kt; ++k) {
    cb_wait_front(cb_in0, 1);  // Wait for A tile
    cb_wait_front(cb_in1, 1);  // Wait for B tile

    matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);  // C[0] += A[0] @ B[0]

    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);
}

// Pack result from DST to CB
tile_regs_commit();
tile_regs_wait();
cb_reserve_back(cb_out, 1);
pack_tile(0, cb_out);  // DST[0] → CB[out]
cb_push_back(cb_out, 1);
tile_regs_release();
```

**Characteristics:**
- **Tile-level operations**: `matmul_tiles`, `add_tiles`, `mul_tiles`, etc.
- **Destination registers (DST)**: Tile-sized accumulators for matmul
- **Explicit lifecycle**: Acquire→Compute→Commit→Pack→Release
- **Tilize/Untilize**: Convert between row-major and tile layout

---

## Summary Comparison Table

| Aspect | GPU (CUDA/HIP) | Tenstorrent (TT-Metalium) |
|--------|----------------|---------------------------|
| **Execution** | Transient blocks, hardware-scheduled | Persistent per-core loops, static partitioning |
| **Parallelism** | Massive (thousands of blocks) | Moderate (64-120 Tensix cores) |
| **Memory** | L1/L2 caches + explicit shared | Explicit L1 circular buffers |
| **Data Movement** | Implicit caching + `__shared__` | Explicit NOC operations + CBs |
| **Tile Size** | Flexible (warp-level fragments) | Fixed 32×32 tiles |
| **Compute** | Tensor Cores (WMMA) | Tile intrinsics (`matmul_tiles`) |
| **Synchronization** | `__syncthreads()`, barriers | CB producer-consumer protocol |
| **Programming Model** | SIMT (Single Instruction, Multiple Threads) | SPMD (Single Program, Multiple Data) + Dataflow |

---

## Implications for TileLang

### What Works Naturally
- Grid-style kernels with block indices (`bx`, `by`)
- Matrix multiplication and convolution patterns
- Tile-aligned shapes (multiples of 32)

### What Requires Adaptation
- **Persistent loop wrapper**: Grid blocks → per-core tile iteration
- **Circular buffer management**: Allocate CBs for `T.copy` operations
- **Tile padding**: Non-tile-aligned shapes need padding + masking
- **Explicit scheduling**: Specify tile-to-core mapping (vs GPU's dynamic scheduling)
- **Three-kernel pattern**: Split reader/compute/writer for pipelined execution

### Developer Experience
TileLang provides a GPU-like surface that lowers to Tenstorrent's model:

```python
import tilelang.language as T
import tilelang.tenstorrent as tt

@T.prim_func
def matmul(A: T.Buffer, B: T.Buffer, C: T.Buffer):
    with T.Kernel(T.ceildiv(M, 32), T.ceildiv(N, 32)) as (bx, by):
        # GPU-style grid kernel (user code stays familiar)
        # ...

# Apply TT backend
mod = tvm.IRModule.from_expr(matmul)
mod = tt.apply_tt_defaults(mod)         # Inject TT annotations
mod = tt.apply_tt_metadata_passes(mod)  # Lower to persistent + CBs
artifacts = tt.emit_tt_artifacts(mod)   # Generate reader/compute/writer
```

The backend handles the transformation to Tenstorrent's execution model transparently.
