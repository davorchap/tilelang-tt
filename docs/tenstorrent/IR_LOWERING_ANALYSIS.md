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

**Location:** `tilelang/engine/tt/lower.py` → `apply_tt_defaults()`

```python
# Apply default TT annotations (Target Registration)
# - Schedule: policy="contiguous", order="row_major"
# - Layout: DRAM interleaved, 32×32 tiles
mod = apply_tt_defaults(mod)
```

### Phase 2: Frontend Lowering (Shared with CUDA)

**Location:** `tilelang/engine/tt/lower.py` → `LowerAndLegalizeTT()`

```python
# Calls the same LowerAndLegalize() as CUDA
with target:
    mod = LowerAndLegalize(mod, target)
```

**Same 15 passes as CUDA** (see Phase 1 above).

### Phase 3: TT-Specific Optimizations

**Location:** `tilelang/engine/tt/lower.py` → `OptimizeForTargetTT()`

```python
def OptimizeForTargetTT(mod: IRModule, target: Target) -> IRModule:
    # === Metadata Inference: Schedule and Sharding Inference ===
    mod = infer_default_tt_schedule(mod)  # Compute per-core tile ranges
    mod = infer_default_tt_shard(mod)     # DRAM layout descriptors

    # === Transform Pipeline: TT-Specific TIR Transformations ===
    mod = grid_to_persistent_tt(mod)      # Grid → persistent loop
    mod = tt_tiles_to_core_map(mod)       # Tile assignments → core (x, y)
    mod = memory_space_lower_tt(mod)      # DRAM → L1 circular buffers
    mod = tile_pad_tt(mod)                # Pad to 32×32 tiles

    # ⭐ CRITICAL: Tensorization (pattern detection)
    mod = tensorize_tt(mod)

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
- **infer_default_tt_schedule**: Compute per-core tile assignments
- **infer_default_tt_shard**: Generate DRAM sharding descriptors
- **grid_to_persistent_tt**: Transform GPU grid to TT persistent loops
- **tt_tiles_to_core_map**: Map tile assignments to NOC grid coordinates
- **memory_space_lower_tt**: Lower DRAM allocations to L1 circular buffers
- **tile_pad_tt**: Pad buffers to 32×32 tile boundaries
- **tensorize_tt**: ⭐ Pattern detection and intrinsic annotation
- **verify_tt_ir**: Verify TT constraints (grid size, CB counts, etc.)

### Phase 4: TensorizeTT Pass (INCOMPLETE)

**Location:** `src/transform/tt/tensorize_tt.cc`

**Current Implementation:**

```cpp
class TensorizeTTMutator : public StmtMutator {
  Stmt VisitStmt_(const AttrStmtNode* op) override {
    // ⭐ Only handles T.gemm() intrinsic calls
    if (op->attr_key == "pragma_gemm" ||
        op->attr_key == "tl.gemm" ||
        op->attr_key == "gemm_operation") {

      matmul_count_++;
      PrimExpr matmul_id = IntImm(DataType::Int(32), matmul_count_ - 1);
      Stmt new_body = VisitStmt(op->body);

      // Add annotation: tt.matmul_intrinsic
      return AttrStmt(op->node, "tt.matmul_intrinsic", matmul_id, new_body);
    }

    return StmtMutator::VisitStmt_(op);
  }
};
```

**What's Missing:**
- ❌ No pattern matching for manual matmul loops (K-loop with A[m,k] * B[k,n])
- ❌ No detection of element-wise operations
- ❌ No annotation of loop bodies with intrinsic metadata

**What Should Be Added:**

```cpp
class TensorizeTTMutator : public StmtMutator {
  Stmt VisitStmt_(const ForNode* op) override {
    // Detect K-loop pattern
    std::string loop_var = op->loop_var->name_hint;
    bool is_k_loop = (loop_var == "kt" || loop_var == "k" ||
                      loop_var.find("_k") != std::string::npos);

    if (is_k_loop) {
      // Analyze loop body to detect matmul pattern
      // Look for: C[m,n] += A[m,k] * B[k,n]
      MatmulPatternMatcher matcher;
      if (matcher.Match(op->body)) {
        // Annotate with tt.matmul_intrinsic
        PrimExpr matmul_id = IntImm(DataType::Int(32), matmul_count_++);

        // Create annotated body
        Stmt annotated_body = AttrStmt(
          op->loop_var,
          "tt.matmul_k_loop",
          matmul_id,
          op->body
        );

        // Return new For loop with annotated body
        return For(
          op->loop_var,
          op->min,
          op->extent,
          op->kind,
          annotated_body
        );
      }
    }

    // Detect element-wise pattern
    TileLoopAnalyzer analyzer;
    if (analyzer.IsElementwiseTileLoop(op)) {
      // Annotate with tt.elementwise_intrinsic
      PrimExpr eltwise_id = IntImm(DataType::Int(32), eltwise_count_++);
      Stmt annotated_body = AttrStmt(
        op->loop_var,
        "tt.elementwise_op",
        eltwise_id,
        op->body
      );
      return For(op->loop_var, op->min, op->extent, op->kind, annotated_body);
    }

    return StmtMutator::VisitStmt_(op);
  }
};
```

### Phase 5: Device Splitting (TT 3-Kernel Architecture)

**Location:** `tilelang/engine/tt/lower.py` → `SplitTTKernels()`

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

**Location:** `src/target/tt/codegen_tt_*.cc`

**Current Implementation (INCOMPLETE):**

```cpp
// Compute visitor
Stmt VisitStmt_(const ForNode* op) override {
  std::string loop_var = op->loop_var->name_hint;
  bool is_k_loop = (loop_var == "kt" || loop_var.find("kt") != std::string::npos);

  if (is_k_loop) {
    // Emit K-loop scaffolding
    if (!dst_acquired_) {
      EmitTileRegsAcquire();
    }
    if (!matmul_init_emitted_) {
      EmitLine("mm_init(cb_in0, cb_in1, cb_out0);");
    }

    EmitLine("for (uint32_t " + loop_var + " = ...) {");

    // ❌ PROBLEM: Visits body, emits raw IR
    VisitStmt(op->body);  // Should check for annotations!

    EmitLine("}");
  }
}
```

**What Should Happen:**

```cpp
Stmt VisitStmt_(const ForNode* op) override {
  std::string loop_var = op->loop_var->name_hint;

  // Check for tt.matmul_k_loop annotation
  if (auto* attr = op->body.as<AttrStmtNode>()) {
    if (attr->attr_key == "tt.matmul_k_loop") {
      // K-loop detected via annotation
      if (!dst_acquired_) {
        EmitTileRegsAcquire();
      }
      if (!matmul_init_emitted_) {
        EmitLine("mm_init(cb_in0, cb_in1, cb_out0);");
      }

      EmitLine("for (uint32_t " + loop_var + " = 0; " +
               loop_var + " < Kt; ++" + loop_var + ") {");

      // ✅ Emit intrinsics instead of visiting body
      EmitLine("  cb_wait_front(cb_in0, 1);");
      EmitLine("  cb_wait_front(cb_in1, 1);");
      EmitLine("  ");
      EmitLine("  bool accumulate = (" + loop_var + " > 0);");
      EmitLine("  matmul_tiles(cb_in0, cb_in1, 0, 0, 0, accumulate);");
      EmitLine("  ");
      EmitLine("  cb_pop_front(cb_in0, 1);");
      EmitLine("  cb_pop_front(cb_in1, 1);");

      EmitLine("}");

      // After K-loop
      EmitLine("tile_regs_commit();");
      EmitLine("tile_regs_wait();");
      EmitLine("pack_tile(0, cb_out0);");

      return;  // Don't visit body!
    }
  }

  // Fallback: visit normally
  return StmtMutator::VisitStmt_(op);
}
```

## Comparison Summary

| Aspect | GPU (CUDA) | Tenstorrent (Current) | Tenstorrent (Should Be) |
|--------|------------|----------------------|-------------------------|
| **Pattern Detection** | Transform pass (`InferFragment`) | Codegen heuristics (variable name) | Transform pass (`tensorize_tt`) ✅ |
| **Annotation Method** | AttrStmt with fragment metadata | None (codegen guesses) | AttrStmt with intrinsic ID ✅ |
| **Intrinsic Insertion** | Frontend (`T.gemm()` → intrinsics) | None (manual loops) | Transform pass annotations ✅ |
| **Codegen Role** | Read annotations, emit code | Detect patterns AND emit code ❌ | Read annotations, emit code ✅ |
| **IR Representation** | Intrinsic calls (tvm_mma_sync) | Raw loops and array ops | Annotated loops ✅ |
| **Device Splitting** | IR transform (SplitHostDevice) | Codegen (3 visitors) | Codegen (3 visitors) ✅ |

## Recommended Architecture Changes

### 1. Extend `tensorize_tt.cc`

Add pattern matching for:
- Manual matmul loops (K-loop detection)
- Element-wise operations
- Other TT intrinsics (copy, etc.)

### 2. Update Compute Visitor

Change from:
```cpp
if (is_k_loop) {
  // Emit scaffolding
  VisitStmt(op->body);  // ❌ Emits raw IR
}
```

To:
```cpp
if (HasAnnotation(op->body, "tt.matmul_k_loop")) {
  // Emit scaffolding
  EmitMatmulIntrinsics();  // ✅ Emit intrinsics
  return;  // Don't visit body
}
```

### 3. Annotation Format

```cpp
// Example annotated IR after tensorize_tt pass:
For(kt, 0, Kt,
  AttrStmt(kt, "tt.matmul_k_loop", 0,
    AttrStmt(_, "tt.input_buffers", {A, B},
      AttrStmt(_, "tt.output_buffer", C,
        // Original loop body (not emitted by codegen)
        C[m,n] += A[m,k] * B[k,n]
      )
    )
  )
)
```

Codegen reads `tt.matmul_k_loop` annotation and emits:
```cpp
for (uint32_t kt = 0; kt < Kt; ++kt) {
  cb_wait_front(cb_in0, 1);
  cb_wait_front(cb_in1, 1);
  matmul_tiles(cb_in0, cb_in1, 0, 0, 0, kt > 0);
  cb_pop_front(cb_in0, 1);
  cb_pop_front(cb_in1, 1);
}
```

## Conclusion

**Current Gap:** TT codegen tries to do pattern detection (heuristics based on variable names) AND code emission, which is fragile and incomplete.

**Solution:** Follow GPU architecture:
1. **Transform pass** (`tensorize_tt`) detects patterns and annotates IR
2. **Codegen** reads annotations and emits intrinsics (no pattern detection)

This separation makes the compiler more maintainable and allows IR optimizations to work correctly.
