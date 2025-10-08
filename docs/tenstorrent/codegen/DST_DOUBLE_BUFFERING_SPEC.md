# DST Register Double Buffering Specification

**Date**: 2025-10-08
**Purpose**: Model Tensix packer and DST register handshaking in generated compute kernels
**Priority**: Foundation for all compute kernel codegen

---

## Overview

The **Destination (DST) registers** in Tensix cores are shared between:
1. **FPU (math engine)**: Writes computation results to DST
2. **Packer**: Reads from DST and packs tiles to L1 circular buffers

To enable pipelining, DST supports **double buffering** with explicit handshake protocol:
- **acquire_dst()**: FPU reserves half of DST for computation
- **commit_dst()**: FPU signals computation complete
- **wait_for_tile()**: Packer waits for FPU to finish
- **release_dst()**: FPU releases DST back to packer

---

## Metalium Reference

From tt-metal programming examples:

```cpp
// Element-wise add example (conceptual)
void MAIN {
    for (uint32_t i = 0; i < num_tiles; ++i) {
        // 1. Acquire DST for this tile
        acquire_dst();

        // 2. Wait for input tiles in CBs
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);

        // 3. Perform computation (FPU writes to DST)
        add_tiles_init();
        add_tiles(cb_a, cb_b, 0, 0, 0);  // A[0] + B[0] → DST[0]

        // 4. Commit DST (mark ready for packer)
        cb_reserve_back(cb_c, 1);
        commit_dst();

        // 5. Pack from DST to output CB
        pack_tile(0, cb_c);  // Packer waits internally
        cb_push_back(cb_c, 1);

        // 6. Pop input tiles
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);

        // 7. Release DST for next iteration
        release_dst();
    }
}
```

For matmul with K-loop:

```cpp
void MAIN {
    for (uint32_t tile_idx = 0; tile_idx < num_output_tiles; ++tile_idx) {
        // Acquire DST once for entire K-loop
        acquire_dst();

        // Initialize accumulator
        matmul_tiles_init(cb_c);

        // K-loop: accumulate in DST
        for (uint32_t k = 0; k < Kt; ++k) {
            cb_wait_front(cb_a, 1);
            cb_wait_front(cb_b, 1);

            matmul_tiles(cb_a, cb_b, 0, 0, 0, false /* transpose */);

            cb_pop_front(cb_a, 1);
            cb_pop_front(cb_b, 1);
        }

        // Commit after all K iterations
        cb_reserve_back(cb_c, 1);
        commit_dst();

        // Pack result
        pack_tile(0, cb_c);
        cb_push_back(cb_c, 1);

        release_dst();
    }
}
```

---

## Pattern Classification

### Pattern 1: Element-Wise Operations
**Characteristics**:
- One output tile per input tile(s)
- No accumulation across loop iterations
- DST acquired and released **per tile**

**DST Lifecycle**:
```
acquire_dst() → compute_single_tile() → commit_dst() → pack() → release_dst()
```

**Examples**:
- Element-wise add: `C = A + B`
- Element-wise multiply: `C = A * B`
- Activation functions: `C = relu(A)`, `C = gelu(A)`
- Type conversions: `C = cast(A, dtype)`

---

### Pattern 2: Reduction Operations (Local)
**Characteristics**:
- Accumulation within a tile
- May span multiple input tiles
- DST acquired once, released after reduction complete

**DST Lifecycle**:
```
acquire_dst() → init_accumulator() →
  for inputs: compute_and_accumulate() →
commit_dst() → pack() → release_dst()
```

**Examples**:
- Row/column reductions: `sum(A, axis=1)`
- Softmax statistics: `max(A)`, `sum(exp(A))`

---

### Pattern 3: Matrix Operations (K-loop)
**Characteristics**:
- Accumulation across K dimension
- DST holds partial results through loop
- Acquired before K-loop, released after

**DST Lifecycle**:
```
acquire_dst() → matmul_init() →
  for k: matmul_accumulate() →
commit_dst() → pack() → release_dst()
```

**Examples**:
- GEMM: `C = A @ B`
- Batched GEMM variants

---

## Implementation Strategy

### Step 1: IR Analysis
Identify pattern from TIR structure:

```python
# Element-wise pattern (Pattern 1)
for tile_id in range(num_tiles):
    load_tile(A)
    load_tile(B)
    C_tile = A_tile + B_tile  # ← Single operation, no accumulation
    store_tile(C)

# K-loop pattern (Pattern 3)
for tile_id in range(num_output_tiles):
    C_acc = zero_init()
    for k in range(Kt):
        load_tile(A, k)
        load_tile(B, k)
        C_acc += matmul(A_tile, B_tile)  # ← Accumulation across K
    store_tile(C_acc)
```

### Step 2: Codegen Insertion Points

In `CodeGenTTComputeVisitor`:

#### For Element-Wise (Pattern 1):
```cpp
void VisitStmt_(const ForNode* op) {
    if (IsOuterTileLoop(op)) {
        EmitString("for (uint32_t i = 0; i < count; ++i) {");
        Indent();

        // Recover (bx, by) from tile ID
        EmitTileIndexRecovery();

        // Element-wise body
        EmitString("acquire_dst();");
        EmitString("cb_wait_front(cb_a, 1);");
        EmitString("cb_wait_front(cb_b, 1);");
        EmitString("add_tiles_init();");
        EmitString("add_tiles(cb_a, cb_b, 0, 0, 0);");
        EmitString("cb_reserve_back(cb_c, 1);");
        EmitString("commit_dst();");
        EmitString("pack_tile(0, cb_c);");
        EmitString("cb_push_back(cb_c, 1);");
        EmitString("cb_pop_front(cb_a, 1);");
        EmitString("cb_pop_front(cb_b, 1);");
        EmitString("release_dst();");

        Dedent();
        EmitString("}");
    }
}
```

#### For K-Loop GEMM (Pattern 3):
```cpp
void VisitStmt_(const ForNode* op) {
    if (IsOuterTileLoop(op)) {
        // Outer loop over output tiles
        EmitString("for (uint32_t tile_idx = 0; tile_idx < num_output_tiles; ++tile_idx) {");
        Indent();

        EmitString("acquire_dst();");  // ← Acquire BEFORE K-loop
        EmitString("matmul_tiles_init(cb_c);");

        // K-loop
        EmitString("for (uint32_t k = 0; k < Kt; ++k) {");
        Indent();
        EmitString("cb_wait_front(cb_a, 1);");
        EmitString("cb_wait_front(cb_b, 1);");
        EmitString("matmul_tiles(cb_a, cb_b, 0, 0, 0, false);");
        EmitString("cb_pop_front(cb_a, 1);");
        EmitString("cb_pop_front(cb_b, 1);");
        Dedent();
        EmitString("}");

        EmitString("cb_reserve_back(cb_c, 1);");
        EmitString("commit_dst();");  // ← Commit AFTER K-loop
        EmitString("pack_tile(0, cb_c);");
        EmitString("cb_push_back(cb_c, 1);");
        EmitString("release_dst();");

        Dedent();
        EmitString("}");
    }
}
```

---

## Testing Strategy

### Test 1: Element-Wise Add
**TileLang Input**:
```python
@T.prim_func
def elem_add(A, B, C):
    with T.Kernel(8, 8) as (bx, by):
        A_tile = T.alloc_fragment((32, 32), "float16")
        B_tile = T.alloc_fragment((32, 32), "float16")
        C_tile = T.alloc_fragment((32, 32), "float16")

        T.copy(A[by*32:(by+1)*32, bx*32:(bx+1)*32], A_tile)
        T.copy(B[by*32:(by+1)*32, bx*32:(bx+1)*32], B_tile)
        C_tile = A_tile + B_tile
        T.copy(C_tile, C[by*32:(by+1)*32, bx*32:(bx+1)*32])
```

**Expected Compute Kernel** (verify DST lifecycle):
```cpp
void kernel_main() {
    uint32_t start_id = get_arg_val<uint32_t>(0);
    uint32_t count = get_arg_val<uint32_t>(1);
    uint32_t grid_x = get_arg_val<uint32_t>(2);

    for (uint32_t i = 0; i < count; ++i) {
        uint32_t tid = start_id + i;
        uint32_t by = tid / grid_x;
        uint32_t bx = tid % grid_x;

        acquire_dst();  // ✓
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        add_tiles_init();
        add_tiles(cb_a, cb_b, 0, 0, 0);
        commit_dst();  // ✓
        cb_reserve_back(cb_c, 1);
        pack_tile(0, cb_c);
        cb_push_back(cb_c, 1);
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
        release_dst();  // ✓
    }
}
```

### Test 2: GEMM K-Loop
**TileLang Input**:
```python
@T.prim_func
def matmul(A, B, C):
    with T.Kernel(8, 8) as (bx, by):
        C_acc = T.alloc_fragment((32, 32), "float16")
        T.clear(C_acc)

        for k in range(8):
            A_tile = T.alloc_fragment((32, 32), "float16")
            B_tile = T.alloc_fragment((32, 32), "float16")
            T.copy(A[by*32:(by+1)*32, k*32:(k+1)*32], A_tile)
            T.copy(B[k*32:(k+1)*32, bx*32:(bx+1)*32], B_tile)
            T.gemm(A_tile, B_tile, C_acc)

        T.copy(C_acc, C[by*32:(by+1)*32, bx*32:(bx+1)*32])
```

**Expected Compute Kernel** (verify DST held across K-loop):
```cpp
void kernel_main() {
    // ... runtime args ...

    for (uint32_t tile_idx = 0; tile_idx < num_output_tiles; ++tile_idx) {
        acquire_dst();  // ✓ Before K-loop
        matmul_tiles_init(cb_c);

        for (uint32_t k = 0; k < Kt; ++k) {
            cb_wait_front(cb_a, 1);
            cb_wait_front(cb_b, 1);
            matmul_tiles(cb_a, cb_b, 0, 0, 0, false);
            cb_pop_front(cb_a, 1);
            cb_pop_front(cb_b, 1);
        }

        cb_reserve_back(cb_c, 1);
        commit_dst();  // ✓ After K-loop
        pack_tile(0, cb_c);
        cb_push_back(cb_c, 1);
        release_dst();  // ✓
    }
}
```

---

## Success Criteria

- [ ] Element-wise kernels have per-tile DST lifecycle
- [ ] GEMM kernels have per-output-tile DST lifecycle (spans K-loop)
- [ ] Generated code matches Metalium examples
- [ ] acquire/commit/release calls are balanced
- [ ] DST not released while packer might be using it
- [ ] Tests pass for both patterns

---

## Implementation Files

**Primary**:
- `src/target/tt/codegen_tt_compute_visitor.cc` - Main compute kernel codegen
- `src/target/tt/codegen_tt_compute_visitor.h` - Class definition

**Tests**:
- `testing/python/tt/test_dst_double_buffering.py` - New test file

**Documentation**:
- This spec document

---

## Next Steps

1. Implement Pattern 1 (element-wise) in compute visitor
2. Test with `examples/elementwise/example_elementwise_add.py`
3. Verify generated code structure
4. Create PR: "Implement DST Double Buffering for Element-Wise Ops"
5. Implement Pattern 3 (K-loop matmul)
6. Test with simple GEMM
7. Create PR: "Implement DST Double Buffering for GEMM K-Loop"

---

**References**:
- [Metalium Programming Guide](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md)
- [GPU vs Tenstorrent Architecture](../GPU_vs_Tenstorrent.md)
- [Examples Plan](../TILELANG_TO_TT_EXAMPLES_PLAN.md)

---

**Status**: Specification Complete → Ready for Implementation
