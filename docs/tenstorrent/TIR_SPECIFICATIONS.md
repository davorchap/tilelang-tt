# TileLang Tenstorrent Backend: TIR Specifications

**Version:** 1.0
**Date:** 2025-10-07
**Purpose:** Detailed Tensor IR specifications for each workstream transformation

---

## Overview

This document provides detailed TVM Tensor IR (TIR) specifications showing the exact IR structure before and after each workstream transformation. These specs are essential for:
- Understanding transform semantics
- Implementing IR-driven codegen (Phase 2)
- Validating transform correctness
- Debugging IR issues

### Reading This Document

- **Notation**: Python-like pseudocode represents TIR AST structure
- **Types**: `PrimFunc`, `Stmt`, `PrimExpr`, `Buffer`, `Var`, `IntImm`, `StringImm`, `Array`
- **Focus**: 256×256 fp16 matmul example throughout

---

## Example Kernel: 256×256 Matmul

### TileLang Source

```python
@T.prim_func
def matmul(A: T.Buffer[(256, 256), "float16"],
           B: T.Buffer[(256, 256), "float16"],
           C: T.Buffer[(256, 256), "float16"]):
    with T.Kernel(8, 8) as (bx, by):  # 8×8 tile grid
        A_tile = T.alloc_fragment((32, 32), "float16")
        B_tile = T.alloc_fragment((32, 32), "float16")
        C_tile = T.alloc_fragment((32, 32), "float16")

        T.clear(C_tile)

        for k in range(8):  # K-loop
            T.copy(A[by*32:(by+1)*32, k*32:(k+1)*32], A_tile)
            T.copy(B[k*32:(k+1)*32, bx*32:(bx+1)*32], B_tile)
            T.gemm(A_tile, B_tile, C_tile)

        T.copy(C_tile, C[by*32:(by+1)*32, bx*32:(bx+1)*32])
```

This will be our running example throughout all transformations.

---

## WS1: Target Registration & Default Annotations

### Transformation Type
**Attribute stamping only** - IR body unchanged

### Input TIR (From Frontend)

```python
PrimFunc(
  params=[
    Buffer(var=A, shape=[256, 256], dtype=DataType::Float(16)),
    Buffer(var=B, shape=[256, 256], dtype=DataType::Float(16)),
    Buffer(var=C, shape=[256, 256], dtype=DataType::Float(16)),
  ],
  buffer_map={
    A: Buffer(var=A, ...),
    B: Buffer(var=B, ...),
    C: Buffer(var=C, ...)
  },
  body=
    # Thread extent attributes (from T.Kernel(8, 8))
    AttrStmt(
      attr_key="thread_extent",
      node=IterVar(var=bx, dom=Range(min=0, extent=8), iter_type=kThreadIndex, thread_tag="blockIdx.x"),
      value=IntImm(dtype=int32, value=8),
      body=
        AttrStmt(
          attr_key="thread_extent",
          node=IterVar(var=by, dom=Range(min=0, extent=8), iter_type=kThreadIndex, thread_tag="blockIdx.y"),
          value=IntImm(dtype=int32, value=8),
          body=
            # Allocations
            Allocate(
              buffer_var=A_tile,
              dtype=DataType::Float(16),
              extents=[IntImm(32), IntImm(32)],
              condition=IntImm(dtype=bool, value=1),
              body=
                Allocate(
                  buffer_var=B_tile,
                  dtype=DataType::Float(16),
                  extents=[IntImm(32), IntImm(32)],
                  condition=IntImm(dtype=bool, value=1),
                  body=
                    Allocate(
                      buffer_var=C_tile,
                      dtype=DataType::Float(16),
                      extents=[IntImm(32), IntImm(32)],
                      condition=IntImm(dtype=bool, value=1),
                      body=
                        # Clear C_tile
                        BufferStore(
                          buffer=C_tile,
                          value=FloatImm(dtype=float16, value=0.0),
                          indices=[IntImm(0), IntImm(0)]
                        )

                        # K-loop
                        For(
                          loop_var=Var(name=k, dtype=int32),
                          min=IntImm(dtype=int32, value=0),
                          extent=IntImm(dtype=int32, value=8),
                          kind=ForKind::kSerial,
                          body=
                            # Copy A[by*32:, k*32:] → A_tile
                            BufferStore(...)

                            # Copy B[k*32:, bx*32:] → B_tile
                            BufferStore(...)

                            # C_tile += A_tile @ B_tile
                            AttrStmt(
                              attr_key="tl.gemm",
                              value={...},
                              body=Evaluate(IntImm(0))
                            )
                        )

                        # Copy C_tile → C[by*32:, bx*32:]
                        BufferStore(...)
                    )
                )
            )
        )
    ),
  ret_type=VoidType(),
  attrs={}  # ⬅ EMPTY INITIALLY
)
```

### Output TIR (After WS1)

```python
PrimFunc(
  params=[...],  # UNCHANGED
  buffer_map={...},  # UNCHANGED
  body=...,  # UNCHANGED
  ret_type=VoidType(),  # UNCHANGED
  attrs={  # ⬅ NEW ATTRIBUTES ADDED
    "tt_schedule_policy": StringImm("contiguous"),
    "tt_schedule_order": StringImm("row_major"),
    "tt_layout_type": StringImm("dram_interleaved"),
    "tt_tile_height": IntImm(dtype=int32, value=32),
    "tt_tile_width": IntImm(dtype=int32, value=32),
  }
)
```

### Transformation Semantics

**apply_tt_defaults(mod: IRModule) -> IRModule:**
```python
def apply_tt_defaults(mod):
    for gv, func in mod.functions.items():
        if isinstance(func, PrimFunc):
            new_attrs = dict(func.attrs) if func.attrs else {}

            # Stamp default TT attributes
            new_attrs["tt_schedule_policy"] = "contiguous"
            new_attrs["tt_schedule_order"] = "row_major"
            new_attrs["tt_layout_type"] = "dram_interleaved"
            new_attrs["tt_tile_height"] = 32
            new_attrs["tt_tile_width"] = 32

            mod[gv] = func.with_attrs(new_attrs)

    return mod
```

---

## WS2: Schedule & Sharding Metadata Inference

### Transformation Type
**Metadata inference** - IR body unchanged, adds schedule + sharding attrs

### Input TIR (After WS1)

```python
PrimFunc(
  params=[A, B, C],
  body=
    AttrStmt(  # Grid: T.Kernel(8, 8)
      attr_key="thread_extent",
      node=IterVar(var=bx, dom=Range(0, 8), thread_tag="blockIdx.x"),
      value=8,
      body=
        AttrStmt(
          attr_key="thread_extent",
          node=IterVar(var=by, dom=Range(0, 8), thread_tag="blockIdx.y"),
          value=8,
          body=
            # ... kernel body ...
        )
    ),
  attrs={
    "tt_schedule_policy": "contiguous",
    "tt_layout_type": "dram_interleaved",
    "tt_tile_height": 32,
    "tt_tile_width": 32,
  }
)
```

### Output TIR (After WS2)

```python
PrimFunc(
  params=[A, B, C],
  body=...,  # UNCHANGED
  attrs={
    # WS1 attrs PRESERVED
    "tt_schedule_policy": StringImm("contiguous"),
    "tt_layout_type": StringImm("dram_interleaved"),
    "tt_tile_height": IntImm(32),
    "tt_tile_width": IntImm(32),

    # WS2: SCHEDULE METADATA (NEW)
    "tt_grid_x": IntImm(8),
    "tt_grid_y": IntImm(8),
    "tt_grid_z": IntImm(1),
    "tt_num_tiles": IntImm(64),  # 8×8
    "tt_num_cores": IntImm(64),  # Grayskull: 8×8 Tensix cores
    "tt_tiles_per_core": Array([
      Array([IntImm(0), IntImm(1)]),   # Core 0: [start_id=0, count=1]
      Array([IntImm(1), IntImm(1)]),   # Core 1: [start_id=1, count=1]
      Array([IntImm(2), IntImm(1)]),   # Core 2: [start_id=2, count=1]
      # ... 61 more entries ...
      Array([IntImm(63), IntImm(1)]),  # Core 63: [start_id=63, count=1]
    ]),

    # WS2: BUFFER SHARDING METADATA (NEW)
    "tt_buffer_A_layout": StringImm("dram_interleaved"),
    "tt_buffer_A_tile_shape": Array([IntImm(32), IntImm(32)]),
    "tt_buffer_A_num_tiles_height": IntImm(8),  # 256 / 32
    "tt_buffer_A_num_tiles_width": IntImm(8),
    "tt_buffer_A_needs_padding": IntImm(0),  # Boolean as int

    "tt_buffer_B_layout": StringImm("dram_interleaved"),
    "tt_buffer_B_tile_shape": Array([IntImm(32), IntImm(32)]),
    "tt_buffer_B_num_tiles_height": IntImm(8),
    "tt_buffer_B_num_tiles_width": IntImm(8),
    "tt_buffer_B_needs_padding": IntImm(0),

    "tt_buffer_C_layout": StringImm("dram_interleaved"),
    "tt_buffer_C_tile_shape": Array([IntImm(32), IntImm(32)]),
    "tt_buffer_C_num_tiles_height": IntImm(8),
    "tt_buffer_C_num_tiles_width": IntImm(8),
    "tt_buffer_C_needs_padding": IntImm(0),
  }
)
```

### Transformation Semantics

**InferDefaultTTSchedule(func: PrimFunc) -> PrimFunc:**

Algorithm:
1. Extract grid dimensions from thread extent attributes
2. Compute total tiles: `num_tiles = grid_x × grid_y × grid_z`
3. Assign tiles to cores using policy (contiguous/interleaved/etc.)
4. Stamp schedule metadata as attributes

**InferDefaultTTShard(func: PrimFunc) -> PrimFunc:**

Algorithm:
1. For each buffer in params:
   - Extract buffer shape from type
   - Compute tile counts: `num_tiles_h = ceil(height / tile_height)`
   - Determine padding needs: `needs_padding = (height % tile_height != 0)`
   - Stamp sharding metadata with buffer name prefix

---

## WS3: TIR Transform Pipeline (GridToPersistentTT)

### Transformation Type
**IR rewriting** - Transforms grid-style kernel to persistent loop

### Input TIR (After WS2)

```python
PrimFunc(
  params=[
    Buffer(var=A, shape=[256, 256], dtype=float16),
    Buffer(var=B, shape=[256, 256], dtype=float16),
    Buffer(var=C, shape=[256, 256], dtype=float16),
  ],
  buffer_map={A: ..., B: ..., C: ...},
  body=
    # Grid binding via thread extent
    AttrStmt(
      attr_key="thread_extent",
      node=IterVar(var=bx, dom=Range(0, 8), thread_tag="blockIdx.x"),
      value=8,
      body=
        AttrStmt(
          attr_key="thread_extent",
          node=IterVar(var=by, dom=Range(0, 8), thread_tag="blockIdx.y"),
          value=8,
          body=
            # Original kernel body
            Allocate(buffer_var=A_tile, dtype=float16, extents=[32, 32],
              body=
                Allocate(buffer_var=B_tile, dtype=float16, extents=[32, 32],
                  body=
                    Allocate(buffer_var=C_tile, dtype=float16, extents=[32, 32],
                      body=
                        # Clear
                        BufferStore(buffer=C_tile, value=0.0, ...)

                        # K-loop
                        For(loop_var=k, min=0, extent=8, kind=kSerial,
                          body=
                            # Load A[by, k]
                            BufferStore(buffer=A_tile, value=BufferLoad(buffer=A, indices=[by*32, k*32]), ...)

                            # Load B[k, bx]
                            BufferStore(buffer=B_tile, value=BufferLoad(buffer=B, indices=[k*32, bx*32]), ...)

                            # Gemm
                            AttrStmt(attr_key="tl.gemm", ...)
                        )

                        # Store C[by, bx]
                        BufferStore(buffer=C, value=BufferLoad(buffer=C_tile, ...), indices=[by*32, bx*32])
                    )
                )
            )
        )
    ),
  attrs={
    # WS1+WS2 attrs...
    "tt_grid_x": 8,
    "tt_grid_y": 8,
    "tt_tiles_per_core": [[0, 1], [1, 1], ..., [63, 1]],
    # ...
  }
)
```

### Output TIR (After WS3 - GridToPersistentTT)

```python
PrimFunc(
  params=[
    Buffer(var=A, shape=[256, 256], dtype=float16),
    Buffer(var=B, shape=[256, 256], dtype=float16),
    Buffer(var=C, shape=[256, 256], dtype=float16),
    # NEW: Runtime arguments for persistent loop
    Var(name=tt_start_id, dtype=int32),
    Var(name=tt_count, dtype=int32),
    Var(name=grid_x, dtype=int32),
    Var(name=grid_y, dtype=int32),
  ],
  buffer_map={A: ..., B: ..., C: ...},
  body=
    # NEW: Persistent outer loop (replaces thread extent)
    For(
      loop_var=Var(name=i, dtype=int32),
      min=IntImm(dtype=int32, value=0),
      extent=Var(name=tt_count, dtype=int32),  # Runtime arg
      kind=ForKind::kSerial,
      body=
        # NEW: Compute tile ID
        LetStmt(
          var=Var(name=tile_id, dtype=int32),
          value=Add(Var(name=tt_start_id), Var(name=i)),
          body=
            # NEW: Recover block indices
            LetStmt(
              var=Var(name=bx, dtype=int32),
              value=FloorMod(Var(name=tile_id), Var(name=grid_x)),
              body=
                LetStmt(
                  var=Var(name=by, dtype=int32),
                  value=FloorDiv(Var(name=tile_id), Var(name=grid_x)),
                  body=
                    # Original kernel body (bx, by now computed variables)
                    Allocate(buffer_var=A_tile, dtype=float16, extents=[32, 32],
                      body=
                        Allocate(buffer_var=B_tile, dtype=float16, extents=[32, 32],
                          body=
                            Allocate(buffer_var=C_tile, dtype=float16, extents=[32, 32],
                              body=
                                # Clear
                                BufferStore(buffer=C_tile, value=0.0, ...)

                                # K-loop (unchanged)
                                For(loop_var=k, min=0, extent=8, kind=kSerial,
                                  body=
                                    # Load A[by, k] (by is now computed var)
                                    BufferStore(buffer=A_tile, value=BufferLoad(buffer=A, indices=[Mul(by, 32), Mul(k, 32)]), ...)

                                    # Load B[k, bx] (bx is now computed var)
                                    BufferStore(buffer=B_tile, value=BufferLoad(buffer=B, indices=[Mul(k, 32), Mul(bx, 32)]), ...)

                                    # Gemm
                                    AttrStmt(attr_key="tl.gemm", ...)
                                )

                                # Store C[by, bx]
                                BufferStore(buffer=C, value=BufferLoad(buffer=C_tile, ...), indices=[Mul(by, 32), Mul(bx, 32)])
                            )
                        )
                    )
                )
            )
        )
    ),
  ret_type=VoidType(),
  attrs={
    # WS1+WS2 attrs PRESERVED
    "tt_grid_x": 8,
    "tt_grid_y": 8,
    "tt_tiles_per_core": [[0, 1], ..., [63, 1]],
    # ...

    # WS3: NEW ATTRS
    "tt_persistent_loop": IntImm(dtype=int32, value=1),  # Boolean as int
    "tt_runtime_args": Array([
      StringImm("tt_start_id"),
      StringImm("tt_count"),
      StringImm("grid_x"),
      StringImm("grid_y")
    ]),
  }
)
```

### Transformation Semantics

**GridToPersistentTT(func: PrimFunc) -> PrimFunc:**

Algorithm:
1. Extract grid dimensions from attrs: `grid_x`, `grid_y`
2. Add new runtime arg parameters to function signature
3. Find thread extent AttrStmt nodes (blockIdx.x, blockIdx.y)
4. Create persistent for-loop: `for i in range(tt_count)`
5. Inside loop:
   - Compute `tile_id = tt_start_id + i`
   - Recover `bx = tile_id % grid_x`
   - Recover `by = tile_id / grid_x`
6. Replace thread extent nodes with LetStmt bindings for bx, by
7. Wrap in persistent loop
8. Stamp new attributes

**C++ Implementation Sketch:**
```cpp
class GridToPersistentTT : public StmtExprMutator {
public:
    PrimFunc Transform(PrimFunc func) {
        // 1. Read metadata
        int grid_x = GetIntAttr(func, "tt_grid_x");
        int grid_y = GetIntAttr(func, "tt_grid_y");

        // 2. Create new params
        Var tt_start_id("tt_start_id", DataType::Int(32));
        Var tt_count("tt_count", DataType::Int(32));
        Var grid_x_var("grid_x", DataType::Int(32));
        Var grid_y_var("grid_y", DataType::Int(32));

        Array<Var> new_params = func->params;
        new_params.push_back(tt_start_id);
        new_params.push_back(tt_count);
        new_params.push_back(grid_x_var);
        new_params.push_back(grid_y_var);

        // 3. Create persistent loop
        Var i("i", DataType::Int(32));
        PrimExpr tile_id = tt_start_id + i;

        // 4. Create bx/by bindings
        Var bx("bx", DataType::Int(32));
        Var by("by", DataType::Int(32));
        Stmt bx_binding = LetStmt(bx, floormod(tile_id, grid_x_var), body);
        Stmt by_binding = LetStmt(by, floordiv(tile_id, grid_x_var), bx_binding);

        // 5. Replace thread extent nodes
        Stmt new_body = ReplaceThreadExtent(func->body, bx, by);

        // 6. Wrap in for-loop
        Stmt persistent_loop = For(i, 0, tt_count, ForKind::kSerial, by_binding);

        // 7. Create new function
        Map<String, ObjectRef> new_attrs = func->attrs;
        new_attrs.Set("tt_persistent_loop", Integer(1));
        new_attrs.Set("tt_runtime_args", Array<String>({
            "tt_start_id", "tt_count", "grid_x", "grid_y"
        }));

        return PrimFunc(new_params, func->buffer_map, persistent_loop,
                        func->ret_type, new_attrs);
    }

private:
    Stmt ReplaceThreadExtent(Stmt stmt, Var bx, Var by) {
        // Walk IR, find AttrStmt with "thread_extent" and blockIdx tags
        // Replace with computed bx/by variables
        // ...
    }
};
```

---

## WS4-6: Code Generation (Phase 1 Template)

### Input

PrimFunc with all WS1-3 metadata + persistent loop structure

### Output

**Not TIR** - Generates C++ source code strings:
- `compute.cpp` - Compute kernel
- `reader.cpp` - Reader kernel
- `writer.cpp` - Writer kernel
- `main.cpp` - Host program
- `tt.plan.json` - Metadata JSON

### Codegen Approach (Phase 1)

**Template-based**: Reads metadata from `func->attrs`, emits fixed code patterns

```cpp
std::string EmitTTComputeKernel(const PrimFunc& func) {
    // Read metadata
    auto grid_x = func->attrs.GetAttr<Integer>("tt_grid_x");
    auto grid_y = func->attrs.GetAttr<Integer>("tt_grid_y");
    auto tiles_per_core = func->attrs.GetAttr<Array>("tt_tiles_per_core");

    MatmulDims dims = ExtractMatmulDims(func);

    // ⚠️ TEMPLATE: Emit fixed structure
    // Does NOT walk func->body

    std::ostringstream code;
    code << "void MAIN() {\n";
    code << "    uint32_t out_tile_start_id = get_arg_val<uint32_t>(0);\n";
    code << "    uint32_t num_output_tiles = get_arg_val<uint32_t>(1);\n";
    code << "    uint32_t Kt = get_arg_val<uint32_t>(2);\n";
    code << "    matmul_tiles_init(CB_A, CB_B, CB_C);\n";
    code << "    for (uint32_t out_tile = 0; out_tile < num_output_tiles; ++out_tile) {\n";
    code << "        for (uint32_t kt = 0; kt < Kt; ++kt) {\n";
    code << "            cb_wait_front(CB_A, 1);\n";
    code << "            cb_wait_front(CB_B, 1);\n";
    code << "            matmul_tiles(CB_A, CB_B, CB_C, kt > 0);\n";
    code << "            cb_pop_front(CB_A, 1);\n";
    code << "            cb_pop_front(CB_B, 1);\n";
    code << "        }\n";
    code << "        cb_push_back(CB_C, 1);\n";
    code << "    }\n";
    code << "}\n";

    return code.str();
}
```

### Codegen Approach (Phase 2 Preview)

**IR-driven**: Walks `func->body` using visitor pattern

```cpp
class TTComputeCodegen : public StmtExprVisitor {
private:
    std::ostringstream code_;
    int indent_ = 0;

public:
    void VisitStmt_(const ForNode* op) override {
        Indent();
        code_ << "for (uint32_t " << op->loop_var->name_hint
              << " = " << PrintExpr(op->min) << "; "
              << op->loop_var->name_hint << " < " << PrintExpr(op->extent)
              << "; ++" << op->loop_var->name_hint << ") {\n";

        indent_++;
        VisitStmt(op->body);
        indent_--;

        Indent();
        code_ << "}\n";
    }

    void VisitStmt_(const AttrStmtNode* op) override {
        if (op->attr_key == "tt.matmul_tiles") {
            // Extract operands from attr value
            auto a_cb = GetCBID(op, "A");
            auto b_cb = GetCBID(op, "B");
            auto c_cb = GetCBID(op, "C");
            bool acc = GetBool(op, "accumulate");

            Indent();
            code_ << "matmul_tiles(CB_" << a_cb << ", CB_" << b_cb
                  << ", CB_" << c_cb << ", " << (acc ? "true" : "false") << ");\n";
        }
        else if (op->attr_key == "tt.cb_wait_front") {
            // ...
        }

        VisitStmt(op->body);
    }

    void VisitStmt_(const BufferStoreNode* op) override {
        // Generate store operation
        // ...
    }

    std::string Generate(const PrimFunc& func) {
        EmitHeader();
        VisitStmt(func->body);  // ✅ Walk IR!
        return code_.str();
    }
};
```

---

## Summary Table

| Workstream | IR Changes | Attrs Changes | Body Changes |
|------------|-----------|---------------|--------------|
| **WS1** | None | +5 defaults | None |
| **WS2** | None | +schedule +sharding | None |
| **WS3** | +4 params | +2 metadata | ✅ Rewrite to persistent |
| **WS4-6** | N/A (codegen) | N/A | N/A |

---

## Validation

To validate transforms, check:

1. **Attrs preservation**: WS1 attrs must survive through all transforms
2. **Param count**: WS3 adds 4 params
3. **Loop structure**: WS3 adds outer `For` node
4. **Variable bindings**: bx/by become `LetStmt` in WS3
5. **No thread extent**: WS3 removes `AttrStmt(thread_extent)`

---

**Document Version**: 1.0
**Author**: Claude Code
**Date**: 2025-10-07
