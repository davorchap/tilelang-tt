# Task: Sharding Inference Pass

## Goal
Implement `InferDefaultTTShard` - a TVM pass that generates DRAM interleaved tensor layout descriptors and identifies padding requirements for non-tile-aligned dimensions.

## Context
- **Workstream:** WS2 - Schedule & Sharding Metadata
- **Dependencies:** Schedule inference pass (needs tile counts)
- **File:** `src/tt/transform/infer_tt_shard.cc`
- **Priority:** High (required for WS3 transforms)

## What This Pass Does

### Inputs
- PrimFunc with buffer parameters (e.g., A, B, C for GEMM)
- Default layout annotations from WS1:
  - `tt_layout_type = "dram_interleaved"`
  - `tt_tile_height = 32`
  - `tt_tile_width = 32`
- Schedule metadata from schedule inference pass:
  - `tt_num_tiles`, `tt_grid_x`, `tt_grid_y`

### Processing
1. **Iterate over buffer parameters** (inputs and outputs)
2. **For each buffer:**
   - Check dimensions against tile size (32×32)
   - Compute number of tiles needed: `(M/32) × (N/32)`
   - Mark if padding required (M or N not multiple of 32)
   - Generate interleaved layout metadata
3. **Create TensorAccessor configuration:**
   - Interleaved DRAM layout
   - Tile stride and offset calculations
   - Bank/channel assignment hints (MVP: defer to TT runtime)

### Outputs
Enhanced buffer parameters with sharding metadata:

```cpp
// Attributes attached to each buffer parameter:
buffer->attrs.Set("tt_layout", String("dram_interleaved"));
buffer->attrs.Set("tt_tile_shape", Array<IntImm>({32, 32}));
buffer->attrs.Set("tt_needs_padding", Bool(needs_padding));
buffer->attrs.Set("tt_padded_shape", Array<IntImm>({...}));  // If padding needed
buffer->attrs.Set("tt_num_tiles_height", IntImm(tiles_h));
buffer->attrs.Set("tt_num_tiles_width", IntImm(tiles_w));
```

## Implementation Plan

### Step 1: Pass Registration
```cpp
// src/tt/transform/infer_tt_shard.cc
namespace tvm {
namespace tir {
namespace transform {

Pass InferDefaultTTShard() {
  auto pass_func = [](PrimFunc func, IRModule mod, PassContext ctx) {
    return InferDefaultTTShardImpl(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InferDefaultTTShard", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InferDefaultTTShard")
.set_body_typed(InferDefaultTTShard);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
```

### Step 2: Core Logic
```cpp
PrimFunc InferDefaultTTShardImpl(PrimFunc func) {
  // 1. Read tt_layout_type from func->attrs (should be "dram_interleaved")
  // 2. Read tt_tile_height and tt_tile_width (should be 32, 32)

  // 3. Iterate over func->params (buffer parameters)
  Array<Var> new_params;
  for (const auto& param : func->params) {
    Buffer buffer = func->buffer_map[param];

    // 4. Get buffer shape
    Array<PrimExpr> shape = buffer->shape;

    // 5. Compute padding requirements
    auto shard_info = ComputeShardInfo(shape, tile_height, tile_width);

    // 6. Attach metadata to buffer
    buffer = AttachShardMetadata(buffer, shard_info);

    new_params.push_back(param);
  }

  // 7. Return updated func with enhanced buffer map
  return func;
}
```

### Step 3: Padding Computation
```cpp
struct ShardInfo {
  int tiles_height;
  int tiles_width;
  bool needs_padding;
  Array<IntImm> padded_shape;  // Only if needs_padding
};

ShardInfo ComputeShardInfo(
    const Array<PrimExpr>& shape,
    int tile_height,
    int tile_width) {

  ShardInfo info;

  // Assume 2D tensor [M, N] for GEMM
  // TODO: Handle 3D/4D for batched operations
  ICHECK(shape.size() >= 2) << "Expected at least 2D tensor";

  int M = shape[shape.size() - 2].as<IntImmNode>()->value;
  int N = shape[shape.size() - 1].as<IntImmNode>()->value;

  // Compute tiles needed
  info.tiles_height = (M + tile_height - 1) / tile_height;  // Ceiling division
  info.tiles_width = (N + tile_width - 1) / tile_width;

  // Check if padding needed
  bool M_aligned = (M % tile_height == 0);
  bool N_aligned = (N % tile_width == 0);
  info.needs_padding = !M_aligned || !N_aligned;

  if (info.needs_padding) {
    int M_padded = info.tiles_height * tile_height;
    int N_padded = info.tiles_width * tile_width;
    info.padded_shape = Array<IntImm>({IntImm(M_padded), IntImm(N_padded)});
  }

  return info;
}
```

### Step 4: Metadata Attachment
```cpp
Buffer AttachShardMetadata(Buffer buffer, const ShardInfo& info) {
  auto n = make_object<BufferNode>(*buffer.get());

  n->attrs.Set("tt_layout", String("dram_interleaved"));
  n->attrs.Set("tt_tile_shape", Array<IntImm>({32, 32}));
  n->attrs.Set("tt_num_tiles_height", IntImm(info.tiles_height));
  n->attrs.Set("tt_num_tiles_width", IntImm(info.tiles_width));
  n->attrs.Set("tt_needs_padding", Bool(info.needs_padding));

  if (info.needs_padding) {
    n->attrs.Set("tt_padded_shape", info.padded_shape);
  }

  return Buffer(n);
}
```

## Testing Strategy

### C++ Unit Test
**File:** `tests/cpp/tt/test_infer_tt_shard.cc`

```cpp
TEST(InferTTShard, AlignedDimensions) {
  // Create PrimFunc with buffers: A[256, 256], B[256, 256], C[256, 256]
  // Apply InferDefaultTTShard pass
  // Verify:
  //   - tt_num_tiles_height = 8 (256/32)
  //   - tt_num_tiles_width = 8
  //   - tt_needs_padding = false
  //   - No tt_padded_shape attribute
}

TEST(InferTTShard, NonAlignedDimensions) {
  // Create PrimFunc with buffers: A[100, 100], B[100, 100], C[100, 100]
  // Apply InferDefaultTTShard pass
  // Verify:
  //   - tt_num_tiles_height = 4 (ceil(100/32) = 4)
  //   - tt_num_tiles_width = 4
  //   - tt_needs_padding = true
  //   - tt_padded_shape = [128, 128] (4*32 = 128)
}

TEST(InferTTShard, RectangularMatrix) {
  // Create PrimFunc with A[512, 256], B[256, 128], C[512, 128]
  // Verify correct tile counts for non-square matrices
}

TEST(InferTTShard, InterleavedLayout) {
  // Verify tt_layout = "dram_interleaved" set correctly
  // (Actual TensorAccessor logic deferred to WS4 codegen)
}
```

## DRAM Interleaved Layout

### What It Means
- Tiles distributed across DRAM banks/channels
- TT-metal `TensorAccessor` handles address calculation
- Avoids manual swizzling in generated kernels

### MVP Approach
For WS2, we only attach **metadata** indicating interleaved layout.

Actual address calculation deferred to **WS4 codegen** where we:
- Generate reader/writer kernels using TensorAccessor API
- Let TT runtime handle interleaving details

### Metadata Format (MVP)
```cpp
// Simple metadata for WS2:
buffer->attrs["tt_layout"] = "dram_interleaved"
buffer->attrs["tt_tile_shape"] = [32, 32]

// WS4 will expand this to full TensorAccessor config
```

## Padding Strategy

### When Padding Needed
- Matrix dimension not multiple of 32
- Example: M=100 → need 4 tiles (128 elements), pad with 28 zeros

### Where Padding Happens
- **WS2:** Only **detect** and **mark** padding requirements
- **WS3:** `TilePadTT` pass will **insert** actual padding operations
- **WS4:** Reader kernels will **implement** zero-fill for padded regions

### Metadata for Padding
```cpp
// WS2 attaches:
buffer->attrs["tt_needs_padding"] = true
buffer->attrs["tt_padded_shape"] = [128, 128]  // Padded dimensions

// WS3 uses this to insert pad ops
// WS4 uses this to generate zero-fill in reader
```

## TVM Integration Points

### 1. Buffer Access
```cpp
// Access buffer parameters
for (const auto& param : func->params) {
  Buffer buffer = func->buffer_map[param];
  // ... process buffer
}
```

### 2. Shape Extraction
```cpp
// Get buffer shape
Array<PrimExpr> shape = buffer->shape;

// For static shapes, extract integer values
const auto* imm = shape[0].as<IntImmNode>();
int dim0 = imm->value;
```

### 3. Buffer Mutation
```cpp
// Buffers are immutable, create new one with metadata
auto n = make_object<BufferNode>(*buffer.get());
n->attrs.Set("key", value);
Buffer new_buffer(n);
```

## Open Questions

1. **TensorAccessor headers:** Do we need TT-metal headers in WS2?
   - **Recommendation:** No, defer to WS4 codegen. WS2 only attaches metadata.

2. **Batched operations:** How to handle 3D/4D tensors?
   - **Recommendation:** MVP supports 2D only. Add TODO for batched.

3. **Different tile sizes:** Will TT support non-32×32 tiles?
   - **Recommendation:** Hardcode 32×32 for MVP. Make configurable later.

4. **Mixed precision:** bf16, fp16, fp32, int8?
   - **Recommendation:** MVP uses bf16. Dtype agnostic for WS2 (just count tiles).

## Acceptance Criteria

- ✅ C++ pass implemented and registered with TVM
- ✅ Pass correctly identifies padding needs
- ✅ C++ unit tests pass (aligned, non-aligned, rectangular)
- ✅ Metadata correctly attached to buffer parameters
- ✅ Integration with schedule inference pass (can run sequentially)

## Next Steps

After shard inference complete:
1. Add Python bindings for both schedule and shard passes
2. Create Python integration test combining both passes
3. Begin WS3 TIR transform pipeline (depends on WS2 metadata)

## References

- [Unified MVP Plan](../UNIFIED_MATMUL_MVP_PLAN.md) - WS2 specification
- [WS2 Status](WS2_STATUS.md) - Overall WS2 progress
- [Schedule Inference](ws2_schedule_inference.md) - Prerequisite pass
- TT-metal TensorAccessor docs (when available)
