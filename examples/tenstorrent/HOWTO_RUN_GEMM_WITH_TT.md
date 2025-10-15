# How to Run example_gemm.py with Tenstorrent Backend

## Quick Start: Two Simple Changes

To run the original `examples/gemm/example_gemm.py` with the Tenstorrent backend, you only need **TWO changes**:

### 1. Import the TT target
```python
from tilelang.utils.target import TENSTORRENT_TARGET
```

### 2. Add target parameter to @tilelang.jit
```python
@tilelang.jit(target=TENSTORRENT_TARGET, out_idx=[-1])
#             ^^^^^^^^^^^^^^^^^^^^^^^^
#             Add this parameter!
```

That's it! Your kernel will now generate TT artifacts instead of CUDA code.

## Complete Example

Here's the modified `example_gemm.py`:

```python
import tilelang
import tilelang.language as T
from tilelang.utils.target import TENSTORRENT_TARGET  # CHANGE 1

@tilelang.jit(target=TENSTORRENT_TARGET, out_idx=[-1])  # CHANGE 2
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def gemm(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm

# Create kernel with TT-friendly tile sizes (32x32)
kernel = matmul(256, 256, 256, 32, 32, 32)
```

## What Happens with TT Backend?

When you use `target=TENSTORRENT_TARGET`:

1. **No CUDA execution** - Instead of running on GPU, the kernel generates TT artifacts
2. **5 artifacts generated**:
   - `reader.cpp` - Loads tiles from DRAM to L1
   - `compute.cpp` - Performs the matmul computation
   - `writer.cpp` - Stores results back to DRAM
   - `main.cpp` - Host coordination program
   - `tt.plan.json` - Runtime execution plan

3. **DSL mappings** (automatic):
   - `T.alloc_shared` → L1 circular buffers
   - `T.alloc_fragment` → Tile registers
   - `T.clear` → Register initialization
   - `T.Pipelined` → Persistent loop with double buffering
   - `T.copy` → NOC DMA operations
   - `T.gemm` → TT matmul_tiles intrinsic

4. **Parameters handled**:
   - `threads=128` is ignored (TT uses tile-level parallelism)
   - `num_stages` becomes double-buffering in TT

## Inspecting Generated Artifacts

```python
import json

# Get the generated artifacts
source = kernel.get_kernel_source()
artifacts = json.loads(source)

# List all artifacts
for name in artifacts:
    print(f"Generated: {name}")

# Examine specific artifact (e.g., compute kernel)
compute_code = artifacts["compute.cpp"]
print(compute_code)
```

## Best Practices for TT

1. **Use 32x32 tile sizes** - This is TT's native tile size
2. **Matrix dimensions should be multiples of 32** for best performance
3. **The kernel runs in simulation mode** - actual execution requires TT hardware or SDK

## Running with TT Hardware

Once you have TT hardware/SDK:
```python
import torch

# Create tensors (no .cuda() needed)
a = torch.randn(256, 256).half()
b = torch.randn(256, 256).half()

# Execute on TT hardware
c = kernel(a, b)
```

## Summary

Converting any TileLang kernel to TT backend is as simple as:
1. Import `TENSTORRENT_TARGET`
2. Add `target=TENSTORRENT_TARGET` to `@tilelang.jit`

The rest is handled automatically! Your TileLang DSL code remains the same, and all features are mapped to TT's architecture transparently.