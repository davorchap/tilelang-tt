# Task: Python Bindings and Integration Tests

## Goal
Expose WS2 C++ passes to Python and create comprehensive integration tests that validate the end-to-end metadata inference pipeline.

## Context
- **Workstream:** WS2 - Schedule & Sharding Metadata
- **Dependencies:** Schedule and shard inference passes implemented (C++)
- **Files:**
  - `tilelang/tt/__init__.py` (bindings)
  - `tilelang/tt/passes.py` (pass wrappers)
  - `testing/python/tt/test_inferred_metadata.py` (integration tests)
- **Priority:** High (enables Python-level testing and WS3 integration)

---

## Part 1: Python Bindings

### Goal
Make C++ passes callable from Python via TVM's FFI system.

### Implementation

#### File: `tilelang/tt/passes.py`

```python
"""TVM passes for Tenstorrent backend metadata inference."""

from typing import Union
import tvm
from tvm import tir
from tvm.ir import IRModule


def infer_default_tt_schedule(mod: Union[IRModule, tir.PrimFunc]) -> Union[IRModule, tir.PrimFunc]:
    """Infer default Tenstorrent schedule metadata.

    This pass computes contiguous per-core tile ranges and runtime
    argument schemas based on the kernel grid dimensions.

    Args:
        mod: The IRModule or PrimFunc to process

    Returns:
        Enhanced IRModule/PrimFunc with schedule metadata attached

    Example:
        >>> mod = create_gemm_module(M=256, N=256, K=256)
        >>> mod = apply_tt_defaults(mod)  # WS1
        >>> mod = infer_default_tt_schedule(mod)  # WS2
        >>> func = mod["main"]
        >>> assert "tt_num_tiles" in func.attrs
    """
    return _ffi_api.InferDefaultTTSchedule(mod)


def infer_default_tt_shard(mod: Union[IRModule, tir.PrimFunc]) -> Union[IRModule, tir.PrimFunc]:
    """Infer default Tenstorrent sharding metadata.

    This pass generates DRAM interleaved layout descriptors and
    identifies padding requirements for buffer parameters.

    Args:
        mod: The IRModule or PrimFunc to process

    Returns:
        Enhanced IRModule/PrimFunc with sharding metadata attached

    Example:
        >>> mod = create_gemm_module(M=256, N=256, K=256)
        >>> mod = apply_tt_defaults(mod)  # WS1
        >>> mod = infer_default_tt_schedule(mod)  # WS2
        >>> mod = infer_default_tt_shard(mod)  # WS2
        >>> # Now mod has full schedule + shard metadata
    """
    return _ffi_api.InferDefaultTTShard(mod)


# FFI API registration
_ffi_api = tvm._ffi.register_module("tir.transform")
```

#### File: `tilelang/tt/__init__.py` (update)

```python
"""Tenstorrent-specific TileLang utilities and helpers."""

from .target import apply_tt_defaults
from .passes import infer_default_tt_schedule, infer_default_tt_shard

__all__ = [
    "apply_tt_defaults",
    "infer_default_tt_schedule",
    "infer_default_tt_shard",
]
```

### FFI Registration in C++

The passes are already registered via `TVM_REGISTER_GLOBAL` in the C++ files:

```cpp
// In src/tt/transform/infer_tt_schedule.cc
TVM_REGISTER_GLOBAL("tir.transform.InferDefaultTTSchedule")
.set_body_typed(InferDefaultTTSchedule);

// In src/tt/transform/infer_tt_shard.cc
TVM_REGISTER_GLOBAL("tir.transform.InferDefaultTTShard")
.set_body_typed(InferDefaultTTShard);
```

Python accesses these via `_ffi_api.InferDefaultTTSchedule(...)`.

### Testing the Bindings

```python
# Quick smoke test
import tvm
from tilelang.tt import infer_default_tt_schedule, infer_default_tt_shard

# Can import without errors
assert callable(infer_default_tt_schedule)
assert callable(infer_default_tt_shard)
```

---

## Part 2: Integration Tests

### Goal
Validate end-to-end WS1 + WS2 pipeline on realistic TileLang examples.

### File: `testing/python/tt/test_inferred_metadata.py`

```python
"""Integration tests for WS2 metadata inference passes."""

import pytest
import tvm
from tvm import tir

try:
    from tilelang.tt import apply_tt_defaults, infer_default_tt_schedule, infer_default_tt_shard
except ImportError as e:
    pytest.skip(f"TileLang TT backend not available: {e}", allow_module_level=True)


def create_simple_gemm_primfunc(M=256, N=256, K=256):
    """Create a simple GEMM PrimFunc for testing.

    This is a minimal version for metadata testing - not a full TileLang GEMM.
    """
    # Create buffer parameters
    A = tir.decl_buffer((M, K), "float16", name="A")
    B = tir.decl_buffer((K, N), "float16", name="B")
    C = tir.decl_buffer((M, N), "float16", name="C")

    # Create a minimal PrimFunc with T.Kernel metadata
    # (Actual kernel body not important for metadata testing)
    block_size = 32
    grid_x = (N + block_size - 1) // block_size
    grid_y = (M + block_size - 1) // block_size

    func = tir.PrimFunc(
        params=[A, B, C],
        body=tir.Evaluate(0),  # Dummy body for testing
        buffer_map={},
    )

    # Add T.Kernel metadata (simulating TileLang lowering)
    func = func.with_attr("tl_grid_x", grid_x)
    func = func.with_attr("tl_grid_y", grid_y)
    func = func.with_attr("global_symbol", "gemm")

    return func


class TestScheduleInference:
    """Tests for schedule inference pass."""

    def test_schedule_inference_aligned_grid(self):
        """Test schedule inference on tile-aligned 256x256 GEMM."""
        func = create_simple_gemm_primfunc(M=256, N=256, K=256)

        # Apply WS1 defaults
        func = apply_tt_defaults(tvm.IRModule({"main": func}))["main"]

        # Apply WS2 schedule inference
        func = infer_default_tt_schedule(func)

        # Verify schedule metadata
        assert "tt_num_tiles" in func.attrs
        assert "tt_grid_x" in func.attrs
        assert "tt_grid_y" in func.attrs
        assert "tt_tiles_per_core" in func.attrs

        # Grid: 256/32 = 8x8 = 64 tiles
        assert func.attrs["tt_num_tiles"] == 64
        assert func.attrs["tt_grid_x"] == 8
        assert func.attrs["tt_grid_y"] == 8

        # With 64 cores, each core gets 1 tile
        tiles_per_core = func.attrs["tt_tiles_per_core"]
        assert len(tiles_per_core) == 64
        # Each core: [start_id, count=1]
        assert all(range_info[1] == 1 for range_info in tiles_per_core)

    def test_schedule_inference_uneven_distribution(self):
        """Test schedule inference with more tiles than cores."""
        # 512x512 grid = 16x16 = 256 tiles, 64 cores
        func = create_simple_gemm_primfunc(M=512, N=512, K=512)
        func = apply_tt_defaults(tvm.IRModule({"main": func}))["main"]
        func = infer_default_tt_schedule(func)

        assert func.attrs["tt_num_tiles"] == 256

        tiles_per_core = func.attrs["tt_tiles_per_core"]
        assert len(tiles_per_core) == 64

        # 256 tiles / 64 cores = 4 tiles per core
        assert all(range_info[1] == 4 for range_info in tiles_per_core)

    def test_schedule_inference_single_tile(self):
        """Test schedule inference with single tile (1x1 grid)."""
        func = create_simple_gemm_primfunc(M=32, N=32, K=32)
        func = apply_tt_defaults(tvm.IRModule({"main": func}))["main"]
        func = infer_default_tt_schedule(func)

        assert func.attrs["tt_num_tiles"] == 1

        tiles_per_core = func.attrs["tt_tiles_per_core"]
        # Only core 0 active
        assert tiles_per_core[0][1] == 1  # count=1
        # Other cores inactive
        assert all(range_info[1] == 0 for range_info in tiles_per_core[1:])


class TestShardInference:
    """Tests for sharding inference pass."""

    def test_shard_inference_aligned_dimensions(self):
        """Test shard inference on tile-aligned matrices."""
        func = create_simple_gemm_primfunc(M=256, N=256, K=256)
        func = apply_tt_defaults(tvm.IRModule({"main": func}))["main"]
        func = infer_default_tt_schedule(func)
        func = infer_default_tt_shard(func)

        # Check buffer metadata (A, B, C)
        for param_name in ["A", "B", "C"]:
            buffer = func.buffer_map[func.params[param_name]]
            assert "tt_layout" in buffer.attrs
            assert buffer.attrs["tt_layout"] == "dram_interleaved"
            assert "tt_tile_shape" in buffer.attrs
            assert buffer.attrs["tt_tile_shape"] == [32, 32]
            assert "tt_needs_padding" in buffer.attrs
            assert buffer.attrs["tt_needs_padding"] == False  # 256 is multiple of 32

    def test_shard_inference_nonaligned_dimensions(self):
        """Test shard inference on non-tile-aligned matrices."""
        func = create_simple_gemm_primfunc(M=100, N=100, K=100)
        func = apply_tt_defaults(tvm.IRModule({"main": func}))["main"]
        func = infer_default_tt_schedule(func)
        func = infer_default_tt_shard(func)

        # Check buffer A [100, 100]
        buffer_A = func.buffer_map[func.params["A"]]
        assert buffer_A.attrs["tt_needs_padding"] == True
        assert "tt_padded_shape" in buffer_A.attrs
        # 100 → ceil(100/32) = 4 tiles → 128 padded
        assert buffer_A.attrs["tt_padded_shape"] == [128, 128]
        assert buffer_A.attrs["tt_num_tiles_height"] == 4
        assert buffer_A.attrs["tt_num_tiles_width"] == 4

    def test_shard_inference_rectangular_matrix(self):
        """Test shard inference on rectangular (non-square) matrices."""
        func = create_simple_gemm_primfunc(M=512, N=256, K=128)
        func = apply_tt_defaults(tvm.IRModule({"main": func}))["main"]
        func = infer_default_tt_schedule(func)
        func = infer_default_tt_shard(func)

        # A: [512, 128] → [16, 4] tiles
        buffer_A = func.buffer_map[func.params["A"]]
        assert buffer_A.attrs["tt_num_tiles_height"] == 16
        assert buffer_A.attrs["tt_num_tiles_width"] == 4

        # B: [128, 256] → [4, 8] tiles
        buffer_B = func.buffer_map[func.params["B"]]
        assert buffer_B.attrs["tt_num_tiles_height"] == 4
        assert buffer_B.attrs["tt_num_tiles_width"] == 8

        # C: [512, 256] → [16, 8] tiles
        buffer_C = func.buffer_map[func.params["C"]]
        assert buffer_C.attrs["tt_num_tiles_height"] == 16
        assert buffer_C.attrs["tt_num_tiles_width"] == 8


class TestEndToEndPipeline:
    """Test complete WS1 + WS2 pipeline."""

    def test_full_metadata_pipeline(self):
        """Test complete pipeline: defaults → schedule → shard."""
        # Start with raw PrimFunc
        func = create_simple_gemm_primfunc(M=256, N=256, K=256)
        mod = tvm.IRModule({"main": func})

        # WS1: Apply defaults
        mod = apply_tt_defaults(mod)
        func = mod["main"]
        assert "tt_schedule_policy" in func.attrs
        assert "tt_layout_type" in func.attrs

        # WS2: Infer schedule
        mod = infer_default_tt_schedule(mod)
        func = mod["main"]
        assert "tt_num_tiles" in func.attrs
        assert "tt_tiles_per_core" in func.attrs

        # WS2: Infer shard
        mod = infer_default_tt_shard(mod)
        func = mod["main"]

        # Verify all metadata present
        assert "tt_schedule_policy" in func.attrs  # WS1
        assert "tt_num_tiles" in func.attrs  # WS2 schedule
        for param in func.params:
            buffer = func.buffer_map[param]
            assert "tt_layout" in buffer.attrs  # WS2 shard

    def test_pipeline_preserves_existing_metadata(self):
        """Test that pipeline preserves earlier metadata."""
        func = create_simple_gemm_primfunc(M=128, N=128, K=128)
        mod = tvm.IRModule({"main": func})

        # Apply full pipeline
        mod = apply_tt_defaults(mod)
        mod = infer_default_tt_schedule(mod)
        mod = infer_default_tt_shard(mod)

        func = mod["main"]

        # Verify WS1 defaults still present after WS2
        assert func.attrs["tt_schedule_policy"] == "contiguous"
        assert func.attrs["tt_layout_type"] == "dram_interleaved"

        # Verify WS2 schedule still present after shard
        assert "tt_num_tiles" in func.attrs
        assert "tt_tiles_per_core" in func.attrs
```

### Test Coverage Matrix

| Test | Grid Size | Cores | Tiles/Core | Padding | Purpose |
|------|-----------|-------|------------|---------|---------|
| aligned_grid | 8x8 (64 tiles) | 64 | 1 | No | Even distribution |
| uneven_distribution | 16x16 (256 tiles) | 64 | 4 | No | Load balancing |
| single_tile | 1x1 (1 tile) | 64 | 0-1 | No | Minimal case |
| nonaligned_dimensions | 100x100 | - | - | Yes | Padding detection |
| rectangular_matrix | 512x256 | - | - | No | Non-square |
| full_pipeline | 256x256 | 64 | 1 | No | End-to-end |

---

## Running the Tests

### Local execution:
```bash
source .venv/bin/activate
export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH
pytest testing/python/tt/test_inferred_metadata.py -v
```

### CI integration:
Tests automatically run via Tenstorrent CI workflow:
```yaml
# In .github/workflows/tenstorrent-ci.yml
- name: Run Tenstorrent metadata inference tests
  run: |
    export LD_LIBRARY_PATH=$(pwd)/build/tvm:$LD_LIBRARY_PATH
    pytest testing/python/tt/test_inferred_metadata.py -v --tb=short
```

---

## Acceptance Criteria

### Bindings
- ✅ `infer_default_tt_schedule` callable from Python
- ✅ `infer_default_tt_shard` callable from Python
- ✅ Passes work on both `IRModule` and `PrimFunc`
- ✅ Proper error handling for invalid inputs

### Tests
- ✅ All test classes pass (Schedule, Shard, EndToEnd)
- ✅ Tests cover edge cases (1x1, large grids, non-aligned)
- ✅ Full pipeline test validates WS1 + WS2 integration
- ✅ Tests run successfully in CI

---

## Next Steps

After Python integration complete:
1. Update CI to run WS2 tests
2. Update CLAUDE.md with WS2 completion status
3. Begin WS3: TIR Transform Pipeline (depends on WS2 metadata)

---

## References

- [WS2 Status](WS2_STATUS.md) - Overall WS2 progress
- [Schedule Inference](ws2_schedule_inference.md) - C++ pass details
- [Shard Inference](ws2_shard_inference.md) - C++ pass details
- [Project Plan](../project_1.md) - WS2 specification
