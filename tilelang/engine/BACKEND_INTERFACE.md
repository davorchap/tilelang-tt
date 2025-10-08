# Backend Interface Guide

**Purpose:** This document defines the unified lowering architecture for TileLang backends, making it easy to add new hardware targets while maximizing code sharing.

**Status:** Task 8 - Unified Architecture Complete

**Supported Backends:** CUDA, HIP (ROCm), Tenstorrent, C, LLVM, WebGPU

---

## Architecture Overview

TileLang uses a **phased lowering pipeline** with shared and backend-specific components:

```
┌─────────────────────────────────────────────────────────┐
│                    TileLang DSL                         │
│                    (Python API)                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│              Phase 1: Frontend Lowering                 │
│              (Shared Across All Backends)               │
│  - LowerAndLegalize(mod, target)                        │
│    • LetInline, AddWrapperForSingleBufStore             │
│    • InjectAssumes, Simplify                            │
│    • LayoutReducer, LayoutInference                     │
│    • LowerTileOp, LowerL2Persistent                     │
│    • LegalizeVectorizedLoop, LegalizeSafeMemoryAccess   │
│    • LoopVectorizeDynamic                               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
         ┌───────────┴────────────┐
         │  Backend Selection     │
         │  (Based on Target)     │
         └───────────┬────────────┘
                     │
         ┌───────────┴────────────┐
         │                        │
         ↓                        ↓
┌─────────────────┐      ┌─────────────────────┐
│  CUDA Backend   │      │  Tenstorrent        │
│                 │      │  Backend            │
│ OptimizeFor     │      │  OptimizeFor        │
│ Target()        │      │  TargetTT()         │
│                 │      │                     │
│ - WarpSpecial   │      │  - WS2 Inference    │
│ - TMA Lowering  │      │  - WS3 Transforms   │
│ - SharedMem     │      │  - Common Opts      │
│ - Common Opts   │      │  - TT Verification  │
│ - SplitHost     │      │  - SplitTTKernels   │
│   Device        │      │                     │
└────────┬────────┘      └──────────┬──────────┘
         │                          │
         ↓                          ↓
┌─────────────────┐      ┌─────────────────────┐
│  CUDA Codegen   │      │  TT Codegen         │
│  (PTX/CUBIN)    │      │  (3 Kernels +       │
│                 │      │   Host Program)     │
└─────────────────┘      └─────────────────────┘
```

---

## Backend Interface Contract

To add a new backend, implement the following interface:

### Required Components

#### 1. Backend-Specific Lowering Function

**Location:** `tilelang/engine/{backend}/lower.py`

**Signature:**
```python
def lower(
    mod: IRModule,
    params: Optional[List[KernelParam]],
    target: Union[str, Target],
    target_host: Optional[Union[str, Target]],
    *,
    runtime_only: bool,
    enable_host_codegen: bool,
    enable_device_compile: bool,
) -> CompiledArtifact:
    """Lower module for {backend} backend.

    Args:
        mod: TVM IRModule to lower
        params: Optional kernel parameters
        target: Target device (should be {backend} target)
        target_host: Optional host target
        runtime_only: Whether to generate runtime-only code
        enable_host_codegen: Whether to enable host code generation
        enable_device_compile: Whether to enable device compilation

    Returns:
        CompiledArtifact with host_mod, device_mod, params, kernel_source
    """
```

**Template:**
```python
def lower(mod, params, target, target_host, **kwargs) -> CompiledArtifact:
    # Step 1: Apply backend defaults (if needed)
    mod = apply_{backend}_defaults(mod)

    # Step 2: Frontend lowering (SHARED - use as-is)
    with target:
        mod = LowerAndLegalize(mod, target)

    # Step 3: Backend-specific optimizations (IMPLEMENT THIS)
    mod = OptimizeFor{Backend}(mod, target)

    # Step 4: Device splitting (IMPLEMENT THIS)
    device_mod, host_mod = Split{Backend}Kernels(mod)

    # Step 5: Generate kernel source (IMPLEMENT THIS)
    kernel_source = generate_{backend}_code(device_mod)

    # Step 6: Return compiled artifact
    return CompiledArtifact(
        host_mod=host_mod,
        device_mod=device_mod,
        params=params or [],
        kernel_source=kernel_source,
        rt_mod=None,  # Optional runtime module
    )
```

#### 2. Target-Specific Optimization Phase

**Location:** Can be in `tilelang/engine/phase.py` or `tilelang/engine/{backend}/`

**Signature:**
```python
def OptimizeFor{Backend}(mod: IRModule, target: Target) -> IRModule:
    """Apply backend-specific optimizations.

    This function should:
    1. Apply backend-specific metadata inference
    2. Apply backend-specific transformations
    3. Apply common optimizations (shared with other backends)
    4. Verify backend-specific constraints

    Returns transformed IRModule ready for codegen.
    """
```

**Pattern:**
```python
def OptimizeFor{Backend}(mod: IRModule, target: Target) -> IRModule:
    # === Backend-Specific Metadata & Transforms ===
    mod = backend_specific_pass_1(mod)
    mod = backend_specific_pass_2(mod)
    # ...

    # === Common Optimizations (SHARED) ===
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tir.transform.Simplify()(mod)

    pass_ctx = tilelang.transform.get_pass_context()
    from tilelang.engine.phase import allow_vectorize
    mod = tilelang.transform.VectorizeLoop(
        enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)

    mod = tilelang.transform.StorageRewrite()(mod)
    mod = tir.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.RemoveNoOp()(mod)
    mod = tir.transform.RewriteUnsafeSelect()(mod)
    mod = tir.transform.HoistIfThenElse()(mod)

    # === Backend-Specific Verification ===
    mod = tir.transform.VerifyMemory()(mod)
    mod = verify_{backend}_ir(mod)  # Backend-specific validation

    return mod
```

#### 3. Device Splitting Function

**Signature:**
```python
def Split{Backend}Kernels(mod: IRModule) -> Tuple[IRModule, IRModule]:
    """Split module into device and host components.

    Returns:
        Tuple of (device_mod, host_mod)
    """
```

**Examples:**

**CUDA Pattern:**
```python
def SplitCUDAKernels(mod: IRModule) -> Tuple[IRModule, IRModule]:
    # Annotate device regions
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)

    # Split into host and device
    mod = tir.transform.SplitHostDevice()(mod)

    # Filter functions
    _is_host_call = lambda f: not has_device_kernel_launch(f.attrs)
    _is_device_call = lambda f: has_device_kernel_launch(f.attrs)

    host_mod = tir.transform.Filter(_is_host_call)(mod)
    device_mod = tir.transform.Filter(_is_device_call)(mod)

    return device_mod, host_mod
```

**Tenstorrent Pattern:**
```python
def SplitTTKernels(mod: IRModule) -> Tuple[IRModule, IRModule]:
    # Annotate device regions (same as CUDA)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)

    # NOTE: TT's 3-kernel split happens during CODEGEN, not IR transformation
    # So we return the same module for both device and host
    # The codegen visitors will generate reader/compute/writer kernels
    return mod, mod
```

#### 4. Register Backend in Main Lowering Entry Point

**Location:** `tilelang/engine/lower.py`

**Pattern:**
```python
from tilelang.utils.target import {BACKEND}_TARGET
from tilelang.engine.{backend} import lower_{backend}

def lower(func_or_mod, target="auto", ...):
    # ... (existing setup code)

    if get_target_kind(target) == {BACKEND}_TARGET:
        return lower_{backend}(
            mod, params, target, target_host,
            runtime_only=runtime_only,
            enable_host_codegen=enable_host_codegen,
            enable_device_compile=enable_device_compile,
        )

    # ... (continue with default CUDA/HIP flow)
```

---

## Shared Components (Use These!)

### Frontend Lowering (Phase 1)

**Function:** `tilelang.engine.phase.LowerAndLegalize(mod, target)`

**What it does:**
- Binds target information to module
- Inlines let expressions
- Adds safety wrappers
- Simplifies expressions
- **Infers memory layouts** (CRITICAL)
- **Lowers high-level tile operations** (CRITICAL)
- Legalizes vectorized loops
- Adds memory safety checks
- Attempts dynamic loop vectorization

**Usage:** Always call this first after applying backend defaults!

```python
# In your backend's lower() function:
with target:
    mod = LowerAndLegalize(mod, target)
```

### Common Optimizations

**Functions:** Available in `tilelang.transform` and `tir.transform`

**Include these in OptimizeFor{Backend}():**

```python
# Buffer and index optimization
mod = tilelang.transform.FlattenBuffer()(mod)
mod = tilelang.transform.ConfigIndexBitwidth()(mod)

# Expression simplification
mod = tir.transform.Simplify()(mod)

# Vectorization (respects PassContext settings)
pass_ctx = tilelang.transform.get_pass_context()
from tilelang.engine.phase import allow_vectorize
mod = tilelang.transform.VectorizeLoop(
    enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)

# Storage and loop optimization
mod = tilelang.transform.StorageRewrite()(mod)
mod = tir.transform.UnrollLoop()(mod)
mod = tir.transform.RenormalizeSplitPattern()(mod)

# Cleanup
mod = tir.transform.Simplify()(mod)
mod = tir.transform.RemoveNoOp()(mod)
mod = tir.transform.RewriteUnsafeSelect()(mod)
mod = tir.transform.HoistIfThenElse()(mod)

# Verification
mod = tir.transform.VerifyMemory()(mod)
```

---

## Reference Implementations

### CUDA Backend (Reference)

**Location:** `tilelang/engine/lower.py` (lines 248-267)

**Pipeline:**
1. `LowerAndLegalize(mod, target)` - Frontend (shared)
2. `OptimizeForTarget(mod, target)` - CUDA-specific + common opts
3. Filter into host/device modules
4. Codegen

**Key Files:**
- `tilelang/engine/phase.py::OptimizeForTarget()` - CUDA optimizations

### Tenstorrent Backend (Reference)

**Location:** `tilelang/engine/tt/lower.py`

**Pipeline:**
1. `apply_tt_defaults(mod)` - TT defaults
2. `LowerAndLegalize(mod, target)` - Frontend (shared)
3. `OptimizeForTargetTT(mod, target)` - TT-specific + common opts
4. `SplitTTKernels(mod)` - Device annotation
5. Return CompiledArtifact

**Key Files:**
- `tilelang/engine/tt/lower.py::OptimizeForTargetTT()` - TT optimizations
- `tilelang/engine/tt/lower.py::SplitTTKernels()` - Device annotation

---

## Adding a New Backend: Step-by-Step

### Step 1: Create Backend Directory

```bash
mkdir -p tilelang/engine/mybackend
touch tilelang/engine/mybackend/__init__.py
touch tilelang/engine/mybackend/lower.py
```

### Step 2: Register Target

In `tilelang/utils/target.py`:

```python
MYBACKEND_TARGET = "mybackend"

def target_available(target_name: str) -> bool:
    # ... existing code ...
    if target_name == MYBACKEND_TARGET:
        return os.environ.get("TILELANG_MYBACKEND_ENABLED", "0") == "1"
```

### Step 3: Implement Backend Lowering

In `tilelang/engine/mybackend/lower.py`:

```python
from typing import List, Optional, Union, Tuple
from tvm.target import Target
from tilelang import tvm as tvm
from tilelang.engine.param import CompiledArtifact, KernelParam
from tilelang.engine.phase import LowerAndLegalize
import tilelang.transform
from tvm import tir

def OptimizeForMyBackend(mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    """Apply MyBackend-specific optimizations."""

    # === MyBackend-Specific Transforms ===
    # TODO: Add your backend-specific passes here
    # mod = my_backend_pass_1(mod)
    # mod = my_backend_pass_2(mod)

    # === Common Optimizations (COPY FROM TT OR CUDA) ===
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tir.transform.Simplify()(mod)

    pass_ctx = tilelang.transform.get_pass_context()
    from tilelang.engine.phase import allow_vectorize
    mod = tilelang.transform.VectorizeLoop(
        enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)

    mod = tilelang.transform.StorageRewrite()(mod)
    mod = tir.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.RemoveNoOp()(mod)
    mod = tir.transform.RewriteUnsafeSelect()(mod)
    mod = tir.transform.HoistIfThenElse()(mod)

    mod = tir.transform.VerifyMemory()(mod)

    return mod

def SplitMyBackendKernels(mod: tvm.IRModule) -> Tuple[tvm.IRModule, tvm.IRModule]:
    """Prepare module for MyBackend codegen."""

    # Annotate device regions
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)

    # TODO: Add your device splitting logic here
    # Option 1: Split in IR (like CUDA)
    # Option 2: Keep intact, split in codegen (like TT)

    return mod, mod

def lower(
    mod: tvm.IRModule,
    params: Optional[List[KernelParam]],
    target: Union[str, Target],
    target_host: Optional[Union[str, Target]],
    *,
    runtime_only: bool,
    enable_host_codegen: bool,
    enable_device_compile: bool,
) -> CompiledArtifact:
    """Lower module for MyBackend."""

    # Validate target
    from tilelang.engine.lower import get_target_kind
    from tilelang.utils.target import MYBACKEND_TARGET

    target_kind = get_target_kind(target)
    if target_kind != MYBACKEND_TARGET:
        raise ValueError(f"MyBackend lowering called with invalid target: {target_kind}")

    # Convert to Target object
    if isinstance(target, str):
        target = tvm.target.Target(target)

    # Create composite target with host
    if target_host is not None:
        if isinstance(target_host, str):
            target_host = tvm.target.Target(target_host)
        target = tvm.target.Target(target, target_host)

    # === Phase 1: Frontend Lowering (SHARED) ===
    with target:
        mod = LowerAndLegalize(mod, target)

    # === Phase 2: Backend-Specific Optimizations ===
    mod = OptimizeForMyBackend(mod, target)

    # === Phase 3: Device Splitting ===
    device_mod, host_mod = SplitMyBackendKernels(mod)

    # === Phase 4: Generate Kernel Source ===
    kernel_source = "// MyBackend kernel placeholder\n"
    # TODO: Implement your codegen here

    # === Phase 5: Return Compiled Artifact ===
    return CompiledArtifact(
        host_mod=host_mod,
        device_mod=device_mod,
        params=params or [],
        kernel_source=kernel_source,
        rt_mod=None,
    )
```

### Step 4: Register in Main Lowering

In `tilelang/engine/lower.py`:

```python
from tilelang.utils.target import MYBACKEND_TARGET
from tilelang.engine.mybackend import lower as lower_mybackend

def lower(func_or_mod, target="auto", ...):
    # ... (existing setup)

    if get_target_kind(target) == MYBACKEND_TARGET:
        return lower_mybackend(
            mod, params, target, target_host,
            runtime_only=runtime_only,
            enable_host_codegen=enable_host_codegen,
            enable_device_compile=enable_device_compile,
        )

    # ... (continue with existing flow)
```

### Step 5: Add Tests

Create `testing/python/mybackend/test_lowering.py`:

```python
import pytest
import tilelang.language as T
from tilelang.engine import lower
from tilelang.utils.target import MYBACKEND_TARGET

def test_mybackend_lowering():
    @T.prim_func
    def simple_kernel(A: T.Buffer((32, 32), "float32")):
        with T.Kernel(1, 1) as (bx, by):
            for i, j in T.Parallel(32, 32):
                A[i, j] = 0.0

    mod = T.IRModule({"main": simple_kernel})

    result = lower(
        mod,
        params=None,
        target=MYBACKEND_TARGET,
        target_host=None,
        runtime_only=False,
        enable_host_codegen=True,
        enable_device_compile=False,
    )

    assert result is not None
    assert result.host_mod is not None
    assert result.device_mod is not None
```

---

## Code Sharing Summary

### What's Shared (60-70%)

✅ **Frontend Lowering** - `LowerAndLegalize()` (15+ passes)
✅ **Common Optimizations** - FlattenBuffer, UnrollLoop, etc. (11 passes)
✅ **Device Annotation** - `AnnotateDeviceRegions()`
✅ **Verification** - `VerifyMemory()`
✅ **Parameter Extraction** - `extrac_params()`
✅ **Compiled Artifact** - `CompiledArtifact` structure

### What's Backend-Specific (30-40%)

🔧 **Target-Specific Transforms** - Hardware-specific IR transformations
🔧 **Memory Model** - CUDA shared memory vs TT circular buffers
🔧 **Device Splitting** - How to split host/device code
🔧 **Codegen** - Final code generation
🔧 **Intrinsics** - Hardware-specific operations
🔧 **Verification** - Backend-specific IR validation

---

## Best Practices

### DO ✅

- **Reuse `LowerAndLegalize()`** - Don't duplicate frontend lowering
- **Include common optimizations** - FlattenBuffer, UnrollLoop, etc.
- **Use `AnnotateDeviceRegions()`** - Mark device code
- **Return `CompiledArtifact`** - Standard return type
- **Validate target** - Check you're getting the right target
- **Document backend-specific behavior** - Explain deviations from CUDA

### DON'T ❌

- **Don't skip frontend lowering** - Critical passes like LayoutInference
- **Don't reinvent common optimizations** - Use shared implementations
- **Don't break existing backends** - Keep CUDA/TT working
- **Don't commit without tests** - Add comprehensive test coverage
- **Don't forget documentation** - Update this guide!

---

## Testing Your Backend

### Unit Tests

Test each component individually:
- Default annotation application
- Frontend lowering compatibility
- Backend-specific transformations
- Device splitting logic
- Codegen output

### Integration Tests

Test end-to-end pipeline:
- Simple kernels (element-wise)
- Complex kernels (GEMM, attention)
- Multiple kernels in one module
- Various grid sizes
- Error handling

### Regression Tests

Ensure existing backends still work:
```bash
pytest testing/python/tt/ -v  # TT backend
pytest testing/python/cuda/ -v  # CUDA backend (if exists)
```

---

## Summary

The TileLang unified lowering architecture enables:

1. **~60% code sharing** across backends (frontend + common optimizations)
2. **Clean separation** of shared vs backend-specific code
3. **Easy backend addition** following this guide
4. **Consistent behavior** across all backends
5. **Maintainability** through shared components

**Key Insight:** You don't need to implement everything from scratch. The frontend lowering and common optimizations are already done. Just add your backend-specific transforms and codegen!

**Reference Backends:**
- **CUDA**: `tilelang/engine/phase.py::OptimizeForTarget()`
- **Tenstorrent**: `tilelang/engine/tt/lower.py`

**Questions?** Check the reference implementations or existing tests for examples.

---

**Document Version:** 1.0 (Task 8 Complete)
**Last Updated:** 2025-10-08
**Maintained By:** TileLang Core Team
