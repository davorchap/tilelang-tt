# TileLang→TT Examples: Use-Case Driven Development Plan

**Date**: 2025-10-08
**Goal**: Implement full TileLang→Metalium compiler stack to support all TileLang GPU examples
**Approach**: Use-case driven - implement transforms/codegen only when needed by examples
**Mode**: Mock/dry-test only (no hardware execution until all examples generate)

---

## Architecture Foundation

### Key Tenstorrent Concepts to Model

1. **DST Register Double Buffering** (Priority #1)
   - Acquire: Reserve DST registers for computation
   - Commit: Mark computation complete, registers ready for packer
   - Wait: Packer waits for FPU completion
   - Release: Free DST registers back to FPU
   - **Handshake between FPU (math) and Packer**

2. **Circular Buffer (CB) Management**
   - Multi-buffering depths (2, 4, 8 tiles)
   - Producer/consumer synchronization
   - cb_reserve_back, cb_push_back, cb_wait_front, cb_pop_front

3. **Kernel Roles**
   - Reader: DRAM → L1 CBs
   - Compute: CB → DST → CB
   - Writer: L1 CBs → DRAM

4. **Work Distribution**
   - Static tile assignment per core
   - Persistent outer loop: `for i in 0..count`
   - Recover `(bx, by)` from linear tile ID

---

## Example Progression (Simple → Complex)

### Phase 1: Foundation (Weeks 19-20)

#### 1.1 Elementwise Add ✅ **START HERE**
**File**: `examples/elementwise/example_elementwise_add.py`

**Why First**:
- Simplest TileLang pattern
- Tests basic CB management
- Tests DST acquire/release for element-wise ops
- No K-loop, no accumulation

**Required Transforms**:
- [ ] None (existing transforms sufficient)

**Required Codegen**:
- [ ] DST double buffering for element-wise ops
  - `acquire_dst()` before computation
  - `commit_dst()` after computation
  - `wait_for_packer()` before release
  - `release_dst()` to free registers
- [ ] CB management for A, B inputs
- [ ] CB management for C output

**Test**: Generate mock code, verify DST handshake

---

#### 1.2 Elementwise (Multi-operand)
**File**: `examples/elementwise/*.py` (all variants)

**What's New**:
- Multiple inputs (A + B + C, etc.)
- More complex element-wise expressions

**Required Transforms**:
- [ ] None

**Required Codegen**:
- [ ] Extend DST management for multi-operand ops

---

#### 1.3 Simple GEMM (no K-loop pipelining)
**File**: `examples/gemm/example_tilelang_gemm_simple.py` (if exists, else create)

**What's New**:
- K-loop with accumulation
- DST double buffering for matmul tiles
- Packer waits for all K iterations

**Required Transforms**:
- [ ] None

**Required Codegen**:
- [ ] DST double buffering for matmul:
  - Acquire DST at start of (bx, by) computation
  - For k in K-loop: matmul_tiles (accumulate in DST)
  - Commit DST after K-loop complete
  - Wait for packer
  - Release DST
- [ ] CB pipelining for A, B tiles

---

### Phase 2: Optimizations (Weeks 21-22)

#### 2.1 GEMM with CB Double-Buffering
**File**: `examples/gemm/*.py` (optimized variants)

**What's New**:
- K-loop software pipelining
- Double-buffered CBs (depth=2)
- Overlap compute with data movement

**Required Transforms**:
- [ ] **New**: CB depth inference
  - Analyze K-loop structure
  - Insert cb_reserve_back/cb_push_back for producer (reader)
  - Insert cb_wait_front/cb_pop_front for consumer (compute)

**Required Codegen**:
- [ ] CB double-buffering in reader kernel
- [ ] CB synchronization in compute kernel

---

#### 2.2 Cast / Type Conversion
**File**: `examples/cast/*.py`

**What's New**:
- Data type conversions (FP32 → BF16, INT8 → FP16, etc.)
- May use DST for conversion or direct packer path

**Required Transforms**:
- [ ] None

**Required Codegen**:
- [ ] Type conversion in DST
- [ ] Pack to different output type

---

#### 2.3 Reduction Operations
**File**: `examples/norm/*.py`, `examples/online_softmax/*.py`

**What's New**:
- Reduction patterns (sum, max, etc.)
- Potential multi-stage reductions
- Synchronization across cores for global reductions

**Required Transforms**:
- [ ] **New**: Reduction pattern recognition
  - Identify reduction axis
  - Plan multi-core reduction strategy

**Required Codegen**:
- [ ] Reduction in DST (local)
- [ ] Inter-core reduction via NoC (if needed)

---

### Phase 3: Advanced Patterns (Weeks 23-24)

#### 3.1 GEMV (Matrix-Vector)
**File**: `examples/gemv/*.py`

**What's New**:
- Non-square tile patterns
- Vector broadcast patterns
- Potentially different CB depths for matrix vs vector

**Required Transforms**:
- [ ] **New**: Non-square tile handling
  - Adjust for vector tiles (1×32, 32×1)

**Required Codegen**:
- [ ] Vector broadcast patterns
- [ ] Non-square DST tile handling

---

#### 3.2 Convolution
**File**: `examples/convolution/*.py`

**What's New**:
- Im2col patterns or direct convolution
- Weight reuse via multicast
- Complex indexing for sliding windows

**Required Transforms**:
- [ ] **New**: Convolution pattern recognition
  - Im2col transform OR
  - Direct convolution with window indexing
- [ ] **New**: Weight multicast planning

**Required Codegen**:
- [ ] Multicast for weight tiles
- [ ] Window indexing for input tiles

---

#### 3.3 Split-K GEMM
**File**: `examples/gemm_splitk/*.py`

**What's New**:
- K-dimension split across cores
- Partial results written to temp buffer
- Second kernel to reduce partial sums

**Required Transforms**:
- [ ] **New**: Split-K planning
  - Divide K-dimension
  - Plan two-phase execution

**Required Codegen**:
- [ ] Partial accumulation with atomic adds (or two-pass)

---

### Phase 4: Attention Mechanisms (Weeks 25-26)

#### 4.1 Flash Attention (Forward)
**File**: `examples/flash_attention/*.py`

**What's New**:
- Online softmax pattern
- QK^T matmul + softmax + PV matmul
- Tile-level statistics (max, sum)
- Rescaling patterns

**Required Transforms**:
- [ ] **New**: Flash Attention pattern recognition
  - Identify QK, softmax, PV stages
  - Insert online statistics tracking
- [ ] **New**: Tile rescaling insertion

**Required Codegen**:
- [ ] Statistics tracking in DST
- [ ] Rescaling in DST
- [ ] Multi-stage pipeline (QK → softmax → PV)

---

#### 4.2 Flash Attention (Backward)
**File**: `examples/flash_attention/*bwd*.py`

**What's New**:
- Backward pass gradient computation
- More complex data dependencies
- Potentially requires saved statistics from forward

**Required Transforms**:
- [ ] **New**: Backward pass pattern
  - Gradient flow analysis
  - Statistics reuse from forward

---

#### 4.3 Grouped Query Attention (GQA)
**File**: `examples/attention_sink/*.py`, `examples/flash_decoding/*.py`

**What's New**:
- Grouped queries (shared KV across query groups)
- Potentially different tensor layouts

**Required Transforms**:
- [ ] **New**: GQA pattern recognition
  - Identify query groups
  - Plan KV reuse via multicast

---

#### 4.4 Linear Attention
**File**: `examples/linear_attention/*.py`

**What's New**:
- Different attention mechanism
- May avoid softmax
- Kernel fusion opportunities

---

#### 4.5 Block Sparse Attention
**File**: `examples/blocksparse_attention/*.py`

**What's New**:
- Sparse patterns
- Conditional tile loading based on mask/indices
- Skip unnecessary computations

**Required Transforms**:
- [ ] **New**: Sparse pattern analysis
  - Identify which tiles to load/compute
  - Insert conditional guards

**Required Codegen**:
- [ ] Conditional CB loading
- [ ] Sparse tile indexing

---

### Phase 5: Specialized Operations (Weeks 27-28)

#### 5.1 FP8 GEMM
**File**: `examples/gemm_fp8/*.py`

**What's New**:
- FP8 data types (E4M3, E5M2)
- Potential scaling factors
- Mixed precision patterns

**Required Transforms**:
- [ ] **New**: FP8 scaling factor handling

---

#### 5.2 Fused MoE (Mixture of Experts)
**File**: `examples/fusedmoe/*.py`

**What's New**:
- Expert routing
- Multiple GEMM operations
- Conditional execution based on routing

**Required Transforms**:
- [ ] **New**: MoE pattern recognition
  - Expert routing analysis
  - Conditional kernel execution

---

#### 5.3 TopK / Sampling
**File**: `examples/topk/*.py`

**What's New**:
- Sorting / selection algorithms
- May require iterative reduction

**Required Transforms**:
- [ ] **New**: TopK pattern
  - Multi-stage sorting
  - Partial results aggregation

---

### Phase 6: Complex Architectures (Weeks 29-30)

#### 6.1 DeepSeek NSA (Non-Standard Attention)
**File**: `examples/deepseek_nsa/*.py`

**What's New**:
- Custom attention variant
- Potentially unique fusion patterns

---

#### 6.2 DeepSeek MLA (Multi-Latent Attention)
**File**: `examples/deepseek_mla/*.py`

**What's New**:
- Multiple latent representations
- Complex tensor manipulation

---

#### 6.3 Warp Specialization Patterns
**File**: `examples/warp_specialize/*.py`

**What's New**:
- Producer/consumer thread patterns
- Maps to Reader/Compute/Writer separation in TT

---

---

## Transform Specification Process

For each example that requires a new transform:

### 1. Document Requirements
**File**: `docs/tenstorrent/transforms/TRANSFORM_NAME_SPEC.md`

**Contents**:
- Purpose and motivation
- Input IR pattern
- Output IR pattern
- Examples (before/after)
- Edge cases

### 2. Implement Transform
**File**: `src/transform/tt/transform_name.cc`

**Structure**:
```cpp
// Header with documentation
class TransformName : public StmtExprMutator {
  // Visit methods for relevant IR nodes
  // Helper methods
};

// Registration
TVM_REGISTER_GLOBAL("tir.transform.TransformName")
.set_body_typed(TransformName);
```

### 3. Add Python Binding
**File**: `tilelang/tt/passes.py`

```python
def transform_name(mod):
    """Apply TransformName pass."""
    return _ffi_api.TransformName(mod)
```

### 4. Write Tests
**File**: `testing/python/tt/test_transform_name.py`

**Test cases**:
- Basic functionality
- Edge cases
- Integration with other transforms

### 5. Create PR
**Title**: `Add TransformName for [use case]`

**Body**:
- Link to spec document
- Test results (all passing)
- Example before/after IR

---

## Codegen Enhancement Process

For each codegen improvement:

### 1. Document Pattern
**File**: `docs/tenstorrent/codegen/PATTERN_NAME.md`

**Contents**:
- TileLang input pattern
- Generated Metalium code
- DST register management
- CB usage
- Performance considerations

### 2. Implement in Visitor
**File**: `src/target/tt/codegen_tt_*_visitor.cc`

**Update relevant visitor**:
- Compute visitor: DST management, compute patterns
- Reader visitor: CB pushing, NOC reads
- Writer visitor: CB popping, NOC writes

### 3. Write Tests
**File**: `testing/python/tt/test_codegen_*.py`

### 4. Create PR

---

## Success Criteria

For each example:
- [ ] TileLang code compiles without errors
- [ ] Generated Metalium code has proper DST double buffering
- [ ] Generated Metalium code has proper CB management
- [ ] Generated code structure matches Metalium examples
- [ ] Mock API usage is correct (verifiable by inspection)
- [ ] Tests pass in mock mode

---

## Development Workflow

### For Each Example:

1. **Plan** (1-2 hours)
   - Analyze TileLang example
   - Identify required transforms
   - Document in spec

2. **Implement** (4-8 hours)
   - Write transform (if needed)
   - Update codegen
   - Add tests

3. **Validate** (1-2 hours)
   - Generate mock code
   - Inspect for correctness
   - Verify DST/CB patterns

4. **PR** (30 min)
   - Create PR with spec + implementation
   - Merge

5. **Next Example**
   - Move to next in progression

### Estimated Timeline:
- **Phase 1** (Foundation): 2 weeks, 3 examples
- **Phase 2** (Optimizations): 2 weeks, 4 examples
- **Phase 3** (Advanced): 2 weeks, 5 examples
- **Phase 4** (Attention): 2 weeks, 5 examples
- **Phase 5** (Specialized): 2 weeks, 4 examples
- **Phase 6** (Complex): 2 weeks, 3 examples

**Total**: 12 weeks, ~24 examples

---

## Current Status (2025-10-08)

**Completed**:
- ✅ WS1-3: Target registration, metadata, basic transforms
- ✅ WS4-6: Basic codegen infrastructure (IR-driven visitors)
- ✅ External SDK integration

**Starting Point**:
- **Example 1.1**: Elementwise Add
- **Focus**: DST double buffering foundation

**Next PR**: `DST Double Buffering for Element-Wise Ops`

---

## References

- [GPU vs Tenstorrent Architecture](GPU_vs_Tenstorrent.md)
- [Kernel Authoring Comparison](kernel_authoring_comparison.md)
- [IR-Driven Codegen Plan](IR_DRIVEN_CODEGEN_PLAN.md)
- [Metalium Guide](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md)

---

**Maintainer**: TileLang Tenstorrent Team
**Status**: Planning → Implementation Phase 1
