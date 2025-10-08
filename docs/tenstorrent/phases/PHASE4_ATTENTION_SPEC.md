# Phase 4: Attention Mechanisms - FlashAttention & Variants

**Timeline**: Weeks 25-26 (2 weeks)
**Priority**: HIGH - Critical for LLM workloads
**Status**: ⏳ Not Started (Blocked by Phase 3)

---

## Overview

Phase 4 implements attention mechanisms essential for modern LLMs:
- **FlashAttention (Forward)**: Online softmax with QK^T matmul + PV matmul
- **FlashAttention (Backward)**: Gradient computation for training
- **Grouped Query Attention (GQA)**: Shared KV across query groups
- **Linear Attention**: Alternative to softmax-based attention
- **Block Sparse Attention**: Sparse patterns for long sequences

---

## Examples Covered

### 4.1 Flash Attention (Forward)
**File**: `examples/flash_attention/*.py`

**Key Patterns**:
- Online softmax with tile-level statistics (max, sum)
- QK^T matmul → softmax → PV matmul pipeline
- Rescaling patterns for numerical stability

**Required Transforms**:
- 🆕 **NEW**: Flash Attention pattern recognition
  - Identify QK, softmax, PV stages
  - Insert online statistics tracking
- 🆕 **NEW**: Tile rescaling insertion

**Expected Pattern**:
```cpp
// Simplified FlashAttention forward
for (uint32_t q_tile = 0; q_tile < num_q_tiles; ++q_tile) {
    float m_i = -INFINITY;  // Running max
    float l_i = 0.0f;        // Running sum

    for (uint32_t kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        // QK^T matmul
        acquire_dst();
        matmul_tiles(CB_Q, CB_K, 0, 0, 0, true /* transpose K */);

        // Online softmax statistics
        float m_ij = reduce_max_tile(DST);
        float m_i_new = max(m_i, m_ij);

        // Rescale previous accumulator
        rescale_tiles(CB_O, exp(m_i - m_i_new));

        // Softmax and attention weights
        softmax_tiles(DST, m_i_new);

        // P @ V matmul
        matmul_tiles(DST, CB_V, 0, 0, 0, false);

        // Update statistics
        m_i = m_i_new;
        l_i = l_i * exp(m_i - m_i_new) + reduce_sum_tile(DST);

        commit_dst();
        release_dst();
    }

    // Finalize output
    normalize_tiles(CB_O, l_i);
}
```

**Codegen Additions**:
- Statistics tracking in registers
- Online rescaling operations
- Softmax tile intrinsics
- Multi-stage pipeline (QK → softmax → PV)

---

### 4.2 Flash Attention (Backward)
**File**: `examples/flash_attention/*bwd*.py`

**What's New**:
- Backward pass gradient computation
- Reuse statistics from forward pass
- Complex data dependencies

**Required Transforms**:
- 🆕 **NEW**: Backward pass pattern
  - Gradient flow analysis
  - Statistics reuse from forward

---

### 4.3 Grouped Query Attention (GQA)
**File**: `examples/attention_sink/*.py`, `examples/flash_decoding/*.py`

**What's New**:
- Grouped queries (shared KV across query groups)
- KV cache management
- Different tensor layouts

**Required Transforms**:
- 🆕 **NEW**: GQA pattern recognition
  - Identify query groups
  - Plan KV reuse via multicast

**Expected Pattern**:
```cpp
// GQA: num_q_groups > num_kv_groups
// Multiple Q tiles share same K, V tiles
for (uint32_t group = 0; group < num_kv_groups; ++group) {
    // Load KV for this group (shared across queries)
    load_kv_tiles(group);

    // Process all queries in this group
    for (uint32_t q_in_group = 0; q_in_group < queries_per_group; ++q_in_group) {
        // Q @ K^T (reuse K)
        // Apply softmax
        // @ V (reuse V)
    }
}
```

---

### 4.4 Linear Attention
**File**: `examples/linear_attention/*.py`

**What's New**:
- Avoid softmax (kernel function instead)
- Different attention mechanism
- Potential for kernel fusion

**Expected Pattern**:
```cpp
// Linear attention: attn = (Q @ K^T) @ V (no softmax)
// Can be reordered: Q @ (K^T @ V) for efficiency
```

---

### 4.5 Block Sparse Attention
**File**: `examples/blocksparse_attention/*.py`

**What's New**:
- Sparse attention patterns (block diagonal, strided, etc.)
- Conditional tile loading based on mask/indices
- Skip unnecessary computations

**Required Transforms**:
- 🆕 **NEW**: Sparse pattern analysis
  - Identify which tiles to load/compute
  - Insert conditional guards

**Expected Pattern**:
```cpp
// Only process blocks in sparse pattern
for (uint32_t q_tile = 0; q_tile < num_q_tiles; ++q_tile) {
    for (uint32_t kv_tile : sparse_pattern[q_tile]) {
        // Only load and compute for non-zero blocks
        if (is_in_pattern(q_tile, kv_tile)) {
            // QK^T @ V as usual
        }
    }
}
```

---

## Implementation Checklist

### 4.1 FlashAttention Forward
- [ ] Online softmax statistics tracking
- [ ] Rescaling operation codegen
- [ ] Multi-stage pipeline (QK → softmax → PV)
- [ ] Test with attention examples
- [ ] PR: "Implement FlashAttention Forward"

### 4.2 FlashAttention Backward
- [ ] Backward pass pattern transform
- [ ] Gradient computation codegen
- [ ] Test gradient correctness
- [ ] PR: "Implement FlashAttention Backward"

### 4.3 GQA
- [ ] Query grouping pattern recognition
- [ ] KV multicast for shared groups
- [ ] Test GQA patterns
- [ ] PR: "Implement Grouped Query Attention"

### 4.4 Linear Attention
- [ ] Kernel function support (no softmax)
- [ ] Reordering for efficiency
- [ ] Test linear attention
- [ ] PR: "Implement Linear Attention"

### 4.5 Block Sparse Attention
- [ ] Sparse pattern analysis
- [ ] Conditional tile loading
- [ ] Test various sparse patterns
- [ ] PR: "Implement Block Sparse Attention"

---

## Timeline Estimate

| Task | Duration | Dependencies |
|------|----------|--------------|
| 4.1 FlashAttention Fwd | 10-12 hours | Phase 1-3 complete |
| 4.2 FlashAttention Bwd | 8-10 hours | 4.1 complete |
| 4.3 GQA | 6-8 hours | 4.1 complete, Phase 3.2 (multicast) |
| 4.4 Linear Attention | 4-6 hours | Phase 1, 3 complete |
| 4.5 Sparse Attention | 6-8 hours | 4.1 complete |
| **Total Phase 4** | **34-44 hours** | **7-9 days** |

---

**Status**: ⏳ Not Started
**Blocked By**: Phase 3 completion
**Last Updated**: 2025-10-08
