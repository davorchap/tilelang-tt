# Phase 5: Specialized Operations - FP8, MoE, TopK

**Timeline**: Weeks 27-28 (2 weeks)
**Priority**: MEDIUM - Specialized LLM operations
**Status**: ⏳ Not Started (Blocked by Phase 4)

---

## Overview

Phase 5 implements specialized operations for modern LLMs:
- **FP8 GEMM**: Low-precision matmul with E4M3/E5M2 formats
- **Fused MoE**: Mixture of Experts with routing
- **TopK/Sampling**: Selection and sampling operations

---

## Examples Covered

### 5.1 FP8 GEMM
**File**: `examples/gemm_fp8/*.py`

**What's New**:
- FP8 data types (E4M3 float8_e4m3fn, E5M2 float8_e5m2)
- Scaling factors for quantization
- Mixed precision patterns (FP8 compute, FP16/BF16 accumulate)

**Required Codegen**:
- FP8 tile size handling (same as FP16: 32×32)
- Scaling factor application
- Type conversion FP8 ↔ FP16/BF16

**Expected Pattern**:
```cpp
// FP8 GEMM with scaling
for (uint32_t tile_idx = 0; tile_idx < num_output_tiles; ++tile_idx) {
    acquire_dst();
    matmul_tiles_init(CB_A_FP8, CB_B_FP8, CB_C_FP16);

    // Load scaling factors
    float scale_a = get_arg_val<float>(scale_a_idx);
    float scale_b = get_arg_val<float>(scale_b_idx);

    for (uint32_t k = 0; k < Kt; ++k) {
        cb_wait_front(CB_A_FP8, 1);
        cb_wait_front(CB_B_FP8, 1);

        // Convert FP8 → FP16, apply scale, matmul
        convert_fp8_to_fp16_tiles(CB_A_FP8, scale_a, 0);
        convert_fp8_to_fp16_tiles(CB_B_FP8, scale_b, 1);
        matmul_tiles(0, 1, 2, false);  // FP16 accumulate

        cb_pop_front(CB_A_FP8, 1);
        cb_pop_front(CB_B_FP8, 1);
    }

    cb_reserve_back(CB_C_FP16, 1);
    commit_dst();
    pack_tile(0, CB_C_FP16);
    cb_push_back(CB_C_FP16, 1);
    release_dst();
}
```

---

### 5.2 Fused MoE (Mixture of Experts)
**File**: `examples/fusedmoe/*.py`

**What's New**:
- Expert routing (top-K expert selection per token)
- Multiple GEMM operations (one per expert)
- Conditional execution based on routing

**Required Transforms**:
- 🆕 **NEW**: MoE pattern recognition
  - Expert routing analysis
  - Conditional kernel execution planning

**Expected Pattern**:
```cpp
// MoE: Each token routed to top-K experts
for (uint32_t token = 0; token < num_tokens; ++token) {
    // Get expert assignments for this token
    uint32_t expert_ids[K];
    float expert_weights[K];
    get_top_k_experts(token, expert_ids, expert_weights);

    // Process each assigned expert
    for (uint32_t k = 0; k < K; ++k) {
        uint32_t expert_id = expert_ids[k];
        float weight = expert_weights[k];

        // GEMM with expert_id's weights
        acquire_dst();
        load_expert_weights(expert_id);  // W_expert

        matmul_tiles_init(CB_TOKEN, CB_EXPERT_W, CB_OUT);
        // ... standard GEMM ...

        // Scale by expert weight
        scale_tiles(CB_OUT, weight);

        // Accumulate to final output
        add_tiles(CB_OUT, CB_FINAL, 0, 0);

        commit_dst();
        release_dst();
    }
}
```

**Challenges**:
- Dynamic expert routing (different per token)
- Efficient weight loading (cache frequently used experts)
- Load balancing across cores

---

### 5.3 TopK / Sampling
**File**: `examples/topk/*.py`

**What's New**:
- Sorting / selection algorithms
- May require iterative reduction
- Multi-stage operations (local topk → global merge)

**Required Transforms**:
- 🆕 **NEW**: TopK pattern
  - Multi-stage sorting
  - Partial results aggregation

**Expected Pattern**:
```cpp
// TopK: Find K largest elements
// Stage 1: Local TopK per core
for (uint32_t tile = 0; tile < tiles_per_core; ++tile) {
    acquire_dst();
    load_tile(CB_IN, tile);

    // Sort tile, keep top-K
    topk_sort_tile(CB_IN, K, CB_LOCAL_TOPK);

    commit_dst();
    release_dst();
}

// Stage 2: Merge local TopK across cores (if multi-core)
// Single core reduces all local TopK results
merge_topk_results(CB_LOCAL_TOPK, K, CB_GLOBAL_TOPK);
```

---

## Implementation Checklist

### 5.1 FP8 GEMM
- [ ] FP8 type support
- [ ] Scaling factor handling
- [ ] Type conversion FP8 ↔ FP16/BF16
- [ ] Test FP8 matmul patterns
- [ ] PR: "Implement FP8 GEMM"

### 5.2 Fused MoE
- [ ] Expert routing pattern recognition
- [ ] Conditional execution codegen
- [ ] Expert weight loading
- [ ] Test MoE patterns
- [ ] PR: "Implement Fused MoE"

### 5.3 TopK/Sampling
- [ ] TopK pattern recognition
- [ ] Multi-stage sorting
- [ ] Merge operations
- [ ] Test top-k selection
- [ ] PR: "Implement TopK and Sampling"

---

## Timeline Estimate

| Task | Duration | Dependencies |
|------|----------|--------------|
| 5.1 FP8 GEMM | 6-8 hours | Phase 1, 2.2 (type conversion) |
| 5.2 Fused MoE | 8-10 hours | Phase 1, 4.3 (routing logic) |
| 5.3 TopK/Sampling | 6-8 hours | Phase 2.3 (reduction) |
| **Total Phase 5** | **20-26 hours** | **4-5 days** |

---

**Status**: ⏳ Not Started
**Blocked By**: Phase 4 completion
**Last Updated**: 2025-10-08
