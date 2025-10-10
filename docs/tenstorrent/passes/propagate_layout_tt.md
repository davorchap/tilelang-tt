# PropagateTTLayout Pass

**Status**: 🟡 Partial (defaults only)  
**Priority**: P0  
**File**: Python helper in `tilelang/tt/passes.py`

---

## Purpose

Consume `tt.buffer.*` metadata and stamp matching circular-buffer descriptors on
each PrimFunc. The current implementation provides a basic bridge between
layout metadata and later passes:

- Computes page size from tile shape and dtype (32×32 tiles assumed).
- Emits default `depth=2` and a TT-compatible `data_format` string.
- Skips role-specific tuning (reader vs writer) and advanced heuristics for now.

---

## Metadata Written

Each buffer gains a `tt.cb.<name>` attribute such as:

```json
{
  "page_size": 2048,
  "depth": 2,
  "data_format": "Float16_b"
}
```

Notes:
- Page size derives from tile shape × element bytes.
- Depth is currently hard-coded; future work should choose depth based on access pattern.
- Only buffers with `tt.buffer.*` metadata receive a CB entry.

---

## High-Level Algorithm

1. Iterate over buffers recorded in the PrimFunc’s `buffer_map`.
2. Look up the corresponding `tt.buffer.<name>` metadata (emitted by
   `InferTTLayout`).
3. Compute page size and pick a default `depth=2`.
4. Translate dtype → TT Metalium data format string (best effort).
5. Attach the resulting dictionary as `tt.cb.<name>`.
6. TODO: refine depth heuristics and warn when dtype is unsupported.

---

## Diagnostics

None today. Future improvements should warn when:
- `tt.buffer.*` metadata is missing for a buffer that participates in DRAM↔L1 transfers.
- Dtype cannot be mapped to a TT data format.

---

## Tests

Covered by `testing/python/tt/test_layout_aware_metadata.py` (default DRAM case).
Additional coverage (e.g., dtype failures, depth overrides) should be added once
policy knobs are introduced.

---

## Dependencies

- Runs after `InferTTLayout` (requires `tt.buffer.*`).
- Shares dtype → data-format helper logic with TT codegen.

---

## Downstream Consumers

- `memory_space_lower_tt` (C++) consumes `tt.cb.*` to size circular buffers.
- Reader/Writer codegen will eventually use these attributes when configuring
  runtime kernels.
