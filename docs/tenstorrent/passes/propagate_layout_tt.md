# PropagateTTLayout Pass

**Status**: 🟡 Partial (defaults only)  
**Priority**: P0  
**File**: Python helper in `tilelang/tt/passes.py`

---

## Purpose

Consume `tt.buffer.*` metadata and stamp matching circular-buffer descriptors on
PrimFuncs. The current implementation bridges layout metadata to downstream
passes by:

- Computing page size from tile shape and dtype (32×32 tiles assumed).
- Emitting a default `depth=2` for all buffers (policy tuning pending).
- Mapping dtype strings to TT Metalium data formats.

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
- `page_size` derives from `tile_shape[0] * tile_shape[1] * bytes_per_element`.
- `depth` remains a fixed constant until role-aware heuristics are added.
- Buffers missing `tt.buffer.*` metadata are skipped.

---

## High-Level Algorithm

1. Iterate over buffers recorded in the PrimFunc’s `buffer_map`.
2. Look up `tt.buffer.<name>` metadata (emitted by `InferTTLayout`).
3. Compute page size and assign default depth/data format.
4. Attach the resulting dictionary as `tt.cb.<name>`.
5. TODO: vary depth based on access role (reader vs writer) and surface
   diagnostics for unsupported dtypes.

---

## Diagnostics

None currently emitted; future revisions should warn when dtype conversion
fails or when `tt.buffer.*` metadata is missing.

---

## Tests

`testing/python/tt/test_layout_aware_metadata.py` exercises the default DRAM
path. Additional coverage will be added once role-aware heuristics exist.

---

## Dependencies

- Runs after `InferTTLayout` (requires `tt.buffer.*`).
- Shares dtype → data-format helper logic with TT codegen.

---

## Downstream Consumers

- `memory_space_lower_tt` (C++) consumes `tt.cb.*` to size circular buffers.
- Reader/Writer codegen will eventually use these attributes when configuring
  runtime kernels.
