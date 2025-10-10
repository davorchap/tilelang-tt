# PropagateTTLayout Pass

**Status**: 🚧 Planned  
**Priority**: P0  
**File**: `src/transform/tt/propagate_layout_tt.cc`

---

## Purpose

Derive circular-buffer metadata for DRAM↔L1 transfers so that data-movement kernels can rely on a consistent cache-block (CB) configuration. After `InferTTLayout` normalizes buffer attributes, this pass:

- Inspects explicit copy ops (`T.copy`, `T.Read`, `T.Write` lowering results) that bridge DRAM and L1.
- Computes the tile-sized page geometry (bytes, depth, data format) for each logical buffer.
- Attaches `tt.cb.<buffer_name>` attributes to the PrimFunc for later consumption by persistent lowering and codegen.

---

## Metadata Written

Each relevant buffer gets a CB configuration dictionary:

```json
{
  "page_size": <tile_bytes>,
  "depth": <num_pages>,
  "data_format": "BFloat16_b" | "Float16_b" | "Float32_b" | "..."
}
```

Guidelines:
- `page_size` = `tile_shape[0] * tile_shape[1] * bytes_per_element`.
- `depth` is determined via helper heuristic (initially a simple default such as 2 for inputs, 1 for accumulators; future work can expose policy knobs).
- `data_format` follows TT-Metalium enum naming.

---

## High-Level Algorithm

1. Collect buffer metadata emitted by `InferTTLayout`.
2. Visit copy statements inside each PrimFunc. For each source/destination buffer pair:
   - Look up dtype and tile shape to compute `tile_bytes`.
   - Determine an appropriate `depth` based on access role (reader vs writer) or fallback default.
   - Translate dtype to TT data format string (e.g., `bf16` → `BFloat16_b`).
3. Attach matching `tt.cb.<buffer>` attributes to the enclosing PrimFunc. For symmetric copies (e.g., DRAM↔L1), stamp both endpoints.
4. Skip pure L1↔L1 copies for now (covered by CB lowering pass).

---

## Diagnostics

- Warn or fail if dtype is unsupported for TT data format lookup.
- Fail if tile shape metadata is missing (indicates `InferTTLayout` did not run).

---

## Tests

- DRAM interleaved path: ensure `page_size == 2048` for bf16 tiles and default `depth`.
- DRAM sharded path: confirm CB metadata still matches tile byte size.
- L1 only copies: currently ignored (no attribute written).
- Negative test for missing layout metadata (expect diagnostic).

Additions live in `tests/tt/test_layouts.py`.

---

## Dependencies

- Must run after `InferTTLayout` (requires `tt.buffer.*`).
- Shares dtype→format helper with codegen (`ToTTDataFormat`).

---

## Downstream Consumers

- `memory_space_lower_tt` consults `tt.cb.*` when shaping circular buffers.
- Reader/writer kernels rely on `page_size`/`depth` to size CB reservations.
