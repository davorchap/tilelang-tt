# Python to C++ Pass Migration Roadmap

**Version:** 1.0
**Date:** 2025-10-15
**Status:** Migration Planning Document

---

## Overview

This document outlines the migration strategy for porting TileLang Tenstorrent backend passes from Python to C++. The migration is necessary to achieve production-quality performance, maintain consistency with the TVM infrastructure, and eliminate the Python/C++ boundary overhead.

---

## Current State Analysis

### Python Passes (To Migrate)

| Pass | Location | Lines | Complexity | Dependencies |
|------|----------|-------|------------|--------------|
| InferTTLayout | `tilelang/tenstorrent/passes/infer_tt_layout.py` | ~200 | Medium | TVM Python API |
| PropagateTTLayout | `tilelang/tenstorrent/passes/propagate_tt_layout.py` | ~150 | Low | InferTTLayout |
| LayoutAwareWorkPartitionTT | `tilelang/tenstorrent/passes/layout_aware_work_partition_tt.py` | ~300 | High | Both above |
| InferDefaultTTSchedule | `tilelang/tenstorrent/passes/infer_default_tt_schedule.py` | ~100 | Low | Legacy (remove) |
| InferDefaultTTShard | `tilelang/tenstorrent/passes/infer_default_tt_shard.py` | ~100 | Low | Legacy (remove) |

### C++ Passes (Existing)

| Pass | Location | Status | Needs Update |
|------|----------|--------|--------------|
| GridToPersistentTT | `src/transform/tenstorrent/grid_to_persistent_tt.cc` | ✅ Working | Yes - metadata |
| MemorySpaceLowerTT | `src/transform/tenstorrent/memory_space_lower_tt.cc` | 🟡 Partial | Yes - CB sizing |
| LowerGemmToTTIntrinsics | `src/transform/tenstorrent/lower_gemm_to_tt_intrinsics.cc` | 🟡 Partial | Yes - remove heuristics |
| VerifyTTIR | `src/transform/tenstorrent/verify_tt_ir.cc` | ✅ Working | Yes - new schema |

---

## Migration Strategy

### Phase 1: Infrastructure Setup (Week 1)

#### 1.1 Create C++ Base Classes

```cpp
// src/transform/tenstorrent/tt_pass_base.h
namespace tvm {
namespace tir {
namespace tenstorrent {

class TTMetadataPass : public PrimFuncPass {
protected:
  // Common metadata manipulation utilities
  Map<String, ObjectRef> GetBufferLayout(const Buffer& buf);
  void SetBufferLayout(Buffer& buf, const Map<String, ObjectRef>& layout);

  Map<String, ObjectRef> GetCBDescriptor(const String& cb_name);
  void SetCBDescriptor(const String& cb_name, const Map<String, ObjectRef>& desc);

  // Validation utilities
  void ValidateL1Capacity(const Map<String, ObjectRef>& layout);
  void ValidateTileAlignment(const Array<PrimExpr>& shape);
};

class TTTransformPass : public StmtExprMutator {
protected:
  // Common IR transformation utilities
  Stmt InsertIntrinsicCall(const String& intrinsic, const Array<PrimExpr>& args);
  PrimExpr CalculateTileId(const Array<PrimExpr>& indices);
  Buffer CreateCircularBuffer(const String& name, const Array<PrimExpr>& shape);
};

} // namespace tenstorrent
} // namespace tir
} // namespace tvm
```

#### 1.2 Setup Testing Framework

```cpp
// tests/cpp/tenstorrent/tt_pass_test_base.h
class TTPassTestBase : public ::testing::Test {
protected:
  IRModule CreateTestModule(const String& tir_script);
  void CompareWithPythonOutput(const IRModule& cpp_output,
                               const String& python_pass_name);
  void ValidateMetadata(const PrimFunc& func,
                       const Map<String, ObjectRef>& expected);
};
```

### Phase 2: Pass-by-Pass Migration (Weeks 2-3)

#### 2.1 InferTTLayout Migration

**Python to C++ Mapping**:

| Python Construct | C++ Equivalent |
|-----------------|----------------|
| `@T.prim_func` decorator | `PrimFunc` class |
| `func.attrs` | `func->attrs` |
| `Buffer.annotate()` | `WithAttr(buffer, key, value)` |
| `tir.transform.function_pass` | `tir::PrimFuncPass` |
| `isinstance()` | `node->IsInstance<Type>()` |
| Dict comprehension | `Map` with iteration |

**Implementation Template**:

```cpp
// src/transform/tenstorrent/infer_tt_layout.cc
class InferTTLayout : public PrimFuncPass {
public:
  PrimFunc operator()(PrimFunc f, IRModule m, PassContext ctx) {
    // Step 1: Process user annotations
    auto annotations = ExtractAnnotations(f);

    // Step 2: Apply defaults
    for (auto& [param, buffer] : f->buffer_map) {
      auto layout = GetOrCreateLayout(buffer, annotations);
      ApplyDefaults(layout);
      ValidateLayout(layout);

      // Step 3: Attach to buffer
      buffer = WithAttr(buffer, "tt.layout_desc", layout);
    }

    return f;
  }

private:
  Map<String, ObjectRef> GetOrCreateLayout(const Buffer& buf,
                                          const Map<String, ObjectRef>& annotations) {
    Map<String, ObjectRef> layout;

    // Check for user annotation
    String key = "tt.buffer." + buf->name;
    if (annotations.count(key)) {
      layout = Downcast<Map<String, ObjectRef>>(annotations[key]);
    }

    // Apply defaults
    if (!layout.count("memory")) {
      layout.Set("memory", String("DRAM"));
    }
    if (!layout.count("layout")) {
      layout.Set("layout", String("interleaved"));
    }
    if (!layout.count("tile_shape")) {
      layout.Set("tile_shape", Array<Integer>({32, 32}));
    }
    if (!layout.count("dtype")) {
      layout.Set("dtype", String("bf16"));
    }

    return layout;
  }

  void ValidateLayout(const Map<String, ObjectRef>& layout) {
    // L1 validation
    if (layout["memory"] == "L1") {
      auto tile_shape = Downcast<Array<Integer>>(layout["tile_shape"]);
      CHECK(tile_shape[0] * tile_shape[1] <= 1024)
          << "L1 tiles must fit in 2KB pages";

      if (layout["layout"] == "sharded") {
        CHECK(layout.count("nd_shard"))
            << "L1 sharded buffers require nd_shard specification";
      }
    }

    // Reject halo
    if (layout.count("halo")) {
      LOG(FATAL) << "Halo metadata not supported in v1";
    }
  }
};

// Registration
TVM_REGISTER_GLOBAL("tir.tenstorrent.InferTTLayout")
.set_body_typed([]() {
  return InferTTLayout();
});
```

**Migration Checklist**:
- [ ] Port annotation extraction logic
- [ ] Port default application logic
- [ ] Port validation logic
- [ ] Add C++ unit tests
- [ ] Cross-validate with Python implementation
- [ ] Update Python wrapper to use C++ pass

#### 2.2 PropagateTTLayout Migration

```cpp
// src/transform/tenstorrent/propagate_tt_layout.cc
class PropagateTTLayout : public PrimFuncPass {
public:
  PrimFunc operator()(PrimFunc f, IRModule m, PassContext ctx) {
    // Extract buffer layouts
    Map<String, Map<String, ObjectRef>> buffer_layouts;
    for (auto& [param, buffer] : f->buffer_map) {
      if (buffer->attrs.count("tt.layout_desc")) {
        buffer_layouts.Set(buffer->name,
                          Downcast<Map<String, ObjectRef>>(
                              buffer->attrs["tt.layout_desc"]));
      }
    }

    // Generate CB descriptors
    Map<String, Map<String, ObjectRef>> cb_descriptors;
    int cb_id = 0;

    for (auto& [name, layout] : buffer_layouts) {
      auto cb_desc = GenerateCBDescriptor(layout);
      cb_descriptors.Set("cb_" + std::to_string(cb_id++), cb_desc);
    }

    // Attach to function
    f = WithAttr(f, "tt.cb_descriptors", cb_descriptors);

    return f;
  }

private:
  Map<String, ObjectRef> GenerateCBDescriptor(const Map<String, ObjectRef>& layout) {
    Map<String, ObjectRef> cb_desc;

    // Calculate page size
    auto tile_shape = Downcast<Array<Integer>>(layout["tile_shape"]);
    auto dtype = Downcast<String>(layout["dtype"]);
    int bytes_per_elem = GetDtypeBytes(dtype);
    int page_size = tile_shape[0].IntValue() *
                   tile_shape[1].IntValue() *
                   bytes_per_elem;

    cb_desc.Set("page_size", Integer(page_size));
    cb_desc.Set("depth", Integer(2));  // Default double-buffering
    cb_desc.Set("data_format", ConvertToMetaliumFormat(dtype));

    return cb_desc;
  }
};
```

#### 2.3 LayoutAwareWorkPartitionTT Migration

This is the most complex pass to migrate:

```cpp
// src/transform/tenstorrent/layout_aware_work_partition_tt.cc
class LayoutAwareWorkPartitionTT : public PrimFuncPass {
public:
  PrimFunc operator()(PrimFunc f, IRModule m, PassContext ctx) {
    // Step 1: Analyze layouts to determine partition mode
    auto partition_mode = DeterminePartitionMode(f);

    // Step 2: Calculate work distribution
    WorkPartition partition;
    if (partition_mode == "global") {
      partition = ComputeGlobalPartition(f);
    } else {
      partition = ComputeLocalShardPartition(f);
    }

    // Step 3: Generate runtime args
    auto runtime_args = GenerateRuntimeArgs(partition_mode, partition);

    // Step 4: Attach all metadata
    f = WithAttr(f, "tt.partition_mode", partition_mode);
    f = WithAttr(f, "tt.core_grid", partition.core_grid);
    f = WithAttr(f, "tt.core_ranges", partition.core_ranges);
    f = WithAttr(f, "tt.grid_tiles", partition.grid_tiles);
    f = WithAttr(f, "tt.work_partition", partition.assignments);
    f = WithAttr(f, "tt.runtime_args", runtime_args);

    if (partition_mode == "local_shard") {
      f = WithAttr(f, "tt.shard_grid", partition.shard_grid);
      f = WithAttr(f, "tt.local_shape_tiles", partition.local_tiles);
    }

    return f;
  }

private:
  struct WorkPartition {
    Array<Integer> core_grid;
    Array<Array<Integer>> core_ranges;
    Array<Integer> grid_tiles;
    Map<String, Array<Array<Integer>>> assignments;
    Array<Integer> shard_grid;
    Array<Integer> local_tiles;
  };

  String DeterminePartitionMode(const PrimFunc& f) {
    // Check if any buffer is L1 sharded
    for (auto& [param, buffer] : f->buffer_map) {
      if (buffer->attrs.count("tt.layout_desc")) {
        auto layout = Downcast<Map<String, ObjectRef>>(
            buffer->attrs["tt.layout_desc"]);
        if (layout["memory"] == "L1" && layout["layout"] == "sharded") {
          return "local_shard";
        }
      }
    }
    return "global";
  }

  WorkPartition ComputeGlobalPartition(const PrimFunc& f) {
    WorkPartition result;

    // Standard 8x8 grid
    result.core_grid = {8, 8};
    result.core_ranges = {{0, 0, 7, 7}};

    // Calculate global tiles
    auto shape = GetOutputShape(f);
    result.grid_tiles = {
      (shape[0] + 31) / 32,
      (shape[1] + 31) / 32
    };

    // Distribute work
    int total_tiles = result.grid_tiles[0].IntValue() *
                     result.grid_tiles[1].IntValue();
    int tiles_per_core = (total_tiles + 63) / 64;

    // Assign tiles to cores
    int tile_id = 0;
    for (int cy = 0; cy < 8; cy++) {
      for (int cx = 0; cx < 8; cx++) {
        Array<Array<Integer>> core_tiles;
        for (int i = 0; i < tiles_per_core && tile_id < total_tiles; i++) {
          int ty = tile_id / result.grid_tiles[1].IntValue();
          int tx = tile_id % result.grid_tiles[1].IntValue();
          core_tiles.push_back({ty, tx});
          tile_id++;
        }
        String core_key = "core_" + std::to_string(cy) + "_" + std::to_string(cx);
        result.assignments.Set(core_key, core_tiles);
      }
    }

    return result;
  }
};
```

### Phase 3: Testing & Validation (Week 4)

#### 3.1 Cross-Validation Framework

```python
# testing/python/tenstorrent/test_cpp_migration.py

def validate_cpp_pass(pass_name, test_cases):
    """Validate C++ pass against Python implementation"""
    python_pass = get_python_pass(pass_name)
    cpp_pass = get_cpp_pass(pass_name)

    for test_case in test_cases:
        # Run both implementations
        python_result = python_pass(test_case)
        cpp_result = cpp_pass(test_case)

        # Compare metadata
        assert_metadata_equal(python_result.attrs, cpp_result.attrs)

        # Compare IR structure
        assert_ir_equal(python_result.body, cpp_result.body)

@pytest.mark.parametrize("kernel", [
    simple_gemm(),
    sharded_buffer(),
    l1_resident(),
    nd_sharding()
])
def test_infer_tt_layout_migration(kernel):
    validate_cpp_pass("InferTTLayout", [kernel])
```

#### 3.2 Performance Validation

```cpp
// benchmarks/tenstorrent/pass_performance.cc
BENCHMARK(InferTTLayout_Python);
BENCHMARK(InferTTLayout_CPP);
BENCHMARK(PropagateTTLayout_Python);
BENCHMARK(PropagateTTLayout_CPP);
```

### Phase 4: Cutover & Cleanup (Week 5)

#### 4.1 Python Wrapper Updates

```python
# tilelang/tenstorrent/passes/infer_tt_layout.py (after migration)

def InferTTLayout():
    """Wrapper for C++ implementation"""
    return tvm.get_global_func("tir.tenstorrent.InferTTLayout")()

# Deprecation notice for old implementation
def _infer_tt_layout_python():
    """DEPRECATED: Use C++ implementation via InferTTLayout()"""
    warnings.warn("Python implementation deprecated", DeprecationWarning)
    # ... old implementation kept for reference ...
```

#### 4.2 Documentation Updates

```markdown
# Migration Complete Checklist
- [ ] All Python passes have C++ equivalents
- [ ] All tests pass with C++ implementations
- [ ] Performance benchmarks show improvement
- [ ] Python wrappers updated
- [ ] Documentation updated
- [ ] Legacy code marked deprecated
```

---

## Migration Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Behavioral differences | High | Extensive cross-validation testing |
| Performance regression | Medium | Benchmark before/after |
| API compatibility | Medium | Maintain Python wrappers |
| Missing edge cases | Medium | Port all Python tests to C++ |
| Metadata schema drift | High | Single source of truth in C++ |

---

## Success Criteria

### Functional
- [ ] All migrated passes produce identical output to Python versions
- [ ] All existing tests pass with C++ implementations
- [ ] No performance regressions in compilation time

### Quality
- [ ] C++ code follows TVM coding standards
- [ ] Comprehensive unit test coverage (>90%)
- [ ] Clear documentation and examples

### Performance
- [ ] 2-5x faster pass execution time
- [ ] Reduced memory usage
- [ ] Eliminated Python/C++ boundary overhead

---

## Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Infrastructure | Base classes, test framework |
| 2 | Migration (Part 1) | InferTTLayout, PropagateTTLayout |
| 3 | Migration (Part 2) | LayoutAwareWorkPartitionTT |
| 4 | Validation | Cross-validation, performance tests |
| 5 | Cutover | Update wrappers, deprecate Python |

---

## Appendix: Common Migration Patterns

### Pattern 1: Attribute Access

**Python:**
```python
if "tt.layout" in func.attrs:
    layout = func.attrs["tt.layout"]
```

**C++:**
```cpp
if (func->attrs.count("tt.layout")) {
    auto layout = Downcast<Map<String, ObjectRef>>(func->attrs["tt.layout"]);
}
```

### Pattern 2: Buffer Manipulation

**Python:**
```python
buffer = buffer.with_attr("new_attr", value)
```

**C++:**
```cpp
buffer = WithAttr(buffer, "new_attr", value);
```

### Pattern 3: IR Traversal

**Python:**
```python
class MyVisitor(tir.StmtVisitor):
    def visit_evaluate(self, op):
        # process
        super().visit_evaluate(op)
```

**C++:**
```cpp
class MyVisitor : public StmtExprVisitor {
  void VisitStmt_(const EvaluateNode* op) override {
    // process
    StmtExprVisitor::VisitStmt_(op);
  }
};
```

---

**Document Version:** 1.0
**Last Updated:** 2025-10-15
**Next Review:** After Phase 1 completion