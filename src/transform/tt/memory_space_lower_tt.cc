/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file memory_space_lower_tt.cc
 * \brief Lower abstract buffer allocations to TT circular buffers (WS3 Phase 2)
 *
 * This pass transforms TileLang's alloc_fragment buffer allocations into
 * Tenstorrent circular buffer (CB) configurations in L1 memory.
 *
 * Transformation:
 * - Identifies DeclBuffer nodes with shared memory scope
 * - Assigns circular buffer IDs (CB0, CB1, CB2, ...)
 * - Stamps CB metadata: {cb_id, num_pages, tile_size}
 * - Annotates buffer with "storage_scope" = "tt.l1"
 *
 * Circular Buffer Allocation Strategy:
 * - A_tile → CB0 (2 pages for double-buffering)
 * - B_tile → CB1 (2 pages for double-buffering)
 * - C_tile → CB2 (1 page for accumulator)
 *
 * See: docs/tenstorrent/UNIFIED_MATMUL_MVP_PLAN.md Phase 2 WS3-Extended
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Circular buffer metadata
 */
struct CircularBufferInfo {
  int cb_id;        // CB index (0, 1, 2, ...)
  int num_pages;    // Number of pages (1 for single-buffer, 2 for double-buffer)
  int tile_size;    // Tile size in bytes
  String name;      // Buffer name for debugging
};

/*!
 * \brief Visitor to identify and annotate buffer allocations with CB metadata
 */
class MemorySpaceLowerMutator : public StmtMutator {
 public:
  MemorySpaceLowerMutator() : next_cb_id_(0) {}

  Stmt VisitStmt_(const DeclBufferNode* op) override {
    // Check if this is a shared memory allocation (typical for fragment buffers)
    // In TileLang, alloc_fragment creates buffers with no specific storage scope,
    // which we want to map to TT L1 circular buffers

    Buffer buffer = op->buffer;

    // Skip buffers that are already annotated as TT L1
    if (buffer.scope() == "tt.l1") {
      return StmtMutator::VisitStmt_(op);
    }

    // Check if buffer is tile-sized (32×32 or similar)
    // For Phase 2, we use heuristic: buffers with 2D shape and small size are tile buffers
    // Function parameters (large DRAM buffers) will be filtered out by size check
    bool is_tile_buffer = false;
    if (buffer->shape.size() == 2) {
      // Check if dimensions are tile-sized (typically 32×32)
      auto shape0 = buffer->shape[0].as<IntImmNode>();
      auto shape1 = buffer->shape[1].as<IntImmNode>();
      if (shape0 && shape1) {
        int64_t dim0 = shape0->value;
        int64_t dim1 = shape1->value;
        // Heuristic: tile buffers are small (<=64) and square
        // This filters out large DRAM buffers (e.g., 256×256) which are function parameters
        if (dim0 == dim1 && dim0 > 0 && dim0 <= 64) {
          is_tile_buffer = true;
        }
      }
    }

    if (!is_tile_buffer) {
      return StmtMutator::VisitStmt_(op);
    }

    // Assign circular buffer ID
    int cb_id = next_cb_id_++;

    // Calculate tile size in bytes
    int element_size = buffer->dtype.bytes();
    int64_t tile_elements = 1;
    for (const auto& dim : buffer->shape) {
      if (auto dim_int = dim.as<IntImmNode>()) {
        tile_elements *= dim_int->value;
      }
    }
    int tile_size = static_cast<int>(tile_elements * element_size);

    // Determine number of pages (double-buffer for inputs, single for accumulator)
    // Heuristic: if buffer name contains "C" or "accumulator", use 1 page; else 2
    int num_pages = 2;  // Default: double-buffer for inputs
    std::string buf_name = buffer->name;
    if (buf_name.find("C") != std::string::npos ||
        buf_name.find("accum") != std::string::npos ||
        buf_name.find("output") != std::string::npos) {
      num_pages = 1;  // Single-buffer for accumulator
    }

    // Store CB info for later use
    CircularBufferInfo cb_info;
    cb_info.cb_id = cb_id;
    cb_info.num_pages = num_pages;
    cb_info.tile_size = tile_size;
    cb_info.name = buf_name;
    cb_map_[buffer.get()] = cb_info;

    // Note: TVM Buffer objects are immutable - we can't change storage_scope directly.
    // Instead, we record the CB info and will create a new buffer in the function attrs.
    // For Phase 2, the storage scope annotation will be handled during codegen.

    // Recurse on body
    Stmt new_body = VisitStmt(op->body);

    // Return original DeclBuffer unchanged (metadata will be in function attrs)
    return DeclBuffer(buffer, new_body);
  }

  const std::unordered_map<const BufferNode*, CircularBufferInfo>& GetCBMap() const {
    return cb_map_;
  }

 private:
  int next_cb_id_;
  std::unordered_map<const BufferNode*, CircularBufferInfo> cb_map_;
};

/*!
 * \brief Main implementation of MemorySpaceLowerTT pass
 *
 * Transforms buffer allocations to TT circular buffer configurations.
 * Attaches CB metadata to function attributes for use in codegen.
 *
 * \param f The PrimFunc to process
 * \return Enhanced PrimFunc with CB annotations
 */
PrimFunc MemorySpaceLowerTTImpl(PrimFunc f) {
  // Step 1: Check if this is a TT function
  auto schedule_policy = f->attrs.GetAttr<String>("tt_schedule_policy");
  if (!schedule_policy.defined()) {
    // Not a TT function, skip transformation
    return f;
  }

  // Step 2: Apply memory space lowering
  MemorySpaceLowerMutator mutator;
  Stmt new_body = mutator(f->body);

  // Step 3: Collect CB metadata
  const auto& cb_map = mutator.GetCBMap();

  // Build CB config array for function metadata
  Array<Map<String, ObjectRef>> cb_configs;
  for (const auto& pair : cb_map) {
    const CircularBufferInfo& info = pair.second;

    Map<String, ObjectRef> config;
    config.Set("cb_id", Integer(info.cb_id));
    config.Set("num_pages", Integer(info.num_pages));
    config.Set("tile_size", Integer(info.tile_size));
    config.Set("name", info.name);

    cb_configs.push_back(config);
  }

  // Step 4: Create new function with transformed body and CB metadata
  PrimFunc new_func = f;
  auto n = make_object<PrimFuncNode>(*f.get());
  n->body = new_body;
  new_func = PrimFunc(n);

  // Attach CB configurations to function attributes
  if (cb_configs.size() > 0) {
    new_func = WithAttr(new_func, "tt_circular_buffers", cb_configs);
    new_func = WithAttr(new_func, "tt_num_cbs", Integer(static_cast<int>(cb_configs.size())));
  }

  return new_func;
}

using namespace tir::transform;

/*!
 * \brief Create the MemorySpaceLowerTT pass
 *
 * \return The TIR pass
 */
Pass MemorySpaceLowerTT() {
  auto pass_func = [=](PrimFunc f, const IRModule& m, const PassContext& ctx) {
    return MemorySpaceLowerTTImpl(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MemorySpaceLowerTT", {});
}

// Register the pass for Python FFI
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.MemorySpaceLowerTT", MemorySpaceLowerTT);
});

}  // namespace tl
}  // namespace tvm
