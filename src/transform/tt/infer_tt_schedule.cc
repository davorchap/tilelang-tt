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
 * \file infer_tt_schedule.cc
 * \brief Infer default Tenstorrent schedule metadata (Metadata Inference stage)
 *
 * This pass computes contiguous per-core tile ranges and runtime argument
 * schemas based on the kernel grid dimensions. It implements the schedule
 * inference logic for the Tenstorrent backend MVP.
 *
 * See: docs/tenstorrent/workstream2/ws2_schedule_inference.md
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <tuple>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

// Helper function to extract grid dimensions from function attributes or body
// Returns (grid_x, grid_y, grid_z) by checking attributes first, then analyzing blockIdx
std::tuple<int, int, int> ExtractGridDimensions(const PrimFunc& f) {
  int grid_x = 1, grid_y = 1, grid_z = 1;

  // First, check if grid dimensions are already in function attributes
  // This supports both tl.grid_x (TileLang convention) and tt_grid_x (TT backend)
  if (auto gx = f->attrs.GetAttr<Integer>("tl.grid_x")) {
    grid_x = static_cast<int>(gx.value()->value);
  } else if (auto gx = f->attrs.GetAttr<Integer>("tt_grid_x")) {
    grid_x = static_cast<int>(gx.value()->value);
  }

  if (auto gy = f->attrs.GetAttr<Integer>("tl.grid_y")) {
    grid_y = static_cast<int>(gy.value()->value);
  } else if (auto gy = f->attrs.GetAttr<Integer>("tt_grid_y")) {
    grid_y = static_cast<int>(gy.value()->value);
  }

  if (auto gz = f->attrs.GetAttr<Integer>("tl.grid_z")) {
    grid_z = static_cast<int>(gz.value()->value);
  } else if (auto gz = f->attrs.GetAttr<Integer>("tt_grid_z")) {
    grid_z = static_cast<int>(gz.value()->value);
  }

  // If attributes found, return them
  if (grid_x > 1 || grid_y > 1 || grid_z > 1) {
    return {grid_x, grid_y, grid_z};
  }

  // Otherwise, fall back to extracting from IR body (blockIdx thread extents)
  class GridExtractor : public StmtVisitor {
   public:
    int grid_x = 1, grid_y = 1, grid_z = 1;

    void VisitStmt_(const AttrStmtNode* op) final {
      if (op->attr_key == tir::attr::thread_extent) {
        IterVar iv = Downcast<IterVar>(op->node);
        if (iv->thread_tag == "blockIdx.x") {
          if (auto imm = op->value.as<IntImmNode>()) {
            grid_x = static_cast<int>(imm->value);
          }
        } else if (iv->thread_tag == "blockIdx.y") {
          if (auto imm = op->value.as<IntImmNode>()) {
            grid_y = static_cast<int>(imm->value);
          }
        } else if (iv->thread_tag == "blockIdx.z") {
          if (auto imm = op->value.as<IntImmNode>()) {
            grid_z = static_cast<int>(imm->value);
          }
        }
      }
      StmtVisitor::VisitStmt_(op);
    }
  };

  GridExtractor extractor;
  extractor(f->body);
  return {extractor.grid_x, extractor.grid_y, extractor.grid_z};
}

// Compute contiguous per-core tile ranges
std::vector<std::pair<int, int>> PartitionTilesContiguous(int num_tiles, int num_cores) {
  std::vector<std::pair<int, int>> ranges;

  int tiles_per_core_base = num_tiles / num_cores;
  int remainder = num_tiles % num_cores;

  int current_start = 0;
  for (int core_id = 0; core_id < num_cores; ++core_id) {
    int count = tiles_per_core_base + (core_id < remainder ? 1 : 0);
    ranges.push_back({current_start, count});
    current_start += count;
  }

  return ranges;
}

/*!
 * \brief Infer default Tenstorrent schedule metadata
 *
 * This pass reads T.Kernel grid metadata, computes total tiles,
 * partitions them across available cores using contiguous row-major
 * ordering, and attaches schedule metadata to the PrimFunc.
 *
 * \param f The PrimFunc to process
 * \return Enhanced PrimFunc with schedule metadata
 */
PrimFunc InferDefaultTTScheduleImpl(PrimFunc f) {
  // MVP: Hardcode 64 Tensix cores for Tenstorrent Grayskull/Wormhole
  const int TT_NUM_CORES = 64;

  // Step 1: Extract grid dimensions from T.Kernel (blockIdx extents)
  auto [grid_x, grid_y, grid_z] = ExtractGridDimensions(f);

  // Step 2: Compute total tiles (row-major: tile_id = by * grid_x + bx)
  int num_tiles = grid_x * grid_y * grid_z;

  // Step 3: Partition tiles contiguously across cores
  auto tile_ranges = PartitionTilesContiguous(num_tiles, TT_NUM_CORES);

  // Step 4: Build per-core (start_id, count) array as TVM Array
  Array<Array<Integer>> tiles_per_core;
  for (const auto& [start, count] : tile_ranges) {
    tiles_per_core.push_back({Integer(start), Integer(count)});
  }

  // Step 5: Attach metadata to function
  PrimFunc new_func = f;
  new_func = WithAttr(new_func, "tt_num_tiles", Integer(num_tiles));
  new_func = WithAttr(new_func, "tt_grid_x", Integer(grid_x));
  new_func = WithAttr(new_func, "tt_grid_y", Integer(grid_y));
  new_func = WithAttr(new_func, "tt_grid_z", Integer(grid_z));
  new_func = WithAttr(new_func, "tt_num_cores", Integer(TT_NUM_CORES));
  new_func = WithAttr(new_func, "tt_tiles_per_core", tiles_per_core);

  // Step 6: Consolidated schedule metadata map (for documentation parity)
  Map<String, ObjectRef> schedule_map;

  if (auto policy = f->attrs.GetAttr<String>("tt_schedule_policy")) {
    schedule_map.Set("policy", policy.value());
  }
  if (auto order = f->attrs.GetAttr<String>("tt_schedule_order")) {
    schedule_map.Set("order", order.value());
  }

  Array<Integer> grid_shape;
  grid_shape.push_back(Integer(grid_x));
  grid_shape.push_back(Integer(grid_y));
  grid_shape.push_back(Integer(grid_z));
  schedule_map.Set("grid_shape", grid_shape);
  schedule_map.Set("total_tiles", Integer(num_tiles));
  schedule_map.Set("num_cores", Integer(TT_NUM_CORES));

  Array<ObjectRef> assignments;
  for (int core_id = 0; core_id < static_cast<int>(tile_ranges.size()); ++core_id) {
    const auto& range = tile_ranges[core_id];
    Map<String, ObjectRef> assignment;
    assignment.Set("core_id", Integer(core_id));
    assignment.Set("start_tile", Integer(range.first));
    assignment.Set("tile_count", Integer(range.second));
    assignments.push_back(assignment);
  }
  schedule_map.Set("assignments", assignments);

  new_func = WithAttr(new_func, "tt_schedule", schedule_map);

  // TODO(Metadata Inference stage): Add tt_runtime_args_schema for kernel invocation
  // Format: {start_id, count, grid_x, grid_y, kt_tiles}

  return new_func;
}

using namespace tir::transform;

/*!
 * \brief Create the InferDefaultTTSchedule pass
 *
 * \return The TIR pass
 */
Pass InferDefaultTTSchedule() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return InferDefaultTTScheduleImpl(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InferDefaultTTSchedule", {});
}

// Register the pass for Python FFI
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InferDefaultTTSchedule", InferDefaultTTSchedule);
});

}  // namespace tl
}  // namespace tvm
