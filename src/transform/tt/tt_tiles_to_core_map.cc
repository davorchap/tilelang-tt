/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
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
 * \file tt_tiles_to_core_map.cc
 * \brief Map tile assignments to physical core coordinates (Persistent Transform stage)
 *
 * This pass generates CoreRangeSet topology mappings for Tenstorrent devices.
 * It converts the logical tile-to-core mapping from Metadata Inference stage into physical core
 * coordinates that the TT-Metalium runtime can understand.
 *
 * Tenstorrent Core Grid (Grayskull/Wormhole):
 * - Logical: 8×8 Tensix cores
 * - Physical: CoreCoord(x, y) where x,y ∈ [0, 7]
 * - CoreRangeSet: Collection of CoreRange objects defining execution topology
 *
 * See: docs/tenstorrent/passes/tt_tiles_to_core_map.md for detailed specification
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/function.h>
#include <tvm/tir/transform.h>

#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Core coordinate on Tenstorrent device
 */
struct CoreCoord {
  int x;
  int y;

  CoreCoord(int x, int y) : x(x), y(y) {}
};

/*!
 * \brief Core range representing rectangular region of cores
 */
struct CoreRange {
  CoreCoord start;
  CoreCoord end;  // Inclusive

  CoreRange(CoreCoord s, CoreCoord e) : start(s), end(e) {}

  // Single core range
  explicit CoreRange(CoreCoord c) : start(c), end(c) {}
};

/*!
 * \brief Convert linear core ID to physical core coordinates
 *
 * For Grayskull/Wormhole 8×8 grid:
 * - Row-major layout: core_id = y * 8 + x
 * - x = core_id % 8
 * - y = core_id / 8
 *
 * \param core_id Linear core ID (0-63)
 * \param grid_width Width of core grid (default 8)
 * \return Physical core coordinates
 */
CoreCoord LinearToCoreCoord(int core_id, int grid_width = 8) {
  int x = core_id % grid_width;
  int y = core_id / grid_width;
  return CoreCoord(x, y);
}

/*!
 * \brief Generate core range sets from tile assignments
 *
 * This function converts the logical tile-to-core mapping from Metadata Inference stage into
 * physical CoreRangeSet objects. It attempts to merge adjacent cores into
 * rectangular ranges for efficiency.
 *
 * \param tiles_per_core Array of [start_id, count] per core from Metadata Inference stage
 * \param num_cores Total number of cores (default 64)
 * \return Array of core ranges (as nested arrays for TVM IR)
 */
Array<Array<Integer>> GenerateCoreRanges(const Array<Array<Integer>>& tiles_per_core,
                                         int num_cores = 64) {
  Array<Array<Integer>> core_ranges;

  // For Phase 2 MVP, use simple 1:1 mapping (each core is its own range)
  // Future optimization: merge adjacent cores into rectangular ranges

  for (int core_id = 0; core_id < num_cores; ++core_id) {
    if (core_id < static_cast<int>(tiles_per_core.size())) {
      auto assignment = tiles_per_core[core_id];
      int start_tile = static_cast<int>(assignment[0]->value);
      int count = static_cast<int>(assignment[1]->value);

      if (count > 0) {
        // Convert linear core ID to physical coordinates
        CoreCoord coord = LinearToCoreCoord(core_id);

        // Create single-core range: [x, y, x, y]
        // Format: [start_x, start_y, end_x, end_y]
        Array<Integer> range;
        range.push_back(Integer(coord.x));      // start_x
        range.push_back(Integer(coord.y));      // start_y
        range.push_back(Integer(coord.x));      // end_x (same as start for single core)
        range.push_back(Integer(coord.y));      // end_y
        range.push_back(Integer(start_tile));   // First tile assigned to this core
        range.push_back(Integer(count));        // Number of tiles

        core_ranges.push_back(range);
      }
    }
  }

  return core_ranges;
}

/*!
 * \brief Generate per-core runtime args arrays
 *
 * Creates arrays of runtime arguments (start_tile, num_tiles) for each core.
 * These will be passed to the TT-Metalium runtime during kernel launch.
 *
 * \param tiles_per_core Tile assignments from Metadata Inference stage
 * \param num_cores Total number of cores
 * \return Array of [start_tile, num_tiles] per core
 */
Array<Array<Integer>> GenerateCoreRuntimeArgs(const Array<Array<Integer>>& tiles_per_core,
                                              int num_cores = 64) {
  Array<Array<Integer>> runtime_args;

  for (int core_id = 0; core_id < num_cores; ++core_id) {
    if (core_id < static_cast<int>(tiles_per_core.size())) {
      auto assignment = tiles_per_core[core_id];
      int start_tile = static_cast<int>(assignment[0]->value);
      int count = static_cast<int>(assignment[1]->value);

      Array<Integer> args;
      args.push_back(Integer(start_tile));
      args.push_back(Integer(count));
      runtime_args.push_back(args);
    } else {
      // Inactive core: no tiles assigned
      Array<Integer> args;
      args.push_back(Integer(0));  // start_tile = 0
      args.push_back(Integer(0));  // count = 0
      runtime_args.push_back(args);
    }
  }

  return runtime_args;
}

/*!
 * \brief Main implementation of TTTilesToCoreMap pass
 *
 * Reads Metadata Inference stage tile assignment metadata and generates physical core topology.
 * Adds the following attributes:
 * - tt_core_ranges: Array of [start_x, start_y, end_x, end_y, start_tile, count]
 * - tt_core_runtime_args: Array of [start_tile, num_tiles] per core
 *
 * \param f The PrimFunc to process
 * \return Enhanced PrimFunc with core topology metadata
 */
PrimFunc TTTilesToCoreMapImpl(PrimFunc f) {
  // Step 1: Check for required Metadata Inference stage metadata
  auto tiles_per_core_attr = f->attrs.GetAttr<Array<Array<Integer>>>("tt_tiles_per_core");
  auto num_cores_attr = f->attrs.GetAttr<Integer>("tt_num_cores");

  if (!tiles_per_core_attr.defined() || !num_cores_attr.defined()) {
    // No Metadata Inference stage metadata, skip transformation
    return f;
  }

  Array<Array<Integer>> tiles_per_core = tiles_per_core_attr.value();
  int num_cores = static_cast<int>(num_cores_attr.value()->value);

  // Step 2: Generate core ranges (physical topology)
  Array<Array<Integer>> core_ranges = GenerateCoreRanges(tiles_per_core, num_cores);

  // Step 3: Generate per-core runtime args
  Array<Array<Integer>> core_runtime_args = GenerateCoreRuntimeArgs(tiles_per_core, num_cores);

  // Step 4: Attach metadata to function
  PrimFunc new_func = f;
  new_func = WithAttr(new_func, "tt_core_ranges", core_ranges);
  new_func = WithAttr(new_func, "tt_core_runtime_args", core_runtime_args);

  return new_func;
}

using namespace tir::transform;

/*!
 * \brief Create the TTTilesToCoreMap pass
 *
 * \return The TIR pass
 */
Pass TTTilesToCoreMap() {
  auto pass_func = [=](PrimFunc f, const IRModule& m, const PassContext& ctx) {
    return TTTilesToCoreMapImpl(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.TTTilesToCoreMap", {});
}

// Register the pass for Python FFI
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.TTTilesToCoreMap", TTTilesToCoreMap);
});

}  // namespace tl
}  // namespace tvm
