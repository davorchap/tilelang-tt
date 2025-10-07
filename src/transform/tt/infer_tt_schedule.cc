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
 * \brief Infer default Tenstorrent schedule metadata (WS2)
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

namespace tvm {
namespace tl {

using namespace tir;

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
  // TODO(WS2): Implement schedule inference logic
  //
  // Steps:
  // 1. Read grid_x and grid_y from func->attrs (from T.Kernel)
  // 2. Compute num_tiles = grid_x * grid_y
  // 3. Query or hardcode num_cores (MVP: 64 cores)
  // 4. Partition tiles contiguously across cores
  // 5. Build per-core (start_id, count) array
  // 6. Attach schedule metadata to func->attrs
  //
  // Example metadata to attach:
  // - tt_num_tiles: total tile count
  // - tt_grid_x, tt_grid_y: grid dimensions
  // - tt_num_cores: number of active cores
  // - tt_tiles_per_core: Array of (start_id, count) per core
  // - tt_runtime_args_schema: Schema for kernel invocation

  // For now, just return the function unchanged (stub)
  return f;
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
