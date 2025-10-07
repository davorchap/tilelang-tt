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
 * \file infer_tt_shard.cc
 * \brief Infer default Tenstorrent sharding metadata (WS2)
 *
 * This pass generates DRAM interleaved layout descriptors and identifies
 * padding requirements for non-tile-aligned dimensions. It prepares buffer
 * parameters with sharding metadata for the Tenstorrent backend.
 *
 * See: docs/tenstorrent/workstream2/ws2_shard_inference.md
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Infer default Tenstorrent sharding metadata
 *
 * This pass iterates over buffer parameters, computes tile counts,
 * detects padding requirements, and attaches sharding metadata
 * for DRAM interleaved layout.
 *
 * \param f The PrimFunc to process
 * \return Enhanced PrimFunc with sharding metadata on buffers
 */
PrimFunc InferDefaultTTShardImpl(PrimFunc f) {
  // TODO(WS2): Implement sharding inference logic
  //
  // Steps:
  // 1. Read default layout from func->attrs (tt_layout_type="dram_interleaved")
  // 2. Read tile size (tt_tile_height=32, tt_tile_width=32)
  // 3. Iterate over func->params (buffer parameters)
  // 4. For each buffer:
  //    a. Get buffer shape
  //    b. Compute number of tiles needed (ceiling division by 32)
  //    c. Check if padding needed (dimensions not multiple of 32)
  //    d. Attach metadata to buffer:
  //       - tt_layout: "dram_interleaved"
  //       - tt_tile_shape: [32, 32]
  //       - tt_num_tiles_height, tt_num_tiles_width
  //       - tt_needs_padding: true/false
  //       - tt_padded_shape: [padded_h, padded_w] (if padding needed)
  //
  // Note: Actual TensorAccessor configuration deferred to WS4 codegen
  // This pass only attaches metadata markers.

  // For now, just return the function unchanged (stub)
  return f;
}

using namespace tir::transform;

/*!
 * \brief Create the InferDefaultTTShard pass
 *
 * \return The TIR pass
 */
Pass InferDefaultTTShard() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return InferDefaultTTShardImpl(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InferDefaultTTShard", {});
}

// Register the pass for Python FFI
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InferDefaultTTShard", InferDefaultTTShard);
});

}  // namespace tl
}  // namespace tvm
