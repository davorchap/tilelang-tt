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

#include <algorithm>

namespace tvm {
namespace tl {

using namespace tir;

// Helper function to compute shard metadata for a buffer
struct ShardInfo {
  int tiles_height;
  int tiles_width;
  bool needs_padding;
  int padded_height;
  int padded_width;
};

ShardInfo ComputeShardInfo(const Buffer& buffer, int tile_height, int tile_width) {
  ShardInfo info;

  // Extract buffer shape (assume at least 2D for matrices)
  Array<PrimExpr> shape = buffer->shape;
  if (shape.size() < 2) {
    // Handle 1D buffers (rare) - treat as (1, N)
    int N = 1;
    if (auto imm = shape[0].as<IntImmNode>()) {
      N = static_cast<int>(imm->value);
    }
    info.tiles_height = 1;
    info.tiles_width = (N + tile_width - 1) / tile_width;  // Ceiling division
    info.needs_padding = (N % tile_width != 0);
    info.padded_height = tile_height;
    info.padded_width = info.tiles_width * tile_width;
    return info;
  }

  // Extract M, N from shape (last two dimensions for 2D+)
  int M = 1, N = 1;
  size_t ndim = shape.size();

  if (auto imm = shape[ndim - 2].as<IntImmNode>()) {
    M = static_cast<int>(imm->value);
  }
  if (auto imm = shape[ndim - 1].as<IntImmNode>()) {
    N = static_cast<int>(imm->value);
  }

  // Compute tiles needed (ceiling division)
  info.tiles_height = (M + tile_height - 1) / tile_height;
  info.tiles_width = (N + tile_width - 1) / tile_width;

  // Check if padding needed
  bool M_aligned = (M % tile_height == 0);
  bool N_aligned = (N % tile_width == 0);
  info.needs_padding = !M_aligned || !N_aligned;

  // Compute padded shape
  info.padded_height = info.tiles_height * tile_height;
  info.padded_width = info.tiles_width * tile_width;

  return info;
}

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
  // MVP: Hardcode 32x32 tiles for Tenstorrent
  const int TT_TILE_HEIGHT = 32;
  const int TT_TILE_WIDTH = 32;
  const std::string TT_LAYOUT = "dram_interleaved";

  // Compute sharding info for all buffers and store as function-level attributes
  // Format: "tt_buffer_{name}_{property}" -> value
  // This avoids needing to modify Buffer objects directly

  PrimFunc new_func = f;

  for (const auto& [param, buffer] : f->buffer_map) {
    // Compute shard info for this buffer
    ShardInfo info = ComputeShardInfo(buffer, TT_TILE_HEIGHT, TT_TILE_WIDTH);

    // Build attribute key prefix from buffer name
    std::string buffer_name = buffer->name;

    // Attach sharding metadata as function attributes
    new_func = WithAttr(new_func, "tt_buffer_" + buffer_name + "_layout", String(TT_LAYOUT));
    new_func = WithAttr(new_func, "tt_buffer_" + buffer_name + "_tile_shape",
                        Array<Integer>({Integer(TT_TILE_HEIGHT), Integer(TT_TILE_WIDTH)}));
    new_func = WithAttr(new_func, "tt_buffer_" + buffer_name + "_num_tiles_height",
                        Integer(info.tiles_height));
    new_func = WithAttr(new_func, "tt_buffer_" + buffer_name + "_num_tiles_width",
                        Integer(info.tiles_width));
    new_func = WithAttr(new_func, "tt_buffer_" + buffer_name + "_needs_padding",
                        Bool(info.needs_padding));

    if (info.needs_padding) {
      new_func = WithAttr(new_func, "tt_buffer_" + buffer_name + "_padded_shape",
                          Array<Integer>({Integer(info.padded_height), Integer(info.padded_width)}));
    }
  }

  return new_func;
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
