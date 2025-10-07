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
 * \file tile_pad_tt.cc
 * \brief Insert padding for non-tile-aligned buffers (WS3 Phase 2)
 *
 * This pass handles buffers with dimensions that are not multiples of the
 * tile size (typically 32). It annotates the IR with padding metadata that
 * will be used during codegen to properly handle boundary tiles.
 *
 * Padding Strategy:
 * - For buffers with needs_padding=1 (from WS2), compute padded dimensions
 * - Padded dimension = ceil(original_dim / tile_size) * tile_size
 * - Stamp padded_shape metadata for use in memory allocation
 * - For Phase 2, actual padding logic will be in codegen
 *
 * Example: 250×250 buffer with tile_size=32
 * - Original: 250×250
 * - Num tiles: ceil(250/32) = 8 tiles per dimension
 * - Padded: 8*32 = 256×256
 * - Padding: 6 extra elements per dimension
 *
 * See: docs/tenstorrent/UNIFIED_MATMUL_MVP_PLAN.md Phase 2 WS3-Extended
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/function.h>
#include <tvm/tir/transform.h>

#include <string>
#include <unordered_map>

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Padding information for a buffer
 */
struct PaddingInfo {
  Array<Integer> original_shape;
  Array<Integer> padded_shape;
  Array<Integer> padding_amount;  // Per dimension
  bool needs_padding;
};

/*!
 * \brief Compute padding for non-tile-aligned dimensions
 *
 * \param original_dim Original dimension size
 * \param tile_size Tile size (typically 32)
 * \return Padded dimension size
 */
int64_t ComputePaddedDim(int64_t original_dim, int tile_size = 32) {
  // Compute number of tiles needed
  int64_t num_tiles = (original_dim + tile_size - 1) / tile_size;

  // Padded dimension is num_tiles * tile_size
  return num_tiles * tile_size;
}

/*!
 * \brief Extract padding info from WS2 metadata
 *
 * \param f The PrimFunc with WS2 sharding metadata
 * \param param The buffer parameter to check
 * \param param_name Buffer parameter name
 * \return PaddingInfo structure
 */
PaddingInfo ExtractPaddingInfo(const PrimFunc& f, const Buffer& param, const std::string& param_name) {
  PaddingInfo info;
  info.needs_padding = false;

  // Check for WS2 padding metadata
  std::string needs_padding_key = "tt_buffer_" + param_name + "_needs_padding";
  auto needs_padding_attr = f->attrs.GetAttr<Integer>(needs_padding_key);

  if (!needs_padding_attr.defined()) {
    return info;  // No WS2 metadata for this buffer
  }

  bool needs_padding = needs_padding_attr.value()->value != 0;
  info.needs_padding = needs_padding;

  if (!needs_padding) {
    // Already tile-aligned, no padding needed
    info.original_shape.clear();
    for (const auto& dim : param->shape) {
      if (auto dim_int = dim.as<IntImmNode>()) {
        info.original_shape.push_back(Integer(dim_int->value));
        info.padded_shape.push_back(Integer(dim_int->value));
        info.padding_amount.push_back(Integer(0));
      }
    }
    return info;
  }

  // Extract tile shape from WS2 metadata
  std::string tile_shape_key = "tt_buffer_" + param_name + "_tile_shape";
  auto tile_shape_attr = f->attrs.GetAttr<Array<Integer>>(tile_shape_key);

  int tile_height = 32;  // Default
  int tile_width = 32;

  if (tile_shape_attr.defined() && tile_shape_attr.value().size() >= 2) {
    tile_height = static_cast<int>(tile_shape_attr.value()[0]->value);
    tile_width = static_cast<int>(tile_shape_attr.value()[1]->value);
  }

  // Compute padded dimensions
  for (size_t i = 0; i < param->shape.size(); ++i) {
    if (auto dim_int = param->shape[i].as<IntImmNode>()) {
      int64_t original = dim_int->value;
      int tile_size = (i == 0) ? tile_height : tile_width;
      int64_t padded = ComputePaddedDim(original, tile_size);
      int64_t padding = padded - original;

      info.original_shape.push_back(Integer(original));
      info.padded_shape.push_back(Integer(padded));
      info.padding_amount.push_back(Integer(padding));
    }
  }

  return info;
}

/*!
 * \brief Main implementation of TilePadTT pass
 *
 * Computes padding metadata for non-tile-aligned buffers.
 * Attaches padded shape information to function attributes for use in codegen.
 *
 * \param f The PrimFunc to process
 * \return Enhanced PrimFunc with padding metadata
 */
PrimFunc TilePadTTImpl(PrimFunc f) {
  // Step 1: Check if this is a TT function with WS2 metadata
  auto schedule_policy = f->attrs.GetAttr<String>("tt_schedule_policy");
  if (!schedule_policy.defined()) {
    // Not a TT function, skip transformation
    return f;
  }

  // Step 2: Process each buffer parameter
  std::unordered_map<std::string, PaddingInfo> padding_map;

  for (const auto& param : f->params) {
    if (auto buf = f->buffer_map.Get(param)) {
      std::string param_name = param->name_hint;
      PaddingInfo info = ExtractPaddingInfo(f, buf.value(), param_name);

      if (info.needs_padding) {
        padding_map[param_name] = info;
      }
    }
  }

  // If no buffers need padding, return unchanged
  if (padding_map.empty()) {
    return f;
  }

  // Step 3: Build padding metadata for function attributes
  Map<String, ObjectRef> padding_metadata;

  for (const auto& pair : padding_map) {
    const std::string& buf_name = pair.first;
    const PaddingInfo& info = pair.second;

    // Create metadata for this buffer
    Map<String, ObjectRef> buf_metadata;
    buf_metadata.Set("needs_padding", Bool(info.needs_padding));
    buf_metadata.Set("original_shape", info.original_shape);
    buf_metadata.Set("padded_shape", info.padded_shape);
    buf_metadata.Set("padding_amount", info.padding_amount);

    padding_metadata.Set(buf_name, buf_metadata);
  }

  // Step 4: Attach padding metadata to function
  PrimFunc new_func = WithAttr(f, "tt_padding_info", padding_metadata);

  return new_func;
}

using namespace tir::transform;

/*!
 * \brief Create the TilePadTT pass
 *
 * \return The TIR pass
 */
Pass TilePadTT() {
  auto pass_func = [=](PrimFunc f, const IRModule& m, const PassContext& ctx) {
    return TilePadTTImpl(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.TilePadTT", {});
}

// Register the pass for Python FFI
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.TilePadTT", TilePadTT);
});

}  // namespace tl
}  // namespace tvm
