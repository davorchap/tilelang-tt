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
 * \file codegen_tt.cc
 * \brief Code generator for Tenstorrent backend (WS4)
 *
 * This module generates Metalium-compatible C++ kernels and metadata
 * for dry-run execution on the Tenstorrent backend.
 *
 * See: docs/tenstorrent/workstream4/WS4_STATUS.md
 */

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

#include <sstream>
#include <string>
#include <unordered_map>

namespace tvm {
namespace tl {

using namespace tir;
using tvm::Map;
using tvm::String;

/*!
 * \brief Generate TT compute kernel C++ source
 */
std::string EmitTTComputeKernel(const PrimFunc& func) {
  std::ostringstream code;

  // Read metadata
  auto grid_x = func->attrs.GetAttr<Integer>("tt_grid_x");
  auto grid_y = func->attrs.GetAttr<Integer>("tt_grid_y");
  auto num_cores = func->attrs.GetAttr<Integer>("tt_num_cores");

  if (!grid_x.defined() || !grid_y.defined()) {
    LOG(FATAL) << "Missing TT grid metadata for codegen";
  }

  // Generate kernel header
  code << "// Generated TT Compute Kernel\n";
  code << "// Grid: " << grid_x.value()->value << "x" << grid_y.value()->value << "\n";
  code << "// Cores: " << (num_cores.defined() ? std::to_string(num_cores.value()->value) : "64")
       << "\n\n";

  code << "#include <cstdint>\n\n";

  code << "// Mock TT intrinsics for dry-run\n";
  code << "template<typename T>\n";
  code << "inline T get_arg_val(uint32_t idx) { return T(); }\n\n";

  // Generate MAIN function
  code << "void MAIN() {\n";
  code << "    // Runtime arguments\n";
  code << "    uint32_t tt_start_id = get_arg_val<uint32_t>(0);\n";
  code << "    uint32_t tt_count = get_arg_val<uint32_t>(1);\n";
  code << "    uint32_t grid_x = get_arg_val<uint32_t>(2);\n";
  code << "    uint32_t grid_y = get_arg_val<uint32_t>(3);\n\n";

  code << "    // Persistent loop\n";
  code << "    for (uint32_t i = 0; i < tt_count; ++i) {\n";
  code << "        uint32_t tile_id = tt_start_id + i;\n";
  code << "        uint32_t bx = tile_id % grid_x;\n";
  code << "        uint32_t by = tile_id / grid_x;\n\n";

  code << "        // Compute indices\n";
  code << "        uint32_t tile_m = by;\n";
  code << "        uint32_t tile_n = bx;\n\n";

  code << "        // TODO: Add circular buffer operations\n";
  code << "        // TODO: Add matmul tile operations\n";
  code << "    }\n";
  code << "}\n";

  return code.str();
}

/*!
 * \brief Generate tt.plan.json metadata
 */
std::string EmitTTPlanJSON(const PrimFunc& func) {
  std::ostringstream json;

  // Read metadata
  auto grid_x = func->attrs.GetAttr<Integer>("tt_grid_x");
  auto grid_y = func->attrs.GetAttr<Integer>("tt_grid_y");
  auto grid_z = func->attrs.GetAttr<Integer>("tt_grid_z");
  auto num_tiles = func->attrs.GetAttr<Integer>("tt_num_tiles");
  auto num_cores = func->attrs.GetAttr<Integer>("tt_num_cores");
  auto tiles_per_core = func->attrs.GetAttr<Array<Array<Integer>>>("tt_tiles_per_core");

  json << "{\n";
  json << "  \"version\": \"1.0\",\n";
  json << "  \"target\": \"tenstorrent\",\n";
  json << "  \"kernel\": \"gemm_generated\",\n";

  // Grid section
  json << "  \"grid\": {\n";
  json << "    \"x\": " << (grid_x.defined() ? grid_x.value()->value : 1) << ",\n";
  json << "    \"y\": " << (grid_y.defined() ? grid_y.value()->value : 1) << ",\n";
  json << "    \"z\": " << (grid_z.defined() ? grid_z.value()->value : 1) << ",\n";
  json << "    \"total_tiles\": " << (num_tiles.defined() ? num_tiles.value()->value : 1) << "\n";
  json << "  },\n";

  // Cores section
  json << "  \"cores\": {\n";
  json << "    \"num_cores\": " << (num_cores.defined() ? num_cores.value()->value : 64) << ",\n";
  json << "    \"topology\": \"8x8_grid\",\n";
  json << "    \"assignments\": [\n";

  if (tiles_per_core.defined()) {
    for (size_t i = 0; i < tiles_per_core.value().size(); ++i) {
      auto assignment = tiles_per_core.value()[i];
      int start = static_cast<int>(assignment[0]->value);
      int count = static_cast<int>(assignment[1]->value);

      json << "      {\"core_id\": " << i << ", \"start_tile\": " << start
           << ", \"count\": " << count << "}";
      if (i < tiles_per_core.value().size() - 1) json << ",";
      json << "\n";
    }
  }

  json << "    ]\n";
  json << "  },\n";

  // Schedule section
  json << "  \"schedule\": {\n";
  json << "    \"policy\": \"contiguous\",\n";
  json << "    \"order\": \"row_major\"\n";
  json << "  }\n";

  json << "}\n";

  return json.str();
}

/*!
 * \brief Main codegen entry point - generates all TT artifacts
 */
std::unordered_map<std::string, std::string> CodegenTT(const IRModule& mod, const std::string& target) {
  std::unordered_map<std::string, std::string> artifacts;

  // Get main function
  auto funcs = mod->functions;
  PrimFunc main_func;
  for (const auto& kv : funcs) {
    if (kv.first->name_hint == "main") {
      main_func = Downcast<PrimFunc>(kv.second);
      break;
    }
  }

  if (!main_func.defined()) {
    LOG(FATAL) << "No main function found in module";
  }

  // Generate compute kernel
  std::string compute_kernel = EmitTTComputeKernel(main_func);
  artifacts["compute.cpp"] = compute_kernel;

  // Generate plan JSON
  std::string plan_json = EmitTTPlanJSON(main_func);
  artifacts["tt.plan.json"] = plan_json;

  return artifacts;
}

/*!
 * \brief Python FFI wrapper for CodegenTT
 */
Map<String, String> EmitTTArtifacts(const IRModule& mod, const std::string& target) {
  auto artifacts = CodegenTT(mod, target);

  // Convert std::unordered_map to TVM Map for Python
  Map<String, String> result;
  for (const auto& kv : artifacts) {
    result.Set(kv.first, kv.second);
  }

  return result;
}

// Register the function for Python FFI
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.codegen.EmitTTArtifacts", EmitTTArtifacts);
});

}  // namespace tl
}  // namespace tvm
