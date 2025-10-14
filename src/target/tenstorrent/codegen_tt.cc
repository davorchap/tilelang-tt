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
 * \brief Code generator for Tenstorrent backend (Artifact Generation stage)
 *
 * This module generates Metalium-compatible C++ kernels and metadata
 * for dry-run execution on the Tenstorrent backend.
 *
 * Supports two modes:
 * - Template-based codegen (legacy): Emits fixed hardcoded patterns
 * - IR-driven codegen: Uses visitor pattern to walk actual IR structure
 *
 * See: docs/tenstorrent/IR_DRIVEN_CODEGEN_PLAN.md
 * See: docs/tenstorrent/workstream4/Artifact Generation stage_STATUS.md
 */

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <array>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// IR-driven codegen visitors
#include "codegen_tt_compute_visitor.h"
#include "codegen_tt_reader_visitor.h"
#include "codegen_tt_writer_visitor.h"

namespace tvm {
namespace tl {

using namespace tir;
using tvm::Map;
using tvm::String;

/*!
 * \brief Extract buffer dimensions from PrimFunc for matmul
 * Returns Mt, Kt, Nt (tile counts)
 */
struct MatmulDims {
  int Mt; // M dimension in tiles
  int Kt; // K dimension in tiles
  int Nt; // N dimension in tiles
  int M;  // M dimension in elements
  int K;  // K dimension in elements
  int N;  // N dimension in elements
};

MatmulDims ExtractMatmulDims(const PrimFunc &func) {
  MatmulDims dims;

  // Try to get dimensions from grid metadata first
  auto grid_x = func->attrs.GetAttr<Integer>("tt_grid_x");
  auto grid_y = func->attrs.GetAttr<Integer>("tt_grid_y");

  if (grid_x.defined() && grid_y.defined()) {
    dims.Nt = grid_x.value()->value;
    dims.Mt = grid_y.value()->value;
    dims.Kt = dims.Mt; // Assume square for MVP

    dims.N = dims.Nt * 32;
    dims.M = dims.Mt * 32;
    dims.K = dims.Kt * 32;
  } else {
    // Fallback to default
    dims.Mt = dims.Kt = dims.Nt = 8;
    dims.M = dims.K = dims.N = 256;
  }

  return dims;
}

// =============================================================================
// IR-Driven Codegen
// =============================================================================

/*!
 * \brief Generate TT compute kernel using IR-driven visitor
 */
std::string EmitTTComputeKernelIRDriven(const PrimFunc &func) {
  TTComputeCodegenVisitor visitor(func);
  return visitor.GetFullKernel();
}

/*!
 * \brief Generate TT reader kernel using IR-driven visitor
 */
std::string EmitTTReaderKernelIRDriven(const PrimFunc &func) {
  TTReaderCodegenVisitor visitor(func);
  return visitor.GetFullKernel();
}

/*!
 * \brief Generate TT writer kernel using IR-driven visitor
 */
std::string EmitTTWriterKernelIRDriven(const PrimFunc &func) {
  TTWriterCodegenVisitor visitor(func);
  return visitor.GetFullKernel();
}

/*!
 * \brief Generate TT host program (main.cpp)
 */
std::string EmitTTHostProgram(const PrimFunc &func) {
  std::ostringstream code;

  auto quote = [](const std::string &value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (char ch : value) {
      if (ch == '"' || ch == '\\') {
        escaped.push_back('\\');
      }
      escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
  };

  // Extract matmul dimensions and partition metadata
  MatmulDims dims = ExtractMatmulDims(func);
  std::string partition_mode = "global";
  if (auto mode = func->attrs.GetAttr<String>("tt.partition_mode")) {
    partition_mode = mode.value();
  }

  std::vector<std::string> runtime_arg_names;
  if (auto names_attr =
          func->attrs.GetAttr<Array<String>>("tt.runtime_arg_names")) {
    runtime_arg_names.reserve(names_attr.value().size());
    for (const auto &name : names_attr.value()) {
      runtime_arg_names.emplace_back(name);
    }
  } else if (auto legacy_attr =
                 func->attrs.GetAttr<Array<String>>("tt_runtime_arg_names")) {
    runtime_arg_names.reserve(legacy_attr.value().size());
    for (const auto &name : legacy_attr.value()) {
      runtime_arg_names.emplace_back(name);
    }
  }
  if (runtime_arg_names.empty()) {
    if (partition_mode == "local_shard") {
      runtime_arg_names = {"tt_start_tile",
                           "tt_tile_count",
                           "Mt",
                           "Kt",
                           "Nt",
                           "Sm",
                           "Sn",
                           "Gy",
                           "Gx",
                           "tt_shard_coord_y",
                           "tt_shard_coord_x"};
    } else {
      runtime_arg_names = {"tt_start_tile", "tt_tile_count", "Mt", "Kt", "Nt"};
    }
  }
  std::unordered_map<std::string, size_t> runtime_arg_index;
  for (size_t i = 0; i < runtime_arg_names.size(); ++i) {
    runtime_arg_index[runtime_arg_names[i]] = i;
  }
  size_t arg_count = runtime_arg_names.size();

  std::vector<std::pair<std::string, int64_t>> runtime_constants;
  if (auto constants_attr =
          func->attrs.GetAttr<Map<String, ObjectRef>>("tt.runtime_constants")) {
    for (const auto &kv : constants_attr.value()) {
      std::string key = std::string(kv.first);
      const ObjectRef &value_ref = kv.second;
      int64_t value = 0;
      if (const auto *int_imm = value_ref.as<IntImmNode>()) {
        value = int_imm->value;
      } else if (const auto *float_imm = value_ref.as<FloatImmNode>()) {
        value = static_cast<int64_t>(float_imm->value);
      }
      runtime_constants.emplace_back(key, value);
    }
  }
  std::sort(runtime_constants.begin(), runtime_constants.end(),
            [](const std::pair<std::string, int64_t> &lhs,
               const std::pair<std::string, int64_t> &rhs) {
              return lhs.first < rhs.first;
            });
  std::unordered_map<std::string, int64_t> runtime_constant_lookup;
  for (const auto &kv : runtime_constants) {
    runtime_constant_lookup.emplace(kv.first, kv.second);
  }

  std::vector<std::vector<int64_t>> core_runtime_args;
  if (auto core_attr =
          func->attrs.GetAttr<Array<ObjectRef>>("tt_core_runtime_args")) {
    for (const ObjectRef &row_obj : core_attr.value()) {
      if (!row_obj.as<ffi::ArrayObj>())
        continue;
      Array<Integer> row_array = Downcast<Array<Integer>>(row_obj);
      std::vector<int64_t> row;
      row.reserve(row_array.size());
      for (const Integer &value : row_array) {
        row.push_back(value.IntValue());
      }
      if (!row.empty()) {
        if (row.size() < arg_count) {
          row.resize(arg_count, 0);
        } else if (row.size() > arg_count) {
          row.resize(arg_count);
        }
        core_runtime_args.push_back(std::move(row));
      }
    }
  }

  if (core_runtime_args.empty()) {
    if (auto tiles_attr =
            func->attrs.GetAttr<Array<ObjectRef>>("tt_tiles_per_core")) {
      for (const ObjectRef &row_obj : tiles_attr.value()) {
        if (!row_obj.as<ffi::ArrayObj>())
          continue;
        Array<Integer> row_array = Downcast<Array<Integer>>(row_obj);
        int64_t start = row_array.size() > 0 ? row_array[0].IntValue() : 0;
        int64_t count = row_array.size() > 1 ? row_array[1].IntValue() : 0;
        std::vector<int64_t> row(arg_count, 0);
        if (auto it = runtime_arg_index.find("tt_start_tile");
            it != runtime_arg_index.end()) {
          row[it->second] = start;
        }
        if (auto it = runtime_arg_index.find("tt_tile_count");
            it != runtime_arg_index.end()) {
          row[it->second] = count;
        }
        for (const auto &kv : runtime_constant_lookup) {
          auto it = runtime_arg_index.find(kv.first);
          if (it != runtime_arg_index.end()) {
            row[it->second] = kv.second;
          }
        }
        core_runtime_args.push_back(std::move(row));
      }
    }
  }

  if (core_runtime_args.empty()) {
    core_runtime_args.push_back(std::vector<int64_t>(arg_count, 0));
  }
  for (auto &row : core_runtime_args) {
    for (int64_t &value : row) {
      ICHECK_GE(value, 0) << "Runtime argument values must be non-negative";
    }
  }

  struct BufferInfo {
    std::string name;
    std::string memory;
    std::string layout;
    int tile_rows;
    int tile_cols;
    int shard_grid_y;
    int shard_grid_x;
  };
  std::vector<BufferInfo> buffers;
  buffers.reserve(func->buffer_map.size());
  for (const auto &kv : func->buffer_map) {
    const Buffer &buffer = kv.second;
    BufferInfo info;
    info.name = buffer->name;
    info.memory = "DRAM";
    info.layout = "interleaved";
    info.tile_rows = 32;
    info.tile_cols = 32;
    info.shard_grid_y = 1;
    info.shard_grid_x = 1;

    std::string attr_key = "tt.buffer." + info.name;
    if (auto meta_attr =
            func->attrs.GetAttr<Map<String, ObjectRef>>(attr_key)) {
      const auto &meta = meta_attr.value();
      if (meta.count(String("memory"))) {
        info.memory = std::string(Downcast<String>(meta[String("memory")]));
      }
      if (meta.count(String("layout"))) {
        info.layout = std::string(Downcast<String>(meta[String("layout")]));
      }
      if (meta.count(String("tile_shape"))) {
        Array<Integer> tile_shape =
            Downcast<Array<Integer>>(meta[String("tile_shape")]);
        if (tile_shape.size() >= 2) {
          info.tile_rows = tile_shape[0].IntValue();
          info.tile_cols = tile_shape[1].IntValue();
        }
      }
      if (meta.count(String("nd_shard"))) {
        Map<String, ObjectRef> nd =
            Downcast<Map<String, ObjectRef>>(meta[String("nd_shard")]);
        if (nd.count(String("projected_grid"))) {
          Array<Integer> grid =
              Downcast<Array<Integer>>(nd[String("projected_grid")]);
          if (grid.size() >= 2) {
            info.shard_grid_y = grid[0].IntValue();
            info.shard_grid_x = grid[1].IntValue();
          }
        }
      }
    }
    buffers.push_back(std::move(info));
  }
  std::sort(buffers.begin(), buffers.end(),
            [](const BufferInfo &lhs, const BufferInfo &rhs) {
              return lhs.name < rhs.name;
            });

  size_t core_count = core_runtime_args.size();

  // Generate host program source
  code << "// Generated TT Host Metadata Program\n";
  code << "// Partition-aware host summary derived from layout-aware "
          "metadata\n\n";
  code << "#include <array>\n";
  code << "#include <cstdint>\n";
  code << "#include <iostream>\n";
  code << "#include <stdexcept>\n";
  code << "#include <string>\n\n";

  code << "struct TensorAccessorArgs {\n";
  code << "  bool initialized;\n";
  code << "  const char* buffer;\n";
  code << "  const char* memory;\n";
  code << "  const char* layout;\n";
  code << "  uint32_t tile_rows;\n";
  code << "  uint32_t tile_cols;\n";
  code << "  uint32_t shard_grid_y;\n";
  code << "  uint32_t shard_grid_x;\n";
  code << "  TensorAccessorArgs()\n";
  code << "      : initialized(false), buffer(nullptr), memory(nullptr), "
          "layout(nullptr),\n";
  code << "        tile_rows(0), tile_cols(0), shard_grid_y(0), "
          "shard_grid_x(0) {}\n";
  code << "  static TensorAccessorArgs Create(const char* buffer,\n";
  code << "                                    const char* memory,\n";
  code << "                                    const char* layout,\n";
  code << "                                    uint32_t tile_rows,\n";
  code << "                                    uint32_t tile_cols,\n";
  code << "                                    uint32_t shard_grid_y,\n";
  code << "                                    uint32_t shard_grid_x) {\n";
  code << "    TensorAccessorArgs args;\n";
  code << "    args.initialized = true;\n";
  code << "    args.buffer = buffer;\n";
  code << "    args.memory = memory;\n";
  code << "    args.layout = layout;\n";
  code << "    args.tile_rows = tile_rows;\n";
  code << "    args.tile_cols = tile_cols;\n";
  code << "    args.shard_grid_y = shard_grid_y;\n";
  code << "    args.shard_grid_x = shard_grid_x;\n";
  code << "    return args;\n";
  code << "  }\n";
  code << "};\n\n";

  code << "inline void GuardTensorAccessor(const TensorAccessorArgs& args) {\n";
  code << "  if (!args.initialized) {\n";
  code << "    throw std::runtime_error(\"TensorAccessorArgs must be created "
          "via TensorAccessorArgs::Create\");\n";
  code << "  }\n";
  code << "}\n\n";

  code << "struct RuntimeConstant {\n";
  code << "  const char* name;\n";
  code << "  uint32_t value;\n";
  code << "};\n\n";

  code << "constexpr const char* kPartitionMode = " << quote(partition_mode)
       << ";\n";
  code << "constexpr uint32_t kMt = " << dims.Mt << ";\n";
  code << "constexpr uint32_t kKt = " << dims.Kt << ";\n";
  code << "constexpr uint32_t kNt = " << dims.Nt << ";\n";
  code << "constexpr uint32_t kM = " << dims.M << ";\n";
  code << "constexpr uint32_t kK = " << dims.K << ";\n";
  code << "constexpr uint32_t kN = " << dims.N << ";\n\n";

  code << "constexpr std::array<const char*, " << arg_count
       << "> kRuntimeArgNames = {";
  if (!runtime_arg_names.empty()) {
    code << "{";
    for (size_t i = 0; i < runtime_arg_names.size(); ++i) {
      if (i != 0) {
        code << ", ";
      }
      code << quote(runtime_arg_names[i]);
    }
    code << "}";
  }
  code << "};\n";

  code << "constexpr std::array<RuntimeConstant, " << runtime_constants.size()
       << "> kRuntimeConstants = {";
  if (!runtime_constants.empty()) {
    code << "{\n";
    for (size_t i = 0; i < runtime_constants.size(); ++i) {
      const auto &kv = runtime_constants[i];
      code << "    {" << quote(kv.first) << ", "
           << static_cast<uint32_t>(kv.second) << "}";
      if (i + 1 < runtime_constants.size()) {
        code << ",";
      }
      code << "\n";
    }
    code << "  }";
  }
  code << "};\n";

  code << "constexpr std::array<std::array<uint32_t, " << arg_count << ">, "
       << core_count << "> kCoreRuntimeArgs = {\n";
  for (size_t core = 0; core < core_count; ++core) {
    const auto &row = core_runtime_args[core];
    code << "    {";
    if (!row.empty()) {
      code << "{";
      for (size_t i = 0; i < row.size(); ++i) {
        if (i != 0) {
          code << ", ";
        }
        code << static_cast<uint32_t>(row[i]);
      }
      code << "}";
    }
    code << "}";
    if (core + 1 < core_count) {
      code << ",";
    }
    code << "\n";
  }
  code << "};\n\n";

  code << "static_assert(kCoreRuntimeArgs.size() >= 1, \"At least one core "
          "required\");\n";
  code << "static_assert(kRuntimeArgNames.size() == 0 || "
          "kRuntimeArgNames.size() == kCoreRuntimeArgs[0].size(),\n";
  code << "              \"Runtime argument schema mismatch\");\n\n";

  if (buffers.empty()) {
    code << "TensorAccessorArgs tensor_accessors[] = {};\n\n";
  } else {
    code << "TensorAccessorArgs tensor_accessors[] = {\n";
    for (size_t i = 0; i < buffers.size(); ++i) {
      const BufferInfo &info = buffers[i];
      code << "    TensorAccessorArgs::Create(" << quote(info.name) << ", "
           << quote(info.memory) << ", " << quote(info.layout) << ", "
           << info.tile_rows << ", " << info.tile_cols << ", "
           << info.shard_grid_y << ", " << info.shard_grid_x << ")";
      if (i + 1 < buffers.size()) {
        code << ",";
      }
      code << "\n";
    }
    code << "};\n\n";
  }

  code << "int main() {\n";
  code
      << "  std::cout << \"Tenstorrent Host Metadata Summary\" << std::endl;\n";
  code << "  std::cout << \"Partition mode: \" << kPartitionMode << "
          "std::endl;\n";
  code << "  std::cout << \"Tiled dims (Mt,Kt,Nt): \" << kMt << \", \" << kKt "
          "<< \", \" << kNt << std::endl;\n";
  code << "  std::cout << \"Element dims (M,K,N): \" << kM << \", \" << kK << "
          "\", \" << kN << std::endl;\n\n";

  code << "  for (const auto& ta : tensor_accessors) {\n";
  code << "    GuardTensorAccessor(ta);\n";
  code << "    std::cout << \"  buffer=\" << ta.buffer\n";
  code << "              << \", memory=\" << ta.memory\n";
  code << "              << \", layout=\" << ta.layout\n";
  code << "              << \", tile=\" << ta.tile_rows << \"x\" << "
          "ta.tile_cols\n";
  code << "              << \", shard=\" << ta.shard_grid_y << \"x\" << "
          "ta.shard_grid_x\n";
  code << "              << std::endl;\n";
  code << "  }\n\n";

  code << "  std::cout << \"Runtime constants:\" << std::endl;\n";
  code << "  if (kRuntimeConstants.size() == 0) {\n";
  code << "    std::cout << \"  (none)\" << std::endl;\n";
  code << "  } else {\n";
  code << "    for (const auto& constant : kRuntimeConstants) {\n";
  code << "      std::cout << \"  \" << constant.name << \" = \" << "
          "constant.value << std::endl;\n";
  code << "    }\n";
  code << "  }\n\n";

  code << "  std::cout << \"Runtime args per core (\" << "
          "kCoreRuntimeArgs.size() << \" cores)\" << std::endl;\n";
  code << "  for (size_t core = 0; core < kCoreRuntimeArgs.size(); ++core) {\n";
  code << "    const auto& args = kCoreRuntimeArgs[core];\n";
  code << "    std::cout << \"  core \" << core;\n";
  code << "    if (args.size() == 0) {\n";
  code << "      std::cout << \": (no args)\" << std::endl;\n";
  code << "      continue;\n";
  code << "    }\n";
  code << "    std::cout << \": \";\n";
  code << "    for (size_t idx = 0; idx < args.size(); ++idx) {\n";
  code << "      std::cout << kRuntimeArgNames[idx] << \"=\" << args[idx];\n";
  code << "      if (idx + 1 < args.size()) {\n";
  code << "        std::cout << \", \";\n";
  code << "      }\n";
  code << "    }\n";
  code << "    std::cout << std::endl;\n";
  code << "  }\n\n";

  code << "  return 0;\n";
  code << "}\n";

  return code.str();
}

/*!
 * \brief Generate tt.plan.json metadata
 */
std::string EmitTTPlanJSON(const PrimFunc &func) {
  std::ostringstream json;

  // Read metadata from both new and old formats for compatibility
  int grid_x_val = 1, grid_y_val = 1, grid_z_val = 1;
  int num_tiles_val = 1, num_cores_val = 64;

  // Try new metadata format first (tt.core_grid)
  auto core_grid = func->attrs.GetAttr<Array<Integer>>("tt.core_grid");
  if (core_grid.defined() && core_grid.value().size() >= 2) {
    grid_x_val = core_grid.value()[0]->value;
    grid_y_val = core_grid.value()[1]->value;
    num_tiles_val = grid_x_val * grid_y_val;
    num_cores_val = num_tiles_val; // Assuming 1:1 mapping for simplicity
  } else {
    // Fall back to old format
    auto grid_x = func->attrs.GetAttr<Integer>("tt_grid_x");
    auto grid_y = func->attrs.GetAttr<Integer>("tt_grid_y");
    auto grid_z = func->attrs.GetAttr<Integer>("tt_grid_z");
    auto num_tiles = func->attrs.GetAttr<Integer>("tt_num_tiles");
    auto num_cores = func->attrs.GetAttr<Integer>("tt_num_cores");

    if (grid_x.defined())
      grid_x_val = grid_x.value()->value;
    if (grid_y.defined())
      grid_y_val = grid_y.value()->value;
    if (grid_z.defined())
      grid_z_val = grid_z.value()->value;
    if (num_tiles.defined())
      num_tiles_val = num_tiles.value()->value;
    if (num_cores.defined())
      num_cores_val = num_cores.value()->value;
  }

  auto tiles_per_core =
      func->attrs.GetAttr<Array<Array<Integer>>>("tt_tiles_per_core");

  json << "{\n";
  json << "  \"version\": \"1.0\",\n";
  json << "  \"target\": \"tenstorrent\",\n";
  json << "  \"kernel\": \"gemm_generated\",\n";

  // Grid section
  json << "  \"grid\": {\n";
  json << "    \"x\": " << grid_x_val << ",\n";
  json << "    \"y\": " << grid_y_val << ",\n";
  json << "    \"z\": " << grid_z_val << ",\n";
  json << "    \"total_tiles\": " << num_tiles_val << "\n";
  json << "  },\n";

  // Cores section
  json << "  \"cores\": {\n";
  json << "    \"num_cores\": " << num_cores_val << ",\n";
  json << "    \"topology\": \"8x8_grid\",\n";
  json << "    \"assignments\": [\n";

  if (tiles_per_core.defined()) {
    for (size_t i = 0; i < tiles_per_core.value().size(); ++i) {
      auto assignment = tiles_per_core.value()[i];
      int start = static_cast<int>(assignment[0]->value);
      int count = static_cast<int>(assignment[1]->value);

      json << "      {\"core_id\": " << i << ", \"start_tile\": " << start
           << ", \"count\": " << count << "}";
      if (i < tiles_per_core.value().size() - 1)
        json << ",";
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
std::unordered_map<std::string, std::string>
CodegenTT(const IRModule &mod, const std::string &target) {
  std::unordered_map<std::string, std::string> artifacts;

  // Get main function
  auto funcs = mod->functions;
  PrimFunc main_func;
  for (const auto &kv : funcs) {
    if (kv.first->name_hint == "main") {
      main_func = Downcast<PrimFunc>(kv.second);
      break;
    }
  }

  if (!main_func.defined()) {
    LOG(FATAL) << "No main function found in module";
  }

  // Generate all 3 kernels using IR-driven codegen
  artifacts["reader.cpp"] = EmitTTReaderKernelIRDriven(main_func);
  artifacts["compute.cpp"] = EmitTTComputeKernelIRDriven(main_func);
  artifacts["writer.cpp"] = EmitTTWriterKernelIRDriven(main_func);

  // Generate host program (same for both modes)
  artifacts["main.cpp"] = EmitTTHostProgram(main_func);

  // Generate plan JSON (same for both modes)
  artifacts["tt.plan.json"] = EmitTTPlanJSON(main_func);

  return artifacts;
}

/*!
 * \brief Python FFI wrapper for CodegenTT
 */
Map<String, String> EmitTTArtifacts(const IRModule &mod,
                                    const std::string &target) {
  auto artifacts = CodegenTT(mod, target);

  // Convert std::unordered_map to TVM Map for Python
  Map<String, String> result;
  for (const auto &kv : artifacts) {
    result.Set(kv.first, kv.second);
  }

  return result;
}

/*!
 * \brief Build function for Tenstorrent target
 *
 * This is the entry point called by TVM's build system when
 * target="tenstorrent". It generates TT kernel code and returns a runtime
 * module containing the artifacts.
 *
 * For now, this is a dry-run implementation that generates code but doesn't
 * compile for actual hardware (that requires TT-Metalium SDK).
 */
runtime::Module BuildTileLangTT(IRModule mod, Target target) {
  // For now, just return a null module
  // The actual code generation happens via EmitTTArtifacts which is called
  // from Python in tilelang/engine/tenstorrent/lower.py
  //
  // TODO: When TT-Metalium SDK is available, this should:
  // 1. Generate artifacts via CodegenTT()
  // 2. Compile with TT-Metalium SDK
  // 3. Return a proper runtime module for execution
  return runtime::Module(nullptr);
}

// Register the function for Python FFI
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.codegen.EmitTTArtifacts", EmitTTArtifacts)
      .def("target.build.tilelang_tt", BuildTileLangTT);
});

} // namespace tl
} // namespace tvm
