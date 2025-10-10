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
 * \file grid_to_persistent_tt.cc
 * \brief Transform grid-style kernel to persistent per-core loop (Persistent Transform stage)
 *
 * This pass converts GPU-style grid kernels to Tenstorrent's persistent
 * execution model. Each core runs a persistent loop iterating over its
 * assigned tiles, recovering block indices from the static schedule.
 *
 * See: docs/tenstorrent/workstream3/Persistent Transform stage_STATUS.md
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Mutator to replace block indices with computed values
 *
 * Replaces references to blockIdx.x/y/z variables with expressions
 * that compute them from tile_id and grid dimensions.
 */
class BlockIndexReplacer : public StmtExprMutator {
 public:
  BlockIndexReplacer(Var tile_id_var, int grid_x, int grid_y, int grid_z)
      : tile_id_var_(tile_id_var),
        grid_x_(std::max(grid_x, 1)),
        grid_y_(std::max(grid_y, 1)),
        grid_z_(std::max(grid_z, 1)) {}

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = block_idx_vars_.find(op);
    if (it == block_idx_vars_.end()) {
      return GetRef<PrimExpr>(op);
    }

    const std::string& thread_tag = it->second;
    PrimExpr tile_expr = tile_id_var_;
    DataType dtype = tile_id_var_->dtype;

    if (thread_tag == "blockIdx.x") {
      if (grid_x_ == 1) {
        return make_const(dtype, 0);
      }
      return floormod(tile_expr, make_const(dtype, grid_x_));
    }

    if (thread_tag == "blockIdx.y") {
      if (grid_y_ == 1) {
        return make_const(dtype, 0);
      }
      PrimExpr div_x = floordiv(tile_expr, make_const(dtype, grid_x_));
      return floormod(div_x, make_const(dtype, grid_y_));
    }

    if (thread_tag == "blockIdx.z") {
      if (grid_z_ == 1) {
        return make_const(dtype, 0);
      }
      PrimExpr denom = make_const(dtype, grid_x_ * grid_y_);
      return floordiv(tile_expr, denom);
    }

    return GetRef<PrimExpr>(op);
  }

  void RegisterBlockIdxVar(const Var& var, const std::string& thread_tag) {
    block_idx_vars_[var.get()] = thread_tag;
  }

 private:
  Var tile_id_var_;
  int grid_x_;
  int grid_y_;
  int grid_z_;
  std::unordered_map<const VarNode*, std::string> block_idx_vars_;
};

/*!
 * \brief Main mutator for GridToPersistentTT transformation
 *
 * Wraps kernel body with persistent loop and replaces block indices.
 */
class GridToPersistentMutator : public StmtMutator {
 public:
  GridToPersistentMutator(const Var& start_param, const Var& count_param, int grid_x, int grid_y,
                          int grid_z)
      : start_param_(start_param),
        count_param_(count_param),
        grid_x_(std::max(grid_x, 1)),
        grid_y_(std::max(grid_y, 1)),
        grid_z_(std::max(grid_z, 1)) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      std::string thread_tag = iv->thread_tag;

      if (thread_tag == "blockIdx.x" || thread_tag == "blockIdx.y" ||
          thread_tag == "blockIdx.z") {
        // Register this variable for replacement
        block_idx_vars_.push_back({iv->var, thread_tag});

        // Remove the attr statement, just process body
        return VisitStmt(op->body);
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt Transform(Stmt body) {
    // Visit the body to collect blockIdx variables
    Stmt processed_body = VisitStmt(body);

    // Create tile_id variable and loop variable
    Var tile_id("tt_tile_id", DataType::Int(32));
    Var loop_var("tt_tile_iter", DataType::Int(32));

    // Replace block indices in the body
    BlockIndexReplacer replacer(tile_id, grid_x_, grid_y_, grid_z_);

    for (const auto& [var, tag] : block_idx_vars_) {
      replacer.RegisterBlockIdxVar(var, tag);
    }
    processed_body = replacer(processed_body);

    // Compute tile_id = start_id + i
    Stmt tile_id_compute = LetStmt(tile_id, start_param_ + loop_var, processed_body);

    // Wrap with persistent loop: for (i = 0; i < count; ++i)
    Stmt persistent_loop = For(loop_var, make_const(DataType::Int(32), 0), count_param_,
                               ForKind::kSerial, tile_id_compute);

    return persistent_loop;
  }

 private:
  Var start_param_;
  Var count_param_;
  int grid_x_;
  int grid_y_;
  int grid_z_;
  std::vector<std::pair<Var, std::string>> block_idx_vars_;
};

/*!
 * \brief Transform grid-style kernel to persistent per-core loop
 *
 * This pass reads Metadata Inference stage schedule metadata and wraps the kernel body
 * with a persistent loop that iterates over assigned tiles.
 *
 * \param f The PrimFunc to process
 * \return Enhanced PrimFunc with persistent loop structure
 */
PrimFunc GridToPersistentTTImpl(PrimFunc f) {
  // Step 1: Read grid dimensions from Metadata Inference stage metadata
  auto grid_x_attr = f->attrs.GetAttr<Integer>("tt_grid_x");
  auto grid_y_attr = f->attrs.GetAttr<Integer>("tt_grid_y");
  auto grid_z_attr = f->attrs.GetAttr<Integer>("tt_grid_z");

  if (!grid_x_attr.defined() || !grid_y_attr.defined() || !grid_z_attr.defined()) {
    // No TT metadata, skip transformation
    return f;
  }

  auto existing_persistent = f->attrs.GetAttr<Bool>("tt_persistent_loop");
  if (existing_persistent.defined() && existing_persistent.value()->value) {
    // Already transformed
    return f;
  }

  int grid_x = grid_x_attr.value()->value;
  int grid_y = grid_y_attr.value()->value;
  int grid_z = grid_z_attr.value()->value;

  std::string partition_mode = "global";
  if (auto mode = f->attrs.GetAttr<String>("tt.partition_mode")) {
    partition_mode = mode.value();
  }

  int grid_tiles_m = std::max(grid_y * grid_z, 1);
  int grid_tiles_n = std::max(grid_x, 1);
  int local_tiles_m = grid_tiles_m;
  int local_tiles_n = grid_tiles_n;
  int shard_grid_y = 1;
  int shard_grid_x = 1;

  if (auto grid_tiles_attr = f->attrs.GetAttr<Array<Integer>>("tt.grid_tiles")) {
    const auto& arr = grid_tiles_attr.value();
    if (arr.size() >= 2) {
      grid_tiles_m = static_cast<int>(arr[0].IntValue());
      grid_tiles_n = static_cast<int>(arr[1].IntValue());
    }
  }
  if (auto local_tiles_attr = f->attrs.GetAttr<Array<Integer>>("tt.local_shape_tiles")) {
    const auto& arr = local_tiles_attr.value();
    if (arr.size() >= 2) {
      local_tiles_m = static_cast<int>(arr[0].IntValue());
      local_tiles_n = static_cast<int>(arr[1].IntValue());
    }
  }
  if (auto shard_grid_attr = f->attrs.GetAttr<Array<Integer>>("tt.shard_grid")) {
    const auto& arr = shard_grid_attr.value();
    if (arr.size() >= 2) {
      shard_grid_y = static_cast<int>(arr[0].IntValue());
      shard_grid_x = static_cast<int>(arr[1].IntValue());
    }
  }

  Array<String> runtime_arg_names;
  if (auto names = f->attrs.GetAttr<Array<String>>("tt.runtime_arg_names")) {
    runtime_arg_names = names.value();
  } else if (auto names = f->attrs.GetAttr<Array<String>>("tt_runtime_arg_names")) {
    runtime_arg_names = names.value();
  }
  if (runtime_arg_names.empty()) {
    if (partition_mode == "local_shard") {
      runtime_arg_names = {
          String("start_id"), String("count"), String("Mt"), String("Kt"), String("Nt"),
          String("Sm"),      String("Sn"),   String("Gy"), String("Gx"), String("sy"),
          String("sx")};
    } else {
      runtime_arg_names = {String("start_id"), String("count"), String("Mt"),
                           String("Kt"), String("Nt")};
    }
  }
  while (runtime_arg_names.size() < 2) {
    runtime_arg_names.push_back(
        String("arg" + std::to_string(runtime_arg_names.size())));
  }

  // Create runtime parameter variables (start tile, tile count)
  Var start_param("tt_start_tile", DataType::Int(32));
  Var count_param("tt_tile_count", DataType::Int(32));
  runtime_arg_names.Set(0, String(start_param->name_hint));
  runtime_arg_names.Set(1, String(count_param->name_hint));

  // Step 2: Apply the transformation
  GridToPersistentMutator mutator(start_param, count_param, grid_x, grid_y, grid_z);
  Stmt new_body = mutator.Transform(f->body);

  // Step 3: Create new function with transformed body and params
  auto n = make_object<PrimFuncNode>(*f.get());
  Array<Var> new_params = n->params;
  new_params.push_back(start_param);
  new_params.push_back(count_param);
  n->params = new_params;
  n->body = new_body;
  PrimFunc new_func = PrimFunc(n);

  // Step 4: Attach updated runtime metadata
  Map<String, ObjectRef> runtime_args;

  Map<String, ObjectRef> start_info;
  start_info.Set("name", String(start_param->name_hint));
  start_info.Set("dtype", String("int32"));
  start_info.Set("semantic", String("tile_start"));
  runtime_args.Set("start_tile", start_info);

  Map<String, ObjectRef> count_info;
  count_info.Set("name", String(count_param->name_hint));
  count_info.Set("dtype", String("int32"));
  count_info.Set("semantic", String("tile_count"));
  runtime_args.Set("tile_count", count_info);

  Array<Integer> grid_shape;
  grid_shape.push_back(Integer(grid_x));
  grid_shape.push_back(Integer(grid_y));
  grid_shape.push_back(Integer(grid_z));
  runtime_args.Set("grid_shape", grid_shape);

  Array<Integer> grid_tiles_array;
  grid_tiles_array.push_back(Integer(grid_tiles_m));
  grid_tiles_array.push_back(Integer(grid_tiles_n));
  runtime_args.Set("grid_tiles", grid_tiles_array);

  Array<Integer> local_tiles_array;
  local_tiles_array.push_back(Integer(local_tiles_m));
  local_tiles_array.push_back(Integer(local_tiles_n));
  runtime_args.Set("local_shape_tiles", local_tiles_array);

  Array<Integer> shard_grid_array;
  shard_grid_array.push_back(Integer(shard_grid_y));
  shard_grid_array.push_back(Integer(shard_grid_x));
  runtime_args.Set("shard_grid", shard_grid_array);

  Map<String, ObjectRef> runtime_constants;
  runtime_constants.Set("Mt", Integer(grid_tiles_m));
  runtime_constants.Set("Nt", Integer(grid_tiles_n));
  runtime_constants.Set("Kt", Integer(1));
  if (partition_mode == "local_shard") {
    runtime_constants.Set("Sm", Integer(local_tiles_m));
    runtime_constants.Set("Sn", Integer(local_tiles_n));
    runtime_constants.Set("Gy", Integer(shard_grid_y));
    runtime_constants.Set("Gx", Integer(shard_grid_x));
  }
  runtime_args.Set("runtime_constants", runtime_constants);
  runtime_args.Set("partition_mode", String(partition_mode));

  int iter_ndims = 1;
  if (grid_y > 1) {
    iter_ndims += 1;
  }
  if (grid_z > 1) {
    iter_ndims += 1;
  }
  runtime_args.Set("iteration_ndims", Integer(iter_ndims));
  runtime_args.Set("iteration_order", String("row_major_xyz"));

  Array<String> dim_symbols;
  dim_symbols.push_back(String("bx"));
  if (grid_y > 1) {
    dim_symbols.push_back(String("by"));
  }
  if (grid_z > 1) {
    dim_symbols.push_back(String("bz"));
  }
  runtime_args.Set("iteration_symbols", dim_symbols);

  Array<String> param_order;
  for (const auto& name : runtime_arg_names) {
    param_order.push_back(name);
  }
  runtime_args.Set("param_order", param_order);
  runtime_args.Set("arg_names", runtime_arg_names);

  new_func = WithAttr(new_func, "tt_runtime_args", runtime_args);
  new_func = WithAttr(new_func, "tt_persistent_loop", Bool(true));
  new_func = WithAttr(new_func, "tt_persistent_iteration_ndims", Integer(iter_ndims));
  new_func = WithAttr(new_func, "tt.partition_mode", String(partition_mode));
  new_func = WithAttr(new_func, "tt.grid_tiles", grid_tiles_array);
  new_func = WithAttr(new_func, "tt.local_shape_tiles", local_tiles_array);
  new_func = WithAttr(new_func, "tt.shard_grid", shard_grid_array);
  new_func = WithAttr(new_func, "tt.runtime_constants", runtime_constants);
  new_func = WithAttr(new_func, "tt.runtime_arg_names", runtime_arg_names);

  return new_func;
}

using namespace tir::transform;

/*!
 * \brief Create the GridToPersistentTT pass
 *
 * \return The TIR pass
 */
Pass GridToPersistentTT() {
  auto pass_func = [=](PrimFunc f, const IRModule& m, const PassContext& ctx) {
    return GridToPersistentTTImpl(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.GridToPersistentTT", {});
}

// Register the pass for Python FFI
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.GridToPersistentTT", GridToPersistentTT);
});

}  // namespace tl
}  // namespace tvm
