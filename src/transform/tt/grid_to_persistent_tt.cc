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
 * \brief Transform grid-style kernel to persistent per-core loop (WS3)
 *
 * This pass converts GPU-style grid kernels to Tenstorrent's persistent
 * execution model. Each core runs a persistent loop iterating over its
 * assigned tiles, recovering block indices from the static schedule.
 *
 * See: docs/tenstorrent/workstream3/WS3_STATUS.md
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

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
  BlockIndexReplacer(Var tile_id_var, PrimExpr grid_x, PrimExpr grid_y)
      : tile_id_var_(tile_id_var), grid_x_(grid_x), grid_y_(grid_y) {}

  PrimExpr VisitExpr_(const VarNode* op) final {
    // Check if this is a blockIdx variable
    if (block_idx_vars_.count(op)) {
      std::string thread_tag = block_idx_vars_[op];

      if (thread_tag == "blockIdx.x") {
        // bx = tile_id % grid_x
        return floormod(tile_id_var_, grid_x_);
      } else if (thread_tag == "blockIdx.y") {
        // by = tile_id / grid_x
        return floordiv(tile_id_var_, grid_x_);
      } else if (thread_tag == "blockIdx.z") {
        // bz = tile_id / (grid_x * grid_y)
        return floordiv(tile_id_var_, grid_x_ * grid_y_);
      }
    }
    return GetRef<PrimExpr>(op);
  }

  void RegisterBlockIdxVar(const Var& var, const std::string& thread_tag) {
    block_idx_vars_[var.get()] = thread_tag;
  }

 private:
  Var tile_id_var_;
  PrimExpr grid_x_;
  PrimExpr grid_y_;
  std::unordered_map<const VarNode*, std::string> block_idx_vars_;
};

/*!
 * \brief Main mutator for GridToPersistentTT transformation
 *
 * Wraps kernel body with persistent loop and replaces block indices.
 */
class GridToPersistentMutator : public StmtMutator {
 public:
  explicit GridToPersistentMutator(int grid_x, int grid_y, int grid_z)
      : grid_x_(grid_x), grid_y_(grid_y), grid_z_(grid_z) {}

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

    // Create tile_id variable
    Var tile_id("tile_id", DataType::Int(32));
    Var loop_var("i", DataType::Int(32));

    // Create start_id and count variables (will be runtime args)
    Var start_id("tt_start_id", DataType::Int(32));
    Var count("tt_count", DataType::Int(32));

    // Replace block indices in the body
    BlockIndexReplacer replacer(tile_id, IntImm(DataType::Int(32), grid_x_),
                                 IntImm(DataType::Int(32), grid_y_));

    for (const auto& [var, tag] : block_idx_vars_) {
      replacer.RegisterBlockIdxVar(var, tag);
    }
    processed_body = replacer(processed_body);

    // Compute tile_id = start_id + i
    Stmt tile_id_compute = LetStmt(tile_id, start_id + loop_var, processed_body);

    // Wrap with persistent loop: for (i = 0; i < count; ++i)
    Stmt persistent_loop =
        For(loop_var, IntImm(DataType::Int(32), 0), count, ForKind::kSerial, tile_id_compute);

    return persistent_loop;
  }

 private:
  int grid_x_;
  int grid_y_;
  int grid_z_;
  std::vector<std::pair<Var, std::string>> block_idx_vars_;
};

/*!
 * \brief Transform grid-style kernel to persistent per-core loop
 *
 * This pass reads WS2 schedule metadata and wraps the kernel body
 * with a persistent loop that iterates over assigned tiles.
 *
 * \param f The PrimFunc to process
 * \return Enhanced PrimFunc with persistent loop structure
 */
PrimFunc GridToPersistentTTImpl(PrimFunc f) {
  // Step 1: Read grid dimensions from WS2 metadata
  auto grid_x_attr = f->attrs.GetAttr<Integer>("tt_grid_x");
  auto grid_y_attr = f->attrs.GetAttr<Integer>("tt_grid_y");
  auto grid_z_attr = f->attrs.GetAttr<Integer>("tt_grid_z");

  if (!grid_x_attr.defined() || !grid_y_attr.defined() || !grid_z_attr.defined()) {
    // No TT metadata, skip transformation
    return f;
  }

  int grid_x = grid_x_attr.value()->value;
  int grid_y = grid_y_attr.value()->value;
  int grid_z = grid_z_attr.value()->value;

  // Step 2: Apply the transformation
  GridToPersistentMutator mutator(grid_x, grid_y, grid_z);
  Stmt new_body = mutator.Transform(f->body);

  // Step 3: Create new function with transformed body
  PrimFunc new_func = f;
  auto n = make_object<PrimFuncNode>(*f.get());
  n->body = new_body;
  new_func = PrimFunc(n);

  // Step 4: Attach runtime args metadata
  // Format: {start_id: int32, count: int32, grid_x: int32, grid_y: int32}
  Map<String, ObjectRef> runtime_args;
  runtime_args.Set("start_id", String("int32"));
  runtime_args.Set("count", String("int32"));
  runtime_args.Set("grid_x", String("int32"));
  runtime_args.Set("grid_y", String("int32"));

  new_func = WithAttr(new_func, "tt_runtime_args", runtime_args);
  new_func = WithAttr(new_func, "tt_persistent_loop", Bool(true));

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
