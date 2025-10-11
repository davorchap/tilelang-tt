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
 * \file tensorize_tt.cc
 * \brief Lower high-level matmul to TT intrinsics (Persistent Transform stage)
 *
 * This pass identifies high-level GEMM operations (T.gemm) and annotates them
 * with Tenstorrent-specific intrinsic metadata for use in codegen.
 *
 * Transformation:
 * - Identifies AttrStmt nodes with "pragma_gemm" or similar matmul markers
 * - Annotates with TT matmul intrinsic type (matmul_tiles, matmul_init, etc.)
 * - Stamps operand buffer IDs and accumulation flags
 * - For Phase 2, actual intrinsic lowering happens in codegen
 *
 * TT Matmul Intrinsics:
 * - matmul_tiles_init(cb_a, cb_b, cb_c) - Initialize accumulator
 * - matmul_tiles(cb_a, cb_b, cb_c, accumulate=true/false) - Perform matmul
 *
 * See: docs/tenstorrent/passes/tensorize_tt.md for detailed specification
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <initializer_list>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

inline Stmt MakeIntrinsic(const char *op_name,
                          std::initializer_list<PrimExpr> args = {}) {
  Array<PrimExpr> call_args;
  for (const PrimExpr &arg : args) {
    call_args.push_back(arg);
  }
  PrimExpr call = Call(DataType::Void(), Op::Get(op_name), call_args);
  return Evaluate(call);
}

struct MatmulPatternInfo {
  Map<String, ObjectRef> metadata;
  const BufferStoreNode *store{nullptr};
  const ForNode *reduction_loop{nullptr};
};

struct MatmulCollection {
  std::vector<MatmulPatternInfo> infos;
  std::unordered_map<const BufferStoreNode *, int> store_to_index;
  std::unordered_map<const ForNode *, std::vector<int>>
      reduction_loop_to_indices;
};

struct MatmulMatch {
  Buffer buffer_a;
  Array<PrimExpr> indices_a;
  Buffer buffer_b;
  Array<PrimExpr> indices_b;
  Buffer buffer_c;
  Array<PrimExpr> indices_c;
  bool accumulate{false};
};

inline const BufferLoadNode *AsBufferLoad(const PrimExpr &expr) {
  if (const auto *load = expr.as<BufferLoadNode>()) {
    return load;
  }
  if (const auto *cast = expr.as<CastNode>()) {
    return AsBufferLoad(cast->value);
  }
  return nullptr;
}

inline void CollectVars(const PrimExpr &expr,
                        std::unordered_set<const VarNode *> *vars) {
  PostOrderVisit(expr, [&](const ObjectRef &obj) {
    if (const auto *v = obj.as<VarNode>()) {
      vars->insert(v);
    }
  });
}

inline std::unordered_set<const VarNode *>
CollectVars(const Array<PrimExpr> &indices) {
  std::unordered_set<const VarNode *> vars;
  for (const PrimExpr &idx : indices) {
    CollectVars(idx, &vars);
  }
  return vars;
}

bool TryMatchMatmul(const BufferStoreNode *store, MatmulMatch *match) {
  const auto *add = store->value.as<AddNode>();
  if (add == nullptr) {
    return false;
  }

  const BufferLoadNode *c_load = AsBufferLoad(add->a);

  const MulNode *mul = add->b.as<MulNode>();
  if (c_load == nullptr || !store->buffer.same_as(c_load->buffer)) {
    c_load = AsBufferLoad(add->b);
    mul = add->a.as<MulNode>();
  }

  if (c_load == nullptr || !store->buffer.same_as(c_load->buffer) ||
      mul == nullptr) {
    return false;
  }

  const BufferLoadNode *load_a = AsBufferLoad(mul->a);
  const BufferLoadNode *load_b = AsBufferLoad(mul->b);

  if (load_a == nullptr || load_b == nullptr) {
    return false;
  }

  match->buffer_c = store->buffer;
  match->indices_c = store->indices;
  match->buffer_a = load_a->buffer;
  match->indices_a = load_a->indices;
  match->buffer_b = load_b->buffer;
  match->indices_b = load_b->indices;
  match->accumulate = true;
  return true;
}

class MatmulPatternCollector : public StmtVisitor {
public:
  MatmulCollection Collect(const Stmt &stmt) {
    patterns_.clear();
    store_to_index_.clear();
    reduction_loop_to_indices_.clear();

    VisitStmt(stmt);

    MatmulCollection result;
    result.infos = patterns_;
    result.store_to_index = store_to_index_;
    result.reduction_loop_to_indices = reduction_loop_to_indices_;
    return result;
  }

protected:
  void VisitStmt_(const AttrStmtNode *op) override {
    bool is_gemm =
        (op->attr_key == "pragma_gemm" || op->attr_key == "tl.gemm" ||
         op->attr_key == "gemm_operation");
    size_t before = patterns_.size();
    if (is_gemm) {
      ++gemm_attr_depth_;
    }
    StmtVisitor::VisitStmt_(op);
    if (is_gemm) {
      --gemm_attr_depth_;
      if (patterns_.size() == before) {
        // Fallback: pragma without a recognizable matmul loop
        Map<String, ObjectRef> info;
        info.Set("source", String("pragma"));
        info.Set("buffer_a", String(""));
        info.Set("buffer_b", String(""));
        info.Set("buffer_c", String(""));
        info.Set("A_indices", Array<PrimExpr>());
        info.Set("B_indices", Array<PrimExpr>());
        info.Set("C_indices", Array<PrimExpr>());
        info.Set("accumulate", Bool(true)); // Changed to true
        info.Set("loop_vars", Array<String>());
        info.Set("reduction_var", String(""));
        // CB IDs will be resolved during mutation
        info.Set("cb_in0", Integer(-1));
        info.Set("cb_in1", Integer(-1));
        info.Set("cb_out", Integer(-1));

        MatmulPatternInfo placeholder;
        placeholder.metadata = info;
        patterns_.push_back(std::move(placeholder));
      }
    }
  }

  void VisitStmt_(const ForNode *op) override {
    loop_stack_.push_back(op->loop_var);
    loop_node_stack_.push_back(op);
    StmtVisitor::VisitStmt_(op);
    loop_stack_.pop_back();
    loop_node_stack_.pop_back();
  }

  void VisitStmt_(const BufferStoreNode *op) override {
    MatmulMatch match;
    if (TryMatchMatmul(op, &match)) {
      Map<String, ObjectRef> info;
      info.Set("source", String(gemm_attr_depth_ > 0 ? "pragma" : "loop"));
      info.Set("buffer_a", String(match.buffer_a->name));
      info.Set("buffer_b", String(match.buffer_b->name));
      info.Set("buffer_c", String(match.buffer_c->name));
      info.Set("A_indices", match.indices_a);
      info.Set("B_indices", match.indices_b);
      info.Set("C_indices", match.indices_c);
      info.Set("accumulate", Bool(match.accumulate));

      Array<String> loop_vars;
      for (const Var &v : loop_stack_) {
        loop_vars.push_back(String(v->name_hint));
      }
      info.Set("loop_vars", loop_vars);

      auto vars_a = CollectVars(match.indices_a);
      auto vars_b = CollectVars(match.indices_b);
      auto vars_c = CollectVars(match.indices_c);
      for (const VarNode *v : vars_c) {
        vars_a.erase(v);
        vars_b.erase(v);
      }
      vars_a.insert(vars_b.begin(), vars_b.end());
      if (vars_a.size() == 1) {
        const VarNode *red = *vars_a.begin();
        info.Set("reduction_var", String(red->name_hint));
      }

      // Initialize CB IDs to default values (will be resolved during mutation)
      info.Set("cb_in0", Integer(-1));
      info.Set("cb_in1", Integer(-1));
      info.Set("cb_out", Integer(-1));

      MatmulPatternInfo pattern_info;
      pattern_info.metadata = info;
      pattern_info.store = op;

      String reduction_var;
      if (info.count("reduction_var")) {
        reduction_var = Downcast<String>(info["reduction_var"]);
      }

      const ForNode *reduction_loop = nullptr;
      if (!reduction_var.empty()) {
        for (auto it = loop_node_stack_.rbegin(); it != loop_node_stack_.rend();
             ++it) {
          if ((*it)->loop_var->name_hint == reduction_var) {
            reduction_loop = *it;
            break;
          }
        }
      }
      pattern_info.reduction_loop = reduction_loop;

      int pattern_index = static_cast<int>(patterns_.size());
      patterns_.push_back(std::move(pattern_info));
      store_to_index_[op] = pattern_index;
      if (reduction_loop != nullptr) {
        reduction_loop_to_indices_[reduction_loop].push_back(pattern_index);
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

private:
  std::vector<MatmulPatternInfo> patterns_;
  std::unordered_map<const BufferStoreNode *, int> store_to_index_;
  std::unordered_map<const ForNode *, std::vector<int>>
      reduction_loop_to_indices_;
  std::vector<Var> loop_stack_;
  std::vector<const ForNode *> loop_node_stack_;
  int gemm_attr_depth_{0};
};

} // namespace

/*!
 * \brief Visitor to identify and annotate matmul operations
 */
class TensorizeMutator : public StmtMutator {
public:
  TensorizeMutator(MatmulCollection *collection,
                   const std::unordered_map<std::string, int> &buffer_cb_map)
      : collection_(collection), buffer_cb_map_(buffer_cb_map) {}

  Stmt VisitStmt_(const AttrStmtNode *op) override {
    if (op->attr_key == "pragma_gemm" || op->attr_key == "tl.gemm" ||
        op->attr_key == "gemm_operation") {
      // Strip legacy pragma wrapper; intrinsic sequence will be emitted
      // directly.
      return VisitStmt(op->body);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) override {
    auto it = collection_->store_to_index.find(op);
    if (it == collection_->store_to_index.end()) {
      return StmtMutator::VisitStmt_(op);
    }

    int pattern_index = it->second;
    MatmulPatternInfo &pattern_info = collection_->infos[pattern_index];
    Map<String, ObjectRef> metadata = pattern_info.metadata;

    int cb_in0_id = ResolveCBId(metadata, "buffer_a", /*fallback=*/0);
    int cb_in1_id = ResolveCBId(metadata, "buffer_b", /*fallback=*/1);
    int cb_out_id = ResolveCBId(metadata, "buffer_c", /*fallback=*/16);

    metadata.Set("cb_in0", Integer(cb_in0_id));
    metadata.Set("cb_in1", Integer(cb_in1_id));
    metadata.Set("cb_out", Integer(cb_out_id));
    pattern_info.metadata = metadata;

    PrimExpr cb_in0 = Integer(cb_in0_id);
    PrimExpr cb_in1 = Integer(cb_in1_id);
    PrimExpr cb_out = Integer(cb_out_id);
    PrimExpr one = Integer(1);
    PrimExpr zero = Integer(0);

    Array<Stmt> seq;
    seq.push_back(MakeIntrinsic("tt.cb_wait_front", {cb_in0, one}));
    seq.push_back(MakeIntrinsic("tt.cb_wait_front", {cb_in1, one}));
    seq.push_back(MakeIntrinsic(
        "tt.matmul_tiles", {cb_in0, cb_in1, zero, zero, zero, Integer(0)}));
    seq.push_back(MakeIntrinsic("tt.cb_pop_front", {cb_in0, one}));
    seq.push_back(MakeIntrinsic("tt.cb_pop_front", {cb_in1, one}));

    // Body no longer wrapped in legacy attr.
    return SeqStmt::Flatten(seq);
  }

  Stmt VisitStmt_(const ForNode *op) override {
    bool is_reduction_loop =
        collection_->reduction_loop_to_indices.count(op) > 0;
    Stmt new_stmt = StmtMutator::VisitStmt_(op);
    if (!is_reduction_loop) {
      return new_stmt;
    }
    ICHECK(new_stmt.as<ForNode>() != nullptr)
        << "Expected ForNode after mutation";

    const auto &indices = collection_->reduction_loop_to_indices.at(op);
    ICHECK_EQ(indices.size(), 1)
        << "Multiple matmuls per reduction loop not yet supported";
    int pattern_index = indices[0];
    MatmulPatternInfo &pattern_info = collection_->infos[pattern_index];
    Map<String, ObjectRef> metadata = pattern_info.metadata;

    int cb_in0_id = ResolveCBId(metadata, "buffer_a", /*fallback=*/0);
    int cb_in1_id = ResolveCBId(metadata, "buffer_b", /*fallback=*/1);
    int cb_out_id = ResolveCBId(metadata, "buffer_c", /*fallback=*/16);

    metadata.Set("cb_in0", Integer(cb_in0_id));
    metadata.Set("cb_in1", Integer(cb_in1_id));
    metadata.Set("cb_out", Integer(cb_out_id));
    pattern_info.metadata = metadata;

    PrimExpr cb_in0 = Integer(cb_in0_id);
    PrimExpr cb_in1 = Integer(cb_in1_id);
    PrimExpr cb_out = Integer(cb_out_id);
    PrimExpr one = Integer(1);
    PrimExpr zero = Integer(0);

    Array<Stmt> seq;
    seq.push_back(MakeIntrinsic("tt.tile_regs_acquire"));
    seq.push_back(MakeIntrinsic("tt.mm_init", {cb_in0, cb_in1, cb_out}));
    seq.push_back(new_stmt);
    seq.push_back(MakeIntrinsic("tt.tile_regs_commit"));
    seq.push_back(MakeIntrinsic("tt.tile_regs_wait"));
    seq.push_back(MakeIntrinsic("tt.cb_reserve_back", {cb_out, one}));
    seq.push_back(MakeIntrinsic("tt.pack_tile", {zero, cb_out}));
    seq.push_back(MakeIntrinsic("tt.cb_push_back", {cb_out, one}));
    seq.push_back(MakeIntrinsic("tt.tile_regs_release"));
    return SeqStmt::Flatten(seq);
  }

private:
  int ResolveCBId(const Map<String, ObjectRef> &metadata, const char *key,
                  int fallback) const {
    if (!metadata.count(key)) {
      return fallback;
    }
    std::string buf_name = Downcast<String>(metadata[key]);
    auto it = buffer_cb_map_.find(buf_name);
    if (it != buffer_cb_map_.end()) {
      return it->second;
    }
    // Heuristic fallback: try suffix "_tile" if present in CB metadata.
    auto alt = buffer_cb_map_.find(buf_name + "_tile");
    if (alt != buffer_cb_map_.end()) {
      return alt->second;
    }
    return fallback;
  }

  MatmulCollection *collection_;
  const std::unordered_map<std::string, int> &buffer_cb_map_;
};

/*!
 * \brief Main implementation of TensorizeTT pass
 *
 * Identifies matmul operations and annotates with TT intrinsic metadata.
 * Attaches matmul count to function attributes.
 *
 * \param f The PrimFunc to process
 * \return Enhanced PrimFunc with matmul intrinsic annotations
 */
PrimFunc TensorizeTTImpl(PrimFunc f) {
  // Step 1: Check if this is a TT function
  auto schedule_policy = f->attrs.GetAttr<String>("tt_schedule_policy");
  if (!schedule_policy.defined()) {
    // Not a TT function, skip transformation
    return f;
  }

  // Collect matmul metadata (supports pragmas and manual loops)
  MatmulPatternCollector collector;
  MatmulCollection collection = collector.Collect(f->body);

  std::unordered_map<std::string, int> buffer_cb_map;
  if (auto cb_configs = f->attrs.GetAttr<Array<Map<String, ObjectRef>>>(
          "tt_circular_buffers")) {
    for (const Map<String, ObjectRef> &config : cb_configs.value()) {
      if (!config.count("name") || !config.count("cb_id")) {
        continue;
      }
      std::string buf_name = Downcast<String>(config["name"]);
      int cb_id = Downcast<Integer>(config["cb_id"])->value;
      buffer_cb_map.emplace(std::move(buf_name), cb_id);
    }
  }

  // Step 2: Apply tensorization transformation
  TensorizeMutator mutator(&collection, buffer_cb_map);
  Stmt new_body = mutator(f->body);

  int matmul_count = static_cast<int>(collection.infos.size());

  // If no matmul operations found, return unchanged
  if (matmul_count == 0) {
    return f;
  }

  // Step 3: Create new function with transformed body
  PrimFunc new_func = f;
  auto n = make_object<PrimFuncNode>(*f.get());
  n->body = new_body;
  new_func = PrimFunc(n);

  // Step 4: Attach matmul metadata
  new_func = WithAttr(new_func, "tt_num_matmuls", Integer(matmul_count));
  new_func = WithAttr(new_func, "tt_has_tensorize", Bool(true));

  Array<Map<String, ObjectRef>> patterns_attr;
  patterns_attr.reserve(collection.infos.size());
  for (const MatmulPatternInfo &info : collection.infos) {
    patterns_attr.push_back(info.metadata);
  }
  new_func = WithAttr(new_func, "tt_matmul_patterns", patterns_attr);

  return new_func;
}

using namespace tir::transform;

/*!
 * \brief Create the TensorizeTT pass
 *
 * \return The TIR pass
 */
Pass TensorizeTT() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return TensorizeTTImpl(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.TensorizeTT", {});
}

// Register the pass for Python FFI
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.TensorizeTT", TensorizeTT);
});

} // namespace tl
} // namespace tvm
