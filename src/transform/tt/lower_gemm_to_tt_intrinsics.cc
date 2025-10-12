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
 * \file lower_gemm_to_tt_intrinsics.cc
 * \brief Lower frontend tl.gemm intrinsics to TT tile intrinsics (Persistent
 * Transform stage)
 *
 * This pass consumes the TileLang frontend `tl.gemm` intrinsic (emitted by
 * `T.gemm`) and expands each call into the Tenstorrent tile-intrinsic
 * sequence consumed by the TT codegen visitors. It mirrors the CUDA
 * `InferFragment` pass but targets the TT runtime intrinsics.
 *
 * See: docs/tenstorrent/passes/lower_gemm_to_tt_intrinsics.md for detailed
 * specification.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <initializer_list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../op/builtin.h"

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

std::string ExtractBufferName(const PrimExpr &expr) {
  if (const auto *load = expr.as<BufferLoadNode>()) {
    return load->buffer->name;
  }
  if (const auto *call = expr.as<CallNode>()) {
    if (call->op.same_as(builtin::tvm_access_ptr()) ||
        call->op.same_as(builtin::address_of())) {
      if (!call->args.empty()) {
        for (const PrimExpr &arg : call->args) {
          std::string name = ExtractBufferName(arg);
          if (!name.empty()) {
            return name;
          }
        }
      }
    }
  }
  return "";
}

} // namespace

/*!
 * \brief Visitor to identify and annotate matmul operations
 */
class GemmToTTMutator : public StmtMutator {
public:
  explicit GemmToTTMutator(
      const std::unordered_map<std::string, int> &buffer_cb_map)
      : buffer_cb_map_(buffer_cb_map) {}

  Stmt VisitStmt_(const AttrStmtNode *op) override {
    if (op->attr_key == "pragma_gemm" || op->attr_key == "tl.gemm" ||
        op->attr_key == "gemm_operation") {
      // Strip legacy pragma wrapper; intrinsic sequence will be emitted
      // directly.
      return VisitStmt(op->body);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const EvaluateNode *op) override {
    const auto *call = op->value.as<CallNode>();
    if (call == nullptr) {
      return StmtMutator::VisitStmt_(op);
    }
    if (!call->op.same_as(tl::tl_gemm())) {
      return StmtMutator::VisitStmt_(op);
    }

    std::string buffer_a = call->args.size() > 1
                               ? ExtractBufferName(call->args[1])
                               : std::string();
    std::string buffer_b = call->args.size() > 2
                               ? ExtractBufferName(call->args[2])
                               : std::string();
    std::string buffer_c = call->args.size() > 3
                               ? ExtractBufferName(call->args[3])
                               : std::string();

    int cb_in0_id = ResolveCBId(buffer_a, /*fallback=*/0);
    int cb_in1_id = ResolveCBId(buffer_b, /*fallback=*/1);
    int cb_out_id = ResolveCBId(buffer_c, /*fallback=*/16);

    PrimExpr cb_in0 = Integer(cb_in0_id);
    PrimExpr cb_in1 = Integer(cb_in1_id);
    PrimExpr cb_out = Integer(cb_out_id);
    PrimExpr one = Integer(1);
    PrimExpr zero = Integer(0);

    Array<Stmt> seq;
    seq.push_back(MakeIntrinsic("tt.tile_regs_acquire"));
    seq.push_back(MakeIntrinsic("tt.mm_init", {cb_in0, cb_in1, cb_out}));
    seq.push_back(MakeIntrinsic("tt.cb_wait_front", {cb_in0, one}));
    seq.push_back(MakeIntrinsic("tt.cb_wait_front", {cb_in1, one}));
    seq.push_back(MakeIntrinsic(
        "tt.matmul_tiles", {cb_in0, cb_in1, zero, zero, zero, Integer(0)}));
    seq.push_back(MakeIntrinsic("tt.cb_pop_front", {cb_in0, one}));
    seq.push_back(MakeIntrinsic("tt.cb_pop_front", {cb_in1, one}));
    seq.push_back(MakeIntrinsic("tt.tile_regs_commit"));
    seq.push_back(MakeIntrinsic("tt.tile_regs_wait"));
    seq.push_back(MakeIntrinsic("tt.cb_reserve_back", {cb_out, one}));
    seq.push_back(MakeIntrinsic("tt.pack_tile", {zero, cb_out}));
    seq.push_back(MakeIntrinsic("tt.cb_push_back", {cb_out, one}));
    seq.push_back(MakeIntrinsic("tt.tile_regs_release"));

    Map<String, ObjectRef> metadata;
    metadata.Set("source", String("tl.gemm"));
    metadata.Set("buffer_a", String(buffer_a));
    metadata.Set("buffer_b", String(buffer_b));
    metadata.Set("buffer_c", String(buffer_c));
    metadata.Set("A_indices", Array<PrimExpr>());
    metadata.Set("B_indices", Array<PrimExpr>());
    metadata.Set("C_indices", Array<PrimExpr>());
    metadata.Set("loop_vars", Array<String>());
    metadata.Set("reduction_var", String(""));
    metadata.Set("accumulate", Bool(true));
    metadata.Set("cb_in0", Integer(cb_in0_id));
    metadata.Set("cb_in1", Integer(cb_in1_id));
    metadata.Set("cb_out", Integer(cb_out_id));
    if (call->args.empty()) {
      metadata.Set("tl_gemm_signature", String(""));
    } else {
      metadata.Set("tl_gemm_signature", call->args[0]);
    }

    patterns_.push_back(metadata);

    return SeqStmt::Flatten(seq);
  }

private:
  int ResolveCBId(const std::string &buffer_name, int fallback) const {
    if (buffer_name.empty()) {
      return fallback;
    }
    auto it = buffer_cb_map_.find(buffer_name);
    if (it != buffer_cb_map_.end()) {
      return it->second;
    }
    // Heuristic fallback: try suffix "_tile" if present in CB metadata.
    auto alt = buffer_cb_map_.find(buffer_name + "_tile");
    if (alt != buffer_cb_map_.end()) {
      return alt->second;
    }
    return fallback;
  }

  Array<Map<String, ObjectRef>> patterns_;

public:
  Array<Map<String, ObjectRef>> TakePatterns() { return patterns_; }

private:
  const std::unordered_map<std::string, int> &buffer_cb_map_;
};

/*!
 * \brief Main implementation of LowerGemmToTTIntrinsics pass
 *
 * Identifies matmul operations and annotates with TT intrinsic metadata.
 * Attaches matmul count to function attributes.
 *
 * \param f The PrimFunc to process
 * \return Enhanced PrimFunc with matmul intrinsic annotations
 */
PrimFunc LowerGemmToTTIntrinsicsImpl(PrimFunc f) {
  // Step 1: Check if this is a TT function
  auto schedule_policy = f->attrs.GetAttr<String>("tt_schedule_policy");
  if (!schedule_policy.defined()) {
    // Not a TT function, skip transformation
    return f;
  }

  // Collect matmul metadata (supports pragmas and manual loops)
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
  GemmToTTMutator mutator(buffer_cb_map);
  Stmt new_body = mutator(f->body);

  Array<Map<String, ObjectRef>> collected_patterns = mutator.TakePatterns();
  int matmul_count = static_cast<int>(collected_patterns.size());

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
  patterns_attr.reserve(collected_patterns.size());
  for (const Map<String, ObjectRef> &info : collected_patterns) {
    patterns_attr.push_back(info);
  }
  new_func = WithAttr(new_func, "tt_matmul_patterns", patterns_attr);

  return new_func;
}

using namespace tir::transform;

/*!
 * \brief Create the LowerGemmToTTIntrinsics pass
 *
 * \return The TIR pass
 */
Pass LowerGemmToTTIntrinsics() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return LowerGemmToTTIntrinsicsImpl(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerGemmToTTIntrinsics", {});
}

// Register the pass for Python FFI
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerGemmToTTIntrinsics",
                        LowerGemmToTTIntrinsics);
});

} // namespace tl
} // namespace tvm
