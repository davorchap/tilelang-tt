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
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Visitor to identify and annotate matmul operations
 */
class TensorizeMutator : public StmtMutator {
 public:
  TensorizeMutator() : matmul_count_(0) {}

  Stmt VisitStmt_(const AttrStmtNode* op) override {
    // Check for GEMM/matmul pragma markers
    // In TileLang, T.gemm() typically generates AttrStmt with specific keys
    // For Phase 2, we look for matmul-related attributes

    if (op->attr_key == "pragma_gemm" ||
        op->attr_key == "tl.gemm" ||
        op->attr_key == "gemm_operation") {

      // This is a matmul operation - annotate with TT intrinsic info
      matmul_count_++;

      // Annotate with matmul ID as PrimExpr (AttrStmt value must be PrimExpr)
      // For Phase 2, codegen will use this to generate matmul_tiles() calls
      PrimExpr matmul_id = IntImm(DataType::Int(32), matmul_count_ - 1);

      // Annotate this node with TT intrinsic metadata
      // For Phase 2, we create a new AttrStmt wrapping the original
      Stmt new_body = VisitStmt(op->body);

      return AttrStmt(op->node, "tt.matmul_intrinsic", matmul_id, new_body);
    }

    // Default: recurse
    return StmtMutator::VisitStmt_(op);
  }

  int GetMatmulCount() const { return matmul_count_; }

 private:
  int matmul_count_;
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

  // Step 2: Apply tensorization transformation
  TensorizeMutator mutator;
  Stmt new_body = mutator(f->body);

  int matmul_count = mutator.GetMatmulCount();

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

  return new_func;
}

using namespace tir::transform;

/*!
 * \brief Create the TensorizeTT pass
 *
 * \return The TIR pass
 */
Pass TensorizeTT() {
  auto pass_func = [=](PrimFunc f, const IRModule& m, const PassContext& ctx) {
    return TensorizeTTImpl(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.TensorizeTT", {});
}

// Register the pass for Python FFI
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.TensorizeTT", TensorizeTT);
});

}  // namespace tl
}  // namespace tvm
