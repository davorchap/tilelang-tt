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
 * \file lower_to_sfpu.cc
 * \brief Lower intra-tile parallelism (threadIdx) to SFPU/SIMD operations
 *
 * This pass transforms threadIdx-based parallelism (T.Parallel) into
 * Tenstorrent SFPU (SIMD Floating Point Unit) operations.
 *
 * For now, this is a placeholder that errors out when threadIdx constructs
 * are detected, since SFPU lowering is not yet implemented.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>
#include <unordered_set>
#include <vector>
#include <tvm/runtime/logging.h>

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Visitor to detect threadIdx constructs and track their variables
 */
class ThreadIdxDetector : public StmtVisitor {
public:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      std::string thread_tag = iv->thread_tag;

      if (thread_tag == "threadIdx.x" || thread_tag == "threadIdx.y" ||
          thread_tag == "threadIdx.z") {
        thread_idx_vars_.push_back({iv->var, thread_tag});
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

  const std::vector<std::pair<Var, std::string>> &GetThreadIdxVars() const {
    return thread_idx_vars_;
  }

private:
  std::vector<std::pair<Var, std::string>> thread_idx_vars_;
};

/*!
 * \brief Visitor to detect if variables are actually used
 */
class VarUseDetector : public StmtExprVisitor {
public:
  explicit VarUseDetector(const std::unordered_set<const VarNode *> &vars_to_check)
      : vars_to_check_(vars_to_check) {}

  void VisitExpr_(const VarNode *op) final {
    if (vars_to_check_.count(op) > 0) {
      found_vars_.insert(op);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  const std::unordered_set<const VarNode *> &GetFoundVars() const {
    return found_vars_;
  }

private:
  const std::unordered_set<const VarNode *> &vars_to_check_;
  std::unordered_set<const VarNode *> found_vars_;
};

/*!
 * \brief Lower threadIdx constructs to SFPU operations
 *
 * For now, this is a placeholder that errors out when threadIdx is *used*.
 *
 * \param f The PrimFunc to process
 * \return The PrimFunc (unchanged, or error if threadIdx used)
 */
PrimFunc LowerToSFPUImpl(PrimFunc f) {
  // Detect threadIdx constructs
  ThreadIdxDetector detector;
  detector(f->body);

  const auto &thread_vars = detector.GetThreadIdxVars();
  if (thread_vars.empty()) {
    return f;  // No threadIdx constructs, nothing to do
  }

  // Check if any threadIdx variables are actually used
  std::unordered_set<const VarNode *> thread_var_nodes;
  for (const auto &[var, tag] : thread_vars) {
    thread_var_nodes.insert(var.get());
  }

  VarUseDetector use_detector(thread_var_nodes);
  use_detector(f->body);

  const auto &used_vars = use_detector.GetFoundVars();
  if (!used_vars.empty()) {
    // Build error message with used threadIdx variables
    std::vector<std::string> used_tags;
    for (const auto &[var, tag] : thread_vars) {
      if (used_vars.count(var.get()) > 0) {
        used_tags.push_back(tag + " (" + var->name_hint + ")");
      }
    }

    std::string tags_str;
    for (size_t i = 0; i < used_tags.size(); ++i) {
      if (i > 0) tags_str += ", ";
      tags_str += used_tags[i];
    }

    LOG(FATAL) << "LowerToSFPU: Found threadIdx constructs that require SFPU lowering.\n"
               << "Detected: " << tags_str << "\n"
               << "SFPU (SIMD Floating Point Unit) lowering is not yet implemented.\n"
               << "T.Parallel() constructs will be supported in a future update to map "
               << "intra-tile parallelism to Tenstorrent SFPU operations.\n"
               << "For now, please use only tile-level parallelism (blockIdx via T.Kernel).";
  }

  // threadIdx declared but not used - pass through for now
  return f;
}

using namespace tir::transform;

/*!
 * \brief Create the LowerToSFPU pass
 *
 * \return The TIR pass
 */
Pass LowerToSFPU() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return LowerToSFPUImpl(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerToSFPU", {});
}

// Register the pass for Python FFI
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerToSFPU", LowerToSFPU);
});

} // namespace tl
} // namespace tvm
