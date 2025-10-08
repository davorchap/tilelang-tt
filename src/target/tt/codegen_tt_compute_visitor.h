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
 * \file codegen_tt_compute_visitor.h
 * \brief Compute kernel IR-driven visitor for Tenstorrent backend
 *
 * Specializes TTCodegenVisitor for compute kernel generation.
 * Handles matmul intrinsics, CB operations, and persistent loop structure.
 *
 * Task 2: Compute Kernel Visitor
 * See: docs/tenstorrent/IR_DRIVEN_CODEGEN_PLAN.md
 */

#ifndef TVM_TARGET_TT_CODEGEN_TT_COMPUTE_VISITOR_H_
#define TVM_TARGET_TT_CODEGEN_TT_COMPUTE_VISITOR_H_

#include "codegen_tt_visitor.h"

namespace tvm {
namespace tl {

/*!
 * \brief Compute kernel visitor for TT backend
 *
 * Generates MAIN() function that processes output tiles with:
 * - Runtime argument extraction (start_tile_id, num_tiles, Kt)
 * - Persistent loop over assigned tiles
 * - K-loop for matmul accumulation
 * - Circular buffer operations (wait, pop, push)
 * - Matmul tile intrinsics
 */
class TTComputeCodegenVisitor : public TTCodegenVisitor {
 public:
  /*!
   * \brief Constructor
   * \param func The PrimFunc to generate compute kernel from
   */
  explicit TTComputeCodegenVisitor(const PrimFunc& func);

  /*!
   * \brief Emit complete compute kernel including preamble
   * \return Full C++ source code for compute kernel
   */
  std::string GetFullKernel();

 protected:
  /*!
   * \brief Visit for loop node (override for compute-specific handling)
   * Emits C++ for loop with uint32_t loop variable
   */
  void VisitStmt_(const ForNode* op) override;

  /*!
   * \brief Visit attribute statement (detect matmul intrinsic, copy ops)
   */
  void VisitStmt_(const AttrStmtNode* op) override;

 private:
  /*!
   * \brief Emit kernel preamble (includes, mock APIs, CB defines)
   */
  void EmitPreamble();

  /*!
   * \brief Emit matmul intrinsic operation
   * \param op The attribute statement containing matmul annotation
   */
  void EmitMatmulIntrinsic(const AttrStmtNode* op);

  /*!
   * \brief Emit element-wise add intrinsic operation
   * \param op The attribute statement containing elementwise_add annotation
   */
  void EmitElementwiseAddIntrinsic(const AttrStmtNode* op);

  /*!
   * \brief Emit CB wait operation
   * \param cb_id Circular buffer ID
   * \param ntiles Number of tiles to wait for
   */
  void EmitCBWait(uint32_t cb_id, uint32_t ntiles);

  /*!
   * \brief Emit CB pop operation
   * \param cb_id Circular buffer ID
   * \param ntiles Number of tiles to pop
   */
  void EmitCBPop(uint32_t cb_id, uint32_t ntiles);

  /*!
   * \brief Emit CB push operation
   * \param cb_id Circular buffer ID
   * \param ntiles Number of tiles to push
   */
  void EmitCBPush(uint32_t cb_id, uint32_t ntiles);

  /*!
   * \brief Get matmul ID from annotation value
   * \param value The attribute value
   * \return Matmul ID (0 for init, >0 for accumulate)
   */
  int GetMatmulId(const PrimExpr& value);

  /*!
   * \brief Emit DST register acquire operation
   * Reserves DST registers for computation (FPU access)
   */
  void EmitDSTAcquire();

  /*!
   * \brief Emit DST register commit operation
   * Signals computation complete, DST ready for packer
   */
  void EmitDSTCommit();

  /*!
   * \brief Emit DST register release operation
   * Frees DST registers back to pool
   */
  void EmitDSTRelease();

  /*! \brief Track if matmul_init has been emitted */
  bool matmul_init_emitted_;

  /*! \brief Track current K-loop iteration for accumulate flag */
  int current_k_iter_;

  /*! \brief Track if DST is currently acquired */
  bool dst_acquired_;

  /*! \brief Track loop nesting depth */
  int loop_depth_;
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TARGET_TT_CODEGEN_TT_COMPUTE_VISITOR_H_
