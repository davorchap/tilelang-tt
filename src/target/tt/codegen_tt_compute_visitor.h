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
 * \brief Compute pattern types for code generation
 *
 * Identifies the type of computation being performed to emit
 * appropriate initialization and compute APIs.
 */
enum class ComputePattern {
  UNKNOWN,      //!< Pattern not yet determined
  MATMUL,       //!< Matrix multiplication (T.gemm) - mm_init/matmul_tiles
  REDUCTION,    //!< Sum reduction - reduce_tiles_init/reduce_tiles
  ELEMENTWISE,  //!< Element-wise ops (add, mul) - binary_op_init_common/add_tiles
  GEMV,         //!< Matrix-vector multiply - gemv_init/gemv_tiles
  CUSTOM        //!< User-defined or composite patterns
};

/*!
 * \brief Pattern detector for analyzing loop bodies
 *
 * Traverses IR to identify computation patterns before code generation.
 * This enables emitting correct initialization APIs and compute intrinsics.
 */
class PatternDetector {
 public:
  /*!
   * \brief Detect computation pattern in a loop body
   * \param loop The for loop to analyze
   * \return Detected pattern type
   */
  static ComputePattern DetectPattern(const ForNode* loop);

 private:
  /*!
   * \brief Check if loop body contains T.gemm() call
   * \param body The statement to analyze
   * \return True if T.gemm() is present
   */
  static bool HasGemmCall(const Stmt& body);

  /*!
   * \brief Check if loop body has reduction pattern (accumulation into same variable)
   * \param body The statement to analyze
   * \return True if reduction pattern detected
   */
  static bool HasReductionPattern(const Stmt& body);

  /*!
   * \brief Check if loop body has element-wise operations (independent tile ops)
   * \param body The statement to analyze
   * \return True if element-wise pattern detected
   */
  static bool HasElementwisePattern(const Stmt& body);

  /*!
   * \brief Check if loop body has GEMV pattern (matrix-vector multiply)
   * \param body The statement to analyze
   * \return True if GEMV pattern detected
   */
  static bool HasGemvPattern(const Stmt& body);
};

/*!
 * \brief Compute kernel visitor for TT backend
 *
 * Generates MAIN() function that processes output tiles with:
 * - Runtime argument extraction (start_tile_id, num_tiles, Kt)
 * - Persistent loop over assigned tiles
 * - K-loop for matmul/reduction/other accumulation patterns
 * - Circular buffer operations (wait, pop, push)
 * - Pattern-specific tile intrinsics
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
   * \brief Visit evaluate node (detect T.copy, T.gemm calls)
   */
  void VisitStmt_(const EvaluateNode* op) override;

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
   * \brief Emit T.gemm() intrinsic operation
   * \param call The call node containing T.gemm() call
   */
  void EmitGemmIntrinsic(const CallNode* call);

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
   * \brief Emit tile register acquire operation
   * Reserves tile registers for computation (FPU access)
   */
  void EmitTileRegsAcquire();

  /*!
   * \brief Emit tile register commit operation
   * Signals computation complete, registers ready for packer
   */
  void EmitTileRegsCommit();

  /*!
   * \brief Emit tile register wait operation
   * Waits for computation to complete on tile registers
   */
  void EmitTileRegsWait();

  /*!
   * \brief Emit tile register release operation
   * Frees tile registers back to pool
   */
  void EmitTileRegsRelease();

  /*! \brief Track if matmul_init has been emitted */
  bool matmul_init_emitted_;

  /*! \brief Track if element-wise init has been emitted */
  bool elementwise_init_emitted_;

  /*! \brief Track current K-loop iteration for accumulate flag */
  int current_k_iter_;

  /*! \brief Track K-loop variable name for accumulate flag */
  std::string k_loop_var_;

  /*! \brief Track if DST is currently acquired */
  bool dst_acquired_;

  /*! \brief Track loop nesting depth */
  int loop_depth_;

  /*! \brief Pattern detection: map loop nodes to detected patterns */
  std::map<const ForNode*, ComputePattern> loop_patterns_;

  /*! \brief Pattern detection: currently active pattern */
  ComputePattern current_pattern_;

  /*! \brief Track if reduction init has been emitted */
  bool reduction_init_emitted_;

  /*! \brief Track if GEMV init has been emitted */
  bool gemv_init_emitted_;
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TARGET_TT_CODEGEN_TT_COMPUTE_VISITOR_H_
