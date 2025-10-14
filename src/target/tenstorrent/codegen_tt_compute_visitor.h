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
  explicit TTComputeCodegenVisitor(const PrimFunc &func);

  /*!
   * \brief Emit complete compute kernel including preamble
   * \return Full C++ source code for compute kernel
   */
  std::string GetFullKernel();

protected:
  /*!
   * \brief Visit for loop node (emits uint32_t loops)
   */
  void VisitStmt_(const ForNode *op) override;

  /*!
   * \brief Visit attribute statement (detect matmul intrinsic, copy ops)
   */
  void VisitStmt_(const AttrStmtNode *op) override;

private:
  /*!
   * \brief Emit kernel preamble (includes, mock APIs, CB defines)
   */
  void EmitPreamble();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TARGET_TT_CODEGEN_TT_COMPUTE_VISITOR_H_
