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
 * \file codegen_tt_writer_visitor.h
 * \brief Writer kernel IR-driven visitor for Tenstorrent backend
 *
 * Specializes TTCodegenVisitor for writer kernel generation.
 * Handles L1 CB → DRAM transfers with NOC async writes.
 *
 * Task 4: Writer Kernel Visitor
 * See: docs/tenstorrent/IR_DRIVEN_CODEGEN_PLAN.md
 */

#ifndef TVM_TARGET_TT_CODEGEN_TT_WRITER_VISITOR_H_
#define TVM_TARGET_TT_CODEGEN_TT_WRITER_VISITOR_H_

#include "codegen_tt_visitor.h"

namespace tvm {
namespace tl {

/*!
 * \brief Writer kernel visitor for TT backend
 *
 * Generates kernel_main() function that writes tiles from L1 to DRAM:
 * - Runtime argument extraction (dram_addr_c, start_tile_id, num_tiles, Nt)
 * - Persistent loop over assigned output tiles
 * - NOC async write operations
 * - Circular buffer wait/pop operations
 */
class TTWriterCodegenVisitor : public TTCodegenVisitor {
 public:
  /*!
   * \brief Constructor
   * \param func The PrimFunc to generate writer kernel from
   */
  explicit TTWriterCodegenVisitor(const PrimFunc& func);

  /*!
   * \brief Emit complete writer kernel including preamble
   * \return Full C++ source code for writer kernel
   */
  std::string GetFullKernel();

 protected:
  /*!
   * \brief Visit for loop node (override for writer-specific handling)
   */
  void VisitStmt_(const ForNode* op) override;

  /*!
   * \brief Visit buffer store (detect DRAM writes)
   */
  void VisitStmt_(const BufferStoreNode* op) override;

 private:
  /*!
   * \brief Emit kernel preamble (includes, mock APIs, CB defines)
   */
  void EmitPreamble();

  /*!
   * \brief Emit NOC async write operation
   * \param dst Destination buffer
   * \param tile_idx Tile index expression
   * \param cb_id Circular buffer ID
   */
  void EmitNOCWrite(const Buffer& dst, const std::string& tile_idx, uint32_t cb_id);

  /*!
   * \brief Emit CB wait front operation
   * \param cb_id Circular buffer ID
   * \param ntiles Number of tiles to wait for
   */
  void EmitCBWaitFront(uint32_t cb_id, uint32_t ntiles);

  /*!
   * \brief Emit CB pop front operation
   * \param cb_id Circular buffer ID
   * \param ntiles Number of tiles to pop
   */
  void EmitCBPopFront(uint32_t cb_id, uint32_t ntiles);
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TARGET_TT_CODEGEN_TT_WRITER_VISITOR_H_
