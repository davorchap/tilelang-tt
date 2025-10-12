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
 * \file codegen_tt_reader_visitor.h
 * \brief Reader kernel IR-driven visitor for Tenstorrent backend
 *
 * Specializes TTCodegenVisitor for reader kernel generation.
 * Handles DRAM → L1 CB transfers with NOC async reads.
 *
 * Task 3: Reader Kernel Visitor
 * See: docs/tenstorrent/IR_DRIVEN_CODEGEN_PLAN.md
 */

#ifndef TVM_TARGET_TT_CODEGEN_TT_READER_VISITOR_H_
#define TVM_TARGET_TT_CODEGEN_TT_READER_VISITOR_H_

#include "codegen_tt_visitor.h"

namespace tvm {
namespace tl {

/*!
 * \brief Reader kernel visitor for TT backend
 *
 * Generates kernel_main() function that loads tiles from DRAM to L1:
 * - Runtime argument extraction (dram_addr_a/b, Mt, Kt, Nt, start_tile_id, num_tiles)
 * - Persistent loop over assigned output tiles
 * - K-loop for loading A[m,k] and B[k,n] tiles
 * - NOC async read operations
 * - Circular buffer reserve/push operations
 */
class TTReaderCodegenVisitor : public TTCodegenVisitor {
 public:
  /*!
   * \brief Constructor
   * \param func The PrimFunc to generate reader kernel from
   */
  explicit TTReaderCodegenVisitor(const PrimFunc& func);

  /*!
   * \brief Emit complete reader kernel including preamble
   * \return Full C++ source code for reader kernel
   */
  std::string GetFullKernel();

 protected:
  /*!
   * \brief Visit for loop node (override for reader-specific handling)
   */
  void VisitStmt_(const ForNode* op) override;

  /*!
   * \brief Visit buffer load (detect DRAM reads)
   */
  void VisitExpr_(const BufferLoadNode* op) override;

 private:
  /*!
   * \brief Emit kernel preamble (includes, mock APIs, CB defines)
   */
  void EmitPreamble();

  /*!
   * \brief Emit NOC async read operation
   * \param src Source buffer
   * \param tile_idx Tile index expression
   * \param cb_id Circular buffer ID
   */
  void EmitNOCRead(const Buffer& src, const std::string& tile_idx, uint32_t cb_id);

  /*!
   * \brief Emit CB reserve operation
   * \param cb_id Circular buffer ID
   * \param ntiles Number of tiles to reserve
   */
  void EmitCBReserve(uint32_t cb_id, uint32_t ntiles);

  /*!
   * \brief Emit CB push back operation
   * \param cb_id Circular buffer ID
   * \param ntiles Number of tiles to push
   */
  void EmitCBPushBack(uint32_t cb_id, uint32_t ntiles);

  /*!
   * \brief Calculate tile index from buffer indices
   * \param indices Buffer indices (e.g., [m, k])
   * \param num_tiles_dim Number of tiles in second dimension
   * \return Tile index expression string
   */
  std::string CalculateTileIndex(const Array<PrimExpr>& indices, const std::string& num_tiles_dim);
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TARGET_TT_CODEGEN_TT_READER_VISITOR_H_
