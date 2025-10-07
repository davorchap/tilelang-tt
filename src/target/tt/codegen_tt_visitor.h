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
 * \file codegen_tt_visitor.h
 * \brief IR-driven codegen visitor for Tenstorrent backend
 *
 * This module provides the base visitor class for walking TIR and generating
 * Metalium-compatible C++ code. Unlike template-based codegen, this visitor
 * analyzes the actual IR body structure to support arbitrary kernel patterns.
 *
 * Task 1: Base visitor infrastructure
 * See: docs/tenstorrent/IR_DRIVEN_CODEGEN_PLAN.md
 */

#ifndef TVM_TARGET_TT_CODEGEN_TT_VISITOR_H_
#define TVM_TARGET_TT_CODEGEN_TT_VISITOR_H_

#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

#include <sstream>
#include <string>
#include <unordered_map>

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Base visitor class for IR-driven TT codegen
 *
 * This class walks the TIR body structure and generates C++ code for
 * Tenstorrent kernels. It provides infrastructure for:
 * - Walking statements (loops, conditionals, allocations)
 * - Walking expressions (buffer loads/stores, arithmetic)
 * - Managing indentation and code emission
 * - Tracking buffer metadata (circular buffers, CB IDs)
 *
 * Subclasses override specific visitor methods to generate kernel-specific
 * code (compute, reader, writer).
 */
class TTCodegenVisitor : public StmtExprVisitor {
 public:
  /*!
   * \brief Constructor
   * \param func The PrimFunc to generate code from
   */
  explicit TTCodegenVisitor(const PrimFunc& func);

  /*!
   * \brief Get the generated C++ code
   * \return The complete C++ source code as a string
   */
  std::string GetCode() const;

 protected:
  //--------------------------------------------------------------------
  // Core infrastructure
  //--------------------------------------------------------------------

  /*! \brief Output stream for generated code */
  std::ostringstream code_;

  /*! \brief Reference to the PrimFunc being codegen'd */
  const PrimFunc& func_;

  /*! \brief Current indentation level (0 = no indent) */
  int indent_level_;

  /*!
   * \brief Emit indentation at current level
   * Emits `indent_level_ * 4` spaces
   */
  void Indent();

  /*!
   * \brief Emit a code fragment without newline
   * \param text The text to emit
   */
  void Emit(const std::string& text);

  /*!
   * \brief Emit a complete line with indentation and newline
   * \param line The line to emit (without leading spaces or trailing newline)
   */
  void EmitLine(const std::string& line);

  /*!
   * \brief Increase indentation level
   */
  void IncIndent() { indent_level_++; }

  /*!
   * \brief Decrease indentation level
   */
  void DecIndent() {
    if (indent_level_ > 0) indent_level_--;
  }

  //--------------------------------------------------------------------
  // Statement visitors (override in subclasses as needed)
  //--------------------------------------------------------------------

  /*!
   * \brief Visit a for loop
   * Default: emits C++ for loop with loop variable and body
   */
  void VisitStmt_(const ForNode* op) override;

  /*!
   * \brief Visit an attribute statement
   * Default: looks for TT-specific attributes (e.g., "tt.matmul_intrinsic")
   */
  void VisitStmt_(const AttrStmtNode* op) override;

  /*!
   * \brief Visit an allocation statement
   * Default: emits local variable declaration
   */
  void VisitStmt_(const AllocateNode* op) override;

  /*!
   * \brief Visit a buffer declaration
   * Default: tracks buffer metadata
   */
  void VisitStmt_(const DeclBufferNode* op) override;

  /*!
   * \brief Visit a buffer store
   * Default: emits assignment statement
   */
  void VisitStmt_(const BufferStoreNode* op) override;

  /*!
   * \brief Visit a sequence of statements
   * Default: visits each statement in order
   */
  void VisitStmt_(const SeqStmtNode* op) override;

  /*!
   * \brief Visit an if-then-else statement
   * Default: emits C++ if/else
   */
  void VisitStmt_(const IfThenElseNode* op) override;

  /*!
   * \brief Visit an evaluate node (expression as statement)
   * Default: emits expression followed by semicolon
   */
  void VisitStmt_(const EvaluateNode* op) override;

  //--------------------------------------------------------------------
  // Expression visitors (override in subclasses as needed)
  //--------------------------------------------------------------------

  /*!
   * \brief Visit a buffer load expression
   * Default: emits buffer access syntax
   */
  void VisitExpr_(const BufferLoadNode* op) override;

  /*!
   * \brief Visit a variable reference
   * Default: emits variable name
   */
  void VisitExpr_(const VarNode* op) override;

  /*!
   * \brief Visit an integer immediate
   * Default: emits integer literal
   */
  void VisitExpr_(const IntImmNode* op) override;

  /*!
   * \brief Visit a floating point immediate
   * Default: emits float literal
   */
  void VisitExpr_(const FloatImmNode* op) override;

  /*!
   * \brief Visit an add expression
   * Default: emits a + b
   */
  void VisitExpr_(const AddNode* op) override;

  /*!
   * \brief Visit a multiply expression
   * Default: emits a * b
   */
  void VisitExpr_(const MulNode* op) override;

  //--------------------------------------------------------------------
  // Helper methods
  //--------------------------------------------------------------------

  /*!
   * \brief Get C++ variable name from TVM Var
   * \param var The TVM variable
   * \return The C++ variable name
   */
  virtual std::string GetVarName(const Var& var);

  /*!
   * \brief Get buffer name from TVM Buffer
   * \param buf The TVM buffer
   * \return The buffer name (e.g., "A", "B", "C")
   */
  virtual std::string GetBufferName(const Buffer& buf);

  /*!
   * \brief Generate C++ expression code from TVM expression
   * \param expr The TVM expression
   * \return The C++ expression string
   */
  virtual std::string EmitExpr(const PrimExpr& expr);

  /*!
   * \brief Check if a buffer is a circular buffer (L1 memory)
   * \param buf The buffer to check
   * \return True if buffer has scope "local" or "shared" (indicates L1)
   */
  virtual bool IsCircularBuffer(const Buffer& buf);

  /*!
   * \brief Get circular buffer ID for a buffer
   * \param buf The buffer
   * \return The CB ID (0, 1, 2, ...) or -1 if not a CB
   */
  virtual int GetCBId(const Buffer& buf);

  /*!
   * \brief Check if a buffer is in DRAM (global scope)
   * \param buf The buffer to check
   * \return True if buffer has scope "global"
   */
  virtual bool IsGlobalBuffer(const Buffer& buf);

  /*!
   * \brief Get data type size in bytes
   * \param dtype The data type (e.g., "float16", "float32")
   * \return Size in bytes
   */
  virtual int GetDTypeSize(const DataType& dtype);

  //--------------------------------------------------------------------
  // Metadata tracking
  //--------------------------------------------------------------------

  /*! \brief Map from Buffer to circular buffer ID */
  std::unordered_map<const BufferNode*, int> buffer_to_cb_id_;

  /*! \brief Map from Var name to C++ variable name */
  std::unordered_map<std::string, std::string> var_name_map_;

  /*! \brief Counter for generating unique variable names */
  int var_counter_;

  /*!
   * \brief Initialize buffer metadata from func attributes
   * Called in constructor to populate buffer_to_cb_id_ map
   */
  void InitBufferMetadata();

  /*!
   * \brief Get or create a unique C++ variable name
   * \param original The original variable name from IR
   * \return A unique C++-compatible variable name
   */
  std::string GetOrCreateVarName(const std::string& original);
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TARGET_TT_CODEGEN_TT_VISITOR_H_
