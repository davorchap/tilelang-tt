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
 * \file codegen_tt_visitor.cc
 * \brief Implementation of IR-driven codegen visitor for Tenstorrent backend
 */

#include "codegen_tt_visitor.h"

#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <algorithm>
#include <sstream>

namespace tvm {
namespace tl {

using namespace tir;

//----------------------------------------------------------------------
// Constructor and core infrastructure
//----------------------------------------------------------------------

TTCodegenVisitor::TTCodegenVisitor(const PrimFunc &func)
    : func_(func), indent_level_(0), var_counter_(0) {
  // Partition mode metadata
  if (auto mode = func_->attrs.GetAttr<String>("tt.partition_mode")) {
    partition_mode_ = mode.value();
  } else {
    partition_mode_ = "global";
  }

  if (auto constants = func_->attrs.GetAttr<Map<String, ObjectRef>>(
          "tt.runtime_constants")) {
    runtime_constants_ = constants.value();
  }

  Array<String> runtime_names_attr;
  if (auto names =
          func_->attrs.GetAttr<Array<String>>("tt.runtime_arg_names")) {
    runtime_names_attr = names.value();
  } else if (auto runtime_args = func_->attrs.GetAttr<Map<String, ObjectRef>>(
                 "tt_runtime_args")) {
    if (runtime_args.value().count(String("arg_names"))) {
      runtime_names_attr =
          Downcast<Array<String>>(runtime_args.value()[String("arg_names")]);
    }
  }

  if (!runtime_names_attr.defined() || runtime_names_attr.empty()) {
    runtime_arg_names_ = {"tt_start_tile", "tt_tile_count", "Mt", "Kt", "Nt"};
  } else {
    for (const auto &name : runtime_names_attr) {
      runtime_arg_names_.push_back(name);
    }
  }

  for (size_t i = 0; i < runtime_arg_names_.size(); ++i) {
    runtime_arg_index_[runtime_arg_names_[i]] = static_cast<int>(i);
  }

  InitBufferMetadata();
}

std::string TTCodegenVisitor::GetCode() const { return code_.str(); }

void TTCodegenVisitor::Indent() {
  for (int i = 0; i < indent_level_; ++i) {
    code_ << "    "; // 4 spaces per indent level
  }
}

void TTCodegenVisitor::Emit(const std::string &text) { code_ << text; }

void TTCodegenVisitor::EmitLine(const std::string &line) {
  Indent();
  code_ << line << "\n";
}

//----------------------------------------------------------------------
// Statement visitors
//----------------------------------------------------------------------

void TTCodegenVisitor::VisitStmt_(const ForNode *op) {
  // Emit C++ for loop
  std::string loop_var = GetVarName(op->loop_var);
  std::string min_expr = EmitExpr(op->min);
  std::string extent_expr = EmitExpr(op->extent);

  EmitLine("for (int " + loop_var + " = " + min_expr + "; " + loop_var + " < " +
           min_expr + " + " + extent_expr + "; ++" + loop_var + ") {");
  IncIndent();

  // Visit loop body
  VisitStmt(op->body);

  DecIndent();
  EmitLine("}");
}

void TTCodegenVisitor::VisitStmt_(const AttrStmtNode *op) {
  // Check for TT-specific attributes
  std::string attr_key = op->attr_key;

  if (attr_key == "tt.matmul_intrinsic") {
    // Subclasses can override to handle matmul intrinsic
    EmitLine("// Matmul intrinsic detected (override in subclass)");
    VisitStmt(op->body);
  } else if (attr_key == "tt.copy") {
    // Subclasses can override to handle copy operations
    EmitLine("// Copy operation detected (override in subclass)");
    VisitStmt(op->body);
  } else if (attr_key == "tt.tensorize") {
    // Subclasses can override to handle tensorize operations
    EmitLine("// Tensorize operation detected (override in subclass)");
    VisitStmt(op->body);
  } else {
    // Default: just visit body
    VisitStmt(op->body);
  }
}

void TTCodegenVisitor::VisitStmt_(const AllocateNode *op) {
  // Emit local variable declaration
  std::string var_name = GetVarName(op->buffer_var);
  std::string dtype_str;

  if (op->dtype == DataType::Float(16)) {
    dtype_str = "uint16_t"; // fp16 as uint16_t
  } else if (op->dtype == DataType::Float(32)) {
    dtype_str = "float";
  } else if (op->dtype == DataType::Int(32)) {
    dtype_str = "int32_t";
  } else {
    dtype_str = "auto";
  }

  // Calculate total size
  std::ostringstream size_expr;
  for (size_t i = 0; i < op->extents.size(); ++i) {
    if (i > 0)
      size_expr << " * ";
    size_expr << EmitExpr(op->extents[i]);
  }

  EmitLine(dtype_str + " " + var_name + "[" + size_expr.str() + "];");

  // Visit body
  VisitStmt(op->body);
}

void TTCodegenVisitor::VisitStmt_(const DeclBufferNode *op) {
  // Track buffer metadata (buffer name, CB ID if applicable)
  // This is mostly for metadata tracking, no code emission needed
  VisitStmt(op->body);
}

void TTCodegenVisitor::VisitStmt_(const BufferStoreNode *op) {
  // Emit buffer store as assignment
  std::string buf_name = GetBufferName(op->buffer);
  std::string value_expr = EmitExpr(op->value);

  // Build index expression
  std::ostringstream indices;
  for (size_t i = 0; i < op->indices.size(); ++i) {
    if (i > 0)
      indices << "][";
    indices << EmitExpr(op->indices[i]);
  }

  EmitLine(buf_name + "[" + indices.str() + "] = " + value_expr + ";");
}

void TTCodegenVisitor::VisitStmt_(const SeqStmtNode *op) {
  // Visit each statement in sequence
  for (const Stmt &stmt : op->seq) {
    VisitStmt(stmt);
  }
}

void TTCodegenVisitor::VisitStmt_(const IfThenElseNode *op) {
  // Emit if-then-else
  std::string cond_expr = EmitExpr(op->condition);
  EmitLine("if (" + cond_expr + ") {");
  IncIndent();

  VisitStmt(op->then_case);

  DecIndent();
  if (op->else_case.defined()) {
    EmitLine("} else {");
    IncIndent();
    VisitStmt(op->else_case.value());
    DecIndent();
  }
  EmitLine("}");
}

void TTCodegenVisitor::VisitStmt_(const EvaluateNode *op) {
  // Emit expression as statement
  std::string expr_str = EmitExpr(op->value);
  if (!expr_str.empty()) {
    EmitLine(expr_str + ";");
  }
}

//----------------------------------------------------------------------
// Expression visitors
//----------------------------------------------------------------------

void TTCodegenVisitor::VisitExpr_(const BufferLoadNode *op) {
  // This is called during expression walking, but we need to return void
  // Actual expression code is generated by EmitExpr()
  // Do nothing here - expression emission is handled by EmitExpr()
}

void TTCodegenVisitor::VisitExpr_(const VarNode *op) {
  // Do nothing - expression emission is handled by EmitExpr()
}

void TTCodegenVisitor::VisitExpr_(const IntImmNode *op) {
  // Do nothing - expression emission is handled by EmitExpr()
}

void TTCodegenVisitor::VisitExpr_(const FloatImmNode *op) {
  // Do nothing - expression emission is handled by EmitExpr()
}

void TTCodegenVisitor::VisitExpr_(const AddNode *op) {
  // Do nothing - expression emission is handled by EmitExpr()
}

void TTCodegenVisitor::VisitExpr_(const MulNode *op) {
  // Do nothing - expression emission is handled by EmitExpr()
}

//----------------------------------------------------------------------
// Helper methods
//----------------------------------------------------------------------

std::string TTCodegenVisitor::GetVarName(const Var &var) {
  return GetOrCreateVarName(var->name_hint);
}

std::string TTCodegenVisitor::GetBufferName(const Buffer &buf) {
  // Use buffer name hint directly
  if (buf->name.empty()) {
    return "buf_" + std::to_string(reinterpret_cast<uintptr_t>(buf.get()));
  }
  return buf->name;
}

std::string TTCodegenVisitor::EmitExpr(const PrimExpr &expr) {
  // Recursively generate C++ expression code
  std::ostringstream expr_stream;

  if (auto *int_imm = expr.as<IntImmNode>()) {
    expr_stream << int_imm->value;
  } else if (auto *float_imm = expr.as<FloatImmNode>()) {
    expr_stream << float_imm->value;
  } else if (auto *var = expr.as<VarNode>()) {
    expr_stream << GetVarName(GetRef<Var>(var));
  } else if (auto *add = expr.as<AddNode>()) {
    expr_stream << "(" << EmitExpr(add->a) << " + " << EmitExpr(add->b) << ")";
  } else if (auto *sub = expr.as<SubNode>()) {
    expr_stream << "(" << EmitExpr(sub->a) << " - " << EmitExpr(sub->b) << ")";
  } else if (auto *mul = expr.as<MulNode>()) {
    expr_stream << "(" << EmitExpr(mul->a) << " * " << EmitExpr(mul->b) << ")";
  } else if (auto *div = expr.as<DivNode>()) {
    expr_stream << "(" << EmitExpr(div->a) << " / " << EmitExpr(div->b) << ")";
  } else if (auto *mod = expr.as<ModNode>()) {
    expr_stream << "(" << EmitExpr(mod->a) << " % " << EmitExpr(mod->b) << ")";
  } else if (auto *floor_div = expr.as<FloorDivNode>()) {
    expr_stream << "(" << EmitExpr(floor_div->a) << " / "
                << EmitExpr(floor_div->b) << ")";
  } else if (auto *floor_mod = expr.as<FloorModNode>()) {
    expr_stream << "(" << EmitExpr(floor_mod->a) << " % "
                << EmitExpr(floor_mod->b) << ")";
  } else if (auto *lt = expr.as<LTNode>()) {
    expr_stream << "(" << EmitExpr(lt->a) << " < " << EmitExpr(lt->b) << ")";
  } else if (auto *le = expr.as<LENode>()) {
    expr_stream << "(" << EmitExpr(le->a) << " <= " << EmitExpr(le->b) << ")";
  } else if (auto *gt = expr.as<GTNode>()) {
    expr_stream << "(" << EmitExpr(gt->a) << " > " << EmitExpr(gt->b) << ")";
  } else if (auto *ge = expr.as<GENode>()) {
    expr_stream << "(" << EmitExpr(ge->a) << " >= " << EmitExpr(ge->b) << ")";
  } else if (auto *eq = expr.as<EQNode>()) {
    expr_stream << "(" << EmitExpr(eq->a) << " == " << EmitExpr(eq->b) << ")";
  } else if (auto *ne = expr.as<NENode>()) {
    expr_stream << "(" << EmitExpr(ne->a) << " != " << EmitExpr(ne->b) << ")";
  } else if (auto *buf_load = expr.as<BufferLoadNode>()) {
    std::string buf_name = GetBufferName(buf_load->buffer);
    expr_stream << buf_name << "[";
    for (size_t i = 0; i < buf_load->indices.size(); ++i) {
      if (i > 0)
        expr_stream << "][";
      expr_stream << EmitExpr(buf_load->indices[i]);
    }
    expr_stream << "]";
  } else if (auto *ramp = expr.as<RampNode>()) {
    // Ramp node represents vectorized index: base + stride * lane_id for each
    // lane For TT, this should ideally be handled at statement level with tile
    // operations For now, emit the base (TODO: proper tile operation emission)
    expr_stream << EmitExpr(ramp->base);
  } else if (auto *broadcast = expr.as<BroadcastNode>()) {
    // Broadcast node represents replicating a scalar across vector lanes
    // For TT, just emit the scalar value
    expr_stream << EmitExpr(broadcast->value);
  } else if (auto *cast = expr.as<CastNode>()) {
    // Cast node for type conversion
    // Emit C-style cast
    expr_stream << "((" << cast->dtype << ")" << EmitExpr(cast->value) << ")";
  } else if (auto *call = expr.as<CallNode>()) {
    // Handle function calls
    if (auto* op_node = call->op.as<OpNode>()) {
      std::string call_name = op_node->name;
      if (call_name.rfind("tt.", 0) == 0) {
        call_name = call_name.substr(3);
      }
      expr_stream << call_name << "(";
      for (size_t i = 0; i < call->args.size(); ++i) {
        if (i > 0) expr_stream << ", ";
        expr_stream << EmitExpr(call->args[i]);
      }
      expr_stream << ")";
    } else if (auto* global_var = call->op.as<GlobalVarNode>()) {
      expr_stream << global_var->name_hint << "(";
      for (size_t i = 0; i < call->args.size(); ++i) {
        if (i > 0)
          expr_stream << ", ";
        expr_stream << EmitExpr(call->args[i]);
      }
      expr_stream << ")";
    } else {
      expr_stream << "/* unsupported call */";
    }
  } else {
    // Fallback for unsupported expression types
    expr_stream << "/* unsupported expr: " << expr->GetTypeKey() << " */";
  }

  return expr_stream.str();
}

bool TTCodegenVisitor::IsCircularBuffer(const Buffer &buf) {
  // Check if buffer scope indicates L1 memory
  std::string scope = buf.scope();
  return (scope == "local" || scope == "shared");
}

int TTCodegenVisitor::GetCBId(const Buffer &buf) {
  // Look up CB ID from metadata map
  auto it = buffer_to_cb_id_.find(buf.get());
  if (it != buffer_to_cb_id_.end()) {
    return it->second;
  }
  return -1; // Not a circular buffer
}

bool TTCodegenVisitor::IsGlobalBuffer(const Buffer &buf) {
  std::string scope = buf.scope();
  return (scope == "global" || scope.empty());
}

int TTCodegenVisitor::GetDTypeSize(const DataType &dtype) {
  if (dtype == DataType::Float(16)) {
    return 2; // fp16 = 2 bytes
  } else if (dtype == DataType::Float(32)) {
    return 4; // fp32 = 4 bytes
  } else if (dtype == DataType::Int(32)) {
    return 4;
  } else if (dtype == DataType::Int(16)) {
    return 2;
  } else if (dtype == DataType::Int(8)) {
    return 1;
  } else {
    return 4; // Default
  }
}

void TTCodegenVisitor::InitBufferMetadata() {
  // Extract circular buffer information from func attributes
  // Look for tt_circular_buffers attribute (list of CB configs)
  auto cb_configs = func_->attrs.GetAttr<Array<Map<String, ObjectRef>>>(
      "tt_circular_buffers");

  if (cb_configs.defined()) {
    for (size_t i = 0; i < cb_configs.value().size(); ++i) {
      auto cb_config = cb_configs.value()[i];

      // Extract CB ID and buffer name
      if (cb_config.count("cb_id") && cb_config.count("buffer_name")) {
        auto cb_id_ref = cb_config.at("cb_id");
        auto buf_name_ref = cb_config.at("buffer_name");

        int cb_id = Downcast<Integer>(cb_id_ref)->value;
        std::string buf_name = Downcast<String>(buf_name_ref);

        // Find buffer by name in func params
        for (const auto &param : func_->params) {
          if (param->name_hint == buf_name) {
            // Map buffer to CB ID
            // Note: We need to map BufferNode*, not the name
            // This is a simplified version - real implementation would track
            // buffer nodes more carefully
            break;
          }
        }
      }
    }
  }
}

std::string TTCodegenVisitor::GetOrCreateVarName(const std::string &original) {
  // Check if we already have a mapping
  auto it = var_name_map_.find(original);
  if (it != var_name_map_.end()) {
    return it->second;
  }

  // Create new C++-compatible name
  std::string cpp_name = original;

  // Replace invalid C++ characters
  std::replace(cpp_name.begin(), cpp_name.end(), '.', '_');
  std::replace(cpp_name.begin(), cpp_name.end(), ':', '_');
  std::replace(cpp_name.begin(), cpp_name.end(), '-', '_');

  // Ensure it doesn't start with a digit
  if (!cpp_name.empty() && std::isdigit(cpp_name[0])) {
    cpp_name = "v_" + cpp_name;
  }

  // Check for C++ keywords and add suffix if needed
  static const std::vector<std::string> keywords = {
      "if",   "else", "for",   "while",  "do",
      "void", "int",  "float", "return", "break"};
  if (std::find(keywords.begin(), keywords.end(), cpp_name) != keywords.end()) {
    cpp_name += "_var";
  }

  // Store mapping
  var_name_map_[original] = cpp_name;
  return cpp_name;
}

} // namespace tl
} // namespace tvm
