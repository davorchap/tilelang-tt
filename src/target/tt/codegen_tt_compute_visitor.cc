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
 * \file codegen_tt_compute_visitor.cc
 * \brief Implementation of compute kernel IR-driven visitor
 */

#include "codegen_tt_compute_visitor.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tir;

TTComputeCodegenVisitor::TTComputeCodegenVisitor(const PrimFunc& func)
    : TTCodegenVisitor(func), matmul_init_emitted_(false), current_k_iter_(0) {}

std::string TTComputeCodegenVisitor::GetFullKernel() {
  // Start fresh
  code_.str("");
  code_.clear();
  indent_level_ = 0;

  // Emit preamble
  EmitPreamble();

  // Emit MAIN function header
  EmitLine("void MAIN() {");
  IncIndent();

  // Emit runtime argument extraction
  EmitLine("// Runtime arguments");
  EmitLine("uint32_t out_tile_start_id = get_arg_val<uint32_t>(0);");
  EmitLine("uint32_t num_output_tiles = get_arg_val<uint32_t>(1);");
  EmitLine("uint32_t Kt = get_arg_val<uint32_t>(2);");
  EmitLine("");

  // Walk the function body (should contain persistent loop + K-loop)
  VisitStmt(func_->body);

  // Close MAIN function
  DecIndent();
  EmitLine("}");

  return GetCode();
}

void TTComputeCodegenVisitor::EmitPreamble() {
  EmitLine("// Generated TT Compute Kernel (IR-Driven)");

  // Extract grid metadata
  auto grid_x = func_->attrs.GetAttr<Integer>("tt_grid_x");
  auto grid_y = func_->attrs.GetAttr<Integer>("tt_grid_y");
  auto num_cores = func_->attrs.GetAttr<Integer>("tt_num_cores");

  if (grid_x.defined() && grid_y.defined()) {
    code_ << "// Grid: " << grid_x.value()->value << "x" << grid_y.value()->value << "\n";
  }
  if (num_cores.defined()) {
    code_ << "// Cores: " << num_cores.value()->value << "\n";
  }
  code_ << "\n";

  // Includes and mock APIs
  EmitLine("#include <cstdint>");
  EmitLine("");
  EmitLine("// Mock TT intrinsics for dry-run");
  EmitLine("template<typename T>");
  EmitLine("inline T get_arg_val(uint32_t idx) { return T(); }");
  EmitLine("");
  EmitLine("// Mock TT circular buffer APIs for dry-run");
  EmitLine("inline void cb_wait_front(uint32_t cb_id, uint32_t n_tiles) {}");
  EmitLine("inline void cb_reserve_back(uint32_t cb_id, uint32_t n_tiles) {}");
  EmitLine("inline void cb_pop_front(uint32_t cb_id, uint32_t n_tiles) {}");
  EmitLine("inline void cb_push_back(uint32_t cb_id, uint32_t n_tiles) {}");
  EmitLine("");
  EmitLine("// Mock TT matmul compute APIs for dry-run");
  EmitLine("inline void matmul_tiles_init(uint32_t cb_a, uint32_t cb_b, uint32_t cb_c) {}");
  EmitLine("inline void matmul_tiles(uint32_t cb_a, uint32_t cb_b, uint32_t cb_c, bool accumulate) {}");
  EmitLine("");
  EmitLine("// Circular Buffer Indices");
  EmitLine("constexpr uint32_t CB_A = 0;");
  EmitLine("constexpr uint32_t CB_B = 1;");
  EmitLine("constexpr uint32_t CB_C = 2;");
  EmitLine("");
}

void TTComputeCodegenVisitor::VisitStmt_(const ForNode* op) {
  // Emit for loop
  std::string loop_var = GetVarName(op->loop_var);
  std::string min_expr = EmitExpr(op->min);
  std::string extent_expr = EmitExpr(op->extent);

  // Check if this is a K-loop (kt variable)
  if (loop_var == "kt" || loop_var.find("kt") != std::string::npos ||
      loop_var == "k" || loop_var.find("_k") != std::string::npos) {
    EmitLine("// K-loop: C[m,n] += sum(A[m,k] * B[k,n] for k in Kt)");
  }

  // Emit loop header (build as string to avoid clearing code_ stream)
  std::ostringstream loop_line;
  loop_line << "for (uint32_t " << loop_var << " = " << min_expr << "; ";
  loop_line << loop_var << " < ";
  if (min_expr == "0") {
    loop_line << extent_expr;
  } else {
    loop_line << min_expr << " + " << extent_expr;
  }
  loop_line << "; ++" << loop_var << ") {";
  EmitLine(loop_line.str());

  IncIndent();

  // Visit body
  VisitStmt(op->body);

  DecIndent();
  EmitLine("}");
}

void TTComputeCodegenVisitor::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == "tt.matmul_intrinsic") {
    // Found matmul intrinsic annotation
    EmitMatmulIntrinsic(op);
  } else if (op->attr_key == "tt.copy") {
    // Handle copy operation (CB wait/pop/push)
    // For now, just visit body
    VisitStmt(op->body);
  } else if (op->attr_key == "tt.cb_wait") {
    // CB wait operation
    if (auto* int_imm = op->value.as<IntImmNode>()) {
      uint32_t cb_id = static_cast<uint32_t>(int_imm->value);
      EmitCBWait(cb_id, 1);
    }
    VisitStmt(op->body);
  } else if (op->attr_key == "tt.cb_pop") {
    // CB pop operation
    if (auto* int_imm = op->value.as<IntImmNode>()) {
      uint32_t cb_id = static_cast<uint32_t>(int_imm->value);
      EmitCBPop(cb_id, 1);
    }
    VisitStmt(op->body);
  } else if (op->attr_key == "tt.cb_push") {
    // CB push operation
    if (auto* int_imm = op->value.as<IntImmNode>()) {
      uint32_t cb_id = static_cast<uint32_t>(int_imm->value);
      EmitCBPush(cb_id, 1);
    }
    VisitStmt(op->body);
  } else {
    // Default: visit body
    TTCodegenVisitor::VisitStmt_(op);
  }
}

void TTComputeCodegenVisitor::EmitMatmulIntrinsic(const AttrStmtNode* op) {
  // Get matmul ID to determine if init or accumulate
  int matmul_id = GetMatmulId(op->value);

  if (matmul_id == 0 && !matmul_init_emitted_) {
    // First matmul: emit init
    EmitLine("// Initialize matmul");
    EmitLine("matmul_tiles_init(CB_A, CB_B, CB_C);");
    matmul_init_emitted_ = true;
  }

  // Always emit wait for input tiles
  EmitLine("// Wait for input tiles from reader");
  EmitLine("cb_wait_front(CB_A, 1);");
  EmitLine("cb_wait_front(CB_B, 1);");
  EmitLine("");

  // Emit matmul operation
  // Accumulate if not the first K iteration
  bool accumulate = (current_k_iter_ > 0);
  if (accumulate) {
    EmitLine("// Matmul: accumulate");
    EmitLine("matmul_tiles(CB_A, CB_B, CB_C, true);");
  } else {
    EmitLine("// Matmul: first K iteration");
    EmitLine("matmul_tiles(CB_A, CB_B, CB_C, false);");
  }
  EmitLine("");

  // Pop input tiles
  EmitLine("// Release input tiles");
  EmitLine("cb_pop_front(CB_A, 1);");
  EmitLine("cb_pop_front(CB_B, 1);");

  current_k_iter_++;

  // Visit body if any
  VisitStmt(op->body);
}

void TTComputeCodegenVisitor::EmitCBWait(uint32_t cb_id, uint32_t ntiles) {
  std::ostringstream line;
  line << "cb_wait_front(" << cb_id << ", " << ntiles << ");";
  EmitLine(line.str());
}

void TTComputeCodegenVisitor::EmitCBPop(uint32_t cb_id, uint32_t ntiles) {
  std::ostringstream line;
  line << "cb_pop_front(" << cb_id << ", " << ntiles << ");";
  EmitLine(line.str());
}

void TTComputeCodegenVisitor::EmitCBPush(uint32_t cb_id, uint32_t ntiles) {
  std::ostringstream line;
  line << "cb_push_back(" << cb_id << ", " << ntiles << ");";
  EmitLine(line.str());
}

int TTComputeCodegenVisitor::GetMatmulId(const PrimExpr& value) {
  if (auto* int_imm = value.as<IntImmNode>()) {
    return static_cast<int>(int_imm->value);
  }
  return 0;
}

}  // namespace tl
}  // namespace tvm
