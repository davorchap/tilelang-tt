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
 * \brief Simplified compute kernel visitor that emits TT intrinsics.
 */

#include "codegen_tt_compute_visitor.h"

#include <tvm/runtime/logging.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace tl {

using namespace tir;

TTComputeCodegenVisitor::TTComputeCodegenVisitor(const PrimFunc& func)
    : TTCodegenVisitor(func) {}

std::string TTComputeCodegenVisitor::GetFullKernel() {
  // Reset output stream and indentation
  code_.str("");
  code_.clear();
  indent_level_ = 0;

  EmitPreamble();

  EmitLine("void MAIN() {");
  IncIndent();

  EmitLine("// Runtime arguments");
  bool is_local_shard = partition_mode() == "local_shard";

  int start_idx = GetRuntimeArgIndex("tt_start_tile");
  ICHECK_GE(start_idx, 0) << "Missing tt_start_tile runtime argument";
  EmitLine("uint32_t tt_start_tile = get_arg_val<uint32_t>(" + std::to_string(start_idx) + ");");

  int count_idx = GetRuntimeArgIndex("tt_tile_count");
  ICHECK_GE(count_idx, 0) << "Missing tt_tile_count runtime argument";
  EmitLine("uint32_t tt_tile_count = get_arg_val<uint32_t>(" + std::to_string(count_idx) + ");");

  int kt_idx = GetRuntimeArgIndex("Kt");
  if (kt_idx >= 0) {
    EmitLine("uint32_t Kt = get_arg_val<uint32_t>(" + std::to_string(kt_idx) + ");");
  } else {
    EmitLine("uint32_t Kt = " + std::to_string(GetRuntimeConst<int>("Kt", 1)) + ";");
  }

  bool has_shard_coord_y = HasRuntimeArg("tt_shard_coord_y");
  bool has_shard_coord_x = HasRuntimeArg("tt_shard_coord_x");

  if (is_local_shard) {
    ICHECK(has_shard_coord_y) << "local_shard partition mode requires tt_shard_coord_y arg";
    ICHECK(has_shard_coord_x) << "local_shard partition mode requires tt_shard_coord_x arg";
  }
  if (has_shard_coord_y) {
    EmitLine("uint32_t tt_shard_coord_y = get_arg_val<uint32_t>(" +
             std::to_string(GetRuntimeArgIndex("tt_shard_coord_y")) + ");");
  }
  if (has_shard_coord_x) {
    EmitLine("uint32_t tt_shard_coord_x = get_arg_val<uint32_t>(" +
             std::to_string(GetRuntimeArgIndex("tt_shard_coord_x")) + ");");
  }

  EmitLine("uint32_t out_tile_start_id = tt_start_tile;");
  EmitLine("uint32_t num_output_tiles = tt_tile_count;");
  EmitLine("");

  VisitStmt(func_->body);

  DecIndent();
  EmitLine("}");

  return GetCode();
}

void TTComputeCodegenVisitor::EmitPreamble() {
  EmitLine("// Generated TT Compute Kernel (IR-Driven)");

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

#ifdef TL_USE_REAL_METALIUM
  EmitLine("#include \"ckernel_include.h\"");
  EmitLine("#include \"ckernel_defs.h\"");
  EmitLine("#include \"compute_kernel_api/common.h\"");
  EmitLine("#include \"compute_kernel_api/tile_move_copy.h\"");
  EmitLine("#include \"compute_kernel_api/eltwise_binary.h\"");
  EmitLine("#include \"compute_kernel_api/matmul.h\"");
#else
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
  EmitLine("inline void mm_init(uint32_t cb_in0, uint32_t cb_in1, uint32_t cb_out = 16) {}");
  EmitLine("inline void matmul_tiles(uint32_t cb_in0, uint32_t cb_in1, uint32_t tile_idx_in0, "
           "uint32_t tile_idx_in1, uint32_t dst_tile_idx, bool transpose) {}");
  EmitLine("");
  EmitLine("// Mock TT tile register APIs for dry-run");
  EmitLine("inline void tile_regs_acquire() {}");
  EmitLine("inline void tile_regs_commit() {}");
  EmitLine("inline void tile_regs_wait() {}");
  EmitLine("inline void tile_regs_release() {}");
  EmitLine("");
  EmitLine("// Mock TT element-wise compute APIs for dry-run");
  EmitLine("inline void binary_op_init_common(uint32_t cb_in0, uint32_t cb_in1, "
           "uint32_t cb_out = 16) {}");
  EmitLine("inline void add_tiles_init(uint32_t cb_in0 = 0, uint32_t cb_in1 = 1) {}");
  EmitLine("inline void add_tiles(uint32_t cb_a, uint32_t cb_b, uint32_t idx_a, uint32_t idx_b, "
           "uint32_t idx_dst) {}");
  EmitLine("inline void pack_tile(uint32_t idx_dst, uint32_t cb_out) {}");
#endif
  EmitLine("");
  EmitLine("// Circular Buffer Indices");
  EmitLine("constexpr auto cb_in0 = tt::CBIndex::c_0;");
  EmitLine("constexpr auto cb_in1 = tt::CBIndex::c_1;");
  EmitLine("constexpr auto cb_out0 = tt::CBIndex::c_16;");
  EmitLine("");
}

void TTComputeCodegenVisitor::VisitStmt_(const ForNode* op) {
  std::string loop_var = GetVarName(op->loop_var);
  std::string min_expr = EmitExpr(op->min);
  std::string extent_expr = EmitExpr(op->extent);

  EmitLine("for (uint32_t " + loop_var + " = " + min_expr + "; " + loop_var + " < " +
           min_expr + " + " + extent_expr + "; ++" + loop_var + ") {");
  IncIndent();
  VisitStmt(op->body);
  DecIndent();
  EmitLine("}");
}

void TTComputeCodegenVisitor::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key.rfind("tt.", 0) == 0) {
    VisitStmt(op->body);
    return;
  }
  TTCodegenVisitor::VisitStmt_(op);
}

}  // namespace tl
}  // namespace tvm

