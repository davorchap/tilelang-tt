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

#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/runtime/logging.h>

#include <sstream>

namespace tvm {
namespace tl {

using namespace tir;

TTComputeCodegenVisitor::TTComputeCodegenVisitor(const PrimFunc& func)
    : TTCodegenVisitor(func),
      matmul_init_emitted_(false),
      elementwise_init_emitted_(false),
      current_k_iter_(0),
      k_loop_var_(""),
      dst_acquired_(false),
      loop_depth_(0),
      current_pattern_(ComputePattern::UNKNOWN),
      reduction_init_emitted_(false),
      gemv_init_emitted_(false) {}

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

  if (HasRuntimeArg("tt_shard_coord_y")) {
    EmitLine("uint32_t tt_shard_coord_y = get_arg_val<uint32_t>(" +
             std::to_string(GetRuntimeArgIndex("tt_shard_coord_y")) + ");");
  }
  if (HasRuntimeArg("tt_shard_coord_x")) {
    EmitLine("uint32_t tt_shard_coord_x = get_arg_val<uint32_t>(" +
             std::to_string(GetRuntimeArgIndex("tt_shard_coord_x")) + ");");
  }

  EmitLine("uint32_t out_tile_start_id = tt_start_tile;");
  EmitLine("uint32_t num_output_tiles = tt_tile_count;");
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

  // Includes - Real Metalium headers (Week 16 integration)
#ifdef TL_USE_REAL_METALIUM
  EmitLine("#include \"ckernel_include.h\"");
  EmitLine("#include \"ckernel_defs.h\"");
  EmitLine("#include \"compute_kernel_api/common.h\"");
  EmitLine("#include \"compute_kernel_api/tile_move_copy.h\"");
  EmitLine("#include \"compute_kernel_api/eltwise_binary.h\"");
  EmitLine("#include \"compute_kernel_api/matmul.h\"");
#else
  // Mock APIs for dry-run (backward compatibility)
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
  EmitLine("inline void matmul_tiles(uint32_t cb_in0, uint32_t cb_in1, uint32_t tile_idx_in0, uint32_t tile_idx_in1, uint32_t dst_tile_idx, bool transpose) {}");
  EmitLine("");
  EmitLine("// Mock TT tile register APIs for dry-run");
  EmitLine("inline void tile_regs_acquire() {}");
  EmitLine("inline void tile_regs_commit() {}");
  EmitLine("inline void tile_regs_wait() {}");
  EmitLine("inline void tile_regs_release() {}");
  EmitLine("");
  EmitLine("// Mock TT element-wise compute APIs for dry-run");
  EmitLine("inline void binary_op_init_common(uint32_t cb_in0, uint32_t cb_in1, uint32_t cb_out = 16) {}");
  EmitLine("inline void add_tiles_init(uint32_t cb_in0 = 0, uint32_t cb_in1 = 1) {}");
  EmitLine("inline void add_tiles(uint32_t cb_a, uint32_t cb_b, uint32_t idx_a, uint32_t idx_b, uint32_t idx_dst) {}");
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
  // Emit for loop
  std::string loop_var = GetVarName(op->loop_var);
  std::string min_expr = EmitExpr(op->min);
  std::string extent_expr = EmitExpr(op->extent);

  // Track loop depth
  loop_depth_++;

  // Detect tile-sized loops (32x32 element-wise operations)
  bool is_tile_loop_outer = (extent_expr == "32" && loop_depth_ >= 2);
  if (is_tile_loop_outer) {
    // Check if the body contains a nested 32-sized loop (i,j pattern)
    if (auto* inner_for = op->body.as<ForNode>()) {
      std::string inner_extent = EmitExpr(inner_for->extent);
      if (inner_extent == "32") {
        // Found T.grid(32, 32) pattern - emit element-wise intrinsic instead

        // Emit initialization once before first element-wise operation
        if (!elementwise_init_emitted_) {
          EmitLine("// Initialize element-wise operation (once before all operations)");
          EmitLine("binary_op_init_common(cb_in0, cb_in1, cb_out0);");
          EmitLine("add_tiles_init();");
          EmitLine("");
          elementwise_init_emitted_ = true;
        }

        EmitLine("// Wait for input tiles from reader");
        EmitLine("cb_wait_front(cb_in0, 1);");
        EmitLine("cb_wait_front(cb_in1, 1);");
        EmitLine("");

        // Acquire tile registers for this tile
        EmitLine("// Acquire tile registers for computation");
        EmitLine("tile_regs_acquire();");
        EmitLine("");

        EmitLine("// Compute C = A + B (element-wise)");
        EmitLine("add_tiles(cb_in0, cb_in1, 0, 0, 0);");
        EmitLine("");

        // Commit and wait for computation
        EmitLine("// Commit tile register computation");
        EmitLine("tile_regs_commit();");
        EmitLine("tile_regs_wait();");
        EmitLine("");

        // Pack result to output CB
        EmitLine("// Pack result to output circular buffer");
        EmitLine("cb_reserve_back(cb_out0, 1);");
        EmitLine("pack_tile(0, cb_out0);");
        EmitLine("cb_push_back(cb_out0, 1);");
        EmitLine("");

        EmitLine("// Pop input tiles");
        EmitLine("cb_pop_front(cb_in0, 1);");
        EmitLine("cb_pop_front(cb_in1, 1);");
        EmitLine("");

        // Release tile registers
        EmitLine("// Release tile registers");
        EmitLine("tile_regs_release();");
        EmitLine("");

        // Skip the loop body (already emitted intrinsic)
        loop_depth_--;
        return;
      }
    }
  }

  // Detect K-loop (inner loop for matmul accumulation)
  bool is_k_loop = (loop_var == "kt" || loop_var.find("kt") != std::string::npos ||
                    loop_var == "k" || loop_var.find("_k") != std::string::npos);

  // Detect outer tile loop (persistent loop)
  bool is_outer_loop = (loop_depth_ == 1);

  // For outer tile loop: reset state for each output tile
  if (is_outer_loop) {
    // Reset K-loop iteration counter for each output tile
    current_k_iter_ = 0;
    // Note: matmul_init_emitted_ is NOT reset - mm_init() called once before all loops
  }

  // Emit K-loop comment and init if detected
  if (is_k_loop) {
    // Acquire tile registers before K-loop (matmul pattern)
    if (!dst_acquired_) {
      EmitLine("// Acquire tile registers for matmul accumulation");
      EmitTileRegsAcquire();
      EmitLine("");
    }

    EmitLine("// K-loop: C[m,n] += sum(A[m,k] * B[k,n] for k in Kt)");

    // Store K-loop variable name for accumulate flag emission
    k_loop_var_ = loop_var;

    // Emit mm_init() before entering K-loop (only once for entire kernel)
    if (!matmul_init_emitted_) {
      EmitLine("// Initialize matmul (once before all loops)");
      EmitLine("mm_init(cb_in0, cb_in1, cb_out0);");
      EmitLine("");
      matmul_init_emitted_ = true;
    }
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

  // After loop completes: commit and release tile registers
  if (dst_acquired_) {
    if (is_k_loop) {
      // K-loop pattern: emit pack/commit/release after K-loop
      EmitLine("");
      EmitLine("// After K-loop: pack result");
      EmitTileRegsCommit();
      EmitTileRegsWait();
      EmitLine("cb_reserve_back(cb_out0, 1);");
      EmitLine("pack_tile(0, cb_out0);");
      EmitLine("cb_push_back(cb_out0, 1);");
      EmitTileRegsRelease();
      EmitLine("");
    } else if (is_outer_loop) {
      // Element-wise pattern: emit pack/commit/release after outer loop
      // Note: This path is dead code for element-wise (handled inside loop now)
      EmitLine("");
      EmitLine("// After tile processing: pack result");
      EmitTileRegsCommit();
      EmitTileRegsWait();
      EmitLine("cb_reserve_back(cb_out0, 1);");
      EmitLine("pack_tile(0, cb_out0);");
      EmitLine("cb_push_back(cb_out0, 1);");
      EmitTileRegsRelease();
      EmitLine("");
    }
  }

  loop_depth_--;
}

void TTComputeCodegenVisitor::VisitStmt_(const EvaluateNode* op) {
  // Check if this is a tl.copy or tl.gemm call
  if (auto* call = op->value.as<CallNode>()) {
    if (auto* op_node = call->op.as<OpNode>()) {
      std::string op_name = op_node->name;

      if (op_name == "tl.copy") {
        // T.copy() call - skip in compute kernel (handled by reader/writer)
        EmitLine("// T.copy - handled by reader/writer kernels");
        return;
      } else if (op_name == "tl.gemm") {
        // T.gemm() call - emit matmul intrinsics
        EmitGemmIntrinsic(call);
        return;
      }
    }
  }

  // Default: handle normally
  TTCodegenVisitor::VisitStmt_(op);
}

void TTComputeCodegenVisitor::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == "tt.matmul_intrinsic") {
    // Found matmul intrinsic annotation
    EmitMatmulIntrinsic(op);
  } else if (op->attr_key == "tt.elementwise_add") {
    // Found element-wise add intrinsic annotation
    EmitElementwiseAddIntrinsic(op);
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
    EmitLine("matmul_tiles_init(cb_in0, cb_in1, cb_out0);");
    matmul_init_emitted_ = true;
  }

  // Always emit wait for input tiles
  EmitLine("// Wait for input tiles from reader");
  EmitLine("cb_wait_front(cb_in0, 1);");
  EmitLine("cb_wait_front(cb_in1, 1);");
  EmitLine("");

  // Emit matmul operation
  // Real Metalium signature: matmul_tiles(cb_in0, cb_in1, tile_idx_in0, tile_idx_in1, dst_tile_idx, transpose)
  // Note: Real API automatically accumulates into dst across multiple calls
#ifdef TL_USE_REAL_METALIUM
  EmitLine("// Matmul: process tile (accumulation automatic)");
  EmitLine("matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);");
#else
  // For dry-run: use mock API (accumulation automatic, transpose=false)
  EmitLine("// Matmul: process tile");
  EmitLine("matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);");
#endif
  EmitLine("");

  // Pop input tiles
  EmitLine("// Release input tiles");
  EmitLine("cb_pop_front(cb_in0, 1);");
  EmitLine("cb_pop_front(cb_in1, 1);");

  current_k_iter_++;

  // Visit body if any
  VisitStmt(op->body);
}

void TTComputeCodegenVisitor::EmitGemmIntrinsic(const CallNode* call) {
  // Emit T.gemm() intrinsic
  // Pattern 3 (K-loop GEMM): DST held across K iterations for accumulation
  // Note: matmul_tiles_init() is emitted before K-loop in ForNode visitor

  // Wait for input tiles
  EmitLine("// Wait for input tiles from reader");
  EmitLine("cb_wait_front(cb_in0, 1);");
  EmitLine("cb_wait_front(cb_in1, 1);");
  EmitLine("");

  // Emit matmul operation (accumulation automatic across K iterations)
  EmitLine("// Matmul: process tile (accumulation automatic)");
  EmitLine("matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);");
  EmitLine("");

  // Pop input tiles
  EmitLine("// Release input tiles");
  EmitLine("cb_pop_front(cb_in0, 1);");
  EmitLine("cb_pop_front(cb_in1, 1);");
  EmitLine("");
}

void TTComputeCodegenVisitor::EmitElementwiseAddIntrinsic(const AttrStmtNode* op) {
  // Emit element-wise add pattern
  // Pattern 1 (Element-wise): DST lifecycle per tile, no accumulation

  // Wait for input tiles from reader kernel
  EmitLine("// Wait for input tiles from reader");
  EmitLine("cb_wait_front(cb_in0, 1);");
  EmitLine("cb_wait_front(cb_in1, 1);");
  EmitLine("");

  // Initialize and execute element-wise add
  EmitLine("// Compute C = A + B (element-wise)");
  EmitLine("add_tiles_init();");
  EmitLine("add_tiles(cb_in0, cb_in1, 0, 0, 0);");
  EmitLine("");

  // Visit body (loop body with element-wise operations)
  // We don't visit the body since we're replacing it with intrinsic
  // VisitStmt(op->body);

  // Note: DST commit/pack/release handled by outer loop in VisitStmt_(ForNode*)
  // CB pop for inputs happens here
  EmitLine("// Release input tiles");
  EmitLine("cb_pop_front(cb_in0, 1);");
  EmitLine("cb_pop_front(cb_in1, 1);");
  EmitLine("");
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

void TTComputeCodegenVisitor::EmitTileRegsAcquire() {
  if (!dst_acquired_) {
    EmitLine("// Acquire tile registers for computation");
    EmitLine("tile_regs_acquire();");
    dst_acquired_ = true;
  }
}

void TTComputeCodegenVisitor::EmitTileRegsCommit() {
  if (dst_acquired_) {
    EmitLine("// Commit tile register computation");
    EmitLine("tile_regs_commit();");
  }
}

void TTComputeCodegenVisitor::EmitTileRegsWait() {
  if (dst_acquired_) {
    EmitLine("// Wait for tile register computation to complete");
    EmitLine("tile_regs_wait();");
  }
}

void TTComputeCodegenVisitor::EmitTileRegsRelease() {
  if (dst_acquired_) {
    EmitLine("// Release tile registers");
    EmitLine("tile_regs_release();");
    dst_acquired_ = false;
  }
}

// ==========================
// PatternDetector Implementation
// ==========================

ComputePattern PatternDetector::DetectPattern(const ForNode* loop) {
  if (!loop || !loop->body.defined()) {
    return ComputePattern::UNKNOWN;
  }

  // Check for T.gemm() call (matmul pattern)
  if (HasGemmCall(loop->body)) {
    return ComputePattern::MATMUL;
  }

  // TODO: Implement additional pattern detection
  // For now, all other patterns return UNKNOWN
  // This allows incremental implementation without breaking existing code

  return ComputePattern::UNKNOWN;
}

bool PatternDetector::HasGemmCall(const Stmt& body) {
  // Visitor to detect T.gemm() calls in the statement tree
  class GemmDetector : public StmtExprVisitor {
   public:
    bool found_gemm = false;

    // Make VisitStmt public so we can call it
    using StmtExprVisitor::VisitStmt;

    void VisitStmt_(const AttrStmtNode* op) final {
      // Check for gemm_intrinsic annotation
      if (op->attr_key == "gemm_intrinsic" || op->attr_key == "pragma_gemm" ||
          op->attr_key == "matmul_intrinsic") {
        found_gemm = true;
      }
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitExpr_(const CallNode* op) final {
      // Check for tl.gemm or similar call names
      if (op->op.as<OpNode>()) {
        std::string call_name = op->op.as<OpNode>()->name;
        if (call_name.find("gemm") != std::string::npos ||
            call_name.find("matmul") != std::string::npos) {
          found_gemm = true;
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }
  };

  GemmDetector detector;
  detector.VisitStmt(body);
  return detector.found_gemm;
}

bool PatternDetector::HasReductionPattern(const Stmt& body) {
  // TODO: Implement reduction pattern detection
  // Look for accumulation pattern: var[i] = var[i] + expr
  return false;
}

bool PatternDetector::HasElementwisePattern(const Stmt& body) {
  // TODO: Implement element-wise pattern detection
  // Look for independent tile operations
  return false;
}

bool PatternDetector::HasGemvPattern(const Stmt& body) {
  // TODO: Implement GEMV pattern detection
  // Look for matrix-vector multiply pattern
  return false;
}

}  // namespace tl
}  // namespace tvm
