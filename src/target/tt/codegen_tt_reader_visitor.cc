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
 * \file codegen_tt_reader_visitor.cc
 * \brief Implementation of reader kernel IR-driven visitor
 */

#include "codegen_tt_reader_visitor.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tir;

TTReaderCodegenVisitor::TTReaderCodegenVisitor(const PrimFunc& func) : TTCodegenVisitor(func) {}

std::string TTReaderCodegenVisitor::GetFullKernel() {
  // Start fresh
  code_.str("");
  code_.clear();
  indent_level_ = 0;

  // Emit preamble
  EmitPreamble();

  // Emit kernel_main function header
  EmitLine("void kernel_main() {");
  IncIndent();

  // Emit runtime argument extraction
  EmitLine("// Runtime arguments");
  EmitLine("uint32_t dram_addr_a = get_arg_val<uint32_t>(0);");
  EmitLine("uint32_t dram_addr_b = get_arg_val<uint32_t>(1);");
  EmitLine("uint32_t Mt = get_arg_val<uint32_t>(2);");
  EmitLine("uint32_t Kt = get_arg_val<uint32_t>(3);");
  EmitLine("uint32_t Nt = get_arg_val<uint32_t>(4);");
  EmitLine("uint32_t out_tile_start_id = get_arg_val<uint32_t>(5);");
  EmitLine("uint32_t num_out_tiles = get_arg_val<uint32_t>(6);");
  EmitLine("");

  // Emit persistent loop structure
  EmitLine("// Process output tiles");
  EmitLine("for (uint32_t out_tile = 0; out_tile < num_out_tiles; ++out_tile) {");
  IncIndent();

  EmitLine("uint32_t current_tile_id = out_tile_start_id + out_tile;");
  EmitLine("uint32_t out_m = current_tile_id / Nt;");
  EmitLine("uint32_t out_n = current_tile_id % Nt;");
  EmitLine("");

  EmitLine("// Load tiles for this output: A[out_m,:] and B[:,out_n]");
  EmitLine("for (uint32_t kt = 0; kt < Kt; ++kt) {");
  IncIndent();

  // Emit A tile load: A[out_m, kt]
  EmitLine("// Read A[out_m, kt]");
  EmitLine("uint32_t tile_a_idx = out_m * Kt + kt;");
  EmitCBReserve(0, 1);
  EmitLine("uint32_t l1_write_addr_a = get_write_ptr(cb_in0);");
  EmitLine("noc_async_read_tile(tile_a_idx, dram_addr_a, l1_write_addr_a);");
  EmitLine("noc_async_read_barrier();");
  EmitCBPushBack(0, 1);
  EmitLine("");

  // Emit B tile load: B[kt, out_n]
  EmitLine("// Read B[kt, out_n]");
  EmitLine("uint32_t tile_b_idx = kt * Nt + out_n;");
  EmitCBReserve(1, 1);
  EmitLine("uint32_t l1_write_addr_b = get_write_ptr(cb_in1);");
  EmitLine("noc_async_read_tile(tile_b_idx, dram_addr_b, l1_write_addr_b);");
  EmitLine("noc_async_read_barrier();");
  EmitCBPushBack(1, 1);

  DecIndent();
  EmitLine("}");  // K-loop

  DecIndent();
  EmitLine("}");  // Persistent loop

  // Close kernel_main function
  DecIndent();
  EmitLine("}");

  return GetCode();
}

void TTReaderCodegenVisitor::EmitPreamble() {
  EmitLine("// Generated TT Reader Kernel (IR-Driven)");
  EmitLine("// Matmul Reader: Loads A[m,k] and B[k,n] tiles");
  EmitLine("");

  // Extract matmul dimensions from metadata
  auto grid_x = func_->attrs.GetAttr<Integer>("tt_grid_x");
  auto grid_y = func_->attrs.GetAttr<Integer>("tt_grid_y");

  if (grid_x.defined() && grid_y.defined()) {
    code_ << "// Grid: " << grid_x.value()->value << "x" << grid_y.value()->value << "\n";
  }
  code_ << "\n";

  // Includes - Real Metalium headers (Week 16 integration)
#ifdef TL_USE_REAL_METALIUM
  EmitLine("#include \"dataflow_api.h\"");
#else
  // Mock APIs for dry-run (backward compatibility)
  EmitLine("#include <cstdint>");
  EmitLine("");
  EmitLine("// Mock TT intrinsics for dry-run");
  EmitLine("template<typename T>");
  EmitLine("inline T get_arg_val(uint32_t idx) { return T(); }");
  EmitLine("");
  EmitLine("// Mock TT circular buffer APIs for dry-run");
  EmitLine("inline void cb_reserve_back(uint32_t cb_id, uint32_t n_tiles) {}");
  EmitLine("inline uint32_t get_write_ptr(uint32_t cb_id) { return 0; }");
  EmitLine("inline void noc_async_read_tile(uint32_t tile_idx, uint32_t base_addr, uint32_t l1_addr) {}");
  EmitLine("inline void noc_async_read_barrier() {}");
  EmitLine("inline void cb_push_back(uint32_t cb_id, uint32_t n_tiles) {}");
#endif
  EmitLine("");
  EmitLine("// Circular Buffer Indices");
  EmitLine("constexpr auto cb_in0 = tt::CBIndex::c_0;");
  EmitLine("constexpr auto cb_in1 = tt::CBIndex::c_1;");
  EmitLine("");
  EmitLine("constexpr uint32_t TILE_SIZE_BYTES = 32 * 32 * sizeof(uint16_t);  // fp16");
  EmitLine("");
}

void TTReaderCodegenVisitor::VisitStmt_(const ForNode* op) {
  // For reader, we generate fixed structure, so this is mostly pass-through
  TTCodegenVisitor::VisitStmt_(op);
}

void TTReaderCodegenVisitor::VisitExpr_(const BufferLoadNode* op) {
  // Reader kernel handles buffer loads by emitting NOC reads
  // This is called during IR walking but actual emission is in GetFullKernel
  // Do nothing here - emission is handled by GetFullKernel's fixed structure
}

void TTReaderCodegenVisitor::EmitNOCRead(const Buffer& src, const std::string& tile_idx,
                                         uint32_t cb_id) {
  std::ostringstream buf_name;
  buf_name << GetBufferName(src);

  EmitCBReserve(cb_id, 1);
  EmitLine("uint32_t l1_addr = get_write_ptr(CB_" + buf_name.str() + ");");
  EmitLine("noc_async_read_tile(" + tile_idx + ", dram_addr_" + buf_name.str() + ", l1_addr);");
  EmitLine("noc_async_read_barrier();");
  EmitCBPushBack(cb_id, 1);
}

void TTReaderCodegenVisitor::EmitCBReserve(uint32_t cb_id, uint32_t ntiles) {
  std::ostringstream line;
  line << "cb_reserve_back(";
  if (cb_id == 0) {
    line << "cb_in0";
  } else if (cb_id == 1) {
    line << "cb_in1";
  } else {
    line << cb_id;
  }
  line << ", " << ntiles << ");";
  EmitLine(line.str());
}

void TTReaderCodegenVisitor::EmitCBPushBack(uint32_t cb_id, uint32_t ntiles) {
  std::ostringstream line;
  line << "cb_push_back(";
  if (cb_id == 0) {
    line << "cb_in0";
  } else if (cb_id == 1) {
    line << "cb_in1";
  } else {
    line << cb_id;
  }
  line << ", " << ntiles << ");";
  EmitLine(line.str());
}

std::string TTReaderCodegenVisitor::CalculateTileIndex(const Array<PrimExpr>& indices,
                                                        const std::string& num_tiles_dim) {
  ICHECK_EQ(indices.size(), 2) << "Expected 2D buffer indices";

  std::ostringstream idx;
  idx << "(" << EmitExpr(indices[0]) << " * " << num_tiles_dim << " + ";
  idx << EmitExpr(indices[1]) << ")";
  return idx.str();
}

}  // namespace tl
}  // namespace tvm
