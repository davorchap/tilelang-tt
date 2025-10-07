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
 * \file codegen_tt_writer_visitor.cc
 * \brief Implementation of writer kernel IR-driven visitor
 */

#include "codegen_tt_writer_visitor.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tir;

TTWriterCodegenVisitor::TTWriterCodegenVisitor(const PrimFunc& func) : TTCodegenVisitor(func) {}

std::string TTWriterCodegenVisitor::GetFullKernel() {
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
  EmitLine("uint32_t dram_addr_c = get_arg_val<uint32_t>(0);");
  EmitLine("uint32_t out_tile_start_id = get_arg_val<uint32_t>(1);");
  EmitLine("uint32_t num_out_tiles = get_arg_val<uint32_t>(2);");
  EmitLine("uint32_t Nt = get_arg_val<uint32_t>(3);");
  EmitLine("");

  // Emit persistent loop structure
  EmitLine("// Write output tiles");
  EmitLine("for (uint32_t out_tile = 0; out_tile < num_out_tiles; ++out_tile) {");
  IncIndent();

  EmitLine("uint32_t tile_idx = out_tile_start_id + out_tile;");
  EmitLine("");

  // Emit C tile write
  EmitCBWaitFront(2, 1);
  EmitLine("uint32_t l1_read_addr = get_read_ptr(CB_C);");
  EmitLine("noc_async_write_tile(tile_idx, l1_read_addr, dram_addr_c);");
  EmitLine("noc_async_write_barrier();");
  EmitCBPopFront(2, 1);

  DecIndent();
  EmitLine("}");  // Persistent loop

  // Close kernel_main function
  DecIndent();
  EmitLine("}");

  return GetCode();
}

void TTWriterCodegenVisitor::EmitPreamble() {
  EmitLine("// Generated TT Writer Kernel (IR-Driven)");
  EmitLine("// Matmul Writer: Writes C[m,n] output tiles");
  EmitLine("");

  // Extract matmul dimensions from metadata
  auto grid_x = func_->attrs.GetAttr<Integer>("tt_grid_x");
  auto grid_y = func_->attrs.GetAttr<Integer>("tt_grid_y");

  if (grid_x.defined() && grid_y.defined()) {
    code_ << "// Grid: " << grid_x.value()->value << "x" << grid_y.value()->value << "\n";
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
  EmitLine("inline uint32_t get_read_ptr(uint32_t cb_id) { return 0; }");
  EmitLine("inline void noc_async_write_tile(uint32_t tile_idx, uint32_t l1_addr, uint32_t base_addr) {}");
  EmitLine("inline void noc_async_write_barrier() {}");
  EmitLine("inline void cb_pop_front(uint32_t cb_id, uint32_t n_tiles) {}");
  EmitLine("");
  EmitLine("// Circular Buffer Index");
  EmitLine("constexpr uint32_t CB_C = 2;");
  EmitLine("");
  EmitLine("constexpr uint32_t TILE_SIZE_BYTES = 32 * 32 * sizeof(uint16_t);  // fp16");
  EmitLine("");
}

void TTWriterCodegenVisitor::VisitStmt_(const ForNode* op) {
  // For writer, we generate fixed structure, so this is mostly pass-through
  TTCodegenVisitor::VisitStmt_(op);
}

void TTWriterCodegenVisitor::VisitStmt_(const BufferStoreNode* op) {
  // Writer kernel handles buffer stores by emitting NOC writes
  // This is called during IR walking but actual emission is in GetFullKernel
  // Do nothing here - emission is handled by GetFullKernel's fixed structure
}

void TTWriterCodegenVisitor::EmitNOCWrite(const Buffer& dst, const std::string& tile_idx,
                                          uint32_t cb_id) {
  std::ostringstream buf_name;
  buf_name << GetBufferName(dst);

  EmitCBWaitFront(cb_id, 1);
  EmitLine("uint32_t l1_addr = get_read_ptr(CB_" + buf_name.str() + ");");
  EmitLine("noc_async_write_tile(" + tile_idx + ", l1_addr, dram_addr_" + buf_name.str() + ");");
  EmitLine("noc_async_write_barrier();");
  EmitCBPopFront(cb_id, 1);
}

void TTWriterCodegenVisitor::EmitCBWaitFront(uint32_t cb_id, uint32_t ntiles) {
  std::ostringstream line;
  line << "cb_wait_front(CB_C, " << ntiles << ");";
  EmitLine(line.str());
}

void TTWriterCodegenVisitor::EmitCBPopFront(uint32_t cb_id, uint32_t ntiles) {
  std::ostringstream line;
  line << "cb_pop_front(CB_C, " << ntiles << ");";
  EmitLine(line.str());
}

}  // namespace tl
}  // namespace tvm
