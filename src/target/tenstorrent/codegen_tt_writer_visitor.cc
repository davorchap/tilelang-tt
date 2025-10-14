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
#include <tvm/runtime/logging.h>

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
  bool is_local_shard = partition_mode() == "local_shard";
  int start_idx = GetRuntimeArgIndex("tt_start_tile");
  ICHECK_GE(start_idx, 0) << "Missing tt_start_tile runtime argument";
  EmitLine("uint32_t out_tile_start_id = get_arg_val<uint32_t>(" + std::to_string(start_idx) + ");");

  int count_idx = GetRuntimeArgIndex("tt_tile_count");
  ICHECK_GE(count_idx, 0) << "Missing tt_tile_count runtime argument";
  EmitLine("uint32_t num_out_tiles = get_arg_val<uint32_t>(" + std::to_string(count_idx) + ");");

  int nt_idx = GetRuntimeArgIndex("Nt");
  if (nt_idx >= 0) {
    EmitLine("uint32_t Nt = get_arg_val<uint32_t>(" + std::to_string(nt_idx) + ");");
  } else {
    EmitLine("uint32_t Nt = " + std::to_string(GetRuntimeConst<int>("Nt", 1)) + ";");
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
    EmitLine("(void)tt_shard_coord_y;");
  }
  if (has_shard_coord_x) {
    EmitLine("uint32_t tt_shard_coord_x = get_arg_val<uint32_t>(" +
             std::to_string(GetRuntimeArgIndex("tt_shard_coord_x")) + ");");
    EmitLine("(void)tt_shard_coord_x;");
  }
  EmitLine("");

  // Emit persistent loop structure
  EmitLine("// Write output tiles");
  EmitLine("for (uint32_t out_tile = 0; out_tile < num_out_tiles; ++out_tile) {");
  IncIndent();

  EmitLine("uint32_t tile_idx = out_tile_start_id + out_tile;");
  EmitLine("");

  // Emit C tile write
  EmitCBWaitFront(2, 1);
  EmitLine("uint32_t l1_read_addr = get_read_ptr(cb_out0);");
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
  EmitLine("inline void cb_wait_front(uint32_t cb_id, uint32_t n_tiles) {}");
  EmitLine("inline uint32_t get_read_ptr(uint32_t cb_id) { return 0; }");
  EmitLine("inline void noc_async_write_tile(uint32_t tile_idx, uint32_t l1_addr, uint32_t base_addr) {}");
  EmitLine("inline void noc_async_write_barrier() {}");
  EmitLine("inline void cb_pop_front(uint32_t cb_id, uint32_t n_tiles) {}");
#endif
  EmitLine("");
  EmitLine("// Circular Buffer Index");
  EmitLine("constexpr auto cb_out0 = tt::CBIndex::c_16;");
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
  line << "cb_wait_front(cb_out0, " << ntiles << ");";
  EmitLine(line.str());
}

void TTWriterCodegenVisitor::EmitCBPopFront(uint32_t cb_id, uint32_t ntiles) {
  std::ostringstream line;
  line << "cb_pop_front(cb_out0, " << ntiles << ");";
  EmitLine(line.str());
}

}  // namespace tl
}  // namespace tvm
