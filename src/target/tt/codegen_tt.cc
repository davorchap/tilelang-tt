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
 * \file codegen_tt.cc
 * \brief Code generator for Tenstorrent backend (Artifact Generation stage)
 *
 * This module generates Metalium-compatible C++ kernels and metadata
 * for dry-run execution on the Tenstorrent backend.
 *
 * Supports two modes:
 * - Template-based codegen (legacy): Emits fixed hardcoded patterns
 * - IR-driven codegen: Uses visitor pattern to walk actual IR structure
 *
 * See: docs/tenstorrent/IR_DRIVEN_CODEGEN_PLAN.md
 * See: docs/tenstorrent/workstream4/Artifact Generation stage_STATUS.md
 */

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

#include <sstream>
#include <string>
#include <unordered_map>

// IR-driven codegen visitors
#include "codegen_tt_compute_visitor.h"
#include "codegen_tt_reader_visitor.h"
#include "codegen_tt_writer_visitor.h"

namespace tvm {
namespace tl {

using namespace tir;
using tvm::Map;
using tvm::String;

/*!
 * \brief Extract buffer dimensions from PrimFunc for matmul
 * Returns Mt, Kt, Nt (tile counts)
 */
struct MatmulDims {
  int Mt;  // M dimension in tiles
  int Kt;  // K dimension in tiles
  int Nt;  // N dimension in tiles
  int M;   // M dimension in elements
  int K;   // K dimension in elements
  int N;   // N dimension in elements
};

MatmulDims ExtractMatmulDims(const PrimFunc& func) {
  MatmulDims dims;

  // Try to get dimensions from grid metadata first
  auto grid_x = func->attrs.GetAttr<Integer>("tt_grid_x");
  auto grid_y = func->attrs.GetAttr<Integer>("tt_grid_y");

  if (grid_x.defined() && grid_y.defined()) {
    dims.Nt = grid_x.value()->value;
    dims.Mt = grid_y.value()->value;
    dims.Kt = dims.Mt;  // Assume square for MVP

    dims.N = dims.Nt * 32;
    dims.M = dims.Mt * 32;
    dims.K = dims.Kt * 32;
  } else {
    // Fallback to default
    dims.Mt = dims.Kt = dims.Nt = 8;
    dims.M = dims.K = dims.N = 256;
  }

  return dims;
}

// =============================================================================
// IR-Driven Codegen
// =============================================================================

/*!
 * \brief Generate TT compute kernel using IR-driven visitor
 */
std::string EmitTTComputeKernelIRDriven(const PrimFunc& func) {
  TTComputeCodegenVisitor visitor(func);
  return visitor.GetFullKernel();
}

/*!
 * \brief Generate TT reader kernel using IR-driven visitor
 */
std::string EmitTTReaderKernelIRDriven(const PrimFunc& func) {
  TTReaderCodegenVisitor visitor(func);
  return visitor.GetFullKernel();
}

/*!
 * \brief Generate TT writer kernel using IR-driven visitor
 */
std::string EmitTTWriterKernelIRDriven(const PrimFunc& func) {
  TTWriterCodegenVisitor visitor(func);
  return visitor.GetFullKernel();
}

/*!
 * \brief Generate TT host program (main.cpp)
 */
std::string EmitTTHostProgram(const PrimFunc& func) {
  std::ostringstream code;

  // Extract matmul dimensions
  MatmulDims dims = ExtractMatmulDims(func);
  int Mt = dims.Mt;
  int Kt = dims.Kt;
  int Nt = dims.Nt;
  int M = dims.M;
  int K = dims.K;
  int N = dims.N;

  auto num_cores = func->attrs.GetAttr<Integer>("tt_num_cores");
  int total_tiles = Mt * Nt;

  // Generate host program header
  code << "// Generated TT Host Program\n";
  code << "// Matmul: M=" << M << ", K=" << K << ", N=" << N << "\n";
  code << "// Grid: " << Nt << "x" << Mt << " (" << total_tiles << " output tiles)\n\n";

  code << "#include <cstdint>\n";
  code << "#include <vector>\n";
  code << "#include <iostream>\n";
  code << "#include <memory>\n\n";

#ifdef TL_USE_REAL_METALIUM
  // Real Metalium APIs
  code << "// Real TT-Metalium APIs\n";
  code << "#include \"tt_metal/host_api.hpp\"\n";
  code << "#include \"tt_metal/impl/device/device.hpp\"\n";
  code << "#include \"tt_metal/impl/device/mesh_device.hpp\"\n";
  code << "#include \"tt_metal/impl/buffers/buffer.hpp\"\n";
  code << "#include \"tt_metal/impl/buffers/mesh_buffer.hpp\"\n";
  code << "#include \"tt_metal/impl/program/program.hpp\"\n";
  code << "#include \"tt_metal/impl/program/mesh_program.hpp\"\n\n";
  code << "using namespace tt::tt_metal;\n\n";
#else
  // Mock TT device APIs for dry-run
  code << "// Mock TT device APIs for dry-run compilation\n";
  code << "class Device {\n";
  code << "public:\n";
  code << "    static Device* Instance() { static Device dev; return &dev; }\n";
  code << "};\n\n";

  code << "class Program {\n";
  code << "public:\n";
  code << "    void AddKernel(const char* name, const char* source) {}\n";
  code << "    void Build() {}\n";
  code << "};\n\n";

  code << "class CommandQueue {\n";
  code << "public:\n";
  code << "    void EnqueueProgram(Program* prog, bool blocking) {}\n";
  code << "    void Finish() {}\n";
  code << "};\n\n";

  code << "class CircularBufferConfig {\n";
  code << "public:\n";
  code << "    CircularBufferConfig(uint32_t cb_id, uint32_t tile_size, uint32_t num_pages) {\n";
  code << "        std::cout << \"  CB\" << cb_id << \": \" << num_pages << \" pages x \" << tile_size << \" bytes\\n\";\n";
  code << "    }\n";
  code << "};\n\n";
#endif

  // Main function
  code << "int main() {\n";
#ifdef TL_USE_REAL_METALIUM
  code << "    std::cout << \"TT Host Program - Real Metalium\" << std::endl;\n\n";

  // 1. Device setup (Real Metalium)
  code << "    // 1. Device setup (Real Metalium)\n";
  code << "    auto device = MeshDevice::create_unit_mesh(/*device_id*/0);\n";
  code << "    CommandQueue& cq = device->mesh_command_queue(/*cq_id*/0);\n";
  code << "    std::cout << \"Device initialized (Mesh)\" << std::endl;\n\n";
#else
  code << "    std::cout << \"TT Host Program - Mock (Dry Run)\" << std::endl;\n\n";

  // 1. Device setup (Mock)
  code << "    // 1. Device setup (Mock)\n";
  code << "    Device* device = Device::Instance();\n";
  code << "    std::cout << \"Device initialized (Mock)\" << std::endl;\n\n";
#endif

  // 2. Common tile configuration
  code << "    // 2. Tile configuration\n";
  code << "    constexpr uint32_t TILE_H = 32;\n";
  code << "    constexpr uint32_t TILE_W = 32;\n";
  code << "    constexpr uint32_t TILE_SIZE_FP16 = TILE_H * TILE_W * sizeof(uint16_t);\n";
  code << "    constexpr uint32_t CB_NUM_PAGES = 2;  // Double buffering\n\n";

#ifdef TL_USE_REAL_METALIUM
  // Real Metalium: Create program first, then CBs
  code << "    // 3. Create program (Real Metalium)\n";
  code << "    Program program = CreateProgram();\n";
  code << "    std::cout << \"Program created\" << std::endl;\n\n";

  code << "    // 4. Create circular buffers (Real Metalium)\n";
  code << "    CoreCoord core{0, 0};  // Single core for MVP\n\n";

  code << "    constexpr uint32_t cb_in0_index = 0;\n";
  code << "    constexpr uint32_t cb_in1_index = 1;\n";
  code << "    constexpr uint32_t cb_out0_index = 2;\n\n";

  code << "    auto cb_in0 = CreateCircularBuffer(\n";
  code << "        program, core,\n";
  code << "        CircularBufferConfig(\n";
  code << "            CB_NUM_PAGES * TILE_SIZE_FP16,\n";
  code << "            {{cb_in0_index, tt::DataFormat::Float16_b}})\n";
  code << "        .set_page_size(cb_in0_index, TILE_SIZE_FP16)\n";
  code << "    );\n\n";

  code << "    auto cb_in1 = CreateCircularBuffer(\n";
  code << "        program, core,\n";
  code << "        CircularBufferConfig(\n";
  code << "            CB_NUM_PAGES * TILE_SIZE_FP16,\n";
  code << "            {{cb_in1_index, tt::DataFormat::Float16_b}})\n";
  code << "        .set_page_size(cb_in1_index, TILE_SIZE_FP16)\n";
  code << "    );\n\n";

  code << "    auto cb_out0 = CreateCircularBuffer(\n";
  code << "        program, core,\n";
  code << "        CircularBufferConfig(\n";
  code << "            CB_NUM_PAGES * TILE_SIZE_FP16,\n";
  code << "            {{cb_out0_index, tt::DataFormat::Float16_b}})\n";
  code << "        .set_page_size(cb_out0_index, TILE_SIZE_FP16)\n";
  code << "    );\n";
  code << "    std::cout << \"Circular buffers created\" << std::endl;\n\n";

  // Real Metalium: Create kernels
  code << "    // 5. Create kernels (Real Metalium)\n";
  code << "    std::cout << \"Creating kernels...\" << std::endl;\n\n";

  code << "    // Reader kernel (DataMovement)\n";
  code << "    auto reader_kernel = CreateKernel(\n";
  code << "        program,\n";
  code << "        \"reader.cpp\",\n";
  code << "        core,\n";
  code << "        DataMovementConfig{\n";
  code << "            .processor = DataMovementProcessor::RISCV_0,\n";
  code << "            .noc = NOC::RISCV_0_default\n";
  code << "        }\n";
  code << "    );\n\n";

  code << "    // Compute kernel\n";
  code << "    auto compute_kernel = CreateKernel(\n";
  code << "        program,\n";
  code << "        \"compute.cpp\",\n";
  code << "        core,\n";
  code << "        ComputeConfig{\n";
  code << "            .math_fidelity = MathFidelity::HiFi4,\n";
  code << "            .fp32_dest_acc_en = false,\n";
  code << "            .math_approx_mode = false\n";
  code << "        }\n";
  code << "    );\n\n";

  code << "    // Writer kernel (DataMovement)\n";
  code << "    auto writer_kernel = CreateKernel(\n";
  code << "        program,\n";
  code << "        \"writer.cpp\",\n";
  code << "        core,\n";
  code << "        DataMovementConfig{\n";
  code << "            .processor = DataMovementProcessor::RISCV_1,\n";
  code << "            .noc = NOC::RISCV_1_default\n";
  code << "        }\n";
  code << "    );\n";
  code << "    std::cout << \"Kernels created successfully\" << std::endl;\n\n";
#else
  // Mock: Circular buffers and program
  code << "    // 3. Circular buffer configuration (Mock)\n";
  code << "    CircularBufferConfig cb_a(0, TILE_SIZE_FP16, CB_NUM_PAGES);\n";
  code << "    CircularBufferConfig cb_b(1, TILE_SIZE_FP16, CB_NUM_PAGES);\n";
  code << "    CircularBufferConfig cb_c(2, TILE_SIZE_FP16, CB_NUM_PAGES);\n";
  code << "    std::cout << \"Circular buffers configured (Mock)\" << std::endl;\n\n";

  code << "    // 4. Create program (Mock)\n";
  code << "    Program program;\n";
  code << "    std::cout << \"Program created (Mock)\" << std::endl;\n\n";

  code << "    // 5. Create kernels (Mock)\n";
  code << "    std::cout << \"Creating kernels (Mock)...\" << std::endl;\n";
  code << "    // Mock kernel creation - simulating reader, compute, writer kernels\n";
  code << "    // In real mode, these would be CreateKernel() calls with actual .cpp files\n";
  code << "    struct MockKernel { std::string name; };\n";
  code << "    MockKernel reader_kernel{\"reader.cpp\"};\n";
  code << "    MockKernel compute_kernel{\"compute.cpp\"};\n";
  code << "    MockKernel writer_kernel{\"writer.cpp\"};\n";
  code << "    program.Build();\n";
  code << "    std::cout << \"Kernels created successfully (Mock)\" << std::endl;\n\n";
#endif

  // 6. Allocate DRAM buffers
  code << "    // 6. Allocate DRAM buffers\n";
  code << "    constexpr uint32_t M = " << M << ";\n";
  code << "    constexpr uint32_t N = " << N << ";\n";
  code << "    constexpr uint32_t K = " << K << ";\n";
  code << "    constexpr uint32_t Mt = " << Mt << ";\n";
  code << "    constexpr uint32_t Kt = " << Kt << ";\n";
  code << "    constexpr uint32_t Nt = " << Nt << ";\n\n";

#ifdef TL_USE_REAL_METALIUM
  // Real Metalium buffer allocation
  code << "    // Real Metalium buffer allocation\n";
  code << "    constexpr uint32_t single_tile_size = TILE_H * TILE_W * sizeof(uint16_t);\n\n";

  code << "    DeviceLocalBufferConfig dram_config{\n";
  code << "        .page_size = single_tile_size,\n";
  code << "        .buffer_type = BufferType::DRAM\n";
  code << "    };\n\n";

  code << "    ReplicatedBufferConfig buffer_config_a{.size = Mt * Kt * single_tile_size};\n";
  code << "    ReplicatedBufferConfig buffer_config_b{.size = Kt * Nt * single_tile_size};\n";
  code << "    ReplicatedBufferConfig buffer_config_c{.size = Mt * Nt * single_tile_size};\n\n";

  code << "    auto buffer_a = MeshBuffer::create(buffer_config_a, dram_config, device.get());\n";
  code << "    auto buffer_b = MeshBuffer::create(buffer_config_b, dram_config, device.get());\n";
  code << "    auto buffer_c = MeshBuffer::create(buffer_config_c, dram_config, device.get());\n\n";

  code << "    // Initialize input buffers (host-side vectors)\n";
  code << "    std::vector<uint16_t> host_a(M * K);\n";
  code << "    std::vector<uint16_t> host_b(K * N);\n";
  code << "    for (size_t i = 0; i < host_a.size(); ++i) host_a[i] = static_cast<uint16_t>(i % 256);\n";
  code << "    for (size_t i = 0; i < host_b.size(); ++i) host_b[i] = static_cast<uint16_t>(i % 256);\n";
  code << "    std::cout << \"Mesh buffers allocated and host data initialized\" << std::endl;\n\n";
#else
  // Mock buffer allocation
  code << "    std::vector<uint16_t> dram_a(M * K);\n";
  code << "    std::vector<uint16_t> dram_b(K * N);\n";
  code << "    std::vector<uint16_t> dram_c(M * N);\n\n";

  code << "    // Initialize input data\n";
  code << "    for (size_t i = 0; i < dram_a.size(); ++i) {\n";
  code << "        dram_a[i] = static_cast<uint16_t>(i % 256);\n";
  code << "    }\n";
  code << "    for (size_t i = 0; i < dram_b.size(); ++i) {\n";
  code << "        dram_b[i] = static_cast<uint16_t>(i % 256);\n";
  code << "    }\n";
  code << "    std::cout << \"DRAM buffers allocated and initialized (Mock)\" << std::endl;\n\n";
#endif

  // 7. Runtime arguments
  code << "    // 7. SetRuntimeArgs for kernels\n";
  code << "    constexpr uint32_t NUM_OUTPUT_TILES = " << total_tiles << ";\n";
  code << "    constexpr uint32_t NUM_CORES = " << (num_cores.defined() ? num_cores.value()->value : 1) << ";\n\n";

  code << "    // For single-core MVP: core 0 processes all tiles\n";
  code << "    uint32_t out_tile_start_id = 0;\n";
  code << "    uint32_t num_out_tiles_per_core = NUM_OUTPUT_TILES;\n\n";

#ifdef TL_USE_REAL_METALIUM
  code << "    // SetRuntimeArgs (Real Metalium)\n";
  code << "    std::cout << \"Setting runtime arguments...\" << std::endl;\n\n";

  code << "    // Reader kernel args: {dram_addr_a, dram_addr_b, Mt, Kt, Nt, start_tile_id, num_tiles}\n";
  code << "    std::vector<uint32_t> reader_args = {\n";
  code << "        reinterpret_cast<uint32_t>(buffer_a->address()),\n";
  code << "        reinterpret_cast<uint32_t>(buffer_b->address()),\n";
  code << "        Mt, Kt, Nt,\n";
  code << "        out_tile_start_id,\n";
  code << "        num_out_tiles_per_core\n";
  code << "    };\n";
  code << "    SetRuntimeArgs(program, reader_kernel, core, reader_args);\n\n";

  code << "    // Compute kernel args: {start_tile_id, num_output_tiles, Kt}\n";
  code << "    std::vector<uint32_t> compute_args = {\n";
  code << "        out_tile_start_id,\n";
  code << "        num_out_tiles_per_core,\n";
  code << "        Kt\n";
  code << "    };\n";
  code << "    SetRuntimeArgs(program, compute_kernel, core, compute_args);\n\n";

  code << "    // Writer kernel args: {dram_addr_c, start_tile_id, num_tiles, Nt}\n";
  code << "    std::vector<uint32_t> writer_args = {\n";
  code << "        reinterpret_cast<uint32_t>(buffer_c->address()),\n";
  code << "        out_tile_start_id,\n";
  code << "        num_out_tiles_per_core,\n";
  code << "        Nt\n";
  code << "    };\n";
  code << "    SetRuntimeArgs(program, writer_kernel, core, writer_args);\n";
  code << "    std::cout << \"Runtime arguments configured successfully\" << std::endl;\n\n";
#else
  code << "    // SetRuntimeArgs (Mock)\n";
  code << "    std::cout << \"Setting runtime arguments (Mock)...\" << std::endl;\n\n";

  code << "    // Mock SetRuntimeArgs function\n";
  code << "    auto SetRuntimeArgs = [](auto& prog, auto& kernel, const std::vector<uint32_t>& args) {\n";
  code << "        // Mock implementation - in real mode, this would configure kernel args\n";
  code << "    };\n\n";

  code << "    // Reader kernel args: {dram_addr_a, dram_addr_b, Mt, Kt, Nt, start_tile_id, num_tiles}\n";
  code << "    std::vector<uint32_t> reader_args = {\n";
  code << "        reinterpret_cast<uint32_t>(dram_a.data()),\n";
  code << "        reinterpret_cast<uint32_t>(dram_b.data()),\n";
  code << "        Mt, Kt, Nt,\n";
  code << "        out_tile_start_id,\n";
  code << "        num_out_tiles_per_core\n";
  code << "    };\n";
  code << "    SetRuntimeArgs(program, reader_kernel, reader_args);\n\n";

  code << "    // Compute kernel args: {start_tile_id, num_output_tiles, Kt}\n";
  code << "    std::vector<uint32_t> compute_args = {\n";
  code << "        out_tile_start_id,\n";
  code << "        num_out_tiles_per_core,\n";
  code << "        Kt\n";
  code << "    };\n";
  code << "    SetRuntimeArgs(program, compute_kernel, compute_args);\n\n";

  code << "    // Writer kernel args: {dram_addr_c, start_tile_id, num_tiles, Nt}\n";
  code << "    std::vector<uint32_t> writer_args = {\n";
  code << "        reinterpret_cast<uint32_t>(dram_c.data()),\n";
  code << "        out_tile_start_id,\n";
  code << "        num_out_tiles_per_core,\n";
  code << "        Nt\n";
  code << "    };\n";
  code << "    SetRuntimeArgs(program, writer_kernel, writer_args);\n\n";

  code << "    std::cout << \"Runtime args configured: \" << NUM_OUTPUT_TILES << \" tiles, Kt=\" << Kt << \" (Mock)\" << std::endl;\n\n";
#endif

  // 8. Launch program
  code << "    // 8. Launch program\n";
#ifdef TL_USE_REAL_METALIUM
  code << "    EnqueueProgram(cq, program, /*blocking*/false);\n";
  code << "    Finish(cq);\n";
  code << "    std::cout << \"Program execution complete (Real Metalium)\" << std::endl;\n\n";
#else
  code << "    CommandQueue cq;\n";
  code << "    cq.EnqueueProgram(&program, true);\n";
  code << "    cq.Finish();\n";
  code << "    std::cout << \"Program execution complete (Mock)\" << std::endl;\n\n";
#endif

  // 9. Verify results (placeholder)
  code << "    // 9. Verify results\n";
#ifdef TL_USE_REAL_METALIUM
  code << "    std::vector<uint16_t> result(M * N);\n";
  code << "    // Read results back from device (Real Metalium)\n";
  code << "    // EnqueueReadBuffer(cq, buffer_c, result.data(), /*blocking*/true);\n";
  code << "    std::cout << \"Results ready for verification (placeholder)\" << std::endl;\n\n";
#else
  code << "    std::cout << \"Results in dram_c (\" << dram_c.size() << \" elements)\" << std::endl;\n";
  code << "    std::cout << \"First 10 elements: \";\n";
  code << "    for (size_t i = 0; i < std::min(size_t(10), dram_c.size()); ++i) {\n";
  code << "        std::cout << dram_c[i] << \" \";\n";
  code << "    }\n";
  code << "    std::cout << std::endl;\n\n";
#endif

  // 10. Cleanup
#ifdef TL_USE_REAL_METALIUM
  code << "    // 10. Cleanup (Real Metalium)\n";
  code << "    device->close();\n";
  code << "    std::cout << \"Device closed\" << std::endl;\n\n";
#endif

  code << "    return 0;\n";
  code << "}\n";

  return code.str();
}

/*!
 * \brief Generate tt.plan.json metadata
 */
std::string EmitTTPlanJSON(const PrimFunc& func) {
  std::ostringstream json;

  // Read metadata
  auto grid_x = func->attrs.GetAttr<Integer>("tt_grid_x");
  auto grid_y = func->attrs.GetAttr<Integer>("tt_grid_y");
  auto grid_z = func->attrs.GetAttr<Integer>("tt_grid_z");
  auto num_tiles = func->attrs.GetAttr<Integer>("tt_num_tiles");
  auto num_cores = func->attrs.GetAttr<Integer>("tt_num_cores");
  auto tiles_per_core = func->attrs.GetAttr<Array<Array<Integer>>>("tt_tiles_per_core");

  json << "{\n";
  json << "  \"version\": \"1.0\",\n";
  json << "  \"target\": \"tenstorrent\",\n";
  json << "  \"kernel\": \"gemm_generated\",\n";

  // Grid section
  json << "  \"grid\": {\n";
  json << "    \"x\": " << (grid_x.defined() ? grid_x.value()->value : 1) << ",\n";
  json << "    \"y\": " << (grid_y.defined() ? grid_y.value()->value : 1) << ",\n";
  json << "    \"z\": " << (grid_z.defined() ? grid_z.value()->value : 1) << ",\n";
  json << "    \"total_tiles\": " << (num_tiles.defined() ? num_tiles.value()->value : 1) << "\n";
  json << "  },\n";

  // Cores section
  json << "  \"cores\": {\n";
  json << "    \"num_cores\": " << (num_cores.defined() ? num_cores.value()->value : 64) << ",\n";
  json << "    \"topology\": \"8x8_grid\",\n";
  json << "    \"assignments\": [\n";

  if (tiles_per_core.defined()) {
    for (size_t i = 0; i < tiles_per_core.value().size(); ++i) {
      auto assignment = tiles_per_core.value()[i];
      int start = static_cast<int>(assignment[0]->value);
      int count = static_cast<int>(assignment[1]->value);

      json << "      {\"core_id\": " << i << ", \"start_tile\": " << start
           << ", \"count\": " << count << "}";
      if (i < tiles_per_core.value().size() - 1) json << ",";
      json << "\n";
    }
  }

  json << "    ]\n";
  json << "  },\n";

  // Schedule section
  json << "  \"schedule\": {\n";
  json << "    \"policy\": \"contiguous\",\n";
  json << "    \"order\": \"row_major\"\n";
  json << "  }\n";

  json << "}\n";

  return json.str();
}

/*!
 * \brief Main codegen entry point - generates all TT artifacts
 */
std::unordered_map<std::string, std::string> CodegenTT(const IRModule& mod, const std::string& target) {
  std::unordered_map<std::string, std::string> artifacts;

  // Get main function
  auto funcs = mod->functions;
  PrimFunc main_func;
  for (const auto& kv : funcs) {
    if (kv.first->name_hint == "main") {
      main_func = Downcast<PrimFunc>(kv.second);
      break;
    }
  }

  if (!main_func.defined()) {
    LOG(FATAL) << "No main function found in module";
  }

  // Generate all 3 kernels using IR-driven codegen
  artifacts["reader.cpp"] = EmitTTReaderKernelIRDriven(main_func);
  artifacts["compute.cpp"] = EmitTTComputeKernelIRDriven(main_func);
  artifacts["writer.cpp"] = EmitTTWriterKernelIRDriven(main_func);

  // Generate host program (same for both modes)
  artifacts["main.cpp"] = EmitTTHostProgram(main_func);

  // Generate plan JSON (same for both modes)
  artifacts["tt.plan.json"] = EmitTTPlanJSON(main_func);

  return artifacts;
}

/*!
 * \brief Python FFI wrapper for CodegenTT
 */
Map<String, String> EmitTTArtifacts(const IRModule& mod, const std::string& target) {
  auto artifacts = CodegenTT(mod, target);

  // Convert std::unordered_map to TVM Map for Python
  Map<String, String> result;
  for (const auto& kv : artifacts) {
    result.Set(kv.first, kv.second);
  }

  return result;
}

/*!
 * \brief Build function for Tenstorrent target
 *
 * This is the entry point called by TVM's build system when target="tenstorrent".
 * It generates TT kernel code and returns a runtime module containing the artifacts.
 *
 * For now, this is a dry-run implementation that generates code but doesn't compile
 * for actual hardware (that requires TT-Metalium SDK).
 */
runtime::Module BuildTileLangTT(IRModule mod, Target target) {
  // For now, just return a null module
  // The actual code generation happens via EmitTTArtifacts which is called
  // from Python in tilelang/engine/tt/lower.py
  //
  // TODO: When TT-Metalium SDK is available, this should:
  // 1. Generate artifacts via CodegenTT()
  // 2. Compile with TT-Metalium SDK
  // 3. Return a proper runtime module for execution
  return runtime::Module(nullptr);
}

// Register the function for Python FFI
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.codegen.EmitTTArtifacts", EmitTTArtifacts)
      .def("target.build.tilelang_tt", BuildTileLangTT);
});

}  // namespace tl
}  // namespace tvm
