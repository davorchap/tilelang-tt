"""
Pass G: CodegenTT (v5 Specification)
Version: 5.0
Date: 2025-10-15

Purpose: Generate C++ code from v5 TIR with complete protocol insertion.
         Produces reader.cpp, compute.cpp, writer.cpp, and main.cpp.

Input: IRModule with 3 split kernels containing all protocols
Output: Dictionary of C++ source files ready for compilation
"""

from __future__ import annotations
from typing import Dict, Any, List
import logging
import os

try:
    import tvm
    from tvm import tir, IRModule
except ImportError:
    tvm = None
    tir = None
    IRModule = object

logger = logging.getLogger(__name__)


def use_real_metalium() -> bool:
    """Check if we should use real Metalium API headers"""
    return os.environ.get("TT_METAL_HOME") is not None


class IntrinsicMapper:
    """Maps TIR intrinsics to Metalium API calls"""

    # NOC/CB Protocol Mappings
    NOC_CB_MAP = {
        "cb_reserve_back": "cb_reserve_back({cb}, {n})",
        "cb_push_back": "cb_push_back({cb}, {n})",
        "cb_wait_front": "cb_wait_front({cb}, {n})",
        "cb_pop_front": "cb_pop_front({cb}, {n})",
        "get_write_ptr": "get_write_ptr({cb})",
        "get_read_ptr": "get_read_ptr({cb})",
        "noc_async_read_tile": "noc_async_read_tile({tile_id}, {accessor}, {addr})",
        "noc_async_write_tile": "noc_async_write_tile({tile_id}, {accessor}, {addr})",
        "noc_async_read_barrier": "noc_async_read_barrier()",
        "noc_async_write_barrier": "noc_async_write_barrier()",
    }

    # Engine Initialization Mappings
    ENGINE_INIT_MAP = {
        "tt.engine.init_common": "binary_op_init_common({cb_in0}, {cb_in1}, {cb_out})",
        "tt.fpu.matmul_init": "mm_init({cb_in0}, {cb_in1}, {cb_out})",
        "tt.fpu.binary_init": "binary_op_init_common({cb_in0}, {cb_in1}, {cb_out})",
        "tt.sfpu.init": "unary_op_init_common({cb_in}, {cb_out})",
    }

    # DST Management Mappings
    DST_MAP = {
        "tt.dst.acquire": "acquire_dst(tt::DstMode::Half)",
        "tt.dst.commit": "pack_tile({dst}, {cb})",
        "tt.dst.wait": "",  # No-op in current Metalium
        "tt.dst.release": "release_dst(tt::DstMode::Half)",
        "pack_tile": "pack_tile({dst}, {cb})",
    }

    # Compute Operation Mappings
    COMPUTE_MAP = {
        "tt.mm.mma": "matmul_tiles({cb_in0}, {cb_in1}, {start0}, {start1}, {dst}, {accumulate})",
        "tt.fpu.add": "add_tiles({cb_in0}, {cb_in1}, {start0}, {start1}, {dst})",
        "tt.fpu.mul": "mul_tiles({cb_in0}, {cb_in1}, {start0}, {start1}, {dst})",
        "tt.fpu.sub": "sub_tiles({cb_in0}, {cb_in1}, {start0}, {start1}, {dst})",
        "tt.sfpu.exp": "exp_tile({cb_in}, {start}, {dst})",
        "tt.sfpu.relu": "relu_tile({cb_in}, {start}, {dst})",
    }

    @classmethod
    def map_intrinsic(cls, intrinsic_name: str, args: List[Any]) -> str:
        """Map a TIR intrinsic to Metalium API call"""

        # Check each mapping category
        if intrinsic_name in cls.NOC_CB_MAP:
            template = cls.NOC_CB_MAP[intrinsic_name]
        elif intrinsic_name in cls.ENGINE_INIT_MAP:
            template = cls.ENGINE_INIT_MAP[intrinsic_name]
        elif intrinsic_name in cls.DST_MAP:
            template = cls.DST_MAP[intrinsic_name]
        elif intrinsic_name in cls.COMPUTE_MAP:
            template = cls.COMPUTE_MAP[intrinsic_name]
        else:
            logger.warning(f"Unknown intrinsic: {intrinsic_name}")
            return f"// TODO: {intrinsic_name}({', '.join(str(a) for a in args)})"

        # Format with arguments
        # This is simplified - real implementation would need proper arg mapping
        return template


class CodeBuffer:
    """Helper for managing generated code with indentation"""

    def __init__(self):
        self.lines = []
        self.indent_level = 0

    def write(self, text: str):
        """Write text with current indentation"""
        if text:
            self.lines.append("    " * self.indent_level + text)
        else:
            self.lines.append("")

    def writeln(self, text: str = ""):
        """Write line with current indentation"""
        self.write(text)

    def indent(self):
        """Increase indentation"""
        self.indent_level += 1

    def dedent(self):
        """Decrease indentation"""
        self.indent_level = max(0, self.indent_level - 1)

    def get_code(self) -> str:
        """Get the generated code as string"""
        return "\n".join(self.lines)


class KernelGenerator:
    """Base class for kernel code generation"""

    def __init__(self, func: "tir.PrimFunc", role: str):
        self.func = func
        self.role = role
        self.code = CodeBuffer()
        self.metadata = self._extract_metadata()

    def _extract_metadata(self) -> Dict[str, Any]:
        """Extract metadata from function attributes"""
        metadata = {}
        if self.func.attrs:
            for key in [
                    "tt.runtime_args", "tt.cb_indices", "tt.persistent_config",
                    "tt.tensor_accessors", "tt.runtime_args_info"
            ]:
                if key in self.func.attrs:
                    metadata[key] = self.func.attrs[key]
        return metadata

    def generate(self) -> str:
        """Generate kernel code"""
        self._generate_includes()
        self._generate_main_function()
        return self.code.get_code()

    def _generate_includes(self):
        """Generate include directives"""
        raise NotImplementedError("Subclass must implement")

    def _generate_main_function(self):
        """Generate MAIN function"""
        raise NotImplementedError("Subclass must implement")


class ReaderKernelGenerator(KernelGenerator):
    """Generator for reader kernels"""

    def _generate_includes(self):
        """Generate includes for reader kernel"""
        if use_real_metalium():
            # Real Metalium uses dataflow_api.h for reader/writer kernels
            self.code.writeln('#include "dataflow_api.h"')
        else:
            # Mock mode uses compute_kernel_api headers
            self.code.writeln('#include "compute_kernel_api/common.h"')
            self.code.writeln('#include "compute_kernel_api/tile_move_copy.h"')
        self.code.writeln()

    def _generate_main_function(self):
        """Generate reader kernel MAIN function"""
        self.code.writeln("void MAIN() {")
        self.code.indent()

        # Generate runtime arg extraction
        self._generate_runtime_args()

        # Generate constant declarations
        self._generate_constants()

        # Generate persistent loop with NOC protocol
        self._generate_persistent_loop()

        self.code.dedent()
        self.code.writeln("}")

    def _generate_runtime_args(self):
        """Generate runtime argument extraction"""
        runtime_args = self.metadata.get("tt.runtime_args", [])
        self.metadata.get("tt.runtime_args_info", {})

        for i, arg_name in enumerate(runtime_args):
            dtype = "uint64_t" if arg_name.endswith("_addr") else "uint32_t"

            self.code.writeln(f"const {dtype} {arg_name} = get_arg_val<{dtype}>({i});")

        self.code.writeln()

    def _generate_constants(self):
        """Generate CB index constants"""
        cb_indices = self.metadata.get("tt.cb_indices", {})

        for cb_name, cb_index in cb_indices.items():
            if "in" in cb_name:
                self.code.writeln(f"constexpr uint32_t {cb_name} = {cb_index};")

        self.code.writeln()

    def _generate_persistent_loop(self):
        """Generate persistent loop with NOC protocol"""
        persistent_config = self.metadata.get("tt.persistent_config", {})

        # Use metadata pattern or default
        loop_pattern = persistent_config.get(
            "pattern", "for (uint32_t tile_id = start_id; tile_id < start_id + count; tile_id++)")

        self.code.writeln(loop_pattern + " {")
        self.code.indent()

        # Generate NOC read protocol
        self._generate_noc_read_protocol()

        self.code.dedent()
        self.code.writeln("}")

    def _generate_noc_read_protocol(self):
        """Generate NOC read protocol sequence"""
        # This would parse the actual TIR body and generate the protocol
        # For now, use template based on common pattern
        self.code.writeln("// NOC read protocol")
        self.code.writeln("cb_reserve_back(cb_in0, 1);")
        self.code.writeln("uint32_t l1_write_addr = get_write_ptr(cb_in0);")
        self.code.writeln("noc_async_read_tile(tile_id, src_accessor, l1_write_addr);")
        self.code.writeln("noc_async_read_barrier();")
        self.code.writeln("cb_push_back(cb_in0, 1);")


class ComputeKernelGenerator(KernelGenerator):
    """Generator for compute kernels"""

    def _generate_includes(self):
        """Generate includes for compute kernel"""
        if use_real_metalium():
            # Real Metalium requires ckernel_include.h first
            self.code.writeln('#include "ckernel_include.h"')
            self.code.writeln('#include "compute_kernel_api/common.h"')

            # Add includes based on compute operations
            if self._has_matmul():
                self.code.writeln('#include "compute_kernel_api/matmul.h"')
            if self._has_binary_ops():
                self.code.writeln('#include "compute_kernel_api/eltwise_binary.h"')
            if self._has_unary_ops():
                self.code.writeln('#include "compute_kernel_api/eltwise_unary.h"')
        else:
            # Mock mode
            self.code.writeln('#include "compute_kernel_api/common.h"')

            # Add includes based on compute operations
            if self._has_matmul():
                self.code.writeln('#include "compute_kernel_api/matmul.h"')
            if self._has_binary_ops():
                self.code.writeln('#include "compute_kernel_api/eltwise_binary.h"')
            if self._has_unary_ops():
                self.code.writeln('#include "compute_kernel_api/eltwise_unary.h"')

        self.code.writeln()

    def _has_matmul(self) -> bool:
        """Check if kernel has matmul operations"""
        # This would analyze the TIR body
        return True  # Simplified for now

    def _has_binary_ops(self) -> bool:
        """Check if kernel has binary operations"""
        return False

    def _has_unary_ops(self) -> bool:
        """Check if kernel has unary operations"""
        return False

    def _generate_main_function(self):
        """Generate compute kernel MAIN function"""
        self.code.writeln("void MAIN() {")
        self.code.indent()

        # Generate runtime args
        self._generate_runtime_args()

        # Generate constants
        self._generate_constants()

        # Generate engine initialization
        self._generate_engine_init()

        # Generate compute with DST management
        self._generate_compute_with_dst()

        self.code.dedent()
        self.code.writeln("}")

    def _generate_runtime_args(self):
        """Generate runtime argument extraction for compute"""
        runtime_args = self.metadata.get("tt.runtime_args", [])

        for i, arg_name in enumerate(runtime_args):
            if arg_name in ["Kt", "Mt", "Nt", "start_id", "count"]:
                self.code.writeln(f"const uint32_t {arg_name} = get_arg_val<uint32_t>({i});")

        self.code.writeln()

    def _generate_constants(self):
        """Generate CB constants for compute"""
        cb_indices = self.metadata.get("tt.cb_indices", {})

        for cb_name, cb_index in cb_indices.items():
            self.code.writeln(f"constexpr uint32_t {cb_name} = {cb_index};")

        self.code.writeln()

    def _generate_engine_init(self):
        """Generate engine initialization"""
        if self._has_matmul():
            self.code.writeln("// Engine initialization")
            self.code.writeln("mm_init(cb_in0, cb_in1, cb_out);")
            self.code.writeln()

    def _generate_compute_with_dst(self):
        """Generate compute with DST management"""
        # This would analyze TIR and generate appropriate pattern
        # For now, generate GEMM accumulation pattern
        self.code.writeln("// DST management and compute")
        self.code.writeln("acquire_dst(tt::DstMode::Half);")
        self.code.writeln()
        self.code.writeln("for (uint32_t kt = 0; kt < Kt; kt++) {")
        self.code.indent()
        self.code.writeln("cb_wait_front(cb_in0, 1);")
        self.code.writeln("cb_wait_front(cb_in1, 1);")
        self.code.writeln("matmul_tiles(cb_in0, cb_in1, 0, 0, 0, kt > 0);")
        self.code.writeln("cb_pop_front(cb_in0, 1);")
        self.code.writeln("cb_pop_front(cb_in1, 1);")
        self.code.dedent()
        self.code.writeln("}")
        self.code.writeln()
        self.code.writeln("cb_reserve_back(cb_out, 1);")
        self.code.writeln("pack_tile(0, cb_out);")
        self.code.writeln("cb_push_back(cb_out, 1);")
        self.code.writeln("release_dst(tt::DstMode::Half);")


class WriterKernelGenerator(KernelGenerator):
    """Generator for writer kernels"""

    def _generate_includes(self):
        """Generate includes for writer kernel"""
        if use_real_metalium():
            # Real Metalium uses dataflow_api.h for reader/writer kernels
            self.code.writeln('#include "dataflow_api.h"')
        else:
            # Mock mode uses compute_kernel_api headers
            self.code.writeln('#include "compute_kernel_api/common.h"')
            self.code.writeln('#include "compute_kernel_api/tile_move_copy.h"')
        self.code.writeln()

    def _generate_main_function(self):
        """Generate writer kernel MAIN function"""
        self.code.writeln("void MAIN() {")
        self.code.indent()

        # Generate runtime args
        self._generate_runtime_args()

        # Generate constants
        self._generate_constants()

        # Generate persistent loop with NOC write
        self._generate_persistent_loop()

        self.code.dedent()
        self.code.writeln("}")

    def _generate_runtime_args(self):
        """Generate runtime args for writer"""
        runtime_args = self.metadata.get("tt.runtime_args", [])

        for i, arg_name in enumerate(runtime_args):
            dtype = "uint64_t" if arg_name.endswith("_addr") else "uint32_t"

            self.code.writeln(f"const {dtype} {arg_name} = get_arg_val<{dtype}>({i});")

        self.code.writeln()

    def _generate_constants(self):
        """Generate CB constants for writer"""
        cb_indices = self.metadata.get("tt.cb_indices", {})

        for cb_name, cb_index in cb_indices.items():
            if "out" in cb_name:
                self.code.writeln(f"constexpr uint32_t {cb_name} = {cb_index};")

        self.code.writeln()

    def _generate_persistent_loop(self):
        """Generate persistent loop with NOC write"""
        persistent_config = self.metadata.get("tt.persistent_config", {})

        loop_pattern = persistent_config.get(
            "pattern", "for (uint32_t tile_id = start_id; tile_id < start_id + count; tile_id++)")

        self.code.writeln(loop_pattern + " {")
        self.code.indent()

        # Generate NOC write protocol
        self.code.writeln("// NOC write protocol")
        self.code.writeln("cb_wait_front(cb_out, 1);")
        self.code.writeln("uint32_t l1_read_addr = get_read_ptr(cb_out);")
        self.code.writeln("noc_async_write_tile(tile_id, dst_accessor, l1_read_addr);")
        self.code.writeln("noc_async_write_barrier();")
        self.code.writeln("cb_pop_front(cb_out, 1);")

        self.code.dedent()
        self.code.writeln("}")


class HostGenerator:
    """Generator for host launcher code"""

    def __init__(self, mod: IRModule):
        self.mod = mod
        self.code = CodeBuffer()
        self.kernels = self._extract_kernels()

    def _extract_kernels(self) -> Dict[str, "tir.PrimFunc"]:
        """Extract kernels by role"""
        kernels = {}
        for _name, func in self.mod.functions_items():
            if isinstance(func, tir.PrimFunc):
                role = func.attrs.get("tt.kernel_role") if func.attrs else None
                if role:
                    kernels[role] = func
        return kernels

    def generate(self) -> str:
        """Generate host launcher code"""
        self._generate_includes()
        self._generate_main()
        return self.code.get_code()

    def _generate_includes(self):
        """Generate include directives"""
        self.code.writeln('#include "tt_metal/host_api.hpp"')
        self.code.writeln('#include "tt_metal/common/constants.hpp"')
        self.code.writeln('#include "tt_metal/detail/tt_metal.hpp"')
        self.code.writeln()
        self.code.writeln("using namespace tt;")
        self.code.writeln("using namespace tt::tt_metal;")
        self.code.writeln()

    def _generate_main(self):
        """Generate main function"""
        self.code.writeln("int main(int argc, char **argv) {")
        self.code.indent()

        # Device setup
        self._generate_device_setup()

        # Program creation
        self._generate_program_creation()

        # Kernel creation
        self._generate_kernel_creation()

        # Runtime args
        self._generate_runtime_args_setup()

        # Execution
        self._generate_execution()

        # Cleanup
        self._generate_cleanup()

        self.code.writeln("return 0;")
        self.code.dedent()
        self.code.writeln("}")

    def _generate_device_setup(self):
        """Generate device setup code"""
        self.code.writeln("// Device setup")
        self.code.writeln("Device *device = CreateDevice(0);")
        self.code.writeln("CommandQueue& cq = device->command_queue();")
        self.code.writeln()

    def _generate_program_creation(self):
        """Generate program creation"""
        self.code.writeln("// Create program")
        self.code.writeln("Program program = CreateProgram();")
        self.code.writeln()

    def _generate_kernel_creation(self):
        """Generate kernel creation"""
        self.code.writeln("// Create kernels")

        if "reader" in self.kernels:
            self.code.writeln("auto reader_kernel = CreateKernel(")
            self.code.indent()
            self.code.writeln('program,')
            self.code.writeln('"reader.cpp",')
            self.code.writeln("CoreRange({0, 0}, {7, 7}),")
            self.code.writeln("DataMovementConfig{.processor = DataMovementProcessor::RISCV_0}")
            self.code.dedent()
            self.code.writeln(");")
            self.code.writeln()

        if "compute" in self.kernels:
            self.code.writeln("auto compute_kernel = CreateKernel(")
            self.code.indent()
            self.code.writeln("program,")
            self.code.writeln('"compute.cpp",')
            self.code.writeln("CoreRange({0, 0}, {7, 7}),")
            self.code.writeln("ComputeConfig{.math_fidelity = MathFidelity::HiFi4}")
            self.code.dedent()
            self.code.writeln(");")
            self.code.writeln()

        if "writer" in self.kernels:
            self.code.writeln("auto writer_kernel = CreateKernel(")
            self.code.indent()
            self.code.writeln("program,")
            self.code.writeln('"writer.cpp",')
            self.code.writeln("CoreRange({0, 0}, {7, 7}),")
            self.code.writeln("DataMovementConfig{.processor = DataMovementProcessor::RISCV_1}")
            self.code.dedent()
            self.code.writeln(");")
            self.code.writeln()

    def _generate_runtime_args_setup(self):
        """Generate runtime arguments setup"""
        self.code.writeln("// Set runtime args")

        # This would extract actual runtime args from metadata
        # Simplified for now
        self.code.writeln("// TODO: Set actual runtime args from metadata")
        self.code.writeln(
            "SetRuntimeArgs(program, reader_kernel, core, {A_addr, start_id, count, Mt, Nt});")
        self.code.writeln("SetRuntimeArgs(program, compute_kernel, core, {Kt});")
        self.code.writeln(
            "SetRuntimeArgs(program, writer_kernel, core, {C_addr, start_id, count, Mt, Nt});")
        self.code.writeln()

    def _generate_execution(self):
        """Generate execution code"""
        self.code.writeln("// Execute")
        self.code.writeln("EnqueueProgram(cq, program, false);")
        self.code.writeln("Finish(cq);")
        self.code.writeln()

    def _generate_cleanup(self):
        """Generate cleanup code"""
        self.code.writeln("// Cleanup")
        self.code.writeln("CloseDevice(device);")
        self.code.writeln()


class CodegenTT:
    """
    Main codegen class for Tenstorrent backend.
    Generates C++ source files from v5 TIR.
    """

    def __init__(self):
        """Initialize the codegen"""
        pass

    def generate(self, mod: IRModule) -> Dict[str, str]:
        """
        Generate all source files from IR module.

        Args:
            mod: IRModule with 3 split kernels

        Returns:
            Dictionary mapping filename to source code
        """
        if tvm is None:
            logger.error("TVM not available")
            return {}

        outputs = {}

        # Import enhanced generators
        from .kernel_generators import create_kernel_generator

        # Check if we have split kernels with roles
        has_split_kernels = False

        # Generate each kernel
        for name, func in mod.functions_items():
            if isinstance(func, tir.PrimFunc):
                role = func.attrs.get("tt.kernel_role") if func.attrs else None

                if role in ["reader", "compute", "writer"]:
                    # Use enhanced generator that visits actual TIR
                    generator = create_kernel_generator(func, role)
                    outputs[f"{role}.cpp"] = generator.generate()
                    logger.info(f"Generated {role}.cpp from {name}")
                    has_split_kernels = True

        # If no split kernels were found, generate stub kernels from the main function
        if not has_split_kernels:
            logger.warning("No split kernels found, generating stub kernels from main function")

            # Find the main function
            main_func = None
            for _name, func in mod.functions_items():
                if isinstance(func, tir.PrimFunc):
                    if func.attrs and func.attrs.get("global_symbol") == "main":
                        main_func = func
                        break
                    # Fallback to any function with metadata
                    if not main_func and func.attrs and "tt.core_grid" in func.attrs:
                        main_func = func

            if main_func:
                # Generate stub kernels with appropriate roles
                # These are simplified templates until kernel splitting is implemented

                # Generate reader kernel
                from tilelang.tenstorrent.attrs import TT_KERNEL_ROLE
                reader_func = main_func.with_attr(TT_KERNEL_ROLE, "reader")
                generator = create_kernel_generator(reader_func, "reader")
                outputs["reader.cpp"] = generator.generate()
                logger.info("Generated stub reader.cpp")

                # Generate compute kernel
                compute_func = main_func.with_attr(TT_KERNEL_ROLE, "compute")
                generator = create_kernel_generator(compute_func, "compute")
                outputs["compute.cpp"] = generator.generate()
                logger.info("Generated stub compute.cpp")

                # Generate writer kernel
                writer_func = main_func.with_attr(TT_KERNEL_ROLE, "writer")
                generator = create_kernel_generator(writer_func, "writer")
                outputs["writer.cpp"] = generator.generate()
                logger.info("Generated stub writer.cpp")

        # Generate host launcher
        host_gen = HostGenerator(mod)
        outputs["main.cpp"] = host_gen.generate()

        # Generate CMakeLists.txt
        outputs["CMakeLists.txt"] = self._generate_cmake()

        # Generate tt.plan.json
        outputs["tt.plan.json"] = self._generate_plan_json(mod)

        logger.info(f"Generated {len(outputs)} source files")

        return outputs

    def _generate_cmake(self) -> str:
        """Generate CMakeLists.txt for building"""
        cmake = """cmake_minimum_required(VERSION 3.16)
project(tt_kernel)

# Find Metalium SDK
set(METALIUM_HOME $ENV{TT_METAL_HOME})
if(NOT METALIUM_HOME)
    message(FATAL_ERROR "TT_METAL_HOME environment variable not set")
endif()

# Include directories
include_directories(
    ${METALIUM_HOME}/include
    ${METALIUM_HOME}/include/compute_kernel_api
)

# Source files
set(SOURCES
    main.cpp
    reader.cpp
    compute.cpp
    writer.cpp
)

# Create executable
add_executable(tt_kernel ${SOURCES})

# Link libraries
target_link_libraries(tt_kernel
    ${METALIUM_HOME}/lib/libtt_metal.so
)

# Set C++ standard
set_property(TARGET tt_kernel PROPERTY CXX_STANDARD 17)
"""
        return cmake

    def _generate_plan_json(self, mod: IRModule) -> str:
        """Generate tt.plan.json with scheduling metadata"""
        import json
        from ..runtime_plan import _as_py

        plan = {
            "version": "5.0",
            "target": "tenstorrent",
            "generator": "tilelang.tenstorrent.codegen",
        }

        # Extract metadata from the main function
        for _name, func in mod.functions_items():
            if isinstance(func, tir.PrimFunc) and func.attrs:
                # Extract core grid dimensions - try both old and new attribute names
                grid_x = None
                grid_y = None

                # First try new v5 attribute names (tt.core_grid)
                if "tt.core_grid" in func.attrs:
                    core_grid = func.attrs["tt.core_grid"]
                    if hasattr(core_grid, "__getitem__") and len(core_grid) >= 2:
                        grid_x = _as_py(core_grid[0])
                        grid_y = _as_py(core_grid[1])
                # Fall back to old test attributes (tt_grid_x/y)
                elif "tt_grid_x" in func.attrs and "tt_grid_y" in func.attrs:
                    grid_x = _as_py(func.attrs["tt_grid_x"])
                    grid_y = _as_py(func.attrs["tt_grid_y"])

                # If we found grid dimensions, add them to plan
                if grid_x is not None and grid_y is not None:
                    total_tiles = grid_x * grid_y
                    plan["grid"] = {
                        "x": grid_x,
                        "y": grid_y,
                        "total_tiles": total_tiles,
                    }

                # Extract core configuration - try both attribute styles
                num_cores = None
                if "tt_num_cores" in func.attrs:
                    num_cores = _as_py(func.attrs["tt_num_cores"])
                elif "tt.core_grid" in func.attrs and grid_x is not None and grid_y is not None:
                    # If not explicitly set, derive from grid
                    num_cores = grid_x * grid_y

                if num_cores is not None:
                    plan["cores"] = {"num_cores": num_cores, "topology": "grid", "assignments": []}

                    # Add tile assignments - only for v4 format (tt_tiles_per_core)
                    # v5 uses tt.work_partition which is handled separately by emit_tt_plan()
                    if "tt_tiles_per_core" in func.attrs:
                        tiles_per_core = func.attrs["tt_tiles_per_core"]
                        for i, assignment in enumerate(tiles_per_core[:num_cores]):
                            if hasattr(assignment, "__getitem__") and len(assignment) >= 2:
                                # Use _as_py() to properly convert TVM expressions to Python primitives
                                start_tile = _as_py(assignment[0])
                                count = _as_py(assignment[1])

                                plan["cores"]["assignments"].append({
                                    "core_id": i,
                                    "start_tile": start_tile,
                                    "count": count,
                                })
                    # Note: "tt.work_partition" is v5 format (dict of core_coords -> WorkItems)
                    # and is incompatible with the v4 format's flat "assignments" list.
                    # The v5 format runtime plan is emitted separately by emit_tt_plan().

                # Add schedule section
                plan["schedule"] = {"policy": "contiguous", "order": "row_major"}

                # Extract other metadata
                if "tt_num_tiles" in func.attrs:
                    plan["num_tiles"] = _as_py(func.attrs["tt_num_tiles"])

                # Only need metadata from one function (they should all have it)
                break

        # Use default=str to handle TVM types that aren't directly JSON serializable
        return json.dumps(plan, indent=2, default=str)

    def __call__(self, mod: IRModule) -> Dict[str, str]:
        """Allow using the class as a function"""
        return self.generate(mod)


# Module-level function for compatibility
def codegen_tt(mod: IRModule) -> Dict[str, str]:
    """Apply CodegenTT to a module."""
    codegen = CodegenTT()
    return codegen(mod)
