"""
End-to-End Tests for Complete v5 Pipeline
Version: 5.0
Date: 2025-10-15

Tests the entire pipeline from TileLang to generated C++ code.
"""

import logging
import os
import sys
from pathlib import Path

# Try to import pytest but don't fail if not available
try:
    import pytest  # noqa: F401
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import TVM through tilelang
try:
    import tilelang
    from tilelang import tvm
    from tvm import tir
    from tvm.script import tir as T
    import tvm.script
    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False
    logger.warning("TVM not available - tests will be skipped")

# Import all v5 passes
if TVM_AVAILABLE:
    from tilelang.tenstorrent.passes import (
        # Stage A: Metadata
        infer_tt_layout_v5,
        propagate_tt_layout_v5,
        attach_tensor_accessor_tt,
        # Stage B: Partitioning
        layout_aware_work_partition_tt_v5,
        grid_to_core_grid_v5,
        # Stage C: Protocol-less
        lower_shared_to_cb_v5,
        lower_tt_tile_intrinsics_v5,
        build_tile_dfg_tt,
        # Stage D: Late Split
        split_device_kernel,
        configure_tensor_accessor_tt,
        lower_cb_intrinsics,
        insert_compute_init_tt,
        insert_dst_management_tt,
        # Stage E: Finalization
        finalize_persistent_signature_tt,
        # Stage F: Verification
        verify_tt_ir,
    )
    # Stage G: Codegen
    from tilelang.tenstorrent.codegen import CodegenTT


def apply_full_pipeline(mod: "tvm.IRModule", skip_verification: bool = False) -> dict:
    """Apply the complete v5 pipeline to a module"""

    pipeline_stages = [
        # Stage A: Metadata Attachment
        ("A1", infer_tt_layout_v5, "InferTTLayout"),
        ("A2", propagate_tt_layout_v5, "PropagateTTLayout"),
        ("A3", attach_tensor_accessor_tt, "AttachTensorAccessor"),

        # Stage B: Partitioning
        ("B1", layout_aware_work_partition_tt_v5, "LayoutAwareWorkPartition"),
        ("B2", grid_to_core_grid_v5, "GridToCoreGrid"),

        # Stage C: Protocol-less Lowering
        ("C1", lower_shared_to_cb_v5, "LowerSharedToCB"),
        ("C2", lower_tt_tile_intrinsics_v5, "LowerTTTileIntrinsics"),
        ("C3", build_tile_dfg_tt, "BuildTileDFG"),

        # Stage D: Late Split & Protocol Insertion
        ("D1", split_device_kernel, "SplitDeviceKernel"),
        ("D2", configure_tensor_accessor_tt, "ConfigureTensorAccessor"),
        ("D3", lower_cb_intrinsics, "LowerCBIntrinsics"),
        ("D4", insert_compute_init_tt, "InsertComputeInit"),
        ("D5", insert_dst_management_tt, "InsertDSTManagement"),

        # Stage E: Finalization
        ("E1", finalize_persistent_signature_tt, "FinalizePersistentSignature"),
    ]

    # Apply each pass
    current_mod = mod
    for stage_id, pass_func, pass_name in pipeline_stages:
        logger.info(f"Applying {stage_id}: {pass_name}")
        try:
            current_mod = pass_func(current_mod)
        except Exception as e:
            logger.error(f"Failed at {stage_id}: {pass_name} - {e}")
            raise

    # Stage F: Verification (optional)
    if not skip_verification:
        logger.info("Applying F: VerifyTTIR")
        errors = verify_tt_ir(current_mod)
        if errors:
            logger.error(f"Verification failed with {len(errors)} errors")
            for error in errors:
                logger.error(f"  - {error}")
            raise ValueError(f"IR verification failed with {len(errors)} errors")

    # Stage G: Codegen
    logger.info("Applying G: CodegenTT")
    codegen = CodegenTT()
    generated_code = codegen.generate(current_mod)

    return {"ir_module": current_mod, "generated_code": generated_code}


class TestE2EPipeline:
    """End-to-end pipeline tests"""

    def test_gemm_full_pipeline(self):
        """Test full pipeline on GEMM kernel"""

        # Create GEMM kernel
        @tvm.script.ir_module
        class GemmModule:

            @T.prim_func
            def gemm(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                     C: T.Buffer((256, 256), "float16")):
                T.func_attr({"global_symbol": "gemm"})
                # Grid of thread blocks
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for bx in T.thread_binding(8, thread="blockIdx.x"):
                        with T.block("compute"):
                            # Shared memory buffers
                            A_shared = T.alloc_buffer([32, 32], dtype="float16", scope="shared")
                            B_shared = T.alloc_buffer([32, 32], dtype="float16", scope="shared")

                            # Load tiles to shared
                            for ty in T.thread_binding(8, thread="threadIdx.y"):
                                for tx in T.thread_binding(8, thread="threadIdx.x"):
                                    A_shared[ty * 4: ty * 4 + 4, tx * 4: tx * 4 + 4] = \
                                        A[by * 32: by * 32 + 32, bx * 32: bx * 32 + 32][
                                            ty * 4: ty * 4 + 4, tx * 4: tx * 4 + 4]

                                    B_shared[ty * 4: ty * 4 + 4, tx * 4: tx * 4 + 4] = \
                                        B[by * 32: by * 32 + 32, bx * 32: bx * 32 + 32][
                                            ty * 4: ty * 4 + 4, tx * 4: tx * 4 + 4]

                            # Compute
                            for k in T.serial(8):
                                for ty in T.thread_binding(8, thread="threadIdx.y"):
                                    for tx in T.thread_binding(8, thread="threadIdx.x"):
                                        for i, j in T.grid(4, 4):
                                            # Tile-level matmul intrinsic
                                            T.evaluate(
                                                T.call_extern("void", "tt.tile.matmul", A_shared.data,
                                                              B_shared.data, C.data,
                                                              by * 32 + ty * 4 + i,
                                                              bx * 32 + tx * 4 + j, k * 32))

        # Apply full pipeline
        result = apply_full_pipeline(GemmModule)

        # Verify we got all expected outputs
        assert "ir_module" in result
        assert "generated_code" in result

        generated = result["generated_code"]
        assert "reader.cpp" in generated
        assert "compute.cpp" in generated
        assert "writer.cpp" in generated
        assert "main.cpp" in generated
        assert "CMakeLists.txt" in generated

        # Check reader kernel
        reader_code = generated["reader.cpp"]
        assert "cb_reserve_back" in reader_code
        assert "noc_async_read_tile" in reader_code
        assert "cb_push_back" in reader_code

        # Check compute kernel
        compute_code = generated["compute.cpp"]
        assert "mm_init" in compute_code or "binary_op_init_common" in compute_code
        assert "acquire_dst" in compute_code
        assert "matmul_tiles" in compute_code or "add_tiles" in compute_code
        assert "pack_tile" in compute_code
        assert "release_dst" in compute_code

        # Check writer kernel
        writer_code = generated["writer.cpp"]
        assert "cb_wait_front" in writer_code
        assert "noc_async_write_tile" in writer_code
        assert "cb_pop_front" in writer_code

        # Check host launcher
        main_code = generated["main.cpp"]
        assert "CreateDevice" in main_code
        assert "CreateKernel" in main_code
        assert "SetRuntimeArgs" in main_code
        assert "EnqueueProgram" in main_code

        logger.info("✅ GEMM full pipeline test passed")

    def test_elementwise_full_pipeline(self):
        """Test full pipeline on element-wise kernel"""

        # Create element-wise add kernel
        @tvm.script.ir_module
        class AddModule:

            @T.prim_func
            def add(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                    C: T.Buffer((256, 256), "float16")):
                T.func_attr({"global_symbol": "add"})
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for bx in T.thread_binding(8, thread="blockIdx.x"):
                        with T.block("compute"):
                            # Shared memory
                            A_shared = T.alloc_buffer([32, 32], dtype="float16", scope="shared")
                            B_shared = T.alloc_buffer([32, 32], dtype="float16", scope="shared")

                            # Load to shared
                            for ty in T.thread_binding(8, thread="threadIdx.y"):
                                for tx in T.thread_binding(8, thread="threadIdx.x"):
                                    for i, j in T.grid(4, 4):
                                        A_shared[ty * 4 + i, tx * 4 + j] = \
                                            A[by * 32 + ty * 4 + i, bx * 32 + tx * 4 + j]
                                        B_shared[ty * 4 + i, tx * 4 + j] = \
                                            B[by * 32 + ty * 4 + i, bx * 32 + tx * 4 + j]

                            # Compute
                            for ty in T.thread_binding(8, thread="threadIdx.y"):
                                for tx in T.thread_binding(8, thread="threadIdx.x"):
                                    # Tile-level add intrinsic
                                    T.evaluate(
                                        T.call_extern("void", "tt.tile.add", A_shared.data,
                                                      B_shared.data, C.data, by * 32 + ty * 4,
                                                      bx * 32 + tx * 4))

        # Apply full pipeline
        result = apply_full_pipeline(AddModule)

        # Verify outputs
        generated = result["generated_code"]
        assert len(generated) == 5  # reader, compute, writer, main, cmake

        # Check compute has add operation
        compute_code = generated["compute.cpp"]
        assert "add_tiles" in compute_code or "binary" in compute_code.lower()

        logger.info("✅ Element-wise full pipeline test passed")

    def test_pipeline_with_validation(self):
        """Test pipeline with validation enabled"""

        # Create a simple valid kernel
        @tvm.script.ir_module
        class SimpleModule:

            @T.prim_func
            def copy(A: T.Buffer((128, 128), "float16"), B: T.Buffer((128, 128), "float16")):
                T.func_attr({"global_symbol": "copy"})
                for i, j in T.grid(128, 128):
                    B[i, j] = A[i, j]

        # Apply with validation
        result = apply_full_pipeline(SimpleModule, skip_verification=False)

        # Should succeed without errors
        assert "ir_module" in result
        assert "generated_code" in result

        logger.info("✅ Pipeline with validation test passed")

    def test_pipeline_stages_metadata_preserved(self):
        """Test that metadata is preserved through pipeline stages"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def compute(A: T.Buffer((64, 64), "float16"), B: T.Buffer((64, 64), "float16")):
                for i, j in T.grid(64, 64):
                    B[i, j] = A[i, j] * T.float16(2.0)

        # Apply pipeline and check metadata at final stage
        result = apply_full_pipeline(TestModule)
        ir_module = result["ir_module"]

        # Check that split kernels exist
        kernel_roles = set()
        for _name, func in ir_module.functions_items():
            if isinstance(func, tir.PrimFunc) and func.attrs:
                role = func.attrs.get("tt.kernel_role")
                if role:
                    kernel_roles.add(role)

        # Should have 3 kernels after splitting
        assert "reader" in kernel_roles
        assert "compute" in kernel_roles
        assert "writer" in kernel_roles

        logger.info("✅ Metadata preservation test passed")

    def test_generated_code_structure(self):
        """Test structure of generated C++ code"""

        @tvm.script.ir_module
        class TestModule:

            @T.prim_func
            def kernel(A: T.Buffer((32, 32), "float16")):
                for i, j in T.grid(32, 32):
                    A[i, j] = A[i, j] + T.float16(1.0)

        result = apply_full_pipeline(TestModule)
        code = result["generated_code"]

        # Check all files have proper structure
        for filename in ["reader.cpp", "compute.cpp", "writer.cpp"]:
            if filename in code:
                content = code[filename]
                # Should have includes
                assert "#include" in content
                # Should have MAIN function
                assert "void MAIN" in content
                # Should have proper C++ syntax
                assert "{" in content and "}" in content

        # Check CMakeLists.txt
        cmake = code["CMakeLists.txt"]
        assert "cmake_minimum_required" in cmake
        assert "project(tt_kernel)" in cmake
        assert "METALIUM_HOME" in cmake

        logger.info("✅ Generated code structure test passed")


class TestCodegenComponents:
    """Test individual codegen components"""

    def test_tir_visitor(self):
        """Test TIR visitor functionality"""
        from tilelang.tenstorrent.codegen.tir_visitor import TIRToMetaliumVisitor
        from tilelang.tenstorrent.codegen.codegen_tt import CodeBuffer

        # Create simple TIR
        @tvm.script.ir_module
        class TestMod:

            @T.prim_func
            def func():
                T.evaluate(T.call_extern("void", "cb_reserve_back", 0, 1))
                T.evaluate(T.call_extern("void", "cb_push_back", 0, 1))

        # Visit and generate code
        code = CodeBuffer()
        visitor = TIRToMetaliumVisitor(code)
        visitor.visit(TestMod["func"].body)

        generated = code.get_code()
        assert "cb_reserve_back(0, 1);" in generated
        assert "cb_push_back(0, 1);" in generated

        logger.info("✅ TIR visitor test passed")

    def test_kernel_generator_factory(self):
        """Test kernel generator factory"""
        from tilelang.tenstorrent.codegen.kernel_generators import create_kernel_generator

        # Create dummy functions with roles
        @tvm.script.ir_module
        class TestMod:

            @T.prim_func
            def reader_func():
                T.evaluate(0)

            @T.prim_func
            def compute_func():
                T.evaluate(0)

            @T.prim_func
            def writer_func():
                T.evaluate(0)

        # Add roles
        reader = TestMod["reader_func"].with_attr("tt.kernel_role", "reader")
        compute = TestMod["compute_func"].with_attr("tt.kernel_role", "compute")
        writer = TestMod["writer_func"].with_attr("tt.kernel_role", "writer")

        # Create generators
        reader_gen = create_kernel_generator(reader, "reader")
        assert reader_gen is not None
        assert reader_gen.role == "reader"

        compute_gen = create_kernel_generator(compute, "compute")
        assert compute_gen is not None
        assert compute_gen.role == "compute"

        writer_gen = create_kernel_generator(writer, "writer")
        assert writer_gen is not None
        assert writer_gen.role == "writer"

        logger.info("✅ Kernel generator factory test passed")


def run_all_tests():
    """Run all tests manually"""
    if not TVM_AVAILABLE:
        logger.error("TVM not available - cannot run tests")
        return False

    success = True
    test_classes = [TestE2EPipeline, TestCodegenComponents]

    for test_class in test_classes:
        test_instance = test_class()
        for method_name in dir(test_instance):
            if method_name.startswith("test_"):
                test_method = getattr(test_instance, method_name)
                try:
                    logger.info(f"\nRunning {test_class.__name__}.{method_name}")
                    test_method()
                except Exception as e:
                    logger.error(f"Failed: {e}")
                    success = False

    return success


if __name__ == "__main__":
    # Set Python path for imports
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Run tests
    if TVM_AVAILABLE:
        logger.info("=" * 60)
        logger.info("Running TT v5 End-to-End Pipeline Tests")
        logger.info("=" * 60)

        if run_all_tests():
            logger.info("\n" + "=" * 60)
            logger.info("✅ All end-to-end tests passed!")
            logger.info("=" * 60)
        else:
            logger.error("\n" + "=" * 60)
            logger.error("❌ Some tests failed")
            logger.error("=" * 60)
            sys.exit(1)
    else:
        logger.error("TVM not available - install TVM to run tests")
        sys.exit(1)
