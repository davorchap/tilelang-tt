"""
Test to verify that no template code is generated.
All code must come from IR traversal.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from tilelang.tenstorrent.codegen.kernel_generators import (EnhancedReaderKernelGenerator,
                                                            EnhancedWriterKernelGenerator,
                                                            EnhancedComputeKernelGenerator)

import tvm
from tvm.script import tir as T


def test_empty_ir_fails_loudly():
    """Test that empty IR causes explicit failure, not template fallback"""

    @tvm.script.ir_module
    class EmptyModule:

        @T.prim_func
        def empty_func():
            T.evaluate(T.int32(0))  # Empty body

    func = EmptyModule["empty_func"]

    # Test reader kernel with empty IR
    reader_gen = EnhancedReaderKernelGenerator(func, "reader")
    with pytest.raises(ValueError, match="Reader kernel has empty or incomplete IR"):
        reader_gen.generate()

    # Test writer kernel with empty IR
    writer_gen = EnhancedWriterKernelGenerator(func, "writer")
    with pytest.raises(ValueError, match="Writer kernel has empty or incomplete IR"):
        writer_gen.generate()

    # Compute kernel should not fail on empty IR (it's valid to have no compute)
    compute_gen = EnhancedComputeKernelGenerator(func, "compute")
    code = compute_gen.generate()
    assert "// Generated TT Compute Kernel" in code
    # Should NOT contain any template markers
    assert "// Reader loop with CB/NOC operations" not in code
    assert "// Writer loop with CB/NOC operations" not in code


def test_no_template_markers_in_generated_code():
    """Test that generated code contains no template markers"""

    # Create a module with actual operations
    @tvm.script.ir_module
    class TestModule:

        @T.prim_func
        def test_func():
            # Simulate some operations from C1/C2
            T.evaluate(T.call_extern("void", "tt.alloc_cb", "cb_in0", 128, 128, "float16"))
            T.evaluate(T.call_extern("void", "tt.alloc_cb", "cb_in1", 128, 128, "float16"))
            T.evaluate(T.call_extern("void", "tt.mm.mma", "cb_in0", "cb_in1", 0, T.bool(True)))

    func = TestModule["test_func"]

    # Test compute kernel
    compute_gen = EnhancedComputeKernelGenerator(func, "compute")
    code = compute_gen.generate()

    # Check that code is generated
    assert len(code) > 0
    assert "void MAIN()" in code

    # Check no template markers
    template_markers = [
        "// Reader loop with CB/NOC operations", "// Writer loop with CB/NOC operations",
        "uint32_t num_out_tiles = tt_tile_count; // Use runtime arg", "// Read input A",
        "// Read input B", "// Write output C", "// TODO: Generate from IR", "TEMPLATE"
    ]

    for marker in template_markers:
        assert marker not in code, f"Found template marker: {marker}"

    # Check that actual intrinsics are present
    assert "cb_reserve_back" in code or "matmul" in code or "mm" in code


def test_intrinsic_registry_usage():
    """Test that intrinsics are properly mapped through the registry"""
    from tilelang.tenstorrent.codegen.intrinsics import INTRINSIC_REGISTRY

    # Verify registry has expected intrinsics
    assert INTRINSIC_REGISTRY.has("tt.alloc_cb")
    assert INTRINSIC_REGISTRY.has("tt.mm.mma")
    assert INTRINSIC_REGISTRY.has("tt.fpu.add")
    assert INTRINSIC_REGISTRY.has("tt.read_to_cb")
    assert INTRINSIC_REGISTRY.has("tt.write_from_cb")

    # Test code generation through registry
    cpp = INTRINSIC_REGISTRY.generate_cpp("tt.mm.mma", ["cb_in0", "cb_in1", "0", "true"])
    assert "matmul" in cpp.lower()

    cpp = INTRINSIC_REGISTRY.generate_cpp("tt.alloc_cb", ["cb_in0", "128", "float16"])
    assert "cb_reserve_back" in cpp


def test_visitor_handles_call_extern():
    """Test that the visitor properly handles call_extern from C1/C2"""
    from tilelang.tenstorrent.codegen.tir_visitor import TIRToMetaliumVisitor
    from tilelang.tenstorrent.codegen.codegen_tt import CodeBuffer

    @tvm.script.ir_module
    class TestModule:

        @T.prim_func
        def test_func():
            # Simulate call_extern from C1/C2
            T.evaluate(T.call_extern("void", "tt.alloc_cb", "cb_in0", 128, 128, "float16"))
            T.evaluate(T.call_extern("void", "tt.mm.mma", "cb_in0", "cb_in1", 0, T.bool(True)))

    func = TestModule["test_func"]

    # Create visitor and code buffer
    code_buffer = CodeBuffer()
    visitor = TIRToMetaliumVisitor(code_buffer)

    # Visit the function body
    visitor.visit(func.body)

    # Get generated code
    code = code_buffer.get_code()

    # Check that intrinsics were handled
    assert "cb_reserve_back" in code or "matmul" in code
    # Check no template markers
    assert "// TODO:" not in code or "Unknown intrinsic" not in code


if __name__ == "__main__":
    # Run tests
    test_empty_ir_fails_loudly()
    print("✅ Empty IR fails loudly - no template fallback")

    test_no_template_markers_in_generated_code()
    print("✅ No template markers in generated code")

    test_intrinsic_registry_usage()
    print("✅ Intrinsic registry working correctly")

    test_visitor_handles_call_extern()
    print("✅ Visitor handles call_extern properly")

    print("\n✅ All tests passed - no template code generation!")
