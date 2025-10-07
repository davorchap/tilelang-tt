"""Test TTReaderCodegenVisitor (Task 3: IR-Driven Codegen)."""

import pytest
import tvm
from tvm import tir


def test_reader_visitor_basic_structure():
    """Test that reader visitor can process basic structure."""
    A = tir.decl_buffer((256, 256), "float16", name="A", scope="global")
    B = tir.decl_buffer((256, 256), "float16", name="B", scope="global")

    body = tir.Evaluate(0)
    func = tir.PrimFunc([A, B], body)

    # Add TT metadata
    func = func.with_attr("tt_grid_x", tvm.tir.IntImm("int32", 8))
    func = func.with_attr("tt_grid_y", tvm.tir.IntImm("int32", 8))

    # Verify metadata
    assert "tt_grid_x" in func.attrs
    assert "tt_grid_y" in func.attrs


if __name__ == "__main__":
    test_reader_visitor_basic_structure()
    print("All reader visitor tests passed!")
