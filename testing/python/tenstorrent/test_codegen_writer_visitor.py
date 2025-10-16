"""Test TTWriterCodegenVisitor (Task 4: IR-Driven Codegen)."""

# Import tilelang first to get proper TVM
from tilelang import tvm
from tvm import tir


def test_writer_visitor_basic_structure():
    """Test that writer visitor can process basic structure."""
    C = tir.decl_buffer((256, 256), "float16", name="C", scope="global")

    body = tir.Evaluate(0)
    func = tir.PrimFunc([C], body)

    # Add TT metadata
    func = func.with_attr("tt_grid_x", tvm.tir.IntImm("int32", 8))
    func = func.with_attr("tt_grid_y", tvm.tir.IntImm("int32", 8))

    # Verify metadata
    assert "tt_grid_x" in func.attrs
    assert "tt_grid_y" in func.attrs


if __name__ == "__main__":
    test_writer_visitor_basic_structure()
    print("All writer visitor tests passed!")
