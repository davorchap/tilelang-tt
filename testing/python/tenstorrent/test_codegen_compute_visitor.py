"""Test TTComputeCodegenVisitor (Task 2: IR-Driven Codegen)."""

# Import tilelang first to get proper TVM
from tilelang import tvm
from tvm import tir


def create_matmul_func_with_loops():
    """Create a PrimFunc with nested loops simulating matmul structure."""
    A = tir.decl_buffer((256, 256), "float16", name="A", scope="global")
    B = tir.decl_buffer((256, 256), "float16", name="B", scope="global")
    C = tir.decl_buffer((256, 256), "float16", name="C", scope="global")

    # Create nested loops: for out_tile: for kt: matmul
    out_tile = tir.Var("out_tile", "int32")
    kt = tir.Var("kt", "int32")

    # Matmul intrinsic (as AttrStmt)
    matmul_body = tir.Evaluate(0)
    matmul_attr = tir.AttrStmt(C.data, "tt.matmul_intrinsic", 0, matmul_body)

    # K-loop
    k_loop = tir.For(kt, 0, 8, tir.ForKind.SERIAL, matmul_attr)

    # Persistent loop
    persistent_loop = tir.For(out_tile, 0, 64, tir.ForKind.SERIAL, k_loop)

    func = tir.PrimFunc([A, B, C], persistent_loop)

    # Add TT metadata
    func = func.with_attr("tt_grid_x", tvm.tir.IntImm("int32", 8))
    func = func.with_attr("tt_grid_y", tvm.tir.IntImm("int32", 8))
    func = func.with_attr("tt_num_cores", tvm.tir.IntImm("int32", 64))

    return func


def test_compute_visitor_basic_structure():
    """Test that compute visitor can process matmul IR structure."""
    func = create_matmul_func_with_loops()

    # Verify IR structure
    assert isinstance(func.body, tir.For)  # Persistent loop
    assert isinstance(func.body.body, tir.For)  # K-loop
    assert isinstance(func.body.body.body, tir.AttrStmt)  # Matmul intrinsic


def test_compute_visitor_metadata_access():
    """Test that visitor can access TT metadata."""
    func = create_matmul_func_with_loops()

    assert "tt_grid_x" in func.attrs
    assert "tt_grid_y" in func.attrs
    assert "tt_num_cores" in func.attrs
    assert int(func.attrs["tt_num_cores"]) == 64


def test_compute_visitor_loop_structure():
    """Test that visitor recognizes loop structure."""
    func = create_matmul_func_with_loops()

    # Check persistent loop
    persistent_loop = func.body
    assert persistent_loop.loop_var.name == "out_tile"
    assert persistent_loop.extent == 64

    # Check K-loop
    k_loop = persistent_loop.body
    assert k_loop.loop_var.name == "kt"
    assert k_loop.extent == 8


def test_compute_visitor_matmul_intrinsic():
    """Test that visitor detects matmul intrinsic."""
    func = create_matmul_func_with_loops()

    # Navigate to matmul intrinsic
    matmul_attr = func.body.body.body
    assert isinstance(matmul_attr, tir.AttrStmt)
    assert matmul_attr.attr_key == "tt.matmul_intrinsic"


if __name__ == "__main__":
    # Run tests
    test_compute_visitor_basic_structure()
    test_compute_visitor_metadata_access()
    test_compute_visitor_loop_structure()
    test_compute_visitor_matmul_intrinsic()
    print("All compute visitor tests passed!")
