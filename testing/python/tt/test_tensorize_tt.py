"""Test TensorizeTT pass (persistent transform stage Phase 2).

This pass lowers high-level matmul operations to TT intrinsics.
"""

import tvm
from tvm import tir
from tvm.tir import stmt_functor


def collect_call_names(stmt):
    """Collect intrinsic call names from a TIR statement."""
    names = set()

    def visitor(node):
        if isinstance(node, tir.Call):
            names.add(node.op.name)

    stmt_functor.post_order_visit(stmt, visitor)
    return names


def has_buffer_store(stmt):
    """Check whether the statement still contains BufferStore nodes."""
    found = {"value": False}

    def visitor(node):
        if isinstance(node, tir.BufferStore):
            found["value"] = True

    stmt_functor.post_order_visit(stmt, visitor)
    return found["value"]


def has_tt_matmul_attr(stmt):
    """Check whether a statement still contains legacy tt.matmul_intrinsic attrs."""
    found = {"value": False}

    def visitor(node):
        if isinstance(node, tir.AttrStmt) and node.attr_key == "tt.matmul_intrinsic":
            found["value"] = True

    stmt_functor.post_order_visit(stmt, visitor)
    return found["value"]


def build_matmul_loop(A, B, C, suffix=""):
    """Construct a tiled matmul loop nest."""
    i = tir.Var(f"i{suffix}", "int32")
    j = tir.Var(f"j{suffix}", "int32")
    k = tir.Var(f"k{suffix}", "int32")

    c_load = tir.BufferLoad(C, [i, j])
    a_load = tir.BufferLoad(A, [i, k])
    b_load = tir.BufferLoad(B, [k, j])
    updated = c_load + a_load * b_load

    store = tir.BufferStore(C, updated, [i, j])
    k_loop = tir.For(k, 0, 32, tir.ForKind.SERIAL, store)
    j_loop = tir.For(j, 0, 32, tir.ForKind.SERIAL, k_loop)
    return tir.For(i, 0, 32, tir.ForKind.SERIAL, j_loop)


def create_func_with_gemm():
    """Create a mock PrimFunc with gemm operations."""
    A = tir.decl_buffer((32, 32), "float16", name="A")
    B = tir.decl_buffer((32, 32), "float16", name="B")
    C = tir.decl_buffer((32, 32), "float16", name="C")

    body = build_matmul_loop(A, B, C)
    body = tir.AttrStmt(
        C.data,
        "pragma_gemm",
        tvm.tir.StringImm("matmul"),
        body,
    )

    func = tir.PrimFunc([A, B, C], body)

    # Add TT defaults
    func = func.with_attr("tt_schedule_policy", "contiguous")

    return func


def create_func_with_multiple_gemms():
    """Create a mock PrimFunc with multiple gemm operations."""
    A = tir.decl_buffer((32, 32), "float16", name="A")
    B = tir.decl_buffer((32, 32), "float16", name="B")
    C = tir.decl_buffer((32, 32), "float16", name="C")

    gemm1 = build_matmul_loop(A, B, C, suffix="_0")
    gemm2 = build_matmul_loop(A, B, C, suffix="_1")
    attr1 = tir.AttrStmt(C.data, "pragma_gemm", tvm.tir.StringImm("matmul"), gemm1)
    attr2 = tir.AttrStmt(C.data, "pragma_gemm", tvm.tir.StringImm("matmul"), gemm2)
    body = tir.SeqStmt([attr1, attr2])

    func = tir.PrimFunc([A, B, C], body)
    func = func.with_attr("tt_schedule_policy", "contiguous")

    return func


def create_manual_matmul_func():
    """Create a PrimFunc that performs a simple matmul via explicit loops."""
    A = tir.decl_buffer((32, 32), "float16", name="A")
    B = tir.decl_buffer((32, 32), "float16", name="B")
    C = tir.decl_buffer((32, 32), "float16", name="C")

    func = tir.PrimFunc([A, B, C], build_matmul_loop(A, B, C))
    func = func.with_attr("tt_schedule_policy", "contiguous")

    return func


def test_tensorize_tt_basic():
    """Test TensorizeTT annotates matmul operations."""
    from tilelang.tt.passes import tensorize_tt

    func = create_func_with_gemm()
    mod = tvm.IRModule({"main": func})

    # Apply TensorizeTT
    mod = tensorize_tt(mod)
    func = mod["main"]

    # Verify tensorize metadata attached
    assert func.attrs is not None, "Function should have attributes"
    assert "tt_num_matmuls" in func.attrs, "Should have tt_num_matmuls attribute"
    assert "tt_has_tensorize" in func.attrs, "Should have tt_has_tensorize attribute"


def test_tensorize_tt_matmul_count():
    """Test TensorizeTT counts matmul operations correctly."""
    from tilelang.tt.passes import tensorize_tt

    func = create_func_with_gemm()
    mod = tvm.IRModule({"main": func})

    mod = tensorize_tt(mod)
    func = mod["main"]

    num_matmuls = int(func.attrs["tt_num_matmuls"])
    assert num_matmuls == 1, f"Expected 1 matmul, got {num_matmuls}"


def test_tensorize_tt_multiple_matmuls():
    """Test TensorizeTT handles multiple matmul operations."""
    from tilelang.tt.passes import tensorize_tt

    func = create_func_with_multiple_gemms()
    mod = tvm.IRModule({"main": func})

    mod = tensorize_tt(mod)
    func = mod["main"]

    num_matmuls = int(func.attrs["tt_num_matmuls"])
    assert num_matmuls == 2, f"Expected 2 matmuls, got {num_matmuls}"


def test_tensorize_tt_has_tensorize_flag():
    """Test TensorizeTT sets has_tensorize flag."""
    from tilelang.tt.passes import tensorize_tt

    func = create_func_with_gemm()
    mod = tvm.IRModule({"main": func})

    mod = tensorize_tt(mod)
    func = mod["main"]

    has_tensorize = bool(func.attrs["tt_has_tensorize"])
    assert has_tensorize, "tt_has_tensorize should be True"


def test_tensorize_tt_skip_non_gemm_functions():
    """Test TensorizeTT skips functions without gemm operations."""
    from tilelang.tt.passes import tensorize_tt

    # Create function without gemm
    A = tir.decl_buffer((32, 32), "float16", name="A")
    body = tir.Evaluate(0)  # No gemm pragma
    func = tir.PrimFunc([A], body)
    func = func.with_attr("tt_schedule_policy", "contiguous")

    mod = tvm.IRModule({"main": func})

    mod = tensorize_tt(mod)
    func = mod["main"]

    # Should not add tensorize metadata
    assert "tt_num_matmuls" not in func.attrs, "Should not add matmul count without gemm ops"


def test_tensorize_tt_skip_non_tt_functions():
    """Test TensorizeTT skips functions without TT attributes."""
    from tilelang.tt.passes import tensorize_tt

    # Create function WITHOUT TT attributes
    A = tir.decl_buffer((32, 32), "float16", name="A")
    gemm_body = tir.Evaluate(0)
    body = tir.AttrStmt(None, "pragma_gemm", tvm.tir.StringImm("matmul"), gemm_body)
    func = tir.PrimFunc([A], body)

    mod = tvm.IRModule({"main": func})

    # Apply pass
    mod = tensorize_tt(mod)
    func = mod["main"]

    # Should NOT add tensorize metadata
    assert func.attrs is None or "tt_num_matmuls" not in func.attrs, "Should not tensorize without TT attributes"


def test_tensorize_tt_integration_with_ws1_ws2():
    """Test TensorizeTT integrates with full TT defaults stage→metadata inference stage→TensorizeTT pipeline."""
    from tilelang.tt.passes import apply_tt_metadata_passes, tensorize_tt
    from tilelang.tt.target import apply_tt_defaults

    # Create function with gemm - use actual matmul loop instead of Evaluate(0)
    A = tir.decl_buffer((256, 256), "float16", name="A", scope="global")
    B = tir.decl_buffer((256, 256), "float16", name="B", scope="global")
    C = tir.decl_buffer((256, 256), "float16", name="C", scope="global")

    gemm_body = build_matmul_loop(A, B, C, suffix="_ws")
    body = tir.AttrStmt(None, "pragma_gemm", tvm.tir.StringImm("matmul"), gemm_body)
    func = tir.PrimFunc([A, B, C], body)

    # Add grid metadata (normally from TileLang frontend)
    func = func.with_attr("tl.grid_x", tvm.tir.IntImm("int32", 8))
    func = func.with_attr("tl.grid_y", tvm.tir.IntImm("int32", 8))

    mod = tvm.IRModule({"main": func})

    # Apply TT defaults stage → metadata inference stage → TensorizeTT
    mod = apply_tt_defaults(mod)
    mod = apply_tt_metadata_passes(mod)
    mod = tensorize_tt(mod)

    func = mod["main"]

    # Verify all metadata exists
    assert "tt_schedule_policy" in func.attrs, "Should have TT defaults stage defaults"
    assert "tt_tiles_per_core" in func.attrs, "Should have metadata inference stage schedule metadata"
    assert "tt_num_matmuls" in func.attrs, "Should have TensorizeTT output"
    assert "tt_has_tensorize" in func.attrs, "Should have tensorize flag"


def test_tensorize_tt_records_pattern_metadata():
    """TensorizeTT should attach pattern metadata for pragma-based matmuls."""
    from tilelang.tt.passes import tensorize_tt

    func = create_func_with_gemm()
    mod = tvm.IRModule({"main": func})
    mod = tensorize_tt(mod)
    func = mod["main"]

    patterns = func.attrs["tt_matmul_patterns"]
    assert len(patterns) == 1
    pattern = patterns[0]
    assert str(pattern["source"]) == "pragma"
    assert bool(pattern["accumulate"]) is True
    assert [str(v) for v in pattern["loop_vars"]] == ["i", "j", "k"]
    assert int(func.attrs["tt_num_matmuls"]) == 1
    assert int(pattern["cb_in0"]) == 0
    assert int(pattern["cb_in1"]) == 1
    assert int(pattern["cb_out"]) == 16

    call_names = collect_call_names(func.body)
    assert "tt.mm_init" in call_names
    assert "tt.matmul_tiles" in call_names
    assert not has_buffer_store(func.body)
    assert not has_tt_matmul_attr(func.body)


def test_tensorize_tt_detects_manual_matmul_loops():
    """TensorizeTT should recognise handwritten matmul loop nests."""
    from tilelang.tt.passes import tensorize_tt

    func = create_manual_matmul_func()
    cb_configs = tvm.runtime.convert([
        {
            "cb_id": tvm.tir.IntImm("int32", 4),
            "num_pages": tvm.tir.IntImm("int32", 2),
            "tile_size": tvm.tir.IntImm("int32", 2048),
            "name": "A"
        },
        {
            "cb_id": tvm.tir.IntImm("int32", 6),
            "num_pages": tvm.tir.IntImm("int32", 2),
            "tile_size": tvm.tir.IntImm("int32", 2048),
            "name": "B"
        },
        {
            "cb_id": tvm.tir.IntImm("int32", 9),
            "num_pages": tvm.tir.IntImm("int32", 1),
            "tile_size": tvm.tir.IntImm("int32", 2048),
            "name": "C"
        },
    ])
    func = func.with_attr("tt_circular_buffers", cb_configs)
    func = func.with_attr("tt_num_cbs", tvm.tir.IntImm("int32", 3))
    mod = tvm.IRModule({"main": func})
    mod = tensorize_tt(mod)
    func = mod["main"]

    assert int(func.attrs["tt_num_matmuls"]) == 1
    assert bool(func.attrs["tt_has_tensorize"])

    patterns = func.attrs["tt_matmul_patterns"]
    assert len(patterns) == 1
    pattern = patterns[0]
    assert str(pattern["source"]) == "loop"
    assert str(pattern["reduction_var"]) == "k"
    assert [str(v) for v in pattern["loop_vars"]] == ["i", "j", "k"]
    assert [str(x) for x in pattern["A_indices"]] == ["i", "k"]
    assert [str(x) for x in pattern["B_indices"]] == ["k", "j"]
    assert [str(x) for x in pattern["C_indices"]] == ["i", "j"]
    assert bool(pattern["accumulate"]) is True
    assert int(pattern["cb_in0"]) == 4
    assert int(pattern["cb_in1"]) == 6
    assert int(pattern["cb_out"]) == 9

    call_names = collect_call_names(func.body)
    assert "tt.mm_init" in call_names
    assert "tt.matmul_tiles" in call_names
    assert not has_buffer_store(func.body)
    assert not has_tt_matmul_attr(func.body)


if __name__ == "__main__":
    # Run tests
    test_tensorize_tt_basic()
    test_tensorize_tt_matmul_count()
    test_tensorize_tt_multiple_matmuls()
    test_tensorize_tt_has_tensorize_flag()
    test_tensorize_tt_skip_non_gemm_functions()
    test_tensorize_tt_skip_non_tt_functions()
    test_tensorize_tt_integration_with_ws1_ws2()
    test_tensorize_tt_records_pattern_metadata()
    test_tensorize_tt_detects_manual_matmul_loops()
    print("All TensorizeTT tests passed!")
