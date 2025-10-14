"""Test LowerGemmToTTIntrinsics pass (persistent transform stage Phase 2).

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


def collect_tt_intrinsic_sequence(stmt):
    """Collect TT intrinsic call names in structural order."""
    sequence = []

    def visit(node):
        if isinstance(node, tir.SeqStmt):
            for stmt in node.seq:
                visit(stmt)
        elif isinstance(node, (tir.For, tir.AttrStmt)):
            visit(node.body)
        elif isinstance(node, tir.IfThenElse):
            visit(node.then_case)
            if node.else_case:
                visit(node.else_case)
        elif isinstance(node, tir.BlockRealize):
            visit(node.block.body)
        elif isinstance(node, (tir.Block, tir.LetStmt)):
            visit(node.body)
        elif isinstance(node, tir.Evaluate):
            call = node.value
            if isinstance(call, tir.Call) and isinstance(call.op, tvm.ir.Op):
                name = call.op.name
                if name.startswith("tt."):
                    sequence.append(name)

    visit(stmt)
    return sequence


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


def create_func_with_gemm():
    """Create a mock PrimFunc that issues a single tl.gemm intrinsic."""
    A = tir.decl_buffer((32, 32), "float16", name="A", scope="global")
    B = tir.decl_buffer((32, 32), "float16", name="B", scope="global")
    C = tir.decl_buffer((32, 32), "float16", name="C", scope="global")

    tl_gemm = tir.op.Op.get("tl.tl_gemm")

    call = tir.call_intrin(
        "handle",
        tl_gemm,
        tir.StringImm("tl::gemm_ss<32, 32, 32, 1, 1, false, false, true>"),
        A.access_ptr("r"),
        B.access_ptr("r"),
        C.access_ptr("rw"),
    )

    body = tir.AttrStmt(
        None,
        "pragma_gemm",
        tvm.tir.StringImm("matmul"),
        tir.Evaluate(call),
    )

    func = tir.PrimFunc([A, B, C], body)
    func = func.with_attr("tt_schedule_policy", "contiguous")

    return func


def create_func_with_multiple_gemms():
    """Create a PrimFunc that issues two tl.gemm intrinsics."""
    A = tir.decl_buffer((32, 32), "float16", name="A", scope="global")
    B = tir.decl_buffer((32, 32), "float16", name="B", scope="global")
    C = tir.decl_buffer((32, 32), "float16", name="C", scope="global")

    tl_gemm = tir.op.Op.get("tl.tl_gemm")

    call0 = tir.call_intrin(
        "handle",
        tl_gemm,
        tir.StringImm("tl::gemm_ss<32, 32, 32, 1, 1, false, false, true>"),
        A.access_ptr("r"),
        B.access_ptr("r"),
        C.access_ptr("rw"),
    )
    call1 = tir.call_intrin(
        "handle",
        tl_gemm,
        tir.StringImm("tl::gemm_ss<32, 32, 32, 1, 1, false, false, true>"),
        A.access_ptr("r"),
        B.access_ptr("r"),
        C.access_ptr("rw"),
    )

    body = tir.SeqStmt(
        [
            tir.AttrStmt(
                None, "pragma_gemm", tvm.tir.StringImm("matmul"), tir.Evaluate(call0)
            ),
            tir.AttrStmt(
                None, "pragma_gemm", tvm.tir.StringImm("matmul"), tir.Evaluate(call1)
            ),
        ]
    )

    func = tir.PrimFunc([A, B, C], body)
    func = func.with_attr("tt_schedule_policy", "contiguous")

    return func


def create_manual_matmul_func():
    """Create a PrimFunc with a raw tl.gemm call but without pragma markers."""
    A = tir.decl_buffer((32, 32), "float16", name="A", scope="global")
    B = tir.decl_buffer((32, 32), "float16", name="B", scope="global")
    C = tir.decl_buffer((32, 32), "float16", name="C", scope="global")

    tl_gemm = tir.op.Op.get("tl.tl_gemm")
    call = tir.call_intrin(
        "handle",
        tl_gemm,
        tir.StringImm("tl::gemm_ss<32, 32, 32, 1, 1, false, false, true>"),
        A.access_ptr("r"),
        B.access_ptr("r"),
        C.access_ptr("rw"),
    )

    func = tir.PrimFunc([A, B, C], tir.Evaluate(call))
    func = func.with_attr("tt_schedule_policy", "contiguous")

    return func


def test_lower_gemm_to_tt_intrinsics_basic():
    """Test the TT GEMM lowering annotates matmul operations."""
    from tilelang.tenstorrent.passes import LowerTTTileIntrinsics

    func = create_func_with_gemm()
    mod = tvm.IRModule({"main": func})

    # Apply TT GEMM lowering (now called LowerTTTileIntrinsics)
    mod = LowerTTTileIntrinsics()(mod)
    func = mod["main"]

    # Verify tensorize metadata attached
    assert func.attrs is not None, "Function should have attributes"
    assert "tt_num_matmuls" in func.attrs, "Should have tt_num_matmuls attribute"
    assert "tt_has_tensorize" in func.attrs, "Should have tt_has_tensorize attribute"


def test_lower_gemm_to_tt_intrinsics_matmul_count():
    """Test the TT GEMM lowering counts matmul operations correctly."""
    from tilelang.tenstorrent.passes import LowerTTTileIntrinsics

    func = create_func_with_gemm()
    mod = tvm.IRModule({"main": func})

    mod = LowerTTTileIntrinsics()(mod)
    func = mod["main"]

    num_matmuls = int(func.attrs["tt_num_matmuls"])
    assert num_matmuls == 1, f"Expected 1 matmul, got {num_matmuls}"


def test_lower_gemm_to_tt_intrinsics_multiple_matmuls():
    """Test the TT GEMM lowering handles multiple matmul operations."""
    from tilelang.tenstorrent.passes import LowerTTTileIntrinsics

    func = create_func_with_multiple_gemms()
    mod = tvm.IRModule({"main": func})

    mod = LowerTTTileIntrinsics()(mod)
    func = mod["main"]

    num_matmuls = int(func.attrs["tt_num_matmuls"])
    assert num_matmuls == 2, f"Expected 2 matmuls, got {num_matmuls}"


def test_lower_gemm_to_tt_intrinsics_has_flag():
    """Test the TT GEMM lowering sets has_tensorize flag."""
    from tilelang.tenstorrent.passes import LowerTTTileIntrinsics

    func = create_func_with_gemm()
    mod = tvm.IRModule({"main": func})

    mod = LowerTTTileIntrinsics()(mod)
    func = mod["main"]

    has_tensorize = bool(func.attrs["tt_has_tensorize"])
    assert has_tensorize, "tt_has_tensorize should be True"


def test_lower_gemm_to_tt_intrinsics_skip_non_gemm_functions():
    """Test the TT GEMM lowering skips functions without GEMM operations."""
    from tilelang.tenstorrent.passes import LowerTTTileIntrinsics

    # Create function without gemm
    A = tir.decl_buffer((32, 32), "float16", name="A")
    body = tir.Evaluate(0)  # No gemm pragma
    func = tir.PrimFunc([A], body)
    func = func.with_attr("tt_schedule_policy", "contiguous")

    mod = tvm.IRModule({"main": func})

    mod = LowerTTTileIntrinsics()(mod)
    func = mod["main"]

    # Should not add tensorize metadata
    assert (
        "tt_num_matmuls" not in func.attrs
    ), "Should not add matmul count without gemm ops"


def test_lower_gemm_to_tt_intrinsics_skip_non_tt_functions():
    """Test the TT GEMM lowering skips functions without TT attributes."""
    from tilelang.tenstorrent.passes import LowerTTTileIntrinsics

    # Create function WITHOUT TT attributes
    A = tir.decl_buffer((32, 32), "float16", name="A")
    gemm_body = tir.Evaluate(0)
    body = tir.AttrStmt(None, "pragma_gemm", tvm.tir.StringImm("matmul"), gemm_body)
    func = tir.PrimFunc([A], body)

    mod = tvm.IRModule({"main": func})

    # Apply pass
    mod = LowerTTTileIntrinsics()(mod)
    func = mod["main"]

    # Should NOT add tensorize metadata
    assert (
        func.attrs is None or "tt_num_matmuls" not in func.attrs
    ), "Should not tensorize without TT attributes"


def test_lower_gemm_to_tt_intrinsics_integration_with_ws1_ws2():
    """Test the TT GEMM lowering integrates with the defaults + metadata pipeline."""
    from tilelang.tenstorrent.passes import LowerTTTileIntrinsics
    from tilelang.tenstorrent.passes import (
        InferTTLayout,
        PropagateTTLayout,
        TTTilesToCoreMap,
    )
    from tilelang.tenstorrent.target import apply_tt_defaults

    # Create function with gemm intrinsic
    A = tir.decl_buffer((256, 256), "float16", name="A", scope="global")
    B = tir.decl_buffer((256, 256), "float16", name="B", scope="global")
    C = tir.decl_buffer((256, 256), "float16", name="C", scope="global")

    tl_gemm = tir.op.Op.get("tl.tl_gemm")
    call = tir.call_intrin(
        "handle",
        tl_gemm,
        tir.StringImm("tl::gemm_ss<32, 32, 32, 1, 1, false, false, true>"),
        A.access_ptr("r"),
        B.access_ptr("r"),
        C.access_ptr("rw"),
    )
    gemm_body = tir.Evaluate(call)
    body = tir.AttrStmt(None, "pragma_gemm", tvm.tir.StringImm("matmul"), gemm_body)
    func = tir.PrimFunc([A, B, C], body)

    # Add grid metadata (normally from TileLang frontend)
    func = func.with_attr("tl.grid_x", tvm.tir.IntImm("int32", 8))
    func = func.with_attr("tl.grid_y", tvm.tir.IntImm("int32", 8))

    mod = tvm.IRModule({"main": func})

    # Apply TT defaults stage → metadata inference stage → TT GEMM lowering
    mod = apply_tt_defaults(mod)
    # Apply metadata passes directly
    mod = InferTTLayout()(mod)
    mod = PropagateTTLayout()(mod)
    mod = TTTilesToCoreMap()(mod)
    mod = LowerTTTileIntrinsics()(mod)

    func = mod["main"]

    # Verify all metadata exists
    assert "tt_schedule_policy" in func.attrs, "Should have TT defaults stage defaults"
    assert (
        "tt.work_partition" in func.attrs
    ), "Should have metadata inference stage schedule metadata"
    assert "tt_num_matmuls" in func.attrs, "Should have TT GEMM lowering output"
    assert "tt_has_tensorize" in func.attrs, "Should have tensorize flag"


def test_lower_gemm_to_tt_intrinsics_records_pattern_metadata():
    """Lowering should attach pattern metadata for pragma-based matmuls."""
    from tilelang.tenstorrent.passes import LowerTTTileIntrinsics

    func = create_func_with_gemm()
    mod = tvm.IRModule({"main": func})
    mod = LowerTTTileIntrinsics()(mod)
    func = mod["main"]

    patterns = func.attrs["tt_matmul_patterns"]
    assert len(patterns) == 1
    pattern = patterns[0]
    assert str(pattern["source"]) == "tl.gemm"
    assert bool(pattern["accumulate"]) is True
    assert len(pattern["loop_vars"]) == 0
    assert int(func.attrs["tt_num_matmuls"]) == 1
    assert int(pattern["cb_in0"]) == 0
    assert int(pattern["cb_in1"]) == 1
    assert int(pattern["cb_out"]) == 16

    call_names = collect_call_names(func.body)
    assert "tt.mm_init" in call_names
    assert "tt.matmul_tiles" in call_names
    assert not has_buffer_store(func.body)
    assert not has_tt_matmul_attr(func.body)


def test_lower_gemm_to_tt_intrinsics_emits_sequence():
    """Lowering should rewrite matmul loops into the expected TT intrinsic sequence."""
    from tilelang.tenstorrent.passes import LowerTTTileIntrinsics

    func = create_func_with_gemm()
    mod = tvm.IRModule({"main": func})
    mod = LowerTTTileIntrinsics()(mod)
    func = mod["main"]

    sequence = collect_tt_intrinsic_sequence(func.body)
    expected = [
        "tt.tile_regs_acquire",
        "tt.mm_init",
        "tt.cb_wait_front",
        "tt.cb_wait_front",
        "tt.matmul_tiles",
        "tt.cb_pop_front",
        "tt.cb_pop_front",
        "tt.tile_regs_commit",
        "tt.tile_regs_wait",
        "tt.cb_reserve_back",
        "tt.pack_tile",
        "tt.cb_push_back",
        "tt.tile_regs_release",
    ]

    assert sequence == expected, f"Unexpected TT intrinsic sequence: {sequence}"

    patterns = func.attrs["tt_matmul_patterns"]
    assert len(patterns) == 1
    pattern = patterns[0]
    assert int(pattern["cb_out"]) == 16
    assert bool(func.attrs["tt_has_tensorize"])


def test_lower_gemm_to_tt_intrinsics_ignores_unmarked_calls():
    """Lowering processes all tl.tl_gemm calls, even without pragma wrappers.

    This test verifies that the pass processes explicit tl.tl_gemm intrinsics
    regardless of whether they're wrapped in pragma attributes. The new design
    consumes frontend gemm markers directly."""
    from tilelang.tenstorrent.passes import LowerTTTileIntrinsics

    func = create_manual_matmul_func()
    mod = tvm.IRModule({"main": func})
    mod = LowerTTTileIntrinsics()(mod)
    func = mod["main"]

    # The pass should process all tl.tl_gemm calls
    assert "tt_num_matmuls" in func.attrs
    assert "tt_has_tensorize" in func.attrs
    call_names = collect_call_names(func.body)
    assert "tt.mm_init" in call_names
    assert not has_buffer_store(func.body)
    assert has_tt_matmul_attr(func.body) is False


if __name__ == "__main__":
    # Run tests
    test_lower_gemm_to_tt_intrinsics_basic()
    test_lower_gemm_to_tt_intrinsics_matmul_count()
    test_lower_gemm_to_tt_intrinsics_multiple_matmuls()
    test_lower_gemm_to_tt_intrinsics_has_flag()
    test_lower_gemm_to_tt_intrinsics_skip_non_gemm_functions()
    test_lower_gemm_to_tt_intrinsics_skip_non_tt_functions()
    test_lower_gemm_to_tt_intrinsics_integration_with_ws1_ws2()
    test_lower_gemm_to_tt_intrinsics_records_pattern_metadata()
    test_lower_gemm_to_tt_intrinsics_emits_sequence()
    test_lower_gemm_to_tt_intrinsics_ignores_unmarked_calls()
    print("All TT GEMM lowering tests passed!")
