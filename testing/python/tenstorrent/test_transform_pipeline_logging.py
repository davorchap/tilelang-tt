"""Ensure Tenstorrent pass pipelines print TIR after every stage."""

from __future__ import annotations

from tilelang import tvm
import tilelang.language as T

from tilelang.tenstorrent.passes import apply_tt_metadata_passes, apply_tt_transform_passes
from tilelang.tenstorrent.target import apply_tt_defaults


def _make_test_module() -> tvm.IRModule:
    """Create a simple kernel that exercises the transform pipeline."""

    @T.prim_func
    def kernel(A: T.Buffer((128, 128), "float16")):
        with T.Kernel(4, 4) as (bx, by):
            T.evaluate(A[bx * 32 + by * 32])

    return tvm.IRModule.from_expr(kernel.with_attr("global_symbol", "main"))


def test_pipeline_prints_tir_for_each_pass(capsys) -> None:
    """Run the transform pipeline and verify every pass prints its TIR."""

    mod = _make_test_module()
    mod = apply_tt_defaults(mod)

    mod = apply_tt_metadata_passes(mod)
    metadata_output = capsys.readouterr().out

    for expected in (
        "--- [Tenstorrent] After InferDefaultTTSchedule ---",
        "--- [Tenstorrent] After InferDefaultTTShard ---",
    ):
        assert expected in metadata_output

    assert "@T.prim_func" in metadata_output

    mod = apply_tt_transform_passes(mod)
    transform_output = capsys.readouterr().out

    for expected in (
        "--- [Tenstorrent] After GridToPersistentTT ---",
        "--- [Tenstorrent] After TTTilesToCoreMap ---",
        "--- [Tenstorrent] After MemorySpaceLowerTT ---",
        "--- [Tenstorrent] After TilePadTT ---",
        "--- [Tenstorrent] After LowerGemmToTTIntrinsics ---",
        "--- [Tenstorrent] After VerifyTTIR ---",
    ):
        assert expected in transform_output

    # The persistent lowering pipeline should surface persistent loop metadata in the dump.
    assert "tt_persistent_loop" in transform_output
