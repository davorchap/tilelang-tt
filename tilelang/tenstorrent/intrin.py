"""Tenstorrent-specific intrinsic helpers for TIR construction."""

from __future__ import annotations

from tvm import tir


def _call_void_intrin(op_name: str, *args: tir.PrimExpr) -> tir.Stmt:
    """Return an Evaluate statement wrapping a TT intrinsic call."""
    op = tir.op.Op.get(op_name)
    return tir.Evaluate(tir.call_intrin("void", op, *args))


# --- Matmul pipeline ------------------------------------------------------- #


def mm_init(
    cb_in0: tir.PrimExpr, cb_in1: tir.PrimExpr, cb_out: tir.PrimExpr
) -> tir.Stmt:
    """Emit TT mm_init(cb_in0, cb_in1, cb_out)."""
    return _call_void_intrin("tt.mm_init", cb_in0, cb_in1, cb_out)


def matmul_tiles(
    cb_in0: tir.PrimExpr,
    cb_in1: tir.PrimExpr,
    tile_idx_in0: tir.PrimExpr,
    tile_idx_in1: tir.PrimExpr,
    dst_tile_idx: tir.PrimExpr,
    transpose: tir.PrimExpr,
) -> tir.Stmt:
    """Emit TT matmul_tiles(cb_in0, cb_in1, tile_idx_in0, tile_idx_in1, dst_tile_idx, transpose)."""
    return _call_void_intrin(
        "tt.matmul_tiles",
        cb_in0,
        cb_in1,
        tile_idx_in0,
        tile_idx_in1,
        dst_tile_idx,
        transpose,
    )


def tile_regs_acquire() -> tir.Stmt:
    """Emit TT tile_regs_acquire()."""
    return _call_void_intrin("tt.tile_regs_acquire")


def tile_regs_release() -> tir.Stmt:
    """Emit TT tile_regs_release()."""
    return _call_void_intrin("tt.tile_regs_release")


def tile_regs_commit() -> tir.Stmt:
    """Emit TT tile_regs_commit()."""
    return _call_void_intrin("tt.tile_regs_commit")


def tile_regs_wait() -> tir.Stmt:
    """Emit TT tile_regs_wait()."""
    return _call_void_intrin("tt.tile_regs_wait")


def pack_tile(dst_tile_idx: tir.PrimExpr, cb_out: tir.PrimExpr) -> tir.Stmt:
    """Emit TT pack_tile(dst_tile_idx, cb_out)."""
    return _call_void_intrin("tt.pack_tile", dst_tile_idx, cb_out)


# --- Circular buffer primitives ------------------------------------------- #


def cb_wait_front(cb: tir.PrimExpr, tiles: tir.PrimExpr) -> tir.Stmt:
    """Emit TT cb_wait_front(cb, tiles)."""
    return _call_void_intrin("tt.cb_wait_front", cb, tiles)


def cb_pop_front(cb: tir.PrimExpr, tiles: tir.PrimExpr) -> tir.Stmt:
    """Emit TT cb_pop_front(cb, tiles)."""
    return _call_void_intrin("tt.cb_pop_front", cb, tiles)


def cb_reserve_back(cb: tir.PrimExpr, tiles: tir.PrimExpr) -> tir.Stmt:
    """Emit TT cb_reserve_back(cb, tiles)."""
    return _call_void_intrin("tt.cb_reserve_back", cb, tiles)


def cb_push_back(cb: tir.PrimExpr, tiles: tir.PrimExpr) -> tir.Stmt:
    """Emit TT cb_push_back(cb, tiles)."""
    return _call_void_intrin("tt.cb_push_back", cb, tiles)


# --- Elementwise helpers --------------------------------------------------- #


def binary_op_init_common(
    cb_in0: tir.PrimExpr, cb_in1: tir.PrimExpr, cb_out: tir.PrimExpr
) -> tir.Stmt:
    """Emit TT binary_op_init_common(cb_in0, cb_in1, cb_out)."""
    return _call_void_intrin("tt.binary_op_init_common", cb_in0, cb_in1, cb_out)


def add_tiles_init() -> tir.Stmt:
    """Emit TT add_tiles_init()."""
    return _call_void_intrin("tt.add_tiles_init")


def add_tiles(
    cb_a: tir.PrimExpr,
    cb_b: tir.PrimExpr,
    idx_a: tir.PrimExpr,
    idx_b: tir.PrimExpr,
    idx_dst: tir.PrimExpr,
) -> tir.Stmt:
    """Emit TT add_tiles(cb_a, cb_b, idx_a, idx_b, idx_dst)."""
    return _call_void_intrin("tt.add_tiles", cb_a, cb_b, idx_a, idx_b, idx_dst)


def mul_tiles_init() -> tir.Stmt:
    """Emit TT mul_tiles_init()."""
    return _call_void_intrin("tt.mul_tiles_init")


def mul_tiles(
    cb_a: tir.PrimExpr,
    cb_b: tir.PrimExpr,
    idx_a: tir.PrimExpr,
    idx_b: tir.PrimExpr,
    idx_dst: tir.PrimExpr,
) -> tir.Stmt:
    """Emit TT mul_tiles(cb_a, cb_b, idx_a, idx_b, idx_dst)."""
    return _call_void_intrin("tt.mul_tiles", cb_a, cb_b, idx_a, idx_b, idx_dst)


# --- Layout helpers ------------------------------------------------------- #


def tilize(
    cb_src: tir.PrimExpr, cb_dst: tir.PrimExpr, tile_idx: tir.PrimExpr
) -> tir.Stmt:
    """Emit TT tilize(cb_src, cb_dst, tile_idx)."""
    return _call_void_intrin("tt.tilize", cb_src, cb_dst, tile_idx)


def untilize(
    cb_src: tir.PrimExpr, cb_dst: tir.PrimExpr, tile_idx: tir.PrimExpr
) -> tir.Stmt:
    """Emit TT untilize(cb_src, cb_dst, tile_idx)."""
    return _call_void_intrin("tt.untilize", cb_src, cb_dst, tile_idx)


__all__ = [
    "mm_init",
    "matmul_tiles",
    "tile_regs_acquire",
    "tile_regs_release",
    "tile_regs_commit",
    "tile_regs_wait",
    "pack_tile",
    "cb_wait_front",
    "cb_pop_front",
    "cb_reserve_back",
    "cb_push_back",
    "binary_op_init_common",
    "add_tiles_init",
    "add_tiles",
    "mul_tiles_init",
    "mul_tiles",
    "tilize",
    "untilize",
]
