"""
Tenstorrent-specific TIR passes for the new lowering pipeline.
"""

# Re-export pass constructors for convenience
from .infer_tt_layout import InferTTLayout
from .propagate_tt_layout import PropagateTTLayout
from .tt_tiles_to_core_map import TTTilesToCoreMap
from .lower_tt_tile_intrinsics import LowerTTTileIntrinsics
from .grid_to_persistent_tt import GridToPersistentTT
from .pipeline import build_tt_pipeline, run_pipeline

__all__ = [
    # New pipeline passes
    "InferTTLayout",
    "PropagateTTLayout",
    "TTTilesToCoreMap",
    "LowerTTTileIntrinsics",
    "GridToPersistentTT",
    "build_tt_pipeline",
    "run_pipeline",
]
