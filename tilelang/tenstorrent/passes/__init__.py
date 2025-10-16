"""
Tenstorrent-specific TIR passes for the new lowering pipeline.
"""

# Re-export pass constructors for convenience (old passes)
from .infer_tt_layout import InferTTLayout
from .propagate_tt_layout import PropagateTTLayout
from .tt_tiles_to_core_map import TTTilesToCoreMap
from .lower_tt_tile_intrinsics import LowerTTTileIntrinsics
from .grid_to_persistent_tt import GridToPersistentTT
from .pipeline import build_tt_pipeline, run_pipeline

# V5 passes - Stage A: Metadata
from .infer_tt_layout_v5 import InferTTLayout_v5, infer_tt_layout_v5
from .propagate_tt_layout_v5 import PropagateTTLayout_v5, propagate_tt_layout_v5
from .attach_tensor_accessor_tt import AttachTensorAccessorTT, attach_tensor_accessor_tt

# V5 passes - Stage B: Partitioning
from .layout_aware_work_partition_tt_v5 import LayoutAwareWorkPartitionTT_v5, layout_aware_work_partition_tt_v5
from .grid_to_core_grid_v5 import GridToCoreGrid_v5, grid_to_core_grid_v5

# V5 passes - Stage C: Protocol-less Lowering
from .lower_shared_to_cb_v5 import LowerSharedToCB_v5, lower_shared_to_cb_v5
from .lower_tt_tile_intrinsics_v5 import LowerTTTileIntrinsics_v5, lower_tt_tile_intrinsics_v5
from .build_tile_dfg_tt import build_tile_dfg_tt

# V5 passes - Stage D: Late Split & Protocol Insertion
from .split_device_kernel import split_device_kernel
from .configure_tensor_accessor_tt import configure_tensor_accessor_tt
from .lower_cb_intrinsics import lower_cb_intrinsics
from .insert_compute_init_tt import insert_compute_init_tt
from .insert_dst_management_tt import insert_dst_management_tt

# V5 passes - Stage E: Finalization
from .finalize_persistent_signature_tt import finalize_persistent_signature_tt

# V5 passes - Stage F: Verification
from .verify_tt_ir import verify_tt_ir

__all__ = [
    # Old pipeline passes
    "InferTTLayout",
    "PropagateTTLayout",
    "TTTilesToCoreMap",
    "LowerTTTileIntrinsics",
    "GridToPersistentTT",
    "build_tt_pipeline",
    "run_pipeline",
    # V5 passes - Stage A
    "InferTTLayout_v5",
    "infer_tt_layout_v5",
    "PropagateTTLayout_v5",
    "propagate_tt_layout_v5",
    "AttachTensorAccessorTT",
    "attach_tensor_accessor_tt",
    # V5 passes - Stage B
    "LayoutAwareWorkPartitionTT_v5",
    "layout_aware_work_partition_tt_v5",
    "GridToCoreGrid_v5",
    "grid_to_core_grid_v5",
    # V5 passes - Stage C
    "LowerSharedToCB_v5",
    "lower_shared_to_cb_v5",
    "LowerTTTileIntrinsics_v5",
    "lower_tt_tile_intrinsics_v5",
    "build_tile_dfg_tt",
    # V5 passes - Stage D
    "split_device_kernel",
    "configure_tensor_accessor_tt",
    "lower_cb_intrinsics",
    "insert_compute_init_tt",
    "insert_dst_management_tt",
    # V5 passes - Stage E
    "finalize_persistent_signature_tt",
    # V5 passes - Stage F
    "verify_tt_ir",
]
