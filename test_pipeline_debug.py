"""
Debug script to identify which pass causes segfault
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import TVM through tilelang
import tilelang
from tilelang import tvm
from tvm import tir
from tvm.script import tir as T
import tvm.script

from tilelang.tenstorrent.passes import (
    # Stage A: Metadata
    infer_tt_layout_v5,
    propagate_tt_layout_v5,
    attach_tensor_accessor_tt,
    # Stage B: Partitioning
    layout_aware_work_partition_tt_v5,
    grid_to_core_grid_v5,
    # Stage C: Protocol-less
    lower_shared_to_cb_v5,
    lower_tt_tile_intrinsics_v5,
    build_tile_dfg_tt,
    # Stage D: Late Split
    split_device_kernel,
    configure_tensor_accessor_tt,
    lower_cb_intrinsics,
    insert_compute_init_tt,
    insert_dst_management_tt,
    # Stage E: Finalization
    finalize_persistent_signature_tt,
)

# Create simplified GEMM kernel
@tvm.script.ir_module
class GemmModule:
    @T.prim_func
    def gemm(A: T.Buffer((64, 64), "float16"), B: T.Buffer((64, 64), "float16"),
             C: T.Buffer((64, 64), "float16")):
        T.func_attr({"global_symbol": "gemm"})
        for i, j in T.grid(64, 64):
            C[i, j] = T.float16(0)
            for k in T.serial(64):
                C[i, j] = C[i, j] + A[i, k] * B[k, j]

# Apply passes one by one
current_mod = GemmModule

try:
    logger.info("=" * 60)
    logger.info("Stage A: Metadata")
    logger.info("=" * 60)

    logger.info("A1: InferTTLayout")
    current_mod = infer_tt_layout_v5(current_mod)
    logger.info("✅ A1 passed")

    logger.info("A2: PropagateTTLayout")
    current_mod = propagate_tt_layout_v5(current_mod)
    logger.info("✅ A2 passed")

    logger.info("A3: AttachTensorAccessor")
    current_mod = attach_tensor_accessor_tt(current_mod)
    logger.info("✅ A3 passed")

    logger.info("=" * 60)
    logger.info("Stage B: Partitioning")
    logger.info("=" * 60)

    logger.info("B1: LayoutAwareWorkPartition")
    current_mod = layout_aware_work_partition_tt_v5(current_mod)
    logger.info("✅ B1 passed")

    logger.info("B2: GridToCoreGrid")
    current_mod = grid_to_core_grid_v5(current_mod)
    logger.info("✅ B2 passed")

    logger.info("=" * 60)
    logger.info("Stage C: Protocol-less")
    logger.info("=" * 60)

    logger.info("C1: LowerSharedToCB")
    current_mod = lower_shared_to_cb_v5(current_mod)
    logger.info("✅ C1 passed")

    logger.info("C2: LowerTTTileIntrinsics")
    current_mod = lower_tt_tile_intrinsics_v5(current_mod)
    logger.info("✅ C2 passed")

    logger.info("C3: BuildTileDFG")
    current_mod = build_tile_dfg_tt(current_mod)
    logger.info("✅ C3 passed")

    logger.info("=" * 60)
    logger.info("Stage D: Late Split & Protocol Insertion")
    logger.info("=" * 60)

    logger.info("D1: SplitDeviceKernel")
    current_mod = split_device_kernel(current_mod)
    logger.info("✅ D1 passed")

    logger.info("D2: ConfigureTensorAccessor")
    current_mod = configure_tensor_accessor_tt(current_mod)
    logger.info("✅ D2 passed")

    logger.info("D3: LowerCBIntrinsics")
    current_mod = lower_cb_intrinsics(current_mod)
    logger.info("✅ D3 passed")

    logger.info("D4: InsertComputeInit")
    current_mod = insert_compute_init_tt(current_mod)
    logger.info("✅ D4 passed")

    logger.info("D5: InsertDSTManagement")
    current_mod = insert_dst_management_tt(current_mod)
    logger.info("✅ D5 passed")

    logger.info("=" * 60)
    logger.info("Stage E: Finalization")
    logger.info("=" * 60)

    logger.info("E1: FinalizePersistentSignature")
    current_mod = finalize_persistent_signature_tt(current_mod)
    logger.info("✅ E1 passed")

    logger.info("=" * 60)
    logger.info("✅ ALL STAGES PASSED!")
    logger.info("=" * 60)

except Exception as e:
    logger.error(f"❌ Pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
