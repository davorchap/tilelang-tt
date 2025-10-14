"""
Build and execute the Tenstorrent pass pipeline.
This module provides the main entry point for the TT lowering pipeline.
"""
from __future__ import annotations
import logging
from typing import List, Optional

try:
    import tvm
    from tvm import IRModule
except ImportError:  # pragma: no cover
    tvm = None
    IRModule = object

from .infer_tt_layout import InferTTLayout
from .propagate_tt_layout import PropagateTTLayout
from .tt_tiles_to_core_map import TTTilesToCoreMap
from .lower_tt_tile_intrinsics import LowerTTTileIntrinsics
from .grid_to_persistent_tt import GridToPersistentTT

logger = logging.getLogger(__name__)

def build_tt_pipeline(plan_path: str = "tt.plan.json",
                      target_device: str = "grayskull",
                      partition_strategy: str = "row_major",
                      enable_double_buffer: bool = True,
                      enable_prefetch: bool = True,
                      custom_passes: Optional[List] = None) -> List:
    """
    Build the Tenstorrent lowering pipeline.
    
    Args:
        plan_path: Output path for the runtime plan JSON
        target_device: Target TT device ("grayskull", "wormhole", "blackhole")
        partition_strategy: Work partitioning strategy ("row_major", "column_major", "block")
        enable_double_buffer: Whether to enable double-buffering
        enable_prefetch: Whether to enable prefetching
        custom_passes: Optional list of additional passes to insert
    
    Returns:
        List of pass instances in execution order
    """
    pipeline = [
        # 1. Infer and attach layout metadata
        InferTTLayout(),
        
        # 2. Propagate and normalize layout info
        PropagateTTLayout(),
        
        # 3. Compute core mapping and work partition
        TTTilesToCoreMap(partition_strategy=partition_strategy),
        
        # 4. Lower tile-level intrinsics to device ops
        LowerTTTileIntrinsics(target_device=target_device),
        
        # 5. Final lowering to persistent kernels
        GridToPersistentTT(
            plan_path=plan_path,
            enable_double_buffer=enable_double_buffer,
            enable_prefetch=enable_prefetch
        ),
    ]
    
    # Insert any custom passes before the final lowering
    if custom_passes:
        pipeline = pipeline[:-1] + custom_passes + pipeline[-1:]
    
    return pipeline

def run_pipeline(mod: IRModule, 
                 plan_path: str = "tt.plan.json",
                 target_device: str = "grayskull",
                 partition_strategy: str = "row_major",
                 enable_double_buffer: bool = True,
                 enable_prefetch: bool = True,
                 verbose: bool = False) -> IRModule:
    """
    Execute the full Tenstorrent lowering pipeline on an IRModule.
    
    Args:
        mod: Input IRModule containing PrimFuncs
        plan_path: Output path for the runtime plan JSON
        target_device: Target TT device
        partition_strategy: Work partitioning strategy
        enable_double_buffer: Whether to enable double-buffering
        enable_prefetch: Whether to enable prefetching
        verbose: Whether to enable verbose logging
    
    Returns:
        Transformed IRModule with persistent kernels
    """
    if tvm is None:
        logger.error("TVM not available, cannot run pipeline")
        return mod
    
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    logger.info("Starting Tenstorrent lowering pipeline")
    
    # Build the pipeline
    pipeline = build_tt_pipeline(
        plan_path=plan_path,
        target_device=target_device,
        partition_strategy=partition_strategy,
        enable_double_buffer=enable_double_buffer,
        enable_prefetch=enable_prefetch
    )
    
    # Execute each pass
    for i, pass_instance in enumerate(pipeline):
        pass_name = pass_instance.__class__.__name__
        logger.info(f"Running pass {i+1}/{len(pipeline)}: {pass_name}")
        
        try:
            mod = pass_instance(mod)
        except Exception as e:
            logger.error(f"Pass {pass_name} failed: {e}")
            raise
    
    logger.info("Tenstorrent lowering pipeline completed successfully")
    return mod

def validate_module_for_tt(mod: IRModule) -> List[str]:
    """
    Validate that an IRModule is ready for TT lowering.
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not mod.functions:
        errors.append("Module contains no functions")
        return errors
    
    for gvar, func in mod.functions_items():
        if isinstance(func, tir.PrimFunc):
            # Check for required buffer properties
            for param in func.params:
                buffer = func.buffer_map.get(param, None)
                if buffer:
                    # Check buffer properties
                    if buffer.dtype not in ["float16", "float32", "int8", "uint8", "int32"]:
                        errors.append(f"Buffer {buffer.name} has unsupported dtype {buffer.dtype}")
    
    return errors
