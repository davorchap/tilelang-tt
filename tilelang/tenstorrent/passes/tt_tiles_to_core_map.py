"""
TTTilesToCoreMap: computes CoreRange(s) and a layout-aware work partition.
This pass assigns output tiles to cores based on the specified partitioning strategy.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import logging
import math

try:
    import tvm
    from tvm import tir, IRModule
except ImportError:  # pragma: no cover
    tvm = None
    tir = None
    IRModule = object

from ..attrs import (
    TT_CORE_GRID,
    TT_CORE_RANGES,
    TT_CORE_RANGE,
    TT_WORK_PARTITION,
    CoreRange,
    WorkItem,
)
from ..ir_sugar import with_core_ranges, with_work_partition

logger = logging.getLogger(__name__)


class TTTilesToCoreMap:
    """
    Pass to compute core ranges and build a layout-aware work partition.

    This pass analyzes the computation and assigns tiles to cores,
    creating a work partition that respects layout constraints.
    """

    def __init__(
        self,
        fallback_grid: Tuple[int, int] = (1, 1),
        partition_strategy: str = "row_major",
    ) -> None:
        """
        Initialize the pass.

        Args:
            fallback_grid: Default grid size if not specified in attributes
            partition_strategy: Strategy for partitioning work ("row_major", "column_major", "block")
        """
        self.fallback_grid = tuple(fallback_grid)
        self.partition_strategy = partition_strategy

    def __call__(self, mod: IRModule) -> IRModule:
        """Apply the pass to an IRModule."""
        if tvm is None:
            return mod

        new_funcs = {}
        for gvar, func in mod.functions_items():
            if not isinstance(func, tir.PrimFunc):
                new_funcs[gvar] = func
                continue

            attrs = func.attrs or {}

            # Get or set core grid
            grid = attrs.get(TT_CORE_GRID, None)
            if grid is None:
                grid = tvm.runtime.convert(list(self.fallback_grid))
                func = func.with_attr(TT_CORE_GRID, grid)
                logger.info(f"Set fallback core grid: {self.fallback_grid}")

            gx, gy = list(grid)

            # Get existing core ranges or compute them
            ranges = attrs.get(TT_CORE_RANGES, None) or attrs.get(TT_CORE_RANGE, None)
            if ranges is None:
                # Create a single range covering the whole grid
                cr = CoreRange(start=(0, 0), extent=(int(gx), int(gy)))
                func = with_core_ranges(func, [cr])
                logger.info(f"Created default core range: {cr}")

            # Get or compute work partition
            if attrs.get(TT_WORK_PARTITION, None) is None:
                # Analyze the function to determine tile dimensions
                tile_info = self._analyze_tiles(func)

                # Create work partition based on strategy
                work = self._create_work_partition(
                    grid=(int(gx), int(gy)),
                    tile_info=tile_info,
                    strategy=self.partition_strategy,
                )

                func = with_work_partition(func, work)
                logger.info(f"Created work partition with {len(work)} core assignments")

            new_funcs[gvar] = func

        return tvm.IRModule(new_funcs)

    def _analyze_tiles(self, func: "tir.PrimFunc") -> Dict[str, Any]:
        """
        Analyze the function to extract tile information.

        Returns a dict with:
        - output_tiles_m: number of M-dimension tiles
        - output_tiles_n: number of N-dimension tiles
        - tile_size_m: size of each M tile
        - tile_size_n: size of each N tile
        """
        # Default tile configuration (will be refined by analyzing the IR)
        tile_info = {
            "output_tiles_m": 4,
            "output_tiles_n": 4,
            "tile_size_m": 32,
            "tile_size_n": 32,
            "k_tiles": 4,
        }

        # Try to extract from function attributes or analyze the body
        # This is a simplified version - real implementation would walk the IR
        if hasattr(func, "body"):
            # Look for buffer dimensions to infer tile counts
            for param in func.params:
                buffer = func.buffer_map.get(param, None)
                if buffer and buffer.name in ["C", "output"]:
                    shape = buffer.shape
                    if len(shape) >= 2:
                        # Assume last two dims are M and N
                        m_size = (
                            int(shape[-2]) if hasattr(shape[-2], "value") else shape[-2]
                        )
                        n_size = (
                            int(shape[-1]) if hasattr(shape[-1], "value") else shape[-1]
                        )

                        tile_info["output_tiles_m"] = math.ceil(
                            m_size / tile_info["tile_size_m"]
                        )
                        tile_info["output_tiles_n"] = math.ceil(
                            n_size / tile_info["tile_size_n"]
                        )
                        logger.debug(
                            f"Inferred tile counts from buffer {buffer.name}: "
                            f"M={tile_info['output_tiles_m']}, N={tile_info['output_tiles_n']}"
                        )
                        break

        return tile_info

    def _create_work_partition(
        self, grid: Tuple[int, int], tile_info: Dict[str, Any], strategy: str
    ) -> Dict[str, List[WorkItem]]:
        """
        Create a work partition assigning tiles to cores.

        Args:
            grid: Core grid dimensions (gx, gy)
            tile_info: Information about tile dimensions
            strategy: Partitioning strategy

        Returns:
            Dictionary mapping core coordinates to work items
        """
        gx, gy = grid
        m_tiles = tile_info["output_tiles_m"]
        n_tiles = tile_info["output_tiles_n"]
        k_tiles = tile_info.get("k_tiles", 1)

        work: Dict[str, List[WorkItem]] = {}

        if strategy == "row_major":
            # Distribute tiles in row-major order across cores
            tile_idx = 0
            total_tiles = m_tiles * n_tiles
            tiles_per_core = math.ceil(total_tiles / (gx * gy))

            for cx in range(gx):
                for cy in range(gy):
                    key = f"({cx},{cy})"
                    work[key] = []

                    # Assign tiles to this core
                    for _ in range(tiles_per_core):
                        if tile_idx < total_tiles:
                            io = tile_idx // n_tiles
                            jo = tile_idx % n_tiles
                            work[key].append(
                                WorkItem(
                                    io=io, jo=jo, len_k=k_tiles, tile_order="row_major"
                                )
                            )
                            tile_idx += 1

                    # If no work assigned, give a dummy item
                    if not work[key]:
                        work[key] = [WorkItem(io=0, jo=0, len_k=0)]

        elif strategy == "column_major":
            # Distribute tiles in column-major order
            tile_idx = 0
            total_tiles = m_tiles * n_tiles
            tiles_per_core = math.ceil(total_tiles / (gx * gy))

            for cy in range(gy):
                for cx in range(gx):
                    key = f"({cx},{cy})"
                    work[key] = []

                    for _ in range(tiles_per_core):
                        if tile_idx < total_tiles:
                            jo = tile_idx // m_tiles
                            io = tile_idx % m_tiles
                            work[key].append(
                                WorkItem(
                                    io=io,
                                    jo=jo,
                                    len_k=k_tiles,
                                    tile_order="column_major",
                                )
                            )
                            tile_idx += 1

                    if not work[key]:
                        work[key] = [WorkItem(io=0, jo=0, len_k=0)]

        elif strategy == "block":
            # Block partitioning - each core gets a rectangular block of tiles
            m_blocks = min(gx, m_tiles)
            n_blocks = min(gy, n_tiles)

            m_tiles_per_block = math.ceil(m_tiles / m_blocks)
            n_tiles_per_block = math.ceil(n_tiles / n_blocks)

            for cx in range(gx):
                for cy in range(gy):
                    key = f"({cx},{cy})"
                    work[key] = []

                    # Compute tile range for this core
                    m_start = cx * m_tiles_per_block
                    m_end = min((cx + 1) * m_tiles_per_block, m_tiles)
                    n_start = cy * n_tiles_per_block
                    n_end = min((cy + 1) * n_tiles_per_block, n_tiles)

                    for io in range(m_start, m_end):
                        for jo in range(n_start, n_end):
                            work[key].append(
                                WorkItem(
                                    io=io, jo=jo, len_k=k_tiles, tile_order="block"
                                )
                            )

                    if not work[key]:
                        work[key] = [WorkItem(io=0, jo=0, len_k=0)]

        else:
            # Fallback: single tile per core
            for cx in range(gx):
                for cy in range(gy):
                    key = f"({cx},{cy})"
                    work[key] = [WorkItem(io=cx, jo=cy, tile_order="default")]

        return work
