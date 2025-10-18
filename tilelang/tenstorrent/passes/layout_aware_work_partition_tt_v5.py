"""
Pass B1: LayoutAwareWorkPartitionTT (v5 Specification)
Version: 5.0
Date: 2025-10-15

Purpose: Choose core grid and work partition based on buffer layouts.
         Generates partition metadata and runtime arguments following v5 spec.

Input: Buffer layouts from A1, CB descriptors from A2
Output: tt.core_grid, tt.partition_mode, tt.grid_tiles, tt.work_partition, tt.runtime_args
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import logging
import math

try:
    import tvm
    from tvm import tir, IRModule
except ImportError:
    tvm = None
    tir = None
    IRModule = object

logger = logging.getLogger(__name__)

from tilelang.tenstorrent.attrs import (
    TT_PARTITION_MODE,
    TT_CORE_GRID,
    TT_CORE_RANGES,
    TT_GRID_TILES,
    TT_WORK_PARTITION,
    TT_RUNTIME_ARGS,
    TT_SHARD_GRID,
    TT_LOCAL_SHAPE_TILES,
    TT_BUFFER_PREFIX,
)


class LayoutAwareWorkPartitionTT_v5:
    """
    Pass to choose work partition strategy based on buffer layouts.

    This pass:
    1. Analyzes buffer layouts to determine partition mode
    2. Calculates core grid and work distribution
    3. Generates runtime arguments following v5 schema
    4. Creates work partition assignments per core
    """

    def __init__(self, default_core_grid: Optional[List[int]] = None) -> None:
        """
        Initialize with optional default core grid.

        Args:
            default_core_grid: Default [x, y] core grid (default [8, 8])
        """
        self.default_core_grid = default_core_grid or [8, 8]

    def __call__(self, mod: IRModule) -> IRModule:
        """Apply the pass to an IRModule."""
        if tvm is None:
            return mod

        new_funcs = {}
        for gvar, func in mod.functions_items():
            if not isinstance(func, tir.PrimFunc):
                new_funcs[gvar] = func
                continue

            # Process this function
            func = self._process_function(func)
            new_funcs[gvar] = func

        return tvm.IRModule(new_funcs)

    def _process_function(self, func: "tir.PrimFunc") -> "tir.PrimFunc":
        """Process a single function to add work partition metadata."""

        # Collect buffer layouts
        buffer_layouts = self._collect_buffer_layouts(func)

        # Determine partition mode based on layouts
        partition_mode, shard_info = self._determine_partition_mode(buffer_layouts)

        # Get or use default core grid
        core_grid = self._get_core_grid(func, partition_mode, shard_info)

        # Calculate global tile dimensions
        grid_tiles = self._calculate_grid_tiles(func, buffer_layouts)

        # Generate work partition
        work_partition, core_ranges = self._generate_work_partition(partition_mode, core_grid,
                                                                    grid_tiles, shard_info)

        # Generate runtime arguments
        runtime_args = self._generate_runtime_args(partition_mode, shard_info)

        # Attach all metadata to function
        func = func.with_attr(TT_PARTITION_MODE, partition_mode)
        func = func.with_attr(TT_CORE_GRID, core_grid)
        func = func.with_attr(TT_CORE_RANGES, tvm.runtime.convert(core_ranges))
        func = func.with_attr(TT_GRID_TILES, grid_tiles)
        func = func.with_attr(TT_WORK_PARTITION, tvm.runtime.convert(work_partition))
        func = func.with_attr(TT_RUNTIME_ARGS, runtime_args)

        # Add shard-specific metadata if local_shard mode
        if partition_mode == "local_shard" and shard_info:
            func = func.with_attr(TT_SHARD_GRID, shard_info["grid"])
            func = func.with_attr(TT_LOCAL_SHAPE_TILES, shard_info["local_tiles"])

        logger.info(f"Set partition mode={partition_mode}, core_grid={core_grid}")

        return func

    def _collect_buffer_layouts(self, func: "tir.PrimFunc") -> Dict[str, Dict[str, Any]]:
        """Collect all buffer layouts from function attributes."""

        layouts = {}

        if func.attrs:
            for key in func.attrs.keys():
                if key.startswith(TT_BUFFER_PREFIX):
                    buffer_name = key.replace(TT_BUFFER_PREFIX, "")
                    layout = self._convert_to_dict(func.attrs[key])
                    layouts[buffer_name] = layout

        return layouts

    def _determine_partition_mode(
            self, buffer_layouts: Dict[str, Dict[str, Any]]) -> Tuple[str, Optional[Dict]]:
        """
        Determine partition mode based on buffer layouts.

        Returns:
            (partition_mode, shard_info) where partition_mode is "global" or "local_shard"
        """

        # Check if any buffer is L1 sharded
        for buffer_name, layout in buffer_layouts.items():
            memory = layout.get("memory", "DRAM")
            layout_type = layout.get("layout", "interleaved")

            if memory == "L1" and layout_type == "sharded":
                # Found L1 sharded buffer - use local_shard mode
                nd_shard = layout.get("nd_shard", {})

                shard_info = {
                    "grid": nd_shard.get("projected_grid", [1, 1]),
                    "local_tiles": nd_shard.get("projected_shard_tiles", [1, 1]),
                    "source_buffer": buffer_name
                }

                logger.debug(f"Using local_shard mode due to L1 sharded buffer {buffer_name}")
                return "local_shard", shard_info

        # Default to global mode
        return "global", None

    def _get_core_grid(self, func: "tir.PrimFunc", partition_mode: str,
                       shard_info: Optional[Dict]) -> List[int]:
        """Get or determine the core grid."""

        # Check if already specified
        if func.attrs and "tt.core_grid" in func.attrs:
            grid = func.attrs["tt.core_grid"]
            if isinstance(grid, (list, tuple)) and len(grid) == 2:
                return [int(grid[0]), int(grid[1])]

        # For local_shard mode, use shard grid
        if partition_mode == "local_shard" and shard_info:
            return shard_info["grid"]

        # Use default
        return self.default_core_grid

    def _calculate_grid_tiles(self, func: "tir.PrimFunc",
                              buffer_layouts: Dict[str, Dict[str, Any]]) -> List[int]:
        """Calculate global grid tile dimensions."""

        # Try to infer from output buffer shape
        for param in func.params:
            buffer = func.buffer_map.get(param, None)
            if buffer and buffer.name in buffer_layouts:
                layout = buffer_layouts[buffer.name]

                # Check if this looks like an output buffer
                if "output" in buffer.name.lower() or buffer.name == "C":
                    shape = buffer.shape
                    tile_shape = layout.get("tile_shape", [32, 32])

                    # Calculate tiles needed
                    if len(shape) >= 2:
                        m_tiles = math.ceil(int(shape[0]) / tile_shape[0])
                        n_tiles = math.ceil(int(shape[1]) / tile_shape[1])
                        return [m_tiles, n_tiles]

        # Default based on core grid
        return [8, 8]  # Default 8x8 tiles

    def _generate_work_partition(self, partition_mode: str, core_grid: List[int],
                                 grid_tiles: List[int],
                                 shard_info: Optional[Dict]) -> Tuple[Dict, List]:
        """
        Generate work partition assignments and core ranges.

        Returns:
            (work_partition, core_ranges)
        """

        grid_x, grid_y = core_grid
        mt, nt = grid_tiles
        total_tiles = mt * nt
        total_cores = grid_x * grid_y

        work_partition = {}
        core_ranges = []

        if partition_mode == "local_shard" and shard_info:
            # Local shard mode - each core processes its local shard
            shard_grid = shard_info["grid"]
            local_tiles = shard_info["local_tiles"]
            sm, sn = local_tiles
            sm * sn

            for cy in range(grid_y):
                for cx in range(grid_x):
                    cy * grid_x + cx
                    core_key = f"core_{cy}_{cx}"

                    if cy < shard_grid[0] and cx < shard_grid[1]:
                        # This core has a shard
                        tiles = []
                        for tm in range(sm):
                            for tn in range(sn):
                                # Local tile within shard
                                tiles.append([tm, tn])

                        work_partition[core_key] = tiles
                        core_ranges.append([cx, cy, cx, cy])
                    else:
                        # This core has no work
                        work_partition[core_key] = []
                        core_ranges.append([cx, cy, cx, cy])

        else:
            # Global mode - distribute tiles across all cores
            tiles_per_core = math.ceil(total_tiles / total_cores)

            tile_id = 0
            for cy in range(grid_y):
                for cx in range(grid_x):
                    core_key = f"core_{cy}_{cx}"
                    tiles = []

                    for _ in range(tiles_per_core):
                        if tile_id < total_tiles:
                            # Convert linear tile ID to 2D coordinates
                            ty = tile_id // nt
                            tx = tile_id % nt
                            tiles.append([ty, tx])
                            tile_id += 1

                    work_partition[core_key] = tiles
                    core_ranges.append([cx, cy, cx, cy])

        return work_partition, core_ranges

    def _generate_runtime_args(self, partition_mode: str, shard_info: Optional[Dict]) -> List[str]:
        """Generate runtime argument names following v5 spec."""

        if partition_mode == "local_shard":
            # Local shard mode needs additional shard parameters
            return [
                "start_id",  # Starting tile ID
                "count",  # Number of tiles
                "Mt",  # Global M tiles
                "Kt",  # K tiles (for GEMM)
                "Nt",  # Global N tiles
                "Sm",  # Shard M tiles
                "Sn",  # Shard N tiles
                "Gy",  # Shard grid Y
                "Gx",  # Shard grid X
                "sy",  # Shard coordinate Y
                "sx"  # Shard coordinate X
            ]
        else:
            # Global mode - simpler runtime args
            return [
                "start_id",  # Starting tile ID
                "count",  # Number of tiles
                "Mt",  # Global M tiles
                "Kt",  # K tiles (for GEMM)
                "Nt"  # Global N tiles
            ]

    def _convert_to_dict(self, attr_value: Any) -> Dict[str, Any]:
        """Convert TVM attribute value to Python dict."""

        if isinstance(attr_value, dict):
            return attr_value

        # Handle TVM Map type
        if hasattr(attr_value, "items"):
            result = {}
            for k, v in attr_value.items():
                if hasattr(v, "items"):
                    result[str(k)] = self._convert_to_dict(v)
                elif isinstance(v, (list, tuple)):
                    result[str(k)] = list(v)
                else:
                    result[str(k)] = self._convert_value(v)
            return result

        return {"value": self._convert_value(attr_value)}

    def _convert_value(self, value: Any) -> Any:
        """Convert TVM value to Python type."""

        if hasattr(value, "value"):
            # IntImm, FloatImm, etc.
            return value.value
        elif isinstance(value, (int, float, str, bool)):
            return value
        elif isinstance(value, (list, tuple)):
            # Already a Python list/tuple, return as-is
            return list(value)
        elif hasattr(value, "__iter__") and not isinstance(value, str):
            # TVM Array or other iterable - convert to list
            try:
                return [self._convert_value(x) for x in value]
            except (TypeError, AttributeError):
                return str(value)
        else:
            return str(value)


# Module-level pass function for compatibility
def layout_aware_work_partition_tt_v5(mod: IRModule) -> IRModule:
    """Apply LayoutAwareWorkPartitionTT v5 pass to a module."""
    pass_instance = LayoutAwareWorkPartitionTT_v5()
    return pass_instance(mod)


# Example usage and testing
if __name__ == "__main__":
    import tvm.script
    from tvm.script import tir as T

    # Create test module
    @tvm.script.ir_module
    class TestModule:

        @T.prim_func
        def gemm(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                 C: T.Buffer((256, 256), "float16")):
            for i, j in T.grid(256, 256):
                C[i, j] = A[i, j] + B[i, j]

    func = TestModule["gemm"]

    # Test 1: Global mode (DRAM interleaved)
    print("=== Test 1: Global Mode ===")

    # Add layouts as if A1 ran
    func = func.with_attr("tt.buffer.A", {
        "memory": "DRAM",
        "layout": "interleaved",
        "tile_shape": [32, 32],
        "dtype": "bf16"
    })
    func = func.with_attr("tt.buffer.B", {
        "memory": "DRAM",
        "layout": "interleaved",
        "tile_shape": [32, 32],
        "dtype": "bf16"
    })
    func = func.with_attr("tt.buffer.C", {
        "memory": "DRAM",
        "layout": "interleaved",
        "tile_shape": [32, 32],
        "dtype": "bf16"
    })

    TestModule["gemm"] = func

    # Apply B1 pass
    pass_b1 = LayoutAwareWorkPartitionTT_v5()
    result = pass_b1(TestModule)

    func = result["gemm"]
    print(f"Partition mode: {func.attrs.get('tt.partition_mode')}")
    print(f"Core grid: {func.attrs.get('tt.core_grid')}")
    print(f"Grid tiles: {func.attrs.get('tt.grid_tiles')}")
    print(f"Runtime args: {func.attrs.get('tt.runtime_args')}")

    # Test 2: Local shard mode (L1 sharded)
    print("\n=== Test 2: Local Shard Mode ===")

    func2 = TestModule["gemm"]
    func2 = func2.with_attr(
        "tt.buffer.C", {
            "memory": "L1",
            "layout": "sharded",
            "tile_shape": [32, 32],
            "dtype": "bf16",
            "nd_shard": {
                "axes": ["M", "N"],
                "grid": [2, 4],
                "projected_grid": [2, 4],
                "projected_shard_tiles": [4, 2]
            }
        })

    TestModule["gemm"] = func2

    result2 = pass_b1(TestModule)
    func2 = result2["gemm"]

    print(f"Partition mode: {func2.attrs.get('tt.partition_mode')}")
    print(f"Core grid: {func2.attrs.get('tt.core_grid')}")
    print(f"Shard grid: {func2.attrs.get('tt.shard_grid')}")
    print(f"Local tiles: {func2.attrs.get('tt.local_shape_tiles')}")
    print(f"Runtime args: {func2.attrs.get('tt.runtime_args')}")
