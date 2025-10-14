"""
GridToPersistentTT: late pass that injects persistent outer loops, staging buffers,
and emits host worklists. This is the final lowering step to TT persistent kernels.
"""
from __future__ import annotations
import logging

try:
    import tvm
    from tvm import tir, IRModule
except ImportError:  # pragma: no cover
    tvm = None
    tir = None
    IRModule = object

from ..attrs import TT_CORE_GRID, TT_WORK_PARTITION
from ..runtime_plan import emit_tt_plan

logger = logging.getLogger(__name__)


class GridToPersistentTT:
    """
    Final lowering pass that transforms grid-style IR to persistent kernels.

    This pass:
    1. Builds per-core worklists from metadata
    2. Injects persistent outer loops
    3. Adds staging buffers and double-buffering
    4. Places barriers for synchronization
    5. Emits tt.plan.json for host coordination
    """

    def __init__(self,
                 plan_path: str = "tt.plan.json",
                 enable_double_buffer: bool = True,
                 enable_prefetch: bool = True) -> None:
        """
        Initialize the pass.

        Args:
            plan_path: Output path for the runtime plan JSON
            enable_double_buffer: Whether to enable double-buffering
            enable_prefetch: Whether to enable prefetching
        """
        self.plan_path = plan_path
        self.enable_double_buffer = enable_double_buffer
        self.enable_prefetch = enable_prefetch

    def __call__(self, mod: IRModule) -> IRModule:
        """Apply the pass to an IRModule."""
        if tvm is None:
            return mod

        # Process each function
        new_funcs = {}
        plan_emitted = False

        for gvar, func in mod.functions_items():
            if not isinstance(func, tir.PrimFunc):
                new_funcs[gvar] = func
                continue

            # Check if this function has the required attributes
            attrs = func.attrs or {}
            if TT_CORE_GRID in attrs and TT_WORK_PARTITION in attrs:
                # This is our main compute function
                if not plan_emitted:
                    # Emit the runtime plan (only once)
                    plan = emit_tt_plan(func, out_path=self.plan_path)
                    logger.info(f"Emitted TT runtime plan to {self.plan_path}")
                    plan_emitted = True

                # Transform to persistent kernel
                func = self._transform_to_persistent(func)
                logger.info(f"Transformed function {gvar} to persistent kernel")

            new_funcs[gvar] = func

        return tvm.IRModule(new_funcs)

    def _transform_to_persistent(self, func: "tir.PrimFunc") -> "tir.PrimFunc":
        """
        Transform a grid-style function to a persistent kernel.

        This is currently a placeholder that marks the function as transformed.
        Real implementation would:
        1. Replace grid iteration with persistent loop
        2. Add worklist loading
        3. Insert staging buffers
        4. Add synchronization barriers
        """
        # Mark as transformed
        func = func.with_attr("tt.persistent_kernel", True)
        func = func.with_attr("tt.double_buffer_enabled", self.enable_double_buffer)
        func = func.with_attr("tt.prefetch_enabled", self.enable_prefetch)

        # Add metadata about the transformation
        transform_info = {
            "persistent_loop": True,
            "staging_buffers": self.enable_double_buffer,
            "prefetch": self.enable_prefetch,
            "plan_path": self.plan_path
        }
        func = func.with_attr("tt.transform_info", tvm.runtime.convert(transform_info))

        # TODO: Actual IR transformation would happen here
        # This would involve:
        # 1. Creating a visitor/mutator to walk the IR
        # 2. Replacing launch_core with get_core_id
        # 3. Wrapping the body in a persistent loop
        # 4. Adding L1 buffer allocations
        # 5. Inserting DMA operations and barriers

        return func

    def _create_persistent_loop_structure(self):
        """
        Conceptual structure of the persistent kernel:

        ```
        tid = T.get_core_id()
        worklist = T.tt_load_worklist(tid)

        # Persistent outer loop
        for task_idx in range(worklist.size):
            task = worklist[task_idx]
            io, jo = task.io, task.jo

            # Double-buffering for inputs
            for ko in range(0, K, TK):
                # Prefetch next tiles
                if enable_prefetch and ko + TK < K:
                    a_buf.next = T.dram_to_l1(A, io, ko + TK)
                    b_buf.next = T.dram_to_l1(B, ko + TK, jo)

                # Compute on current tiles
                T.barrier()
                T.tt_mma(a_buf.curr, b_buf.curr, accum)

                # Swap buffers
                a_buf.swap()
                b_buf.swap()

            # Write back result
            T.l1_to_dram(C, io, jo, accum)
            T.barrier()
        ```
        """
        pass
