"""
LowerTTTileIntrinsics: maps high-level tile intrinsics to TT-specific device calls.
This pass transforms tile-level operations into Tenstorrent hardware intrinsics.
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

logger = logging.getLogger(__name__)


class LowerTTTileIntrinsics:
    """
    Pass to lower high-level tile intrinsics to TT-specific implementations.

    This includes:
    - T.gemm -> TT matrix engine ops
    - tile_load/store -> TT DMA operations
    - Epilogue ops -> TT SFPU operations
    """

    def __init__(self, target_device: str = "grayskull") -> None:
        """
        Initialize the pass.

        Args:
            target_device: Target TT device ("grayskull", "wormhole", "blackhole")
        """
        self.target_device = target_device

        # Device-specific configurations
        self.device_config = {
            "grayskull": {
                "tile_size": 32,
                "l1_banks": 12,
                "compute_with_storage": 9,
                "storage_only": 3,
            },
            "wormhole": {
                "tile_size": 32,
                "l1_banks": 12,
                "compute_with_storage": 8,
                "storage_only": 4,
            },
            "blackhole": {
                "tile_size": 32,
                "l1_banks": 16,
                "compute_with_storage": 12,
                "storage_only": 4,
            },
        }

    def __call__(self, mod: IRModule) -> IRModule:
        """Apply the pass to an IRModule."""
        if tvm is None:
            return mod

        # Get device config
        config = self.device_config.get(
            self.target_device, self.device_config["grayskull"]
        )

        # Process each function
        new_funcs = {}
        for gvar, func in mod.functions_items():
            if not isinstance(func, tir.PrimFunc):
                new_funcs[gvar] = func
                continue

            # Check if function has TT attributes (skip non-TT functions)
            if func.attrs is None or "tt_schedule_policy" not in func.attrs:
                new_funcs[gvar] = func
                continue

            # Analyze and transform the function
            matmul_patterns = []
            num_matmuls = 0
            has_tensorize = False

            # Count matmul operations with manual traversal
            def count_matmuls(stmt):
                nonlocal num_matmuls, has_tensorize
                patterns = []

                def visit(node):
                    nonlocal num_matmuls, has_tensorize

                    if isinstance(node, tir.SeqStmt):
                        for s in node.seq:
                            visit(s)
                    elif isinstance(node, tir.AttrStmt):
                        if (
                            node.attr_key == "pragma_gemm"
                            and node.value.value == "matmul"
                        ):
                            # Found a pragma gemm, count it
                            num_matmuls += 1
                            has_tensorize = True
                            # Record pattern metadata
                            pattern = {
                                "source": "tl.gemm",
                                "accumulate": True,
                                "loop_vars": [],
                                "cb_in0": 0,
                                "cb_in1": 1,
                                "cb_out": 16,
                            }
                            patterns.append(pattern)
                            # Don't visit children - we already counted this gemm
                        else:
                            visit(node.body)
                    elif isinstance(node, tir.Evaluate):
                        if (
                            isinstance(node.value, tir.Call)
                            and hasattr(node.value.op, "name")
                            and "tl_gemm" in node.value.op.name
                        ):
                                # Direct tl.tl_gemm call without pragma wrapper
                                num_matmuls += 1
                                has_tensorize = True
                                pattern = {
                                    "source": "tl.gemm",
                                    "accumulate": True,
                                    "loop_vars": [],
                                    "cb_in0": 0,
                                    "cb_in1": 1,
                                    "cb_out": 16,
                                }
                                patterns.append(pattern)
                    elif isinstance(node, tir.For):
                        visit(node.body)
                    elif isinstance(node, tir.IfThenElse):
                        visit(node.then_case)
                        if node.else_case:
                            visit(node.else_case)
                    elif isinstance(node, tir.Block):
                        visit(node.body)
                    elif isinstance(node, tir.BlockRealize):
                        visit(node.block)
                    elif isinstance(node, tir.LetStmt):
                        visit(node.body)

                visit(stmt)
                return patterns

            matmul_patterns = count_matmuls(func.body)

            # Transform the IR if there are matmuls
            if num_matmuls > 0:
                # Here we would normally transform the IR to emit TT intrinsics
                # For now, keep the original body but add metadata
                transformed_body = self._transform_matmuls(func.body)
                func = tir.PrimFunc(
                    func.params,
                    transformed_body,
                    func.ret_type,
                    func.buffer_map,
                    func.attrs,
                )

                # Add metadata
                func = func.with_attr("tt_num_matmuls", num_matmuls)
                func = func.with_attr("tt_has_tensorize", has_tensorize)
                func = func.with_attr(
                    "tt_matmul_patterns", tvm.runtime.convert(matmul_patterns)
                )

            # Mark as processed by adding attributes
            func = func.with_attr("tt.tile_intrinsics_lowered", True)
            func = func.with_attr("tt.target_device", self.target_device)
            func = func.with_attr("tt.device_config", tvm.runtime.convert(config))

            logger.info(
                f"Marked function {gvar} for TT tile intrinsic lowering ({self.target_device})"
            )

            new_funcs[gvar] = func

        return tvm.IRModule(new_funcs)

    def _transform_matmuls(self, stmt):
        """Transform matmul operations to TT intrinsics."""

        # Create IR mutator to transform matmuls
        class MatmulTransformer:
            def __init__(self, parent):
                self.parent = parent

            def transform(self, stmt):
                if isinstance(stmt, tir.SeqStmt):
                    new_seq = []
                    for s in stmt.seq:
                        new_seq.append(self.transform(s))
                    return tir.SeqStmt(new_seq)
                elif isinstance(stmt, tir.AttrStmt):
                    if stmt.attr_key == "pragma_gemm" and stmt.value.value == "matmul":
                        # Replace with TT intrinsic sequence
                        return self._create_tt_matmul_sequence(stmt.body)
                    else:
                        return tir.AttrStmt(
                            stmt.node,
                            stmt.attr_key,
                            stmt.value,
                            self.transform(stmt.body),
                        )
                elif isinstance(stmt, tir.Evaluate):
                    if (
                        isinstance(stmt.value, tir.Call)
                        and hasattr(stmt.value.op, "name")
                        and "tl_gemm" in stmt.value.op.name
                    ):
                        # Replace direct tl.tl_gemm with TT sequence
                        return self._create_tt_matmul_sequence(stmt)
                    return stmt
                elif isinstance(stmt, tir.For):
                    return tir.For(
                        stmt.loop_var,
                        stmt.min,
                        stmt.extent,
                        stmt.kind,
                        self.transform(stmt.body),
                    )
                elif isinstance(stmt, tir.IfThenElse):
                    then_case = self.transform(stmt.then_case)
                    else_case = (
                        self.transform(stmt.else_case) if stmt.else_case else None
                    )
                    return tir.IfThenElse(stmt.condition, then_case, else_case)
                elif isinstance(stmt, tir.Block):
                    return tir.Block(
                        stmt.iter_vars,
                        stmt.reads,
                        stmt.writes,
                        stmt.name_hint,
                        self.transform(stmt.body),
                        stmt.init,
                        stmt.alloc_buffers,
                        stmt.match_buffers,
                        stmt.annotations,
                    )
                elif isinstance(stmt, tir.BlockRealize):
                    return tir.BlockRealize(
                        stmt.iter_values, stmt.predicate, self.transform(stmt.block)
                    )
                elif isinstance(stmt, tir.LetStmt):
                    return tir.LetStmt(stmt.var, stmt.value, self.transform(stmt.body))
                else:
                    return stmt

            def _create_tt_matmul_sequence(self, gemm_stmt):
                """Create TT intrinsic sequence for a matmul."""

                # Get TT intrinsic ops
                def get_op(name):
                    return tir.op.Op.get(name)

                # Create the TT intrinsic sequence
                intrinsics = []

                # 1. Acquire tile registers
                intrinsics.append(
                    tir.Evaluate(
                        tir.call_intrin("handle", get_op("tt.tile_regs_acquire"))
                    )
                )

                # 2. Initialize matrix multiply
                intrinsics.append(
                    tir.Evaluate(
                        tir.call_intrin(
                            "handle",
                            get_op("tt.mm_init"),
                            tir.IntImm("int32", 0),  # in0_cb_id
                            tir.IntImm("int32", 1),  # in1_cb_id
                            tir.IntImm("int32", 0),  # transpose_in0
                            tir.IntImm("int32", 0),
                        )  # transpose_in1
                    )
                )

                # 3. Wait for input data in CB0
                intrinsics.append(
                    tir.Evaluate(
                        tir.call_intrin(
                            "handle",
                            get_op("tt.cb_wait_front"),
                            tir.IntImm("int32", 0),  # cb_id
                            tir.IntImm("int32", 1),
                        )  # num_tiles
                    )
                )

                # 4. Wait for input data in CB1
                intrinsics.append(
                    tir.Evaluate(
                        tir.call_intrin(
                            "handle",
                            get_op("tt.cb_wait_front"),
                            tir.IntImm("int32", 1),  # cb_id
                            tir.IntImm("int32", 1),
                        )  # num_tiles
                    )
                )

                # 5. Perform matrix multiply
                intrinsics.append(
                    tir.Evaluate(
                        tir.call_intrin(
                            "handle",
                            get_op("tt.matmul_tiles"),
                            tir.IntImm("int32", 0),  # in0_cb_id
                            tir.IntImm("int32", 1),  # in1_cb_id
                            tir.IntImm("int32", 0),  # in0_tile_idx
                            tir.IntImm("int32", 0),  # in1_tile_idx
                            tir.IntImm("int32", 0),  # dst_tile_idx
                            tir.IntImm("int32", 1),
                        )  # transpose
                    )
                )

                # 6. Pop input from CB0
                intrinsics.append(
                    tir.Evaluate(
                        tir.call_intrin(
                            "handle",
                            get_op("tt.cb_pop_front"),
                            tir.IntImm("int32", 0),  # cb_id
                            tir.IntImm("int32", 1),
                        )  # num_tiles
                    )
                )

                # 7. Pop input from CB1
                intrinsics.append(
                    tir.Evaluate(
                        tir.call_intrin(
                            "handle",
                            get_op("tt.cb_pop_front"),
                            tir.IntImm("int32", 1),  # cb_id
                            tir.IntImm("int32", 1),
                        )  # num_tiles
                    )
                )

                # 8. Commit tile registers
                intrinsics.append(
                    tir.Evaluate(
                        tir.call_intrin("handle", get_op("tt.tile_regs_commit"))
                    )
                )

                # 9. Wait for tile registers
                intrinsics.append(
                    tir.Evaluate(tir.call_intrin("handle", get_op("tt.tile_regs_wait")))
                )

                # 10. Reserve space in output CB
                intrinsics.append(
                    tir.Evaluate(
                        tir.call_intrin(
                            "handle",
                            get_op("tt.cb_reserve_back"),
                            tir.IntImm("int32", 16),  # cb_id
                            tir.IntImm("int32", 1),
                        )  # num_tiles
                    )
                )

                # 11. Pack tile to output CB
                intrinsics.append(
                    tir.Evaluate(
                        tir.call_intrin(
                            "handle",
                            get_op("tt.pack_tile"),
                            tir.IntImm("int32", 0),  # dst_tile_idx
                            tir.IntImm("int32", 16),
                        )  # cb_id
                    )
                )

                # 12. Push to output CB
                intrinsics.append(
                    tir.Evaluate(
                        tir.call_intrin(
                            "handle",
                            get_op("tt.cb_push_back"),
                            tir.IntImm("int32", 16),  # cb_id
                            tir.IntImm("int32", 1),
                        )  # num_tiles
                    )
                )

                # 13. Release tile registers
                intrinsics.append(
                    tir.Evaluate(
                        tir.call_intrin("handle", get_op("tt.tile_regs_release"))
                    )
                )

                return tir.SeqStmt(intrinsics)

        transformer = MatmulTransformer(self)
        return transformer.transform(stmt)

    def _lower_gemm(self, call_node):
        """Lower a GEMM intrinsic to TT matrix engine operations."""
        # Placeholder for actual lowering logic
        # Would generate TT-specific matmul operations
        pass

    def _lower_tile_load(self, call_node):
        """Lower tile load to TT DMA operations."""
        # Placeholder for DMA configuration
        # Would generate NOC read operations
        pass

    def _lower_tile_store(self, call_node):
        """Lower tile store to TT DMA operations."""
        # Placeholder for DMA configuration
        # Would generate NOC write operations
        pass

    def _lower_epilogue(self, call_node):
        """Lower epilogue operations to TT SFPU ops."""
        # Placeholder for SFPU operations
        # Would map to TT activation functions
        pass
