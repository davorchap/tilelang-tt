"""
Consolidated intrinsic registry for Tenstorrent backend.
Single source of truth for all IR -> C++ mappings.

This replaces scattered template code with IR-driven generation.
"""

from dataclasses import dataclass
from typing import Optional, List, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class Intrinsic:
    """Represents a mapping from IR operation to C++ code"""
    ir_name: str  # IR operation name (e.g., "tt.alloc_cb")
    cpp_func: str  # C++ function name (e.g., "cb_reserve_back")
    template: str  # C++ code template
    arg_count: int  # Expected number of arguments
    category: str  # Category: noc, cb, compute, sync, etc.
    requires_namespace: bool = True  # Whether to prepend namespace
    validator: Optional[Callable] = None  # Optional argument validator


class IntrinsicRegistry:
    """Central registry for all TT intrinsics"""

    def __init__(self):
        self._registry = {}
        self._init_intrinsics()

    def _init_intrinsics(self):
        """Initialize all intrinsic mappings"""

        # ========== CB Management Operations ==========
        self.register(Intrinsic(
            ir_name="tt.alloc_cb",
            cpp_func="cb_reserve_back",
            template="cb_reserve_back({cb_id}, {num_tiles});",
            arg_count=-1,  # Variable: cb_name, shape (1-2 args), dtype (3-4 total)
            category="cb"
        ))

        self.register(Intrinsic(
            ir_name="tir.cb_reserve_back",
            cpp_func="cb_reserve_back",
            template="cb_reserve_back({0}, {1});",
            arg_count=2,
            category="cb"
        ))

        self.register(Intrinsic(
            ir_name="tir.cb_push_back",
            cpp_func="cb_push_back",
            template="cb_push_back({0}, {1});",
            arg_count=2,
            category="cb"
        ))

        self.register(Intrinsic(
            ir_name="tir.cb_pop_front",
            cpp_func="cb_pop_front",
            template="cb_pop_front({0}, {1});",
            arg_count=2,
            category="cb"
        ))

        self.register(Intrinsic(
            ir_name="tir.cb_wait_front",
            cpp_func="cb_wait_front",
            template="cb_wait_front({0}, {1});",
            arg_count=2,
            category="cb"
        ))

        # ========== NOC Operations ==========
        self.register(Intrinsic(
            ir_name="tt.read_to_cb",
            cpp_func="noc_async_read_tile",
            template="noc_async_read_tile({src_addr}, {cb_id});",
            arg_count=2,
            category="noc"
        ))

        self.register(Intrinsic(
            ir_name="tt.write_from_cb",
            cpp_func="noc_async_write_tile",
            template="noc_async_write_tile({cb_id}, {dst_addr});",
            arg_count=2,
            category="noc"
        ))

        self.register(Intrinsic(
            ir_name="tir.noc_async_read_tile",
            cpp_func="noc_async_read_tile",
            template="noc_async_read_tile({0}, {1}, {2});",
            arg_count=3,
            category="noc"
        ))

        self.register(Intrinsic(
            ir_name="tir.noc_async_write_tile",
            cpp_func="noc_async_write_tile",
            template="noc_async_write_tile({0}, {1}, {2});",
            arg_count=3,
            category="noc"
        ))

        self.register(Intrinsic(
            ir_name="tir.noc_async_read_barrier",
            cpp_func="noc_async_read_barrier",
            template="noc_async_read_barrier();",
            arg_count=0,
            category="noc"
        ))

        self.register(Intrinsic(
            ir_name="tir.noc_async_write_barrier",
            cpp_func="noc_async_write_barrier",
            template="noc_async_write_barrier();",
            arg_count=0,
            category="noc"
        ))

        # ========== Compute Operations ==========
        # Matrix multiply
        self.register(Intrinsic(
            ir_name="tt.mm.mma",
            cpp_func="matmul_tiles",
            template="matmul_tiles({cb_in0}, {cb_in1}, {num_tiles}, {dst_reg}, {accumulate});",
            arg_count=4,  # cb_in0, cb_in1, dst_reg, accumulate
            category="compute"
        ))

        self.register(Intrinsic(
            ir_name="tir.mm_init",
            cpp_func="mm_init",
            template="mm_init();",
            arg_count=0,
            category="compute"
        ))

        self.register(Intrinsic(
            ir_name="tir.matmul_tiles",
            cpp_func="matmul_tiles",
            template="matmul_tiles({0}, {1}, {2}, {3}, {4});",
            arg_count=5,
            category="compute"
        ))

        # FPU binary operations
        self.register(Intrinsic(
            ir_name="tt.fpu.add",
            cpp_func="add_tiles",
            template="add_tiles({cb_in0}, {cb_in1}, {num_tiles});",
            arg_count=3,
            category="compute"
        ))

        self.register(Intrinsic(
            ir_name="tt.fpu.multiply",
            cpp_func="mul_tiles",
            template="mul_tiles({cb_in0}, {cb_in1}, {num_tiles});",
            arg_count=3,
            category="compute"
        ))

        self.register(Intrinsic(
            ir_name="tt.fpu.subtract",
            cpp_func="sub_tiles",
            template="sub_tiles({cb_in0}, {cb_in1}, {num_tiles});",
            arg_count=3,
            category="compute"
        ))

        self.register(Intrinsic(
            ir_name="tir.add_tiles",
            cpp_func="add_tiles",
            template="add_tiles({0}, {1}, {2}, {3});",
            arg_count=4,
            category="compute"
        ))

        self.register(Intrinsic(
            ir_name="tir.mul_tiles",
            cpp_func="mul_tiles",
            template="mul_tiles({0}, {1}, {2}, {3});",
            arg_count=4,
            category="compute"
        ))

        # SFPU unary operations
        self.register(Intrinsic(
            ir_name="tt.sfpu.unary",
            cpp_func="sfpu_unary",
            template="sfpu_{op_type}({cb_in}, {dst_reg});",
            arg_count=3,  # op_type, cb_in, dst_reg
            category="compute"
        ))

        self.register(Intrinsic(
            ir_name="tir.sfpu_relu",
            cpp_func="relu_tiles",
            template="relu_tiles({0}, {1});",
            arg_count=2,
            category="compute"
        ))

        self.register(Intrinsic(
            ir_name="tir.sfpu_gelu",
            cpp_func="gelu_tiles",
            template="gelu_tiles({0}, {1});",
            arg_count=2,
            category="compute"
        ))

        # DST management
        self.register(Intrinsic(
            ir_name="tir.acquire_dst",
            cpp_func="acquire_dst",
            template="acquire_dst(tt::DstMode::Half);",
            arg_count=0,
            category="dst"
        ))

        self.register(Intrinsic(
            ir_name="tir.release_dst",
            cpp_func="release_dst",
            template="release_dst(tt::DstMode::Half);",
            arg_count=0,
            category="dst"
        ))

        self.register(Intrinsic(
            ir_name="tir.pack_tile",
            cpp_func="pack_tile",
            template="pack_tile({0}, {1});",
            arg_count=2,
            category="dst"
        ))

        # ========== Synchronization Operations ==========
        self.register(Intrinsic(
            ir_name="tir.barrier",
            cpp_func="barrier",
            template="barrier();",
            arg_count=0,
            category="sync"
        ))

        self.register(Intrinsic(
            ir_name="tir.get_noc_addr",
            cpp_func="get_noc_addr",
            template="get_noc_addr({0}, {1});",
            arg_count=2,
            category="sync"
        ))

        # ========== Helper/Utility Operations ==========
        self.register(Intrinsic(
            ir_name="tir.get_tile_index",
            cpp_func="get_tile_index",
            template="get_tile_index({0});",
            arg_count=1,
            category="util"
        ))

    def register(self, intrinsic: Intrinsic):
        """Register an intrinsic mapping"""
        if intrinsic.ir_name in self._registry:
            logger.warning(f"Overwriting existing intrinsic: {intrinsic.ir_name}")
        self._registry[intrinsic.ir_name] = intrinsic

    def get(self, ir_name: str) -> Optional[Intrinsic]:
        """Get intrinsic by IR name"""
        return self._registry.get(ir_name)

    def has(self, ir_name: str) -> bool:
        """Check if intrinsic is registered"""
        return ir_name in self._registry

    def get_by_category(self, category: str) -> List[Intrinsic]:
        """Get all intrinsics in a category"""
        return [i for i in self._registry.values() if i.category == category]

    def generate_cpp(self, ir_name: str, args: List[str]) -> str:
        """Generate C++ code for an intrinsic call"""
        intrinsic = self.get(ir_name)
        if not intrinsic:
            raise ValueError(f"Unknown intrinsic: {ir_name}")

        # Validate argument count
        if intrinsic.arg_count >= 0 and len(args) != intrinsic.arg_count:
            logger.warning(
                f"Argument count mismatch for {ir_name}: "
                f"expected {intrinsic.arg_count}, got {len(args)}"
            )

        # Format the template
        if "{" in intrinsic.template:
            # Named placeholders
            kwargs = {}
            # Map common argument positions to names
            if len(args) > 0:
                kwargs['cb_id'] = args[0]
                kwargs['cb_in0'] = args[0]
                kwargs['cb_in'] = args[0]
                kwargs['src_addr'] = args[0]
            if len(args) > 1:
                kwargs['num_tiles'] = args[1]
                kwargs['cb_in1'] = args[1]
                kwargs['dst_addr'] = args[1]
            if len(args) > 2:
                kwargs['dst_reg'] = args[2]
                kwargs['op_type'] = args[0]  # For SFPU ops
            if len(args) > 3:
                kwargs['accumulate'] = args[3]

            # Also support positional args
            for i, arg in enumerate(args):
                kwargs[str(i)] = arg

            try:
                return intrinsic.template.format(**kwargs)
            except KeyError as e:
                # Fall back to positional
                return intrinsic.template.format(*args)
        else:
            # No placeholders, return as-is
            return intrinsic.template


# Global registry instance
INTRINSIC_REGISTRY = IntrinsicRegistry()


def get_intrinsic(ir_name: str) -> Optional[Intrinsic]:
    """Get intrinsic from global registry"""
    return INTRINSIC_REGISTRY.get(ir_name)


def generate_cpp_for_intrinsic(ir_name: str, args: List[str]) -> str:
    """Generate C++ code for an intrinsic call"""
    return INTRINSIC_REGISTRY.generate_cpp(ir_name, args)


def is_intrinsic_registered(ir_name: str) -> bool:
    """Check if intrinsic is registered"""
    return INTRINSIC_REGISTRY.has(ir_name)


# Export key functions
__all__ = [
    'Intrinsic',
    'IntrinsicRegistry',
    'INTRINSIC_REGISTRY',
    'get_intrinsic',
    'generate_cpp_for_intrinsic',
    'is_intrinsic_registered'
]