"""
Base utilities for transforming TVM Block structures in v5 passes.
This provides a foundation for handling the Block/BlockRealize pattern
that modern TVM uses for structured IR.
"""

from tvm import tir
from typing import List, Dict, Any, Optional, Tuple


class BlockTransformer:
    """
    Base class for transforming TVM Block structures.

    This class provides the infrastructure for properly traversing and
    transforming TVM's Block/BlockRealize nodes while preserving structure.
    """

    def __init__(self):
        self.metadata = {}

    def visit(self, stmt):
        """
        Main visitor dispatch method.

        Handles all TVM IR node types with special handling for Block structures.
        """
        if stmt is None:
            return None

        # Handle Block structures
        if isinstance(stmt, tir.BlockRealize):
            return self.visit_block_realize(stmt)
        elif isinstance(stmt, tir.Block):
            return self.visit_block(stmt)

        # Handle other common nodes
        elif isinstance(stmt, tir.For):
            return self.visit_for(stmt)
        elif isinstance(stmt, tir.Allocate):
            return self.visit_allocate(stmt)
        elif isinstance(stmt, tir.Evaluate):
            return self.visit_evaluate(stmt)
        elif isinstance(stmt, tir.BufferStore):
            return self.visit_buffer_store(stmt)
        elif isinstance(stmt, tir.BufferLoad):
            return self.visit_buffer_load(stmt)

        # Handle sequence statements
        elif isinstance(stmt, tir.SeqStmt):
            new_seq = []
            changed = False
            for i, s in enumerate(stmt.seq):
                new_s = self.visit(s)
                if new_s is not None:
                    new_seq.append(new_s)
                    if new_s is not s:  # Check object identity, not equality
                        changed = True
                        print(f"[BlockTransformer.visit] SeqStmt child {i} changed")
            if changed:
                print(f"[BlockTransformer.visit] SeqStmt changed, creating new SeqStmt")
                return tir.SeqStmt(new_seq)
            print(f"[BlockTransformer.visit] SeqStmt unchanged")
            return stmt

        # Handle nodes with body
        elif hasattr(stmt, 'body'):
            new_body = self.visit(stmt.body)
            if new_body != stmt.body:
                # Try to use with_body method if available
                if hasattr(stmt, 'with_body'):
                    return stmt.with_body(new_body)
                # Otherwise try to reconstruct
                return self._reconstruct_with_body(stmt, new_body)
            return stmt

        # Default: return unchanged
        return stmt

    def visit_block_realize(self, block_realize: tir.BlockRealize) -> tir.BlockRealize:
        """
        Visit a BlockRealize node.

        BlockRealize wraps a Block with iteration values and predicates.
        """
        # Visit the contained block
        new_block = self.visit(block_realize.block)

        # Reconstruct if changed (use object identity)
        if new_block is not block_realize.block:
            return tir.BlockRealize(block_realize.iter_values, block_realize.predicate, new_block,
                                    block_realize.span)
        return block_realize

    def visit_block(self, block: tir.Block) -> tir.Block:
        """
        Visit a Block node.

        This is where we handle alloc_buffers and the block body.
        """
        # Process alloc_buffers (e.g., shared memory allocations)
        new_alloc_buffers, cb_metadata = self.process_alloc_buffers(block.alloc_buffers)

        # Visit the body
        original_body = block.body
        new_body = self.visit(original_body)

        # Insert any necessary transformations (e.g., CB allocations)
        if cb_metadata:
            new_body = self.insert_cb_allocations(new_body, cb_metadata)

        # Reconstruct block if anything changed (use object identity check)
        if (new_body is not original_body or new_alloc_buffers is not block.alloc_buffers):
            # Use the replace method if available, otherwise reconstruct
            if hasattr(block, 'replace'):
                return block.replace(body=new_body, alloc_buffers=new_alloc_buffers)
            else:
                # Manual reconstruction
                return tir.Block(block.iter_vars, block.reads, block.writes, block.name_hint,
                                 new_body, block.init, new_alloc_buffers, block.match_buffers,
                                 block.annotations, block.span)
        return block

    def process_alloc_buffers(self, alloc_buffers: List) -> Tuple[List, List[Dict]]:
        """
        Process alloc_buffers from a Block.

        Override this in subclasses to handle shared memory, etc.

        Returns:
            - Updated alloc_buffers list
            - Metadata about any transformations (e.g., CB allocations to insert)
        """
        # Default: no transformation
        return alloc_buffers, []

    def insert_cb_allocations(self, body, cb_metadata: List[Dict]) -> tir.Stmt:
        """
        Insert CB allocations at the start of the body.

        Override in subclasses to customize CB allocation generation.
        """
        if not cb_metadata:
            return body

        cb_allocs = []
        for cb_info in cb_metadata:
            alloc = self.create_cb_allocation(cb_info)
            if alloc:
                cb_allocs.append(alloc)

        if cb_allocs:
            if body:
                cb_allocs.append(body)
            return tir.SeqStmt(cb_allocs)
        return body

    def create_cb_allocation(self, cb_info: Dict) -> Optional[tir.Stmt]:
        """
        Create a CB allocation statement.

        Override in subclasses to generate specific CB allocation intrinsics.
        """
        # Default implementation (to be overridden)
        return None

    # Default visit methods for other nodes
    def visit_for(self, for_node: tir.For) -> tir.For:
        """Visit For loop. Override in subclasses for specific behavior."""
        new_body = self.visit(for_node.body)
        if new_body is not for_node.body:  # Use object identity
            return tir.For(for_node.loop_var, for_node.min, for_node.extent, for_node.kind,
                           new_body, for_node.thread_binding, for_node.annotations, for_node.span)
        return for_node

    def visit_allocate(self, allocate: tir.Allocate) -> tir.Allocate:
        """Visit Allocate node. Override in subclasses."""
        new_body = self.visit(allocate.body)
        if new_body is not allocate.body:  # Use object identity
            return tir.Allocate(allocate.buffer_var, allocate.dtype, allocate.extents,
                                allocate.condition, new_body, allocate.annotations, allocate.span)
        return allocate

    def visit_evaluate(self, evaluate: tir.Evaluate) -> tir.Evaluate:
        """Visit Evaluate node. Override in subclasses."""
        return evaluate

    def visit_buffer_store(self, buffer_store: tir.BufferStore) -> tir.BufferStore:
        """Visit BufferStore. Override in subclasses."""
        return buffer_store

    def visit_buffer_load(self, buffer_load: tir.BufferLoad) -> tir.BufferLoad:
        """Visit BufferLoad. Override in subclasses."""
        return buffer_load

    def _reconstruct_with_body(self, stmt, new_body):
        """
        Helper to reconstruct a statement with a new body.

        This is a fallback for nodes that don't have a with_body method.
        """
        # This would need to handle each node type specifically
        # For now, return the original if we can't reconstruct
        return stmt


# Utility functions for working with Blocks and Buffers


def is_shared_buffer(buffer) -> bool:
    """Check if a buffer is in shared memory scope."""
    return buffer.scope() == "shared"


def is_local_buffer(buffer) -> bool:
    """Check if a buffer is in local/register scope."""
    return buffer.scope() in [
        "local", "wmma.matrix_a", "wmma.matrix_b", "wmma.accumulator", "fragment"
    ]


def extract_buffer_info(buffer) -> Dict[str, Any]:
    """Extract relevant information from a Buffer object."""
    return {
        'name': buffer.name,
        'shape': [int(dim) if hasattr(dim, 'value') else dim for dim in buffer.shape],
        'dtype': str(buffer.dtype),
        'scope': buffer.scope(),
        'strides': buffer.strides,
        'elem_offset': buffer.elem_offset if hasattr(buffer, 'elem_offset') else 0
    }


def create_intrinsic_call(name: str, *args) -> tir.Call:
    """Create an intrinsic call node."""
    return tir.call_extern("void", name, *args)


def create_cb_intrinsic(cb_name: str, shape: List[int], dtype: str) -> tir.Call:
    """Create a CB allocation intrinsic call."""
    return tir.call_extern("handle", "tt.alloc_cb", tir.StringImm(cb_name),
                           *[tir.IntImm("int32", dim) for dim in shape], tir.StringImm(dtype))


def find_blocks_in_stmt(stmt) -> List[tir.Block]:
    """Find all Block nodes in a statement tree."""
    blocks = []

    def visitor(node):
        if isinstance(node, tir.Block):
            blocks.append(node)

    tir.stmt_functor.post_order_visit(stmt, visitor)
    return blocks


def find_shared_allocations(block: tir.Block) -> List[Dict[str, Any]]:
    """Find all shared memory allocations in a Block."""
    shared_allocs = []

    for buffer in block.alloc_buffers:
        if is_shared_buffer(buffer):
            shared_allocs.append(extract_buffer_info(buffer))

    return shared_allocs


# Pattern matching utilities for compute operations


def is_matmul_pattern(stmt) -> bool:
    """Check if statement matches a matrix multiplication pattern."""
    # This would need actual pattern matching logic
    # Simplified for now
    if isinstance(stmt, tir.Evaluate) and isinstance(stmt.value, tir.Call):
        call_name = str(stmt.value.op)
        return any(pattern in call_name for pattern in ["gemm", "matmul", "T.gemm"])
    return False


def is_elementwise_pattern(stmt) -> bool:
    """Check if statement matches an element-wise operation pattern."""
    if isinstance(stmt, tir.Evaluate) and isinstance(stmt.value, tir.Call):
        call_name = str(stmt.value.op)
        elementwise_ops = ["add", "multiply", "subtract", "divide", "T.add", "T.multiply"]
        return any(op in call_name for op in elementwise_ops)
    return False


# Example usage
if __name__ == "__main__":
    # Example: Create a simple transformer
    class ExampleTransformer(BlockTransformer):

        def process_alloc_buffers(self, alloc_buffers):
            """Example: Transform shared buffers to CBs"""
            cb_metadata = []

            for buffer in alloc_buffers:
                if is_shared_buffer(buffer):
                    cb_info = extract_buffer_info(buffer)
                    cb_info['cb_name'] = f"cb_{buffer.name}"
                    cb_metadata.append(cb_info)

            return alloc_buffers, cb_metadata

        def create_cb_allocation(self, cb_info):
            """Create CB allocation for shared buffer"""
            return tir.Evaluate(
                create_cb_intrinsic(cb_info['cb_name'], cb_info['shape'], cb_info['dtype']))

    print("BlockTransformer utilities loaded successfully")
