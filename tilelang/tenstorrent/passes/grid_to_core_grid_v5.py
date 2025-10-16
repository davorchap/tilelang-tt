"""
Pass B2: GridToCoreGrid (New Metadata Format) - Fixed with BlockTransformer
Version: 5.1
Date: 2025-10-15

Purpose: Convert GPU-style grid kernel to Tenstorrent core launch model
         using the new metadata format from v5 design.

This version properly handles TVM Block structures using BlockTransformer.

Input: TIR with T.Kernel(gx, gy) grid structure and new metadata
Output: TIR with T.launch_core persistent model
"""

import tvm
from tvm import tir
from tvm.tir import stmt_functor
import sys
import os

# Import our BlockTransformer base class
sys.path.append(os.path.dirname(__file__))
from block_transformer import BlockTransformer


@tvm.tir.transform.prim_func_pass(opt_level=0)
def GridToCoreGrid_v5(func, mod, ctx):
    """
    Transform GPU grid to persistent core model using new metadata.

    This pass:
    1. Reads new metadata format (tt.core_grid, tt.work_partition, etc.)
    2. Converts T.Kernel blocks to T.launch_core
    3. Adds axis mapping attributes
    4. Supports both global and local_shard modes
    """

    # Extract metadata from function attributes
    metadata = extract_metadata(func)

    if not metadata:
        # If no metadata, skip transformation
        print("Warning: No TT metadata found, skipping GridToCoreGrid")
        return func

    class GridTransformer(BlockTransformer):

        def __init__(self, metadata):
            super().__init__()
            self.metadata = metadata
            self.core_grid = metadata.get("core_grid", [8, 8])
            self.partition_mode = metadata.get("partition_mode", "global")
            self.grid_tiles = metadata.get("grid_tiles", [])
            self.work_partition = metadata.get("work_partition", {})
            self.runtime_args = metadata.get("runtime_args", [])

            # Track transformation state
            self.in_kernel = False
            self.kernel_vars = {}
            self.grid_loops_found = []
            self.grid_loop_depth = 0  # Track nesting depth of grid loops
            self.work_distribution_added = False  # Track if we've added work distribution

        def visit_block(self, block):
            """Override to detect grid patterns in blocks"""
            # Process the block
            result = super().visit_block(block)

            # Detect grid patterns in the block
            self._detect_grid_patterns_in_block(block)

            return result

        def visit_for(self, for_node):
            """Transform grid loops to core launch"""

            # Check if this is a grid loop (blockIdx binding)
            if self._is_grid_loop(for_node):
                # Replace with core launch
                return self._create_core_launch(for_node)

            # Otherwise, visit normally
            new_body = self.visit(for_node.body)

            if new_body != for_node.body:
                return tir.For(for_node.loop_var, for_node.min, for_node.extent, for_node.kind,
                               new_body, for_node.thread_binding, for_node.annotations,
                               for_node.span)
            return for_node

        def visit_attr_stmt(self, attr):
            """Handle T.Kernel attribute nodes and thread bindings"""

            # Check if this is a T.Kernel block
            if self._is_kernel_attr(attr):
                # Extract grid dimensions from the kernel
                grid_dims = self._extract_grid_dims(attr)

                # Transform to core launch
                transformed_body = self._transform_to_core_launch(attr.body, grid_dims)

                # Return the transformed body directly (remove T.Kernel wrapper)
                return self.visit(transformed_body)

            # Handle thread binding attributes
            elif self._is_thread_binding_attr(attr):
                # This could be a grid dimension binding
                return self._handle_thread_binding(attr)

            # Visit body
            new_body = self.visit(attr.body)

            if new_body != attr.body:
                return tir.AttrStmt(attr.node, attr.attr_key, attr.value, new_body, attr.span)
            return attr

        def _detect_grid_patterns_in_block(self, block):
            """Detect GPU grid patterns within a block"""

            def visitor(node):
                if isinstance(node, tir.For):
                    if self._is_grid_loop(node):
                        self.grid_loops_found.append(node)
                elif isinstance(node, tir.AttrStmt) and (self._is_kernel_attr(node) or
                                                         self._is_thread_binding_attr(node)):
                    self.grid_loops_found.append(node)

            stmt_functor.post_order_visit(block.body, visitor)

        def _is_kernel_attr(self, attr_node):
            """Check if this is a T.Kernel attribute"""
            # T.Kernel creates an AttrStmt with specific patterns
            if hasattr(attr_node, 'attr_key'):
                attr_key_str = str(attr_node.attr_key)
                return "pragma_kernel" in attr_key_str or "kernel" in attr_key_str.lower()
            return False

        def _is_thread_binding_attr(self, attr_node):
            """Check if this is a thread binding attribute"""
            if hasattr(attr_node, 'attr_key'):
                attr_key_str = str(attr_node.attr_key)
                return "thread_extent" in attr_key_str or "thread_binding" in attr_key_str
            return False

        def _is_grid_loop(self, for_node):
            """Check if this is a grid dimension loop"""
            # Check for thread binding
            if hasattr(for_node, 'thread_binding') and for_node.thread_binding is not None:
                binding_str = str(for_node.thread_binding)
                grid_patterns = ["blockIdx", "gridIdx"]
                return any(pattern in binding_str for pattern in grid_patterns)

            # Check loop variable name
            if hasattr(for_node, 'loop_var'):
                loop_var_name = for_node.loop_var.name if hasattr(
                    for_node.loop_var, 'name') else str(for_node.loop_var)
                grid_patterns = ["blockIdx", "bx", "by", "bz", "gridIdx", "gx", "gy"]
                return any(pattern in loop_var_name for pattern in grid_patterns)

            return False

        def _extract_grid_dims(self, kernel_attr):
            """Extract grid dimensions from T.Kernel"""
            # T.Kernel typically stores grid size in the value
            if hasattr(kernel_attr, 'value') and isinstance(kernel_attr.value, tir.IntImm):
                # Single dimension
                return [kernel_attr.value.value, 1]
            # Default to core grid if can't extract

            return self.core_grid

        def _handle_thread_binding(self, attr):
            """Handle thread binding attributes for grid dimensions"""
            # Extract binding information
            if hasattr(attr, 'value') and hasattr(attr, 'node'):
                # This is a thread extent binding
                extent = attr.value
                var = attr.node

                # Check if this is a grid binding
                attr_key_str = str(attr.attr_key)
                if "blockIdx" in attr_key_str:
                    # Transform to core launch
                    return self._create_core_launch_from_binding(var, extent, attr.body,
                                                                 attr_key_str)

            # Visit body normally
            new_body = self.visit(attr.body)
            if new_body != attr.body:
                return tir.AttrStmt(attr.node, attr.attr_key, attr.value, new_body, attr.span)
            return attr

        def _transform_to_core_launch(self, body, grid_dims):
            """Transform kernel body to use core launch"""

            # Create core launch variables
            tir.Var("cx", "int32")
            tir.Var("cy", "int32")

            # Create launch statements
            stmts = []

            # Add T.launch_core for x dimension
            launch_x = tir.Evaluate(
                tir.call_extern("int32", "T.launch_core", tir.StringImm("coreIdx.x"),
                                tir.IntImm("int32", self.core_grid[0])))
            stmts.append(launch_x)

            # Add T.launch_core for y dimension
            launch_y = tir.Evaluate(
                tir.call_extern("int32", "T.launch_core", tir.StringImm("coreIdx.y"),
                                tir.IntImm("int32", self.core_grid[1])))
            stmts.append(launch_y)

            # Add work distribution logic based on partition mode
            if self.partition_mode == "global":
                stmts.extend(self._add_global_work_distribution())
            else:  # local_shard
                stmts.extend(self._add_local_shard_distribution())

            # Transform the body to use core coordinates
            transformed_body = self._replace_grid_indices(body)
            stmts.append(transformed_body)

            return tir.SeqStmt(stmts)

        def _create_core_launch(self, for_node):
            """Create T.launch_core from grid loop"""
            loop_var = for_node.loop_var

            # Track grid loop depth
            self.grid_loop_depth += 1
            current_depth = self.grid_loop_depth

            # Determine dimension (x or y) from binding or variable name
            dim_str = ""
            if hasattr(for_node, 'thread_binding') and for_node.thread_binding is not None:
                binding_str = str(for_node.thread_binding)
                if "x" in binding_str or ".0" in binding_str:
                    dim_str = "x"
                elif "y" in binding_str or ".1" in binding_str:
                    dim_str = "y"
                elif "z" in binding_str or ".2" in binding_str:
                    dim_str = "z"

            if not dim_str:
                # Fall back to variable name
                loop_var_name = loop_var.name if hasattr(loop_var, 'name') else str(loop_var)
                dim_str = "x" if "x" in loop_var_name or "0" in loop_var_name else "y"

            # Get core count for this dimension
            if dim_str == "x":
                core_count = self.core_grid[0]
            elif dim_str == "y":
                core_count = self.core_grid[1] if len(self.core_grid) > 1 else 1
            else:
                core_count = 1  # z dimension not typically used for cores

            # Create launch_core call
            launch_call = tir.call_extern("int32", "T.launch_core",
                                          tir.StringImm(f"coreIdx.{dim_str}"),
                                          tir.IntImm("int32", core_count))

            # Create variable to hold core index
            core_var = tir.Var(f"c{dim_str}", "int32")

            # Transform body
            new_body = self.visit(for_node.body)

            # Store mapping for later use
            self.kernel_vars[loop_var] = core_var

            # Check if this is the innermost grid loop (for 2D grids, depth == 2)
            # and we haven't added work distribution yet
            expected_depth = len(self.core_grid)  # 2 for 2D grids
            if current_depth == expected_depth and not self.work_distribution_added:
                # Inject work distribution logic before the body
                work_stmts = []

                # Add work distribution based on partition mode
                if self.partition_mode == "global":
                    work_stmts = self._add_global_work_distribution()
                else:  # local_shard
                    work_stmts = self._add_local_shard_distribution()

                # If we have work distribution statements, wrap them around the body
                if work_stmts:
                    # Create a sequence of work distribution + body
                    wrapped_body = self._wrap_body_with_work_distribution(new_body, work_stmts)
                    new_body = wrapped_body
                    self.work_distribution_added = True

            # Create let statement
            let_stmt = tir.LetStmt(core_var, launch_call, new_body)

            # Decrement depth when returning
            self.grid_loop_depth -= 1

            return let_stmt

        def _wrap_body_with_work_distribution(self, body, work_stmts):
            """
            Wrap the body with work distribution LetStmt nodes.

            Takes work_stmts (list of LetStmt with dummy bodies) and chains them
            so the innermost LetStmt contains the actual body.
            """
            if not work_stmts:
                return body

            # Work backwards through the work_stmts, making each LetStmt wrap the next
            current_body = body
            for stmt in reversed(work_stmts):
                if isinstance(stmt, tir.LetStmt):
                    # Replace the dummy body with the current body
                    current_body = tir.LetStmt(stmt.var, stmt.value, current_body)
                else:
                    # If it's not a LetStmt, just prepend it
                    current_body = tir.SeqStmt([stmt, current_body])

            return current_body

        def _create_core_launch_from_binding(self, var, extent, body, attr_key):
            """Create core launch from thread binding attribute"""
            # Similar to _create_core_launch but from AttrStmt

            # Determine dimension
            if "x" in attr_key or ".0" in attr_key:
                dim_str = "x"
                core_count = self.core_grid[0]
            elif "y" in attr_key or ".1" in attr_key:
                dim_str = "y"
                core_count = self.core_grid[1] if len(self.core_grid) > 1 else 1
            else:
                dim_str = "z"
                core_count = 1

            # Create launch_core call
            launch_call = tir.call_extern("int32", "T.launch_core",
                                          tir.StringImm(f"coreIdx.{dim_str}"),
                                          tir.IntImm("int32", core_count))

            # Create variable to hold core index
            core_var = tir.Var(f"c{dim_str}", "int32")

            # Transform body
            new_body = self.visit(body)

            # Store mapping
            self.kernel_vars[var] = core_var

            # Create let statement
            let_stmt = tir.LetStmt(core_var, launch_call, new_body)

            return let_stmt

        def _add_global_work_distribution(self):
            """Add work distribution for global mode"""
            stmts = []

            # In global mode, each core processes assigned tiles
            # Add runtime arg access for start_id and count
            start_id = tir.call_extern(
                "int32",
                "get_arg_val",
                tir.IntImm("int32", 0)  # start_id is typically first arg
            )
            count = tir.call_extern(
                "int32",
                "get_arg_val",
                tir.IntImm("int32", 1)  # count is typically second arg
            )

            # Create variables
            start_var = tir.Var("start_id", "int32")
            count_var = tir.Var("count", "int32")

            stmts.append(tir.LetStmt(start_var, start_id, tir.Evaluate(tir.IntImm("int32", 0))))
            stmts.append(tir.LetStmt(count_var, count, tir.Evaluate(tir.IntImm("int32", 0))))

            # Add tile dimension access
            if "Mt" in self.runtime_args:
                mt_idx = self.runtime_args.index("Mt")
                mt = tir.call_extern("int32", "get_arg_val", tir.IntImm("int32", mt_idx))
                stmts.append(
                    tir.LetStmt(tir.Var("Mt", "int32"), mt, tir.Evaluate(tir.IntImm("int32", 0))))

            if "Nt" in self.runtime_args:
                nt_idx = self.runtime_args.index("Nt")
                nt = tir.call_extern("int32", "get_arg_val", tir.IntImm("int32", nt_idx))
                stmts.append(
                    tir.LetStmt(tir.Var("Nt", "int32"), nt, tir.Evaluate(tir.IntImm("int32", 0))))

            return stmts

        def _add_local_shard_distribution(self):
            """Add work distribution for local shard mode"""
            stmts = []

            # In local shard mode, need shard dimensions
            shard_args = ["Sm", "Sn", "Gy", "Gx", "sy", "sx"]
            for arg_name in shard_args:
                if arg_name in self.runtime_args:
                    idx = self.runtime_args.index(arg_name)
                    value = tir.call_extern("int32", "get_arg_val", tir.IntImm("int32", idx))
                    stmts.append(
                        tir.LetStmt(
                            tir.Var(arg_name, "int32"), value, tir.Evaluate(tir.IntImm("int32",
                                                                                       0))))

            return stmts

        def _replace_grid_indices(self, body):
            """Replace grid indices with core-based computation"""

            class IndexReplacer:

                def __init__(self, kernel_vars):
                    self.kernel_vars = kernel_vars

                def visit(self, stmt):
                    """Main visit dispatch"""
                    if isinstance(stmt, tir.Var):
                        return self.visit_var(stmt)
                    elif isinstance(stmt, tir.BufferLoad):
                        return self.visit_buffer_load(stmt)
                    elif isinstance(stmt, tir.BufferStore):
                        return self.visit_buffer_store(stmt)
                    elif isinstance(stmt, tir.SeqStmt):
                        new_seq = []
                        for s in stmt.seq:
                            new_seq.append(self.visit(s))
                        return tir.SeqStmt(new_seq)
                    elif hasattr(stmt, 'body'):
                        new_body = self.visit(stmt.body)
                        if hasattr(stmt, 'with_body'):
                            return stmt.with_body(new_body)
                    return stmt

                def visit_var(self, op):
                    # Replace grid variables with core variables
                    if op in self.kernel_vars:
                        return self.kernel_vars[op]
                    return op

                def visit_buffer_load(self, op):
                    # Transform buffer indices for core-based access
                    new_indices = []
                    for idx in op.indices:
                        new_idx = self.visit(idx)
                        new_indices.append(new_idx)

                    if new_indices != op.indices:
                        return tir.BufferLoad(op.buffer, new_indices, op.span)
                    return op

                def visit_buffer_store(self, op):
                    # Transform buffer indices for core-based access
                    new_indices = []
                    for idx in op.indices:
                        new_idx = self.visit(idx)
                        new_indices.append(new_idx)
                    new_value = self.visit(op.value)

                    if new_indices != op.indices or new_value != op.value:
                        return tir.BufferStore(op.buffer, new_value, new_indices, op.span)
                    return op

            replacer = IndexReplacer(self.kernel_vars)
            return replacer.visit(body)

    # Apply transformation
    transformer = GridTransformer(metadata)
    new_body = transformer.visit(func.body)

    # Update function with new body
    func = func.with_body(new_body)

    # Add axis mapping attributes
    func = func.with_attr("tt.core_map_x", "coreIdx.x")
    func = func.with_attr("tt.core_map_y", "coreIdx.y")

    # Keep the metadata for later passes
    func = func.with_attr("tt.transformed_to_core", True)

    return func


def extract_metadata(func):
    """Extract v5 metadata from function attributes"""
    metadata = {}

    # List of expected metadata keys
    metadata_keys = [
        "tt.core_grid", "tt.core_ranges", "tt.partition_mode", "tt.grid_tiles", "tt.work_partition",
        "tt.runtime_args", "tt.shard_grid", "tt.local_shape_tiles"
    ]

    # Extract available metadata
    for key in metadata_keys:
        if key in func.attrs:
            metadata[key.replace("tt.", "")] = func.attrs[key]

    return metadata


def validate_core_launch(func):
    """
    Validate that the output has proper core launch structure.
    Should have:
    - T.launch_core calls
    - No T.Kernel blocks
    - Proper runtime arg access
    """

    class CoreLaunchChecker:

        def __init__(self):
            self.has_launch_core = False
            self.has_kernel_block = False
            self.launch_calls = []

        def check(self, stmt):
            """Check for core launch patterns"""

            def visitor(node):
                if isinstance(node, tir.Call):
                    call_name = str(node.op)
                    if "launch_core" in call_name:
                        self.has_launch_core = True
                        self.launch_calls.append(call_name)
                elif isinstance(node, tir.AttrStmt) and hasattr(
                        node, 'attr_key') and "kernel" in str(node.attr_key).lower():
                    self.has_kernel_block = True

            stmt_functor.post_order_visit(stmt, visitor)

    checker = CoreLaunchChecker()
    checker.check(func.body)

    if checker.has_kernel_block:
        raise ValueError("Output still contains T.Kernel blocks")

    if not checker.has_launch_core:
        print("Warning: No T.launch_core calls found (may be OK if no kernel blocks)")

    return True


# Example usage and testing
if __name__ == "__main__":
    import tvm.script
    from tvm.script import tir as T

    # Create a test function with GPU-style grid
    @tvm.script.ir_module
    class TestModule:

        @T.prim_func
        def grid_kernel(A: T.Buffer((256, 256), "float16"), B: T.Buffer((256, 256), "float16"),
                        C: T.Buffer((256, 256), "float16")):
            # GPU-style grid kernel with block structure
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    # Process tile at (bx, by)
                    for i, j in T.grid(32, 32):
                        C[by * 32 + i,
                          bx * 32 + j] = A[by * 32 + i, bx * 32 + j] + B[by * 32 + i, bx * 32 + j]

    # Get function
    func = TestModule["grid_kernel"]

    # Add v5 metadata
    func = func.with_attr("tt.core_grid", [8, 8])
    func = func.with_attr("tt.partition_mode", "global")
    func = func.with_attr("tt.grid_tiles", [8, 8])
    func = func.with_attr("tt.runtime_args", ["start_id", "count", "Mt", "Nt"])
    func = func.with_attr(
        "tt.work_partition",
        {
            "core_0_0": [[0, 0]],
            "core_0_1": [[0, 1]],
            # ... simplified for example
        })

    # Apply transformation
    transformed = GridToCoreGrid_v5(func, TestModule, None)

    print("=== Original Function (GPU Grid) ===")
    print(func.script())
    print("\n=== Transformed Function (Core Launch) ===")
    print(transformed.script())
    print("\n=== New Attributes ===")
    for key, value in transformed.attrs.items():
        if key.startswith("tt."):
            print(f"  {key}: {value}")

    # Validate
    try:
        validate_core_launch(transformed)
        print("\n✅ Pass validation successful: Proper core launch structure")
    except ValueError as e:
        print(f"\n❌ Validation failed: {e}")


# Module-level wrapper function for compatibility with test imports
def grid_to_core_grid_v5(mod):
    """Apply GridToCoreGrid v5 pass to a module."""
    return GridToCoreGrid_v5(mod)
