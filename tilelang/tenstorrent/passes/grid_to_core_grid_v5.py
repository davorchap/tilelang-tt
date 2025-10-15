"""
Pass B2: GridToCoreGrid (New Metadata Format)
Version: 5.0
Date: 2025-10-15

Purpose: Convert GPU-style grid kernel to Tenstorrent core launch model
         using the new metadata format from v5 design.

Input: TIR with T.Kernel(gx, gy) grid structure and new metadata
Output: TIR with T.launch_core persistent model
"""

import tvm
from tvm import tir
from tvm.tir import stmt_functor


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

    class GridTransformer:

        def __init__(self, metadata):
            self.metadata = metadata
            self.core_grid = metadata.get("core_grid", [8, 8])
            self.partition_mode = metadata.get("partition_mode", "global")
            self.grid_tiles = metadata.get("grid_tiles", [])
            self.work_partition = metadata.get("work_partition", {})
            self.runtime_args = metadata.get("runtime_args", [])


        def visit(self, stmt):
            """Generic visit method that dispatches to specific visit methods"""
            if stmt is None:
                return None

            # Dispatch to specific visit methods based on node type
            if isinstance(stmt, tir.For):
                return self.visit_for(stmt)
            elif isinstance(stmt, tir.BufferStore):
                return self.visit_buffer_store(stmt)
            elif isinstance(stmt, tir.BufferLoad):
                return self.visit_buffer_load(stmt)
            elif isinstance(stmt, tir.SeqStmt):
                new_seq = []
                for s in stmt.seq:
                    new_s = self.visit(s)
                    if new_s is not None:
                        new_seq.append(new_s)
                return tir.SeqStmt(new_seq) if new_seq else None
            elif hasattr(stmt, "body"):
                new_body = self.visit(stmt.body)
                if new_body != stmt.body:
                    return stmt.with_body(new_body) if hasattr(stmt, 'with_body') else stmt
                return stmt
            else:
                return stmt

            # Track transformation state
            self.in_kernel = False
            self.kernel_vars = {}

        def visit_attr(self, op):
            """Handle T.Kernel attribute nodes"""

            # Check if this is a T.Kernel block
            if self._is_kernel_attr(op):
                # Extract grid dimensions from the kernel
                grid_dims = self._extract_grid_dims(op)

                # Transform to core launch
                transformed_body = self._transform_to_core_launch(op.body, grid_dims)

                # Return the transformed body directly (remove T.Kernel wrapper)
                return self.visit(transformed_body)

            # Continue visiting (was super().visit_attr)
            return self.visit(op.body) if hasattr(op, "body") else op

        def visit_for(self, op):
            """Transform grid loops to core launch"""

            # Check if this is a grid loop from T.Kernel
            if self._is_grid_loop(op):
                # Replace with core launch
                return self._create_core_launch(op)

            
            return self.visit(op.body) if hasattr(op, "body") else op

        def _is_kernel_attr(self, attr_node):
            """Check if this is a T.Kernel attribute"""
            # T.Kernel creates an AttrStmt with specific patterns
            if hasattr(attr_node, 'attr_key'):
                return attr_node.attr_key == "pragma_kernel" or \
                       str(attr_node.attr_key) == "thread_extent"
            return False

        def _is_grid_loop(self, for_node):
            """Check if this is a grid dimension loop"""
            # Look for blockIdx.x, blockIdx.y patterns
            loop_var = for_node.loop_var.name

            # Common patterns for grid loops
            grid_patterns = ["blockIdx", "bx", "by", "bz", "gridIdx"]
            return any(pattern in loop_var for pattern in grid_patterns)

        def _extract_grid_dims(self, kernel_attr):
            """Extract grid dimensions from T.Kernel"""
            # T.Kernel typically stores grid size in the value
            if hasattr(kernel_attr, 'value'):
                # Parse grid dimensions
                # This depends on how T.Kernel encodes the grid
                return [8, 8]  # Default for now

            return self.core_grid

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

            # Determine dimension (x or y)
            if "x" in loop_var.name or "0" in loop_var.name:
                dim = "x"
                core_count = self.core_grid[0]
            else:
                dim = "y"
                core_count = self.core_grid[1]

            # Create launch_core call
            launch_call = tir.call_extern("int32", "T.launch_core", tir.StringImm(f"coreIdx.{dim}"),
                                          tir.IntImm("int32", core_count))

            # Create variable to hold core index
            core_var = tir.Var(f"c{dim}", "int32")
            let_stmt = tir.LetStmt(core_var, launch_call, self.visit(for_node.body))

            # Store mapping for later use
            self.kernel_vars[loop_var] = core_var

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

            stmts.append(tir.LetStmt(start_var, start_id, tir.Evaluate(0)))
            stmts.append(tir.LetStmt(count_var, count, tir.Evaluate(0)))

            # Add tile dimension access
            if "Mt" in self.runtime_args:
                mt_idx = self.runtime_args.index("Mt")
                mt = tir.call_extern("int32", "get_arg_val", tir.IntImm("int32", mt_idx))
                stmts.append(tir.LetStmt(tir.Var("Mt", "int32"), mt, tir.Evaluate(0)))

            if "Nt" in self.runtime_args:
                nt_idx = self.runtime_args.index("Nt")
                nt = tir.call_extern("int32", "get_arg_val", tir.IntImm("int32", nt_idx))
                stmts.append(tir.LetStmt(tir.Var("Nt", "int32"), nt, tir.Evaluate(0)))

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
                    stmts.append(tir.LetStmt(tir.Var(arg_name, "int32"), value, tir.Evaluate(0)))

            return stmts

        def _replace_grid_indices(self, body):
            """Replace grid indices with core-based computation"""

            class IndexReplacer:

                def __init__(self, kernel_vars):
                    self.kernel_vars = kernel_vars

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
                    return tir.BufferLoad(op.buffer, new_indices, op.span)

                def visit_buffer_store(self, op):
                    # Transform buffer indices for core-based access
                    new_indices = []
                    for idx in op.indices:
                        new_idx = self.visit(idx)
                        new_indices.append(new_idx)
                    new_value = self.visit(op.value)
                    return tir.BufferStore(op.buffer, new_value, new_indices, op.span)

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

        def visit_evaluate(self, op):
            if isinstance(op.value, tir.Call):
                call_name = str(op.value.op)
                if "launch_core" in call_name:
                    self.has_launch_core = True
                    self.launch_calls.append(call_name)
            # Continue visiting evaluate

        def visit_attr(self, op):
            if hasattr(op, 'attr_key'):
                if "kernel" in str(op.attr_key).lower():
                    self.has_kernel_block = True
            # Continue visiting attr

    checker = CoreLaunchChecker()
    checker.visit(func.body)

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
            # GPU-style grid kernel
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
