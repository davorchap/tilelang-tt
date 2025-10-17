"""
Test fixtures and helpers for generating complete IR for Tenstorrent tests.

This module provides utilities to create TVM IRModules with complete IR
that can be successfully processed by the v5 codegen pipeline.
"""

import tvm
from tvm import tir
from tvm.script import tir as T


def create_complete_ir_module_with_split_kernels(grid_x=8, grid_y=8, num_cores=64):
    """
    Create a TVM IRModule with split reader/compute/writer kernels with complete IR.

    This simulates the output after the v5 pipeline's split_device_kernel pass.
    """

    # Create reader kernel with actual NOC/CB operations
    @T.prim_func
    def reader_kernel(
        A_handle: T.handle,
        B_handle: T.handle,
        tt_start_tile: T.int32,
        tt_tile_count: T.int32,
    ):
        T.func_attr({"global_symbol": "reader_kernel", "tt.kernel_role": "reader"})

        # Read loop for input A - using proper integer cb index
        for i in T.serial(tt_tile_count):
            # Reserve space in CB
            T.evaluate(T.call_extern("void", "cb_reserve_back", T.int32(0), T.int32(1)))
            # Get write pointer
            T.evaluate(T.call_extern("int32", "get_write_ptr", T.int32(0)))
            # NOC read
            T.evaluate(T.call_extern("void", "noc_async_read_tile",
                                     tt_start_tile + i, T.int32(0)))
            # Wait for read completion
            T.evaluate(T.call_extern("void", "noc_async_read_barrier"))
            # Push to CB
            T.evaluate(T.call_extern("void", "cb_push_back", T.int32(0), T.int32(1)))

        # Read loop for input B
        for i in T.serial(tt_tile_count):
            # Reserve space in CB
            T.evaluate(T.call_extern("void", "cb_reserve_back", T.int32(1), T.int32(1)))
            # Get write pointer
            T.evaluate(T.call_extern("int32", "get_write_ptr", T.int32(1)))
            # NOC read
            T.evaluate(T.call_extern("void", "noc_async_read_tile",
                                     tt_start_tile + i, T.int32(1)))
            # Wait for read completion
            T.evaluate(T.call_extern("void", "noc_async_read_barrier"))
            # Push to CB
            T.evaluate(T.call_extern("void", "cb_push_back", T.int32(1), T.int32(1)))

    @T.prim_func
    def compute_kernel(
        tt_start_tile: T.int32,
        tt_tile_count: T.int32,
    ):
        T.func_attr({"global_symbol": "compute_kernel", "tt.kernel_role": "compute"})

        # Initialize compute
        T.evaluate(T.call_extern("void", "mm_init"))

        # Main compute loop - using proper integer cb indices
        for i in T.serial(tt_tile_count):
            # Wait for input tiles
            T.evaluate(T.call_extern("void", "cb_wait_front", T.int32(0), T.int32(1)))
            T.evaluate(T.call_extern("void", "cb_wait_front", T.int32(1), T.int32(1)))

            # Acquire DST registers
            T.evaluate(T.call_extern("void", "tile_regs_acquire"))

            # Perform matmul - cb_in0=0, cb_in1=1
            T.evaluate(T.call_extern("void", "matmul_tiles", T.int32(0), T.int32(1), T.int32(0), T.int32(0), T.int32(0)))

            # Commit DST registers
            T.evaluate(T.call_extern("void", "tile_regs_commit"))

            # Pop input tiles
            T.evaluate(T.call_extern("void", "cb_pop_front", T.int32(0), T.int32(1)))
            T.evaluate(T.call_extern("void", "cb_pop_front", T.int32(1), T.int32(1)))

            # Reserve output space - cb_out0=16
            T.evaluate(T.call_extern("void", "cb_reserve_back", T.int32(16), T.int32(1)))

            # Pack output
            T.evaluate(T.call_extern("void", "pack_tile", T.int32(0), T.int32(16)))

            # Push output
            T.evaluate(T.call_extern("void", "cb_push_back", T.int32(16), T.int32(1)))

    @T.prim_func
    def writer_kernel(
        C_handle: T.handle,
        tt_start_tile: T.int32,
        tt_tile_count: T.int32,
    ):
        T.func_attr({"global_symbol": "writer_kernel", "tt.kernel_role": "writer"})

        # Write loop - cb_out0=16
        for i in T.serial(tt_tile_count):
            # Wait for output tile
            T.evaluate(T.call_extern("void", "cb_wait_front", T.int32(16), T.int32(1)))
            # Get read pointer
            T.evaluate(T.call_extern("int32", "get_read_ptr", T.int32(16)))
            # NOC write
            T.evaluate(T.call_extern("void", "noc_async_write_tile",
                                     T.int32(16), tt_start_tile + i))
            # Wait for write completion
            T.evaluate(T.call_extern("void", "noc_async_write_barrier"))
            # Pop output tile
            T.evaluate(T.call_extern("void", "cb_pop_front", T.int32(16), T.int32(1)))

    # Add metadata to each kernel
    num_tiles = grid_x * grid_y
    tiles_per_core = []
    for i in range(num_cores):
        start_id = i % num_tiles
        count = 1
        tiles_per_core.append([tvm.tir.IntImm("int32", start_id), tvm.tir.IntImm("int32", count)])

    runtime_args = ["tt_start_tile", "tt_tile_count", "A_addr", "B_addr", "C_addr"]
    runtime_args_info = {
        "types": {
            "tt_start_tile": "uint32_t",
            "tt_tile_count": "uint32_t",
            "A_addr": "uint64_t",
            "B_addr": "uint64_t",
            "C_addr": "uint64_t"
        }
    }

    attrs = {
        "tt_grid_x": tvm.tir.IntImm("int32", grid_x),
        "tt_grid_y": tvm.tir.IntImm("int32", grid_y),
        "tt_grid_z": tvm.tir.IntImm("int32", 1),
        "tt_num_tiles": tvm.tir.IntImm("int32", num_tiles),
        "tt_num_cores": tvm.tir.IntImm("int32", num_cores),
        "tt_tiles_per_core": tiles_per_core,
        "tt.runtime_args": runtime_args,
        "tt.runtime_args_info": runtime_args_info,
    }

    reader_kernel = reader_kernel.with_attrs(attrs)
    compute_kernel = compute_kernel.with_attrs(attrs)
    writer_kernel = writer_kernel.with_attrs(attrs)

    # Create IRModule with all three kernels
    mod = tvm.IRModule({
        "reader_kernel": reader_kernel,
        "compute_kernel": compute_kernel,
        "writer_kernel": writer_kernel,
    })

    return mod


def create_complete_ir_module(grid_x=8, grid_y=8, num_cores=64):
    """
    Create a TVM IRModule with a single main kernel with complete IR.

    This includes realistic TT operations that can be processed by codegen.
    """

    # Create buffers
    A = tir.decl_buffer((256, 256), "float16", name="A")
    B = tir.decl_buffer((256, 256), "float16", name="B")
    C = tir.decl_buffer((256, 256), "float16", name="C")

    # Create body with actual TT operations
    body_stmts = []

    # Add compute initialization
    body_stmts.append(
        tir.Evaluate(tir.call_extern("void", "mm_init"))
    )

    # Create a simple loop with TT operations
    loop_var = tir.Var("i", "int32")
    loop_body = tir.SeqStmt([
        # CB wait operations - using integer indices
        tir.Evaluate(tir.call_extern("void", "cb_wait_front", tir.IntImm("int32", 0), tir.IntImm("int32", 1))),
        tir.Evaluate(tir.call_extern("void", "cb_wait_front", tir.IntImm("int32", 1), tir.IntImm("int32", 1))),

        # DST management
        tir.Evaluate(tir.call_extern("void", "tile_regs_acquire")),

        # Matmul operation
        tir.Evaluate(tir.call_extern("void", "matmul_tiles",
                                      tir.IntImm("int32", 0), tir.IntImm("int32", 1),
                                      tir.IntImm("int32", 0), tir.IntImm("int32", 0),
                                      tir.IntImm("int32", 0))),

        # DST commit
        tir.Evaluate(tir.call_extern("void", "tile_regs_commit")),

        # CB pop operations
        tir.Evaluate(tir.call_extern("void", "cb_pop_front", tir.IntImm("int32", 0), tir.IntImm("int32", 1))),
        tir.Evaluate(tir.call_extern("void", "cb_pop_front", tir.IntImm("int32", 1), tir.IntImm("int32", 1))),

        # Output operations
        tir.Evaluate(tir.call_extern("void", "cb_reserve_back", tir.IntImm("int32", 16), tir.IntImm("int32", 1))),
        tir.Evaluate(tir.call_extern("void", "pack_tile", tir.IntImm("int32", 0), tir.IntImm("int32", 16))),
        tir.Evaluate(tir.call_extern("void", "cb_push_back", tir.IntImm("int32", 16), tir.IntImm("int32", 1))),
    ])

    # Create for loop
    loop = tir.For(
        loop_var=loop_var,
        min_val=tir.IntImm("int32", 0),
        extent=tir.IntImm("int32", 8),  # Simple fixed loop count
        kind=tir.ForKind.SERIAL,
        body=loop_body
    )
    body_stmts.append(loop)

    # Create complete body
    body = tir.SeqStmt(body_stmts)

    func = tir.PrimFunc(
        params=[A, B, C],
        body=body,
    )

    # Attach metadata
    num_tiles = grid_x * grid_y
    tiles_per_core = []
    for i in range(num_cores):
        start_id = i % num_tiles
        count = 1
        tiles_per_core.append([tvm.tir.IntImm("int32", start_id), tvm.tir.IntImm("int32", count)])

    runtime_args = ["tt_start_tile", "tt_tile_count", "A_addr", "B_addr", "C_addr"]
    runtime_args_info = {
        "types": {
            "tt_start_tile": "uint32_t",
            "tt_tile_count": "uint32_t",
            "A_addr": "uint64_t",
            "B_addr": "uint64_t",
            "C_addr": "uint64_t"
        }
    }

    func = func.with_attrs({
        "global_symbol": "main",
        "tt_grid_x": tvm.tir.IntImm("int32", grid_x),
        "tt_grid_y": tvm.tir.IntImm("int32", grid_y),
        "tt_grid_z": tvm.tir.IntImm("int32", 1),
        "tt_num_tiles": tvm.tir.IntImm("int32", num_tiles),
        "tt_num_cores": tvm.tir.IntImm("int32", num_cores),
        "tt_tiles_per_core": tiles_per_core,
        "tt.runtime_args": runtime_args,
        "tt.runtime_args_info": runtime_args_info,
    })

    # Create IRModule
    mod = tvm.IRModule({"main": func})
    return mod


def create_empty_ir_module_for_fail_test(grid_x=8, grid_y=8, num_cores=64):
    """
    Create a TVM IRModule with empty body to test fail-loud behavior.

    This should cause codegen to raise ValueError.
    """
    A = tir.decl_buffer((256, 256), "float16", name="A")
    B = tir.decl_buffer((256, 256), "float16", name="B")
    C = tir.decl_buffer((256, 256), "float16", name="C")

    # Empty body - should trigger fail-loud codegen
    body = tir.Evaluate(0)

    func = tir.PrimFunc(
        params=[A, B, C],
        body=body,
    )

    # Attach metadata
    num_tiles = grid_x * grid_y
    tiles_per_core = []
    for i in range(num_cores):
        start_id = i % num_tiles
        count = 1
        tiles_per_core.append([tvm.tir.IntImm("int32", start_id), tvm.tir.IntImm("int32", count)])

    runtime_args = ["tt_start_tile", "tt_tile_count", "A_addr", "B_addr", "C_addr"]
    runtime_args_info = {
        "types": {
            "tt_start_tile": "uint32_t",
            "tt_tile_count": "uint32_t",
            "A_addr": "uint64_t",
            "B_addr": "uint64_t",
            "C_addr": "uint64_t"
        }
    }

    func = func.with_attrs({
        "global_symbol": "main",
        "tt_grid_x": tvm.tir.IntImm("int32", grid_x),
        "tt_grid_y": tvm.tir.IntImm("int32", grid_y),
        "tt_grid_z": tvm.tir.IntImm("int32", 1),
        "tt_num_tiles": tvm.tir.IntImm("int32", num_tiles),
        "tt_num_cores": tvm.tir.IntImm("int32", num_cores),
        "tt_tiles_per_core": tiles_per_core,
        "tt.runtime_args": runtime_args,
        "tt.runtime_args_info": runtime_args_info,
    })

    # Create IRModule
    mod = tvm.IRModule({"main": func})
    return mod