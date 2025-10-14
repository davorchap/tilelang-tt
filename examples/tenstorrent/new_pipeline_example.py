#!/usr/bin/env python3
"""
Example demonstrating the new Tenstorrent lowering pipeline.

This shows how to use the new metadata-driven approach to compile
a simple matrix multiplication for Tenstorrent hardware.
"""

import os
from pathlib import Path

try:
    import tvm
    from tvm.script import tir as T
except ImportError:
    print("TVM not found. Please install TVM first.")
    exit(1)

# Import the new Tenstorrent utilities
from tilelang.tenstorrent import (
    CoreRange,
    WorkItem,
    with_core_grid,
    with_core_ranges,
    with_work_partition,
    with_layout_desc,
    run_pipeline,
    load_tt_plan,
    validate_plan,
)


def create_matmul_ir(M: int = 128, N: int = 128, K: int = 128):
    """Create a simple matrix multiplication PrimFunc."""

    @T.prim_func
    def matmul(
            A: T.Buffer((M, K), "float16"),
            B: T.Buffer((K, N), "float16"),
            C: T.Buffer((M, N), "float32"),
    ):
        # Grid iteration (will be transformed to persistent loops)
        for i in T.grid(M):
            for j in T.grid(N):
                with T.block("C"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    T.reads(A[vi, 0:K], B[0:K, vj])
                    T.writes(C[vi, vj])

                    # Initialize accumulator
                    C[vi, vj] = T.float32(0)

                    # Compute
                    for k in range(K):
                        C[vi,
                          vj] = C[vi,
                                  vj] + T.cast(A[vi, k], "float32") * T.cast(B[k, vj], "float32")

    return matmul


def example_basic_pipeline():
    """Basic example: compile a matmul with default settings."""
    print("=" * 60)
    print("Basic Pipeline Example")
    print("=" * 60)

    # Create the IR
    matmul = create_matmul_ir(128, 128, 128)

    # Attach metadata using the new API
    matmul = with_core_grid(matmul, 4, 4)  # 4x4 core grid
    matmul = with_core_ranges(matmul, [CoreRange((0, 0), (4, 4))])  # Use all cores
    matmul = with_layout_desc(
        matmul, {
            "A": {
                "shard": "DRAM",
                "interleave": True
            },
            "B": {
                "shard": "DRAM",
                "interleave": True
            },
            "C": {
                "shard": "L1",
                "tile_id_order": "row_major"
            },
        })

    # Create module and run pipeline
    mod = tvm.IRModule({"main": matmul})

    # Run the pipeline
    output_plan = "matmul_basic.plan.json"
    result_mod = run_pipeline(mod, plan_path=output_plan)

    # Load and validate the generated plan
    if Path(output_plan).exists():
        plan = load_tt_plan(output_plan)
        print(f"\nGenerated plan saved to: {output_plan}")
        print(f"Core grid: {plan['core_grid']}")
        print(f"Number of core ranges: {len(plan['core_ranges'])}")
        print(f"Work items assigned: {sum(len(v) for v in plan['work_partition'].values())}")

        # Validate the plan
        errors = validate_plan(plan)
        if errors:
            print("\nPlan validation errors:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("\n✅ Plan validation passed!")


def example_custom_partition():
    """Advanced example: custom work partitioning."""
    print("\n" + "=" * 60)
    print("Custom Partition Example")
    print("=" * 60)

    # Create the IR
    matmul = create_matmul_ir(256, 256, 128)

    # Custom work partition: assign specific tiles to specific cores
    work_partition = {}
    tile_size = 32
    m_tiles = 256 // tile_size  # 8 tiles
    n_tiles = 256 // tile_size  # 8 tiles

    # Distribute tiles in a custom pattern
    for cx in range(4):
        for cy in range(4):
            core_key = f"({cx},{cy})"
            work_items = []

            # Each core gets 2 tiles in a diagonal pattern
            base_i = (cx * 2) % m_tiles
            base_j = (cy * 2) % n_tiles

            work_items.append(WorkItem(io=base_i, jo=base_j, len_k=128))
            work_items.append(
                WorkItem(io=(base_i + 1) % m_tiles, jo=(base_j + 1) % n_tiles, len_k=128))

            work_partition[core_key] = work_items

    # Attach metadata
    matmul = with_core_grid(matmul, 4, 4)
    matmul = with_core_ranges(matmul, [CoreRange((0, 0), (4, 4))])
    matmul = with_work_partition(matmul, work_partition)
    matmul = with_layout_desc(
        matmul,
        {
            "A": {
                "shard": "DRAM",
                "interleave": False,
                "stride": [256, 1]
            },
            "B": {
                "shard": "DRAM",
                "interleave": True
            },
            "C": {
                "shard": "L1",
                "tile_id_order": "z_order"
            },  # Z-order for cache efficiency
        })

    # Create module and run pipeline
    mod = tvm.IRModule({"main": matmul})

    # Run with custom settings
    output_plan = "matmul_custom.plan.json"
    result_mod = run_pipeline(
        mod,
        plan_path=output_plan,
        target_device="wormhole",  # Target Wormhole device
        enable_double_buffer=True,
        enable_prefetch=True,
    )

    # Display the custom partition
    if Path(output_plan).exists():
        plan = load_tt_plan(output_plan)
        print(f"\nCustom plan saved to: {output_plan}")
        print("\nWork distribution:")
        for core, items in list(plan['work_partition'].items())[:4]:  # Show first 4 cores
            print(f"  {core}: {len(items)} tiles")
            for item in items[:2]:  # Show first 2 items
                print(f"    - Tile ({item['io']}, {item['jo']})")


def example_multi_range():
    """Example with multiple disjoint core ranges."""
    print("\n" + "=" * 60)
    print("Multi-Range Example")
    print("=" * 60)

    # Create the IR
    matmul = create_matmul_ir(128, 128, 64)

    # Define multiple core ranges (e.g., for multi-chip or partitioned execution)
    core_ranges = [
        CoreRange((0, 0), (2, 2)),  # Top-left quadrant
        CoreRange((2, 2), (2, 2)),  # Bottom-right quadrant
    ]

    # Attach metadata
    matmul = with_core_grid(matmul, 4, 4)
    matmul = with_core_ranges(matmul, core_ranges)
    matmul = with_layout_desc(matmul, {
        "A": {
            "shard": "DRAM"
        },
        "B": {
            "shard": "DRAM"
        },
        "C": {
            "shard": "L1"
        },
    })

    # Create module and run pipeline
    mod = tvm.IRModule({"main": matmul})

    output_plan = "matmul_multirange.plan.json"
    result_mod = run_pipeline(mod, plan_path=output_plan, partition_strategy="block")

    if Path(output_plan).exists():
        plan = load_tt_plan(output_plan)
        print(f"\nMulti-range plan saved to: {output_plan}")
        print("Core ranges:")
        for i, cr in enumerate(plan['core_ranges']):
            print(f"  Range {i}: start={cr['start']}, extent={cr['extent']}")


def main():
    """Run all examples."""
    print("Tenstorrent New Pipeline Examples")
    print("=" * 60)

    # Create output directory
    os.makedirs("tt_output", exist_ok=True)
    os.chdir("tt_output")

    try:
        # Run examples
        example_basic_pipeline()
        example_custom_partition()
        example_multi_range()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print(f"Output files saved in: {os.getcwd()}")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
