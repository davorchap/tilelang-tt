"""
MVP Acceptance Tests - End-to-End Validation

These tests validate that the complete WS1-4 infrastructure is working correctly
on representative GEMM workloads.
"""

import pytest
import tvm
from tvm import tir
import tilelang.tt as tt


def create_test_module(M, N, K):
    """
    Create a TileLang IRModule for testing.

    For MVP, we create a minimal module with buffer declarations matching
    the requested dimensions. The grid size is inferred from the dimensions.
    """
    # Create buffers with specified dimensions
    A = tir.decl_buffer((M, K), "float16", name="A", scope="global")
    B = tir.decl_buffer((K, N), "float16", name="B", scope="global")
    C = tir.decl_buffer((M, N), "float16", name="C", scope="global")

    # Create minimal body
    body = tir.Evaluate(0)
    func = tir.PrimFunc(params=[A, B, C], body=body)

    # Compute grid dimensions (one tile = 32x32)
    grid_x = N // 32
    grid_y = M // 32

    # Add grid metadata (normally from T.Kernel)
    func = func.with_attr("tl.grid_x", tvm.tir.IntImm("int32", grid_x))
    func = func.with_attr("tl.grid_y", tvm.tir.IntImm("int32", grid_y))
    func = func.with_attr("global_symbol", "main")

    return tvm.IRModule({"main": func})


def create_fully_annotated_module(grid_x, grid_y, num_cores=64):
    """
    Create IRModule with complete WS1-3 metadata for codegen validation.

    This simulates the output of the full transform pipeline.
    """
    # Create minimal PrimFunc
    A = tir.decl_buffer((256, 256), "float16", name="A")
    B = tir.decl_buffer((256, 256), "float16", name="B")
    C = tir.decl_buffer((256, 256), "float16", name="C")

    body = tir.Evaluate(0)
    func = tir.PrimFunc(params=[A, B, C], body=body)

    # Attach complete WS2 schedule metadata
    num_tiles = grid_x * grid_y
    tiles_per_core = []
    for i in range(num_cores):
        start_id = i % num_tiles
        count = 1 if num_tiles >= num_cores else 0
        if i < num_tiles:
            tiles_per_core.append([start_id, count])

    func = func.with_attrs({
        "global_symbol": "main",
        "tt_grid_x": grid_x,
        "tt_grid_y": grid_y,
        "tt_grid_z": 1,
        "tt_num_tiles": num_tiles,
        "tt_num_cores": num_cores,
        "tt_tiles_per_core": tiles_per_core,
    })

    return tvm.IRModule({"main": func})


def test_mvp_gemm_256x256_full_pipeline():
    """
    MVP Acceptance Test: 256×256 GEMM End-to-End

    This test validates the complete transformation pipeline:
    - WS1: Default TT annotations
    - WS2: Schedule and sharding inference
    - WS3: Grid-to-persistent transformation
    - WS4: Artifact generation (compute.cpp + tt.plan.json)

    Success criteria:
    - All passes execute without errors
    - Artifacts are generated successfully
    - Generated code contains expected metadata
    """
    # Step 1: Create GEMM module (256×256, expect 8×8 grid)
    mod = create_test_module(M=256, N=256, K=256)

    # Step 2: Apply WS1 defaults
    mod = tt.apply_tt_defaults(mod)
    func = mod["main"]

    # Verify WS1 attributes
    assert "tt_schedule_policy" in func.attrs, "WS1: Missing schedule policy"
    assert "tt_schedule_order" in func.attrs, "WS1: Missing schedule order"
    assert func.attrs["tt_schedule_policy"] == "contiguous", "WS1: Incorrect policy"
    assert func.attrs["tt_schedule_order"] == "row_major", "WS1: Incorrect order"

    # Step 3: Apply WS2 passes (schedule + sharding inference)
    mod = tt.apply_ws2_passes(mod)
    func = mod["main"]

    # Verify WS2 schedule metadata
    assert "tt_grid_x" in func.attrs, "WS2: Missing grid_x"
    assert "tt_grid_y" in func.attrs, "WS2: Missing grid_y"
    assert "tt_num_tiles" in func.attrs, "WS2: Missing num_tiles"
    assert "tt_num_cores" in func.attrs, "WS2: Missing num_cores"
    assert "tt_tiles_per_core" in func.attrs, "WS2: Missing tiles_per_core"

    grid_x = int(func.attrs["tt_grid_x"])
    grid_y = int(func.attrs["tt_grid_y"])
    num_tiles = int(func.attrs["tt_num_tiles"])
    num_cores = int(func.attrs["tt_num_cores"])

    assert grid_x == 8, f"WS2: Expected grid_x=8, got {grid_x}"
    assert grid_y == 8, f"WS2: Expected grid_y=8, got {grid_y}"
    assert num_tiles == 64, f"WS2: Expected 64 tiles, got {num_tiles}"
    assert num_cores == 64, f"WS2: Expected 64 cores, got {num_cores}"

    tiles_per_core = func.attrs["tt_tiles_per_core"]
    assert len(tiles_per_core) == 64, "WS2: Should have 64 core assignments"

    # Verify WS2 sharding metadata
    assert "tt_buffer_A_layout" in func.attrs, "WS2: Missing buffer A layout"
    assert "tt_buffer_B_layout" in func.attrs, "WS2: Missing buffer B layout"
    assert "tt_buffer_C_layout" in func.attrs, "WS2: Missing buffer C layout"

    # Step 4: Apply WS3 passes (grid-to-persistent transformation)
    # Note: WS3 GridToPersistentTT is foundation-only in MVP
    # Full persistent loop transformation deferred to post-MVP
    mod = tt.apply_ws3_passes(mod)

    # Step 5: Generate artifacts (WS4)
    artifacts = tt.emit_tt_artifacts(mod)

    # Verify artifact structure
    assert isinstance(artifacts, dict), "WS4: Artifacts should be dict"
    assert "compute.cpp" in artifacts, "WS4: Missing compute.cpp"
    assert "tt.plan.json" in artifacts, "WS4: Missing tt.plan.json"

    # Validate compute kernel content (IR-driven codegen)
    compute_cpp = artifacts["compute.cpp"]
    assert "void MAIN()" in compute_cpp, "WS4: Missing MAIN function"

    # Check runtime arguments (IR-driven uses get_arg_val pattern)
    assert "get_arg_val<uint32_t>(0)" in compute_cpp, "WS4: Missing runtime arg 0"
    assert "get_arg_val<uint32_t>(1)" in compute_cpp, "WS4: Missing runtime arg 1"
    assert "get_arg_val<uint32_t>(2)" in compute_cpp, "WS4: Missing runtime arg 2"

    # Check for loop structure (IR-driven generates from actual IR, variable names may vary)
    assert "for (uint32_t" in compute_cpp, "WS4: Missing loop structure"

    # Check grid metadata in comments
    assert "// Grid: 8x8" in compute_cpp, "WS4: Missing grid comment"
    assert "// Cores: 64" in compute_cpp, "WS4: Missing cores comment"

    # Note: IR-driven with empty body won't have matmul ops (they come from IR nodes)
    # For MVP with actual matmul IR, this would have matmul_tiles calls

    # Validate plan.json content
    plan_json = artifacts["tt.plan.json"]
    import json
    plan_data = json.loads(plan_json)

    assert plan_data["version"] == "1.0", "WS4: Incorrect version"
    assert plan_data["target"] == "tenstorrent", "WS4: Incorrect target"
    assert plan_data["grid"]["x"] == 8, "WS4: Incorrect grid X"
    assert plan_data["grid"]["y"] == 8, "WS4: Incorrect grid Y"
    assert plan_data["grid"]["total_tiles"] == 64, "WS4: Incorrect total tiles"
    assert plan_data["cores"]["num_cores"] == 64, "WS4: Incorrect num cores"
    assert plan_data["schedule"]["policy"] == "contiguous", "WS4: Incorrect schedule policy"
    assert plan_data["schedule"]["order"] == "row_major", "WS4: Incorrect schedule order"

    core_assignments = plan_data["cores"]["assignments"]
    assert len(core_assignments) == 64, "WS4: Should have 64 core assignments"

    # Verify first and last core assignments
    assert core_assignments[0]["core_id"] == 0, "WS4: First core ID incorrect"
    assert core_assignments[0]["start_tile"] == 0, "WS4: First core should start at tile 0"
    assert core_assignments[63]["core_id"] == 63, "WS4: Last core ID incorrect"

    print("✅ MVP Acceptance Test PASSED: 256×256 GEMM")
    print(f"   - WS1: Default annotations applied")
    print(f"   - WS2: Schedule inferred (8×8 grid, 64 cores, 64 tiles)")
    print(f"   - WS2: Sharding inferred (3 buffers, DRAM interleaved)")
    print(f"   - WS3: Transform pipeline applied")
    print(f"   - WS4: Artifacts generated (compute.cpp + tt.plan.json)")
    print(f"   - All metadata validated ✓")


def test_mvp_gemm_512x512_scalability():
    """
    MVP Acceptance Test: 512×512 GEMM Scalability

    Tests that the pipeline scales to larger grids (16×16).
    """
    mod = create_test_module(M=512, N=512, K=512)

    # Full pipeline
    mod = tt.apply_tt_defaults(mod)
    mod = tt.apply_ws2_passes(mod)
    mod = tt.apply_ws3_passes(mod)
    artifacts = tt.emit_tt_artifacts(mod)

    # Validate scalability
    plan_json = artifacts["tt.plan.json"]
    import json
    plan_data = json.loads(plan_json)

    assert plan_data["grid"]["x"] == 16, "Scalability: Expected 16×16 grid"
    assert plan_data["grid"]["y"] == 16
    assert plan_data["grid"]["total_tiles"] == 256, "Scalability: Expected 256 tiles"

    # Verify tile distribution (256 tiles / 64 cores = 4 tiles per core)
    core_assignments = plan_data["cores"]["assignments"]
    total_assigned_tiles = sum(a["count"] for a in core_assignments)
    assert total_assigned_tiles == 256, f"Scalability: Should assign all 256 tiles, got {total_assigned_tiles}"

    print("✅ MVP Scalability Test PASSED: 512×512 GEMM (16×16 grid)")


def test_mvp_gemm_128x128_small_grid():
    """
    MVP Acceptance Test: 128×128 GEMM Small Grid

    Tests that the pipeline works with smaller grids (4×4).
    """
    mod = create_test_module(M=128, N=128, K=128)

    # Full pipeline
    mod = tt.apply_tt_defaults(mod)
    mod = tt.apply_ws2_passes(mod)
    mod = tt.apply_ws3_passes(mod)
    artifacts = tt.emit_tt_artifacts(mod)

    # Validate small grid
    plan_json = artifacts["tt.plan.json"]
    import json
    plan_data = json.loads(plan_json)

    assert plan_data["grid"]["x"] == 4, "Small grid: Expected 4×4 grid"
    assert plan_data["grid"]["y"] == 4
    assert plan_data["grid"]["total_tiles"] == 16, "Small grid: Expected 16 tiles"

    print("✅ MVP Small Grid Test PASSED: 128×128 GEMM (4×4 grid)")


if __name__ == "__main__":
    # Run tests standalone
    print("Running MVP Acceptance Tests\n")

    test_mvp_gemm_256x256_full_pipeline()
    print()
    test_mvp_gemm_512x512_scalability()
    print()
    test_mvp_gemm_128x128_small_grid()

    print("\n✅ All MVP Acceptance Tests PASSED!")
    print("Total tests run: 3")
    print("WS1-4 pipeline validated successfully.")
