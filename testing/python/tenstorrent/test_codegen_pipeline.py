"""
artifact generation stage: Code Generation Integration Tests

Tests for TT kernel codegen and artifact generation.
"""

import tvm
from tvm import tir
import tilelang.tenstorrent as tt


def create_tt_module_with_metadata(grid_x=8, grid_y=8, num_cores=64):
    """
    Create a minimal TVM IRModule with metadata inference stage/persistent transform stage metadata attached for codegen testing.

    This simulates the output of TT defaults stage-3 pipeline without needing actual kernel code.
    """
    # Create minimal PrimFunc
    A = tir.decl_buffer((256, 256), "float16", name="A")
    B = tir.decl_buffer((256, 256), "float16", name="B")
    C = tir.decl_buffer((256, 256), "float16", name="C")

    # Empty body (codegen doesn't need actual operations for these tests)
    body = tir.Evaluate(0)

    func = tir.PrimFunc(
        params=[A, B, C],
        body=body,
    )

    # Attach metadata inference stage schedule metadata
    # Note: Convert Python ints to IntImm for FFI compatibility
    num_tiles = grid_x * grid_y
    tiles_per_core = []
    for i in range(num_cores):
        start_id = i % num_tiles  # Simplified assignment
        count = 1
        # Convert list elements to IntImm for FFI
        tiles_per_core.append([tvm.tir.IntImm("int32", start_id), tvm.tir.IntImm("int32", count)])

    func = func.with_attrs({
        "global_symbol": "main",
        "tt_grid_x": tvm.tir.IntImm("int32", grid_x),
        "tt_grid_y": tvm.tir.IntImm("int32", grid_y),
        "tt_grid_z": tvm.tir.IntImm("int32", 1),
        "tt_num_tiles": tvm.tir.IntImm("int32", num_tiles),
        "tt_num_cores": tvm.tir.IntImm("int32", num_cores),
        "tt_tiles_per_core": tiles_per_core,
    })

    # Create IRModule
    mod = tvm.IRModule({"main": func})
    return mod


def test_emit_tt_artifacts_basic():
    """
    Test 1: Basic artifact generation

    Verifies that emit_tt_artifacts returns the expected artifact files.
    """
    # Create module with metadata inference stage/persistent transform stage metadata
    mod = create_tt_module_with_metadata(grid_x=8, grid_y=8)

    # Generate artifacts
    artifacts = tt.emit_tt_artifacts(mod)

    # Verify artifact structure
    assert isinstance(artifacts, dict), "artifacts should be a dictionary"
    assert "compute.cpp" in artifacts, "compute.cpp artifact missing"
    assert "tt.plan.json" in artifacts, "tt.plan.json artifact missing"

    # Verify compute kernel contains expected elements (IR-driven codegen)
    compute_cpp = artifacts["compute.cpp"]

    # Check IR-driven marker
    assert "// Generated TT Compute Kernel (IR-Driven)" in compute_cpp, "IR-driven marker missing"

    # Check structure
    assert "void MAIN()" in compute_cpp, "MAIN function missing"

    # Check runtime arguments (IR-driven uses get_arg_val pattern)
    assert "get_arg_val<uint32_t>(0)" in compute_cpp, "runtime arg 0 missing"
    assert "get_arg_val<uint32_t>(1)" in compute_cpp, "runtime arg 1 missing"
    # Note: Runtime arg indices depend on metadata; just verify some args are present
    assert "tt_start_tile" in compute_cpp, "tt_start_tile missing"
    assert "tt_tile_count" in compute_cpp, "tt_tile_count missing"

    # Check for basic structure (empty body means no actual operations)
    # Just verify the kernel compiled successfully
    assert "void MAIN()" in compute_cpp, "MAIN function structure invalid"

    # Verify plan JSON contains expected elements
    plan_json = artifacts["tt.plan.json"]
    assert '"version"' in plan_json, "version missing from plan.json"
    assert '"target": "tenstorrent"' in plan_json, "target missing from plan.json"
    assert '"grid"' in plan_json, "grid section missing"
    assert '"cores"' in plan_json, "cores section missing"
    assert '"schedule"' in plan_json, "schedule section missing"

    print("✓ Test 1 passed: Basic artifact generation")


def test_emit_tt_artifacts_grid_metadata():
    """
    Test 2: Grid metadata in generated code

    Verifies that grid dimensions from metadata inference stage appear correctly in generated artifacts.
    """
    mod = create_tt_module_with_metadata(grid_x=8, grid_y=8)

    artifacts = tt.emit_tt_artifacts(mod)

    # Check compute kernel header (IR-driven includes grid metadata in comments)
    compute_cpp = artifacts["compute.cpp"]
    assert "// Grid: 8x8" in compute_cpp, "Grid dimensions missing from kernel header"
    assert "// Cores: 64" in compute_cpp, "Core count missing from kernel header"

    # Check plan JSON grid section
    plan_json = artifacts["tt.plan.json"]
    import json
    plan_data = json.loads(plan_json)

    assert plan_data["grid"]["x"] == 8, "Grid X dimension incorrect"
    assert plan_data["grid"]["y"] == 8, "Grid Y dimension incorrect"
    assert plan_data["grid"]["total_tiles"] == 64, "Total tiles incorrect"
    assert plan_data["cores"]["num_cores"] == 64, "Num cores incorrect"

    print("✓ Test 2 passed: Grid metadata verification")


def test_emit_tt_artifacts_scheduling_metadata():
    """
    Test 3: Scheduling metadata in plan.json

    Verifies that metadata inference stage schedule metadata appears in the plan.json artifact.
    """
    mod = create_tt_module_with_metadata(grid_x=8, grid_y=8)

    artifacts = tt.emit_tt_artifacts(mod)
    plan_json = artifacts["tt.plan.json"]

    import json
    plan_data = json.loads(plan_json)

    # Verify schedule section
    assert "schedule" in plan_data, "Schedule section missing"
    assert plan_data["schedule"]["policy"] == "contiguous", "Schedule policy incorrect"
    assert plan_data["schedule"]["order"] == "row_major", "Schedule order incorrect"

    # Verify core assignments section
    assert "assignments" in plan_data["cores"], "Core assignments missing"
    assignments = plan_data["cores"]["assignments"]
    assert len(assignments) == 64, "Should have 64 core assignments"

    # Verify assignment structure
    first_assignment = assignments[0]
    assert "core_id" in first_assignment, "core_id missing from assignment"
    assert "start_tile" in first_assignment, "start_tile missing from assignment"
    assert "count" in first_assignment, "count missing from assignment"

    # Verify contiguous assignment: first core gets tile 0
    assert first_assignment["start_tile"] == 0, "First core should start at tile 0"
    assert first_assignment["count"] == 1, "Each core should get 1 tile for 64-tile grid"

    print("✓ Test 3 passed: Scheduling metadata verification")


def test_emit_tt_artifacts_various_grid_sizes():
    """
    Test 4: Different grid sizes

    Verifies codegen works correctly for various grid dimensions.
    """
    test_cases = [
        (4, 4),  # 4x4 grid = 16 tiles
        (16, 16),  # 16x16 grid = 256 tiles
    ]

    for grid_x, grid_y in test_cases:
        mod = create_tt_module_with_metadata(grid_x=grid_x, grid_y=grid_y)

        artifacts = tt.emit_tt_artifacts(mod)

        # Verify artifacts exist
        assert "compute.cpp" in artifacts
        assert "tt.plan.json" in artifacts

        # Parse plan.json
        import json
        plan_data = json.loads(artifacts["tt.plan.json"])

        # Verify grid dimensions match
        expected_tiles = grid_x * grid_y

        assert plan_data["grid"]["x"] == grid_x
        assert plan_data["grid"]["y"] == grid_y
        assert plan_data["grid"]["total_tiles"] == expected_tiles

        print(f"✓ Test 4 passed for grid {grid_x}x{grid_y} ({expected_tiles} tiles)")


def test_write_artifacts_to_disk(tmp_path):
    """
    Test 5: Writing artifacts to disk

    Verifies that write_artifacts_to_disk creates the expected files.
    """
    mod = create_tt_module_with_metadata(grid_x=8, grid_y=8)

    artifacts = tt.emit_tt_artifacts(mod)

    # Write to temporary directory
    output_dir = str(tmp_path / "tt_kernels")
    tt.write_artifacts_to_disk(artifacts, output_dir=output_dir)

    # Verify files exist
    import os
    compute_path = os.path.join(output_dir, "compute.cpp")
    plan_path = os.path.join(output_dir, "tt.plan.json")

    assert os.path.exists(compute_path), "compute.cpp not written"
    assert os.path.exists(plan_path), "tt.plan.json not written"

    # Verify file contents match
    with open(compute_path, "r") as f:
        written_compute = f.read()
    assert written_compute == artifacts["compute.cpp"]

    with open(plan_path, "r") as f:
        written_plan = f.read()
    assert written_plan == artifacts["tt.plan.json"]

    print("✓ Test 5 passed: Writing artifacts to disk")


if __name__ == "__main__":
    # Run tests
    print("Running artifact generation stage Code Generation Tests\n")

    test_emit_tt_artifacts_basic()
    test_emit_tt_artifacts_grid_metadata()
    test_emit_tt_artifacts_scheduling_metadata()
    test_emit_tt_artifacts_various_grid_sizes()

    # Skip tmp_path test when running standalone
    print("\n✅ All artifact generation stage codegen tests passed!")
