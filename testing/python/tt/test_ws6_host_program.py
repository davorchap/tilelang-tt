"""
host program stage: Host Program Generation Integration Tests

Tests for TT host program generation, including device setup, circular buffer
configuration, DRAM buffer management, and kernel launch orchestration.
"""

import tvm
from tvm import tir
import tilelang.tt as tt


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
    num_tiles = grid_x * grid_y
    tiles_per_core = []
    for i in range(num_cores):
        start_id = i % num_tiles  # Simplified assignment
        count = 1
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

    # Create IRModule
    mod = tvm.IRModule({"main": func})
    return mod


def test_emit_host_program_basic():
    """
    Test 1: Basic host program generation

    Verifies that main.cpp artifact is generated with correct structure.
    """
    # Create module with metadata inference stage/persistent transform stage metadata
    mod = create_tt_module_with_metadata(grid_x=8, grid_y=8)

    # Generate artifacts
    artifacts = tt.emit_tt_artifacts(mod)

    # Verify main.cpp exists
    assert "main.cpp" in artifacts, "main.cpp artifact missing"

    # Verify host program contains expected elements
    main_cpp = artifacts["main.cpp"]
    assert "// Generated TT Host Program" in main_cpp, "Host program header missing"
    assert "int main()" in main_cpp, "main function missing"
    assert "return 0;" in main_cpp, "main return statement missing"

    # Verify includes
    assert "#include <cstdint>" in main_cpp, "cstdint include missing"
    assert "#include <vector>" in main_cpp, "vector include missing"
    assert "#include <iostream>" in main_cpp, "iostream include missing"

    # Verify mock device APIs
    assert "class Device" in main_cpp, "Device class missing"
    assert "class Program" in main_cpp, "Program class missing"
    assert "class CommandQueue" in main_cpp, "CommandQueue class missing"

    print("✓ Test 1 passed: Basic host program generation")


def test_device_setup_code():
    """
    Test 2: Device setup code generation

    Verifies that device initialization code is present.
    """
    mod = create_tt_module_with_metadata(grid_x=8, grid_y=8)

    artifacts = tt.emit_tt_artifacts(mod)
    main_cpp = artifacts["main.cpp"]

    # Verify device setup
    assert "Device* device = Device::Instance()" in main_cpp, "Device initialization missing"
    assert "Device initialized" in main_cpp, "Device init message missing"

    print("✓ Test 2 passed: Device setup code generation")


def test_cb_config_generation():
    """
    Test 3: Circular buffer configuration code

    Verifies that CB config code is generated correctly.
    """
    mod = create_tt_module_with_metadata(grid_x=8, grid_y=8)

    artifacts = tt.emit_tt_artifacts(mod)
    main_cpp = artifacts["main.cpp"]

    # Verify CB configuration
    assert "class CircularBufferConfig" in main_cpp, "CircularBufferConfig class missing"
    assert "CircularBufferConfig cb_a(0, TILE_SIZE_FP16, CB_NUM_PAGES)" in main_cpp, "CB_A config missing"
    assert "CircularBufferConfig cb_b(1, TILE_SIZE_FP16, CB_NUM_PAGES)" in main_cpp, "CB_B config missing"
    assert "CircularBufferConfig cb_c(2, TILE_SIZE_FP16, CB_NUM_PAGES)" in main_cpp, "CB_C config missing"

    # Verify CB constants
    assert "constexpr uint32_t TILE_H = 32" in main_cpp, "TILE_H constant missing"
    assert "constexpr uint32_t TILE_W = 32" in main_cpp, "TILE_W constant missing"
    assert "constexpr uint32_t TILE_SIZE_FP16 = TILE_H * TILE_W * sizeof(uint16_t)" in main_cpp, \
        "TILE_SIZE_FP16 calculation missing"
    assert "constexpr uint32_t CB_NUM_PAGES = 2" in main_cpp, "CB_NUM_PAGES missing (double buffering)"

    print("✓ Test 3 passed: CB config generation")


def test_dram_buffer_allocation():
    """
    Test 4: DRAM buffer allocation code

    Verifies that DRAM buffer allocation and initialization code is generated.
    """
    mod = create_tt_module_with_metadata(grid_x=8, grid_y=8)

    artifacts = tt.emit_tt_artifacts(mod)
    main_cpp = artifacts["main.cpp"]

    # Verify buffer allocation
    assert "std::vector<uint16_t> dram_a" in main_cpp, "dram_a allocation missing"
    assert "std::vector<uint16_t> dram_b" in main_cpp, "dram_b allocation missing"
    assert "std::vector<uint16_t> dram_c" in main_cpp, "dram_c allocation missing"

    # Verify buffer dimensions
    assert "constexpr uint32_t M = 256" in main_cpp, "M dimension missing (8*32=256)"
    assert "constexpr uint32_t N = 256" in main_cpp, "N dimension missing (8*32=256)"
    assert "constexpr uint32_t K = 256" in main_cpp, "K dimension missing (8*32=256)"

    # Verify initialization
    assert "for (size_t i = 0; i < dram_a.size(); ++i)" in main_cpp, "dram_a initialization loop missing"
    assert "for (size_t i = 0; i < dram_b.size(); ++i)" in main_cpp, "dram_b initialization loop missing"

    print("✓ Test 4 passed: DRAM buffer allocation")


def test_program_creation_and_launch():
    """
    Test 5: Program creation and kernel launch code

    Verifies that program creation and launch code is generated.
    """
    mod = create_tt_module_with_metadata(grid_x=8, grid_y=8)

    artifacts = tt.emit_tt_artifacts(mod)
    main_cpp = artifacts["main.cpp"]

    # Verify program creation
    assert "Program program" in main_cpp, "Program object missing"
    assert "program.Build()" in main_cpp or "CreateProgram()" in main_cpp, "Program creation missing"

    # Verify kernel loading (commented placeholders or actual kernel creation)
    # Note: In refactored version, AddKernel is replaced with CreateKernel placeholders
    assert ("program.Build()" in main_cpp or "CreateKernel" in main_cpp or
            "Kernels created" in main_cpp), "Kernel creation missing"

    # Verify command queue and launch
    assert "CommandQueue cq" in main_cpp or "CommandQueue& cq" in main_cpp, "CommandQueue object missing"
    assert ("cq.EnqueueProgram(&program, true)" in main_cpp or
            "EnqueueProgram(cq, program" in main_cpp), "EnqueueProgram call missing"
    assert "cq.Finish()" in main_cpp or "Finish(cq)" in main_cpp, "cq.Finish() call missing"

    print("✓ Test 5 passed: Program creation and launch")


def test_runtime_args_configuration():
    """
    Test 6: Runtime arguments configuration

    Verifies that runtime arguments are correctly configured from metadata.
    """
    mod = create_tt_module_with_metadata(grid_x=8, grid_y=8)

    artifacts = tt.emit_tt_artifacts(mod)
    main_cpp = artifacts["main.cpp"]

    # reader/writer specialization stage: Runtime args now use matmul dimensions (Mt, Kt, Nt)
    assert "constexpr uint32_t Mt = 8" in main_cpp, "Mt constant missing"
    assert "constexpr uint32_t Nt = 8" in main_cpp, "Nt constant missing"
    assert "constexpr uint32_t Kt = 8" in main_cpp, "Kt constant missing"
    assert "constexpr uint32_t NUM_OUTPUT_TILES = 64" in main_cpp, "NUM_OUTPUT_TILES constant missing (8*8=64)"
    assert "constexpr uint32_t NUM_CORES = " in main_cpp, "NUM_CORES constant missing"

    print("✓ Test 6 passed: Runtime args configuration")


def test_different_grid_sizes():
    """
    Test 7: Host program handles different grid sizes

    Verifies that host program generation works for various grid dimensions.
    """
    test_cases = [
        (4, 4),  # 16 tiles, 4x4 grid
        (16, 16),  # 256 tiles, 16x16 grid
    ]

    for grid_x, grid_y in test_cases:
        mod = create_tt_module_with_metadata(grid_x=grid_x, grid_y=grid_y)

        artifacts = tt.emit_tt_artifacts(mod)
        main_cpp = artifacts["main.cpp"]

        # reader/writer specialization stage: Verify matmul dimensions (Mt, Kt, Nt)
        expected_tiles = grid_x * grid_y
        expected_m = grid_y * 32
        expected_n = grid_x * 32

        assert f"constexpr uint32_t Mt = {grid_y}" in main_cpp, \
            f"Mt {grid_y} not found"
        assert f"constexpr uint32_t Nt = {grid_x}" in main_cpp, \
            f"Nt {grid_x} not found"
        assert f"constexpr uint32_t NUM_OUTPUT_TILES = {expected_tiles}" in main_cpp, \
            f"NUM_OUTPUT_TILES {expected_tiles} not found"

        # Verify buffer dimensions
        assert f"constexpr uint32_t M = {expected_m}" in main_cpp, \
            f"M dimension {expected_m} not found"
        assert f"constexpr uint32_t N = {expected_n}" in main_cpp, \
            f"N dimension {expected_n} not found"

        print(
            f"✓ Test 7 passed for grid {grid_x}x{grid_y} ({expected_tiles} tiles, {expected_m}x{expected_n} buffers)"
        )


def test_full_host_program_structure():
    """
    Test 8: Complete host program structure

    Verifies that the host program has all required sections in correct order.
    """
    mod = create_tt_module_with_metadata(grid_x=8, grid_y=8)

    artifacts = tt.emit_tt_artifacts(mod)
    main_cpp = artifacts["main.cpp"]

    # Verify section order by finding positions (reader/writer specialization stage: updated for matmul dimensions)
    sections = [
        ("includes", "#include <cstdint>"),
        ("device_apis", "class Device"),
        ("main_func", "int main()"),
        ("device_setup", "Device* device"),
        ("cb_config", "CircularBufferConfig cb_a"),
        ("program_create", "Program program"),
        ("dram_alloc", "std::vector<uint16_t> dram_a"),
        ("runtime_args", "constexpr uint32_t Mt"),
        ("launch", "CommandQueue cq"),
        ("return", "return 0;"),
    ]

    positions = {}
    for section_name, search_str in sections:
        pos = main_cpp.find(search_str)
        assert pos != -1, f"Section '{section_name}' not found (searching for '{search_str}')"
        positions[section_name] = pos

    # Verify order (reader/writer specialization stage: runtime_args (dimensions) now come before dram_alloc)
    assert positions["includes"] < positions[
        "device_apis"], "Includes should come before device APIs"
    assert positions["device_apis"] < positions[
        "main_func"], "Device APIs should come before main()"
    assert positions["main_func"] < positions[
        "device_setup"], "main() should come before device setup"
    assert positions["device_setup"] < positions[
        "cb_config"], "Device setup should come before CB config"
    assert positions["cb_config"] < positions[
        "program_create"], "CB config should come before program create"
    assert positions["program_create"] < positions[
        "runtime_args"], "Program create should come before runtime args (dimensions)"
    assert positions["runtime_args"] < positions[
        "dram_alloc"], "Runtime args (dimensions) should come before DRAM alloc"
    assert positions["dram_alloc"] < positions["launch"], "DRAM alloc should come before launch"
    assert positions["launch"] < positions["return"], "Launch should come before return"

    print("✓ Test 8 passed: Full host program structure verification")


if __name__ == "__main__":
    # Run tests
    print("Running host program stage Host Program Generation Tests\n")

    test_emit_host_program_basic()
    test_device_setup_code()
    test_cb_config_generation()
    test_dram_buffer_allocation()
    test_program_creation_and_launch()
    test_runtime_args_configuration()
    test_different_grid_sizes()
    test_full_host_program_structure()

    print("\n✅ All host program stage host program tests passed!")
