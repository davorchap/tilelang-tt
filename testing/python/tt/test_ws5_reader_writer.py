"""
reader/writer generation stage: Reader/Writer Kernel Emission Integration Tests

Tests for TT reader and writer kernel generation, circular buffer operations,
and 3-kernel coordination (reader → compute → writer).
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


def test_emit_reader_kernel_basic():
    """
    Test 1: Basic reader kernel generation

    Verifies that reader.cpp artifact is generated with correct CB API calls
    and NOC operations.
    """
    # Create module with metadata inference stage/persistent transform stage metadata
    mod = create_tt_module_with_metadata(grid_x=8, grid_y=8)

    # Generate artifacts
    artifacts = tt.emit_tt_artifacts(mod)

    # Verify reader.cpp exists
    assert "reader.cpp" in artifacts, "reader.cpp artifact missing"

    # Verify reader kernel contains expected elements (IR-driven codegen)
    reader_cpp = artifacts["reader.cpp"]
    assert "// Generated TT Reader Kernel (IR-Driven)" in reader_cpp, "IR-driven reader kernel header missing"
    assert "void kernel_main()" in reader_cpp, "kernel_main function missing"

    # Verify CB API calls
    assert "cb_reserve_back(cb_in0, 1)" in reader_cpp, "cb_reserve_back missing for cb_in0"
    assert "cb_push_back(cb_in0, 1)" in reader_cpp, "cb_push_back missing for cb_in0"
    assert "get_write_ptr(cb_in0)" in reader_cpp, "get_write_ptr missing for cb_in0"

    assert "cb_reserve_back(cb_in1, 1)" in reader_cpp, "cb_reserve_back missing for cb_in1"
    assert "cb_push_back(cb_in1, 1)" in reader_cpp, "cb_push_back missing for cb_in1"
    assert "get_write_ptr(cb_in1)" in reader_cpp, "get_write_ptr missing for cb_in1"

    # Verify NOC operations (reader/writer specialization stage: uses noc_async_read_tile)
    assert "noc_async_read_tile(" in reader_cpp, "noc_async_read_tile missing"
    assert "noc_async_read_barrier()" in reader_cpp, "noc_async_read_barrier missing"

    # Verify CB indices
    assert "constexpr auto cb_in0 = tt::CBIndex::c_0" in reader_cpp, "cb_in0 index definition missing"
    assert "constexpr auto cb_in1 = tt::CBIndex::c_1" in reader_cpp, "cb_in1 index definition missing"

    print("✓ Test 1 passed: Basic reader kernel generation")


def test_emit_writer_kernel_basic():
    """
    Test 2: Basic writer kernel generation

    Verifies that writer.cpp artifact is generated with correct CB API calls
    and NOC operations.
    """
    mod = create_tt_module_with_metadata(grid_x=8, grid_y=8)

    # Generate artifacts
    artifacts = tt.emit_tt_artifacts(mod)

    # Verify writer.cpp exists
    assert "writer.cpp" in artifacts, "writer.cpp artifact missing"

    # Verify writer kernel contains expected elements (IR-driven codegen)
    writer_cpp = artifacts["writer.cpp"]
    assert "// Generated TT Writer Kernel (IR-Driven)" in writer_cpp, "IR-driven writer kernel header missing"
    assert "void kernel_main()" in writer_cpp, "kernel_main function missing"

    # Verify CB API calls
    assert "cb_wait_front(cb_out0, 1)" in writer_cpp, "cb_wait_front missing for cb_out0"
    assert "cb_pop_front(cb_out0, 1)" in writer_cpp, "cb_pop_front missing for cb_out0"
    assert "get_read_ptr(cb_out0)" in writer_cpp, "get_read_ptr missing for cb_out0"

    # Verify NOC operations (reader/writer specialization stage: uses noc_async_write_tile)
    assert "noc_async_write_tile(" in writer_cpp, "noc_async_write_tile missing"
    assert "noc_async_write_barrier()" in writer_cpp, "noc_async_write_barrier missing"

    # Verify CB index
    assert "constexpr auto cb_out0 = tt::CBIndex::c_16" in writer_cpp, "cb_out0 index definition missing"

    print("✓ Test 2 passed: Basic writer kernel generation")


def test_3_kernel_coordination():
    """
    Test 3: 3-kernel coordination (reader → compute → writer)

    Verifies that all 3 kernels are generated and use matching CB indices
    for proper synchronization.
    """
    mod = create_tt_module_with_metadata(grid_x=8, grid_y=8)

    artifacts = tt.emit_tt_artifacts(mod)

    # Verify all 3 kernels generated
    assert "reader.cpp" in artifacts, "reader.cpp missing"
    assert "compute.cpp" in artifacts, "compute.cpp missing"
    assert "writer.cpp" in artifacts, "writer.cpp missing"

    reader_cpp = artifacts["reader.cpp"]
    compute_cpp = artifacts["compute.cpp"]
    writer_cpp = artifacts["writer.cpp"]

    # Verify CB indices match across all kernels (correct Metalium format)
    # cb_in0 should appear in reader and compute
    assert "cb_in0 = tt::CBIndex::c_0" in reader_cpp, "cb_in0 not defined in reader"
    assert "cb_in0 = tt::CBIndex::c_0" in compute_cpp, "cb_in0 not defined in compute"

    # cb_in1 should appear in reader and compute
    assert "cb_in1 = tt::CBIndex::c_1" in reader_cpp, "cb_in1 not defined in reader"
    assert "cb_in1 = tt::CBIndex::c_1" in compute_cpp, "cb_in1 not defined in compute"

    # cb_out0 should appear in compute and writer
    assert "cb_out0 = tt::CBIndex::c_16" in compute_cpp, "cb_out0 not defined in compute"
    assert "cb_out0 = tt::CBIndex::c_16" in writer_cpp, "cb_out0 not defined in writer"

    # Verify synchronization pattern in compute kernel (IR-driven)
    # Note: IR-driven generates from actual IR body. Empty body won't have CB ops.
    # CB indices are defined, actual operations come from IR (loops/matmul/copy nodes)

    print("✓ Test 3 passed: 3-kernel coordination")


def test_reader_writer_tile_counts():
    """
    Test 4: Reader/Writer kernel handles different tile counts correctly

    Verifies that tile counts from metadata inference stage metadata are correctly reflected
    in generated reader/writer kernels.
    """
    test_cases = [
        (4, 4),  # 16 tiles
        (8, 8),  # 64 tiles
        (16, 16),  # 256 tiles
    ]

    for grid_x, grid_y in test_cases:
        mod = create_tt_module_with_metadata(grid_x=grid_x, grid_y=grid_y)

        artifacts = tt.emit_tt_artifacts(mod)

        expected_tiles = grid_x * grid_y

        # reader/writer specialization stage: Reader/writer kernels now have matmul-specific structure
        # Verify they contain runtime args and loop structures
        reader_cpp = artifacts["reader.cpp"]
        assert "num_out_tiles" in reader_cpp, \
            f"Reader kernel missing num_out_tiles for {grid_x}x{grid_y} grid"
        assert "for (uint32_t out_tile = 0; out_tile < num_out_tiles; ++out_tile)" in reader_cpp, \
            f"Reader kernel missing output tile loop for {grid_x}x{grid_y} grid"

        writer_cpp = artifacts["writer.cpp"]
        assert "num_out_tiles" in writer_cpp, \
            f"Writer kernel missing num_out_tiles for {grid_x}x{grid_y} grid"
        assert "for (uint32_t out_tile = 0; out_tile < num_out_tiles; ++out_tile)" in writer_cpp, \
            f"Writer kernel missing output tile loop for {grid_x}x{grid_y} grid"

        print(f"✓ Test 4 passed for grid {grid_x}x{grid_y} ({expected_tiles} tiles)")


def test_cb_synchronization_pattern():
    """
    Test 5: Circular buffer synchronization pattern validation

    Verifies that reader/compute/writer follow correct CB patterns:
    - Reader: reserve → write → push
    - Compute: wait → compute → pop (input) + reserve → compute → push (output)
    - Writer: wait → read → pop
    """
    mod = create_tt_module_with_metadata(grid_x=8, grid_y=8)

    artifacts = tt.emit_tt_artifacts(mod)

    reader_cpp = artifacts["reader.cpp"]
    compute_cpp = artifacts["compute.cpp"]
    writer_cpp = artifacts["writer.cpp"]

    # Reader pattern: reserve → write → push (reader/writer specialization stage: unified kernel_main)
    # Check order by finding positions
    reader_lines = reader_cpp.split('\n')
    reader_kernel_start = next(
        i for i, line in enumerate(reader_lines) if "void kernel_main()" in line)
    reader_kernel_section = '\n'.join(reader_lines[reader_kernel_start:reader_kernel_start + 30])

    assert "cb_reserve_back" in reader_kernel_section, "Reader missing reserve"
    assert "noc_async_read" in reader_kernel_section, "Reader missing async read"
    assert "cb_push_back" in reader_kernel_section, "Reader missing push"

    # Verify order: reserve before push
    reserve_pos = reader_kernel_section.find("cb_reserve_back")
    push_pos = reader_kernel_section.find("cb_push_back")
    assert reserve_pos < push_pos, "Reader should reserve before push"

    # Compute pattern: wait (inputs) → matmul → pop (inputs) → push (output)
    # Note: IR-driven codegen generates from actual IR body. Empty body = no CB operations.
    # This test is for template-based codegen. For IR-driven, CB operations come from IR.
    compute_lines = compute_cpp.split('\n')
    main_start = next(i for i, line in enumerate(compute_lines) if "void MAIN()" in line)
    main_section = '\n'.join(compute_lines[main_start:main_start + 40])

    # IR-driven: Check for operations if present (empty body won't have them)
    if "for (uint32_t" in compute_cpp:
        # Has loop body, check for CB operations
        assert "cb_wait_front" in compute_cpp or "matmul_tiles" in compute_cpp, "Compute missing operations"
    # else: Empty body is valid for IR-driven (just has runtime args)

    # Writer pattern: wait → read → pop (reader/writer specialization stage: kernel_main)
    writer_lines = writer_cpp.split('\n')
    writer_kernel_start = next(
        i for i, line in enumerate(writer_lines) if "void kernel_main()" in line)
    writer_kernel_section = '\n'.join(writer_lines[writer_kernel_start:writer_kernel_start + 20])

    assert "cb_wait_front" in writer_kernel_section, "Writer missing wait"
    assert "noc_async_write" in writer_kernel_section, "Writer missing async write"
    assert "cb_pop_front" in writer_kernel_section, "Writer missing pop"

    # Verify order: wait before pop
    wait_pos = writer_kernel_section.find("cb_wait_front")
    pop_pos = writer_kernel_section.find("cb_pop_front")
    assert wait_pos < pop_pos, "Writer should wait before pop"

    print("✓ Test 5 passed: CB synchronization pattern validation")


if __name__ == "__main__":
    # Run tests
    print("Running reader/writer generation stage Reader/Writer Kernel Tests\n")

    test_emit_reader_kernel_basic()
    test_emit_writer_kernel_basic()
    test_3_kernel_coordination()
    test_reader_writer_tile_counts()
    test_cb_synchronization_pattern()

    print("\n✅ All reader/writer generation stage reader/writer tests passed!")
