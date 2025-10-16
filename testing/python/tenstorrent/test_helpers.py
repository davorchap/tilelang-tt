"""
Helper functions for migrating tests to the new metadata-driven architecture.

This module provides utilities to help tests transition from the old C++ FFI-based
implementation to the new Python implementation.
"""


def use_new_pipeline(mod, **kwargs):
    """
    Helper to run the v5 pipeline on a module.

    Args:
        mod: IRModule to process
        **kwargs: Optional pipeline configuration

    Returns:
        Transformed IRModule
    """
    from tilelang.tenstorrent.passes import run_pipeline

    # Set defaults
    kwargs.setdefault("plan_path", "test.plan.json")
    kwargs.setdefault("verbose", False)

    return run_pipeline(mod, **kwargs)


def convert_legacy_attrs(func):
    """
    Convert legacy attribute names to new format in tests.

    Use this when asserting on function attributes.
    """
    attr_map = {
        "tt.grid": "tt.core_grid",
        "tt.tiles_per_core": "tt.work_partition",
        "tt.block_shape": "tt.work_partition",  # Derived
        "tt.start_tile": "tt.work_partition",  # Derived
        "tt.runtime_args": "tt.work_partition",  # Converted
    }

    return attr_map


def assert_has_new_attrs(func, required_attrs=None):
    """
    Assert that a function has the required new attributes.

    Args:
        func: PrimFunc to check
        required_attrs: List of required attribute names, or None for defaults
    """
    if required_attrs is None:
        from tilelang.tenstorrent.attrs import (
            TT_CORE_GRID,
            TT_LAYOUT_DESC,
            TT_WORK_PARTITION,
        )

        required_attrs = [TT_CORE_GRID, TT_LAYOUT_DESC, TT_WORK_PARTITION]

    for attr in required_attrs:
        assert attr in func.attrs, f"Missing required attribute: {attr}"

    return True
