import importlib

import pytest

try:
    import tvm
    from tvm.target import Target
except ModuleNotFoundError as exc:
    pytest.skip(f"TVM not available: {exc}", allow_module_level=True)

_target_mod = importlib.import_module("tilelang.utils.target")
_tt_lower = importlib.import_module("tilelang.engine.tenstorrent.lower")
_tt_target = importlib.import_module("tilelang.tenstorrent.target")
CompiledArtifact = importlib.import_module("tilelang.engine.param").CompiledArtifact


@pytest.fixture
def toggle_tt_backend(monkeypatch):
    original = getattr(_target_mod, "_HAS_TENSTORRENT_BACKEND", False)

    def setter(value: bool):
        monkeypatch.setattr(_target_mod, "_HAS_TENSTORRENT_BACKEND", value, raising=False)

    setter(original)
    try:
        yield setter
    finally:
        setter(original)


def test_available_targets_contains_tt():
    assert _target_mod.TENSTORRENT_TARGET in _target_mod.AVALIABLE_TARGETS


def test_determine_target_returns_target_when_backend_enabled(toggle_tt_backend):
    toggle_tt_backend(True)
    scope_name = _target_mod.determine_target(_target_mod.TENSTORRENT_TARGET)
    assert scope_name == _target_mod.TENSTORRENT_TARGET

    target_obj = _target_mod.determine_target(_target_mod.TENSTORRENT_TARGET, return_object=True)
    assert isinstance(target_obj, Target)
    assert target_obj.kind.name == _target_mod.TENSTORRENT_TARGET


def test_determine_target_raises_when_backend_disabled(toggle_tt_backend):
    toggle_tt_backend(False)
    with pytest.raises(ValueError, match="Tenstorrent backend requires"):
        _target_mod.determine_target(_target_mod.TENSTORRENT_TARGET)


def test_tenstorrent_engine_lower_returns_compiled_artifact(toggle_tt_backend):
    """Test that TT lowering returns a CompiledArtifact (Task 1+2 complete)."""
    toggle_tt_backend(True)

    # Use TileLang DSL to create a proper kernel
    import tilelang.language as T

    @T.prim_func
    def simple_copy(A: T.Buffer((32, 32), "float16"), B: T.Buffer((32, 32), "float16")):
        with T.Kernel(1, 1) as (bx, by):
            for i, j in T.Parallel(32, 32):
                B[i, j] = A[i, j]

    # Create IRModule
    mod = tvm.IRModule({"main": simple_copy})

    # Call lowering
    result = _tt_lower.lower(
        mod,
        params=None,
        target=_target_mod.TENSTORRENT_TARGET,
        target_host=None,
        runtime_only=False,
        enable_host_codegen=False,
        enable_device_compile=False,
    )

    # Verify result is a CompiledArtifact with expected attributes
    assert isinstance(result, CompiledArtifact)
    assert hasattr(result, 'host_mod')
    assert hasattr(result, 'device_mod')
    assert hasattr(result, 'params')
    assert hasattr(result, 'kernel_source')
    assert isinstance(result.host_mod, tvm.IRModule)
    assert isinstance(result.device_mod, tvm.IRModule)


def test_tenstorrent_engine_lower_validates_target(toggle_tt_backend):
    toggle_tt_backend(True)
    with pytest.raises(ValueError, match="Tenstorrent lowering called with invalid target"):
        _tt_lower.lower(
            tvm.IRModule(),
            params=None,
            target="cuda",
            target_host=None,
            runtime_only=False,
            enable_host_codegen=False,
            enable_device_compile=False,
        )


def test_apply_tt_defaults_adds_attributes_to_empty_module():
    """Test that default TT attributes are added to PrimFuncs without TT attrs."""
    # Create a simple PrimFunc without any TT attributes
    # Use a basic empty PrimFunc
    func = tvm.tir.PrimFunc([], tvm.tir.Evaluate(0))
    func = func.with_attr("global_symbol", "main")
    mod = tvm.IRModule({"main": func})

    # Apply defaults
    mod_with_defaults = _tt_target.apply_tt_defaults(mod)

    # Check that the function now has TT attributes
    func = mod_with_defaults["main"]
    assert "tt_schedule_policy" in func.attrs
    assert func.attrs["tt_schedule_policy"] == "contiguous"
    assert "tt_schedule_order" in func.attrs
    assert func.attrs["tt_schedule_order"] == "row_major"
    assert "tt_layout_type" in func.attrs
    assert func.attrs["tt_layout_type"] == "dram_interleaved"
    assert "tt_tile_height" in func.attrs
    assert func.attrs["tt_tile_height"] == 32
    assert "tt_tile_width" in func.attrs
    assert func.attrs["tt_tile_width"] == 32


def test_apply_tt_defaults_preserves_existing_attributes():
    """Test that existing TT attributes are not overwritten."""
    # Create a PrimFunc with existing TT attributes
    func = tvm.tir.PrimFunc([], tvm.tir.Evaluate(0))
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("tt_schedule_policy", "custom_policy")
    func = func.with_attr("tt_custom_attr", "custom_value")
    mod = tvm.IRModule({"main": func})

    # Apply defaults
    mod_with_defaults = _tt_target.apply_tt_defaults(mod)

    # Check that existing attributes are preserved
    func = mod_with_defaults["main"]
    assert func.attrs["tt_schedule_policy"] == "custom_policy"
    assert func.attrs["tt_custom_attr"] == "custom_value"

    # Other default attributes should NOT be added since we already have TT attrs
    assert "tt_schedule_order" not in func.attrs


def test_apply_tt_defaults_is_idempotent():
    """Test that applying defaults multiple times produces the same result."""
    # Create a simple PrimFunc
    func = tvm.tir.PrimFunc([], tvm.tir.Evaluate(0))
    func = func.with_attr("global_symbol", "main")
    mod = tvm.IRModule({"main": func})

    # Apply defaults twice
    mod1 = _tt_target.apply_tt_defaults(mod)
    mod2 = _tt_target.apply_tt_defaults(mod1)

    # Check that attributes are the same
    func1 = mod1["main"]
    func2 = mod2["main"]

    for key in func1.attrs.keys():
        if key.startswith("tt_"):
            assert key in func2.attrs
            assert func1.attrs[key] == func2.attrs[key]
