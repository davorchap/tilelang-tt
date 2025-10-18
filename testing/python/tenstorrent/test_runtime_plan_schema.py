from tilelang.tenstorrent.runtime_plan import validate_plan


def test_runtime_plan_schema_validation_success():
    plan = {
        "schema_version": 1,
        "backend_version": "0.0.0",
        "core_grid": [2, 3],
        "core_ranges": [{
            "start": [0, 0],
            "extent": [2, 3]
        }],
        "work_partition": {},
        "layouts": {},
    }
    errors = validate_plan(plan)
    assert errors == []


def test_runtime_plan_schema_validation_missing_version():
    plan = {
        # "schema_version": 1,  # intentionally omitted
        "core_grid": [1, 1],
        "core_ranges": [{
            "start": [0, 0],
            "extent": [1, 1]
        }],
        "work_partition": {},
        "layouts": {},
    }
    errors = validate_plan(plan)
    assert any("schema_version" in e for e in errors)
