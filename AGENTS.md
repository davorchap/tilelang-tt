# Repository Guidelines

## Tenstorrent Backend Snapshot
- Layout-aware metadata, shard-aware persistent lowering, and runtime guardrails are active as of 2025-10-10; host codegen now emits a metadata summary in `main.cpp` that mirrors runtime argument schemas.
- The backend pipeline flows from Python orchestration helpers (`tilelang/tt`) through TT-specific transforms (`src/transform/tt/`) into IR-driven visitors under `src/target/tt/`.
- Mock-mode CI stays the default validation path; real SDK validation (`--with-metalium`) requires an installed TT-Metalium SDK (`TT_METAL_HOME`).

## Project Structure & Module Organization
- `src/target/tt/` holds the C++ code generators (reader/compute/writer visitors, host metadata emitter).
- `src/transform/tt/` implements Tenstorrent-specific passes (layout-aware metadata, persistent lowering, verification).
- `tilelang/tt/` contains Python orchestration helpers (`apply_tt_defaults`, layout-aware wrappers) that wire the C++ passes.
- `testing/python/tt/` hosts pytest coverage for the backend; keep new tests alongside the feature they exercise.
- Automation such as `maint/scripts/local_build_and_test_tt.sh` lives under `maint/scripts/`.
- Reference material lives in `docs/tenstorrent/`; start with the doc set below.

## Core Docs & References
- Architecture & runtime contracts: `docs/tenstorrent/TT_ARCHITECTURE.md`
- CI and local parity workflow: `docs/tenstorrent/CI.md`
- Backend overview and doc index: `docs/tenstorrent/README.md`
- SDK setup for hardware runs: `docs/tenstorrent/METALIUM_SETUP_GUIDE.md`
- Additional guides (e.g., `local_build_guide.md`, `IR_LOWERING_TASKS.md`) expand on build details and pass roadmaps.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — create an isolated environment (recommended for all workflows).
- `pip install -e ".[dev]"` — install TileLang in editable mode with formatting, lint, and pytest extras.
- `bash maint/scripts/local_build_and_test_tt.sh --skip-deps --jobs 4` — full mock-mode build plus Tenstorrent test suite; mirrors Tier‑1 CI.
- `bash maint/scripts/local_build_and_test_tt.sh --with-metalium --skip-deps --jobs 4` — build/tests against a real TT‑Metalium SDK (requires `TT_METAL_HOME`).
- `LD_LIBRARY_PATH=build/tvm:$LD_LIBRARY_PATH pytest testing/python/tt/ -v` — run backend tests; add `-k <pattern>` for focused runs.
- `bash format.sh` — invoke yapf, ruff, codespell, and clang-format; run before committing.

## Coding Style & Naming Conventions
- Python: 4-space indentation, Ruff-compliant imports, snake_case for functions/modules, PascalCase for classes.
- C++: follow the repository clang-format; prefer camelCase methods, UpperCamelCase types, `kConstantName` for constexprs.
- Tests and helper modules should use lowercase filenames with underscores (e.g., `test_ir_to_codegen_integration.py`).
- Mirror public metadata keys in identifiers (`tt_start_tile`, `partition_mode`) to keep host/runtime contracts obvious.

## Testing Guidelines
- Extend pytest suites with descriptive `test_*` functions; organize fixtures in `conftest.py` when reused.
- Whenever runtime arguments or metadata schemas change, add regression coverage in `testing/python/tt/`.
- For native changes, add/adjust gtests in `testing/cpp` and execute with `ctest --output-on-failure` after a CMake build.
- Always run the mock-mode script or equivalent pytest command locally before submitting a PR; CI assumes clean baselines.

## Commit & Pull Request Guidelines
- Write imperative, sentence-case commit subjects (e.g., “Enforce shard guardrails in compute visitor”); keep commits logically scoped.
- Reference updated documentation or scripts in commit bodies when workflows change.
- PRs must summarize key changes, list validation commands (e.g., `pytest testing/python/tt/ -v`), and link relevant docs/issues.
- Target `davorchap/tilelang-tt` → `main`; never open PRs against `tile-ai/tilelang`. Await passing CI before requesting review.
