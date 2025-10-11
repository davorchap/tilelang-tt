# CLAUDE.md

Guidelines for Claude Code when collaborating on this repository.

## Repository Snapshot
- Fork `davorchap/tilelang-tt` hosts the Tenstorrent backend effort: layout-aware metadata passes, shard-aware persistent lowering, and IR-driven reader/compute/writer/host generators.
- Python orchestration lives in `tilelang/tt`, compiler passes in `src/transform/tt/`, and codegen visitors in `src/target/tt/`. Tests reside under `testing/python/tt/`.
- As of 2025-10-10 the host artifact emits a metadata summary (`main.cpp`) aligned with runtime argument schemas; guardrails ensure shard completeness across visitors.

## Build & Test Quickstart
1. Create an isolated environment: `python -m venv .venv && source .venv/bin/activate`.
2. Install development dependencies: `pip install -e ".[dev]"`.
3. Run the mock-mode CI replica: `bash maint/scripts/local_build_and_test_tt.sh --skip-deps --jobs 4`.
4. For SDK-backed runs, export `TT_METAL_HOME` and add `--with-metalium`.
5. Standalone tests: `LD_LIBRARY_PATH=build/tvm:$LD_LIBRARY_PATH pytest testing/python/tt/ -v`.
6. Enforce formatting and lint baselines with `bash format.sh` before staging commits.

## Workflow Expectations
- Branch from `main` and open PRs against `davorchap/tilelang-tt`; upstream `tile-ai/tilelang` stays read-only for TT backend development.
- Keep commits focused, imperative, and documented—reference any touched docs or scripts when workflows change.
- PR descriptions must summarise behavioural changes, list validation commands (e.g., `pytest testing/python/tt/ -v`), and link doc updates such as `docs/tenstorrent/CI.md`.
- Coordinate interface changes (runtime metadata, pass ordering) with matching test and documentation updates in the same PR.
- Use the GitHub CLI (`gh`) to draft and submit pull requests when changes are ready.

## Reference Documents
- Architecture & runtime contracts: `docs/tenstorrent/TT_ARCHITECTURE.md`
- Backend overview & doc index: `docs/tenstorrent/README.md`
- CI/local parity workflows: `docs/tenstorrent/CI.md`
- SDK setup for hardware validation: `docs/tenstorrent/METALIUM_SETUP_GUIDE.md`
- Pass roadmaps & analysis: `docs/tenstorrent/IR_LOWERING_TASKS.md`, `docs/tenstorrent/IR_LOWERING_ANALYSIS.md`, `docs/tenstorrent/PASS_TABLE.md`
- Build deep-dive & scripts: `docs/tenstorrent/local_build_guide.md`

Review these files when making changes; update them if instructions drift from reality.
