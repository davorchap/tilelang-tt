# CLAUDE.md

Guidelines for Claude Code when collaborating on this repository.

## Repository Snapshot
- This fork (`davorchap/tilelang-tt`) tracks the Tenstorrent backend work: layout-aware metadata passes, shard-aware persistent lowering, and the IR-driven code generators (reader/compute/writer/host).
- The codebase mixes C++ (under `src/`) with Python orchestration (`tilelang/tt`). Tests live primarily in `testing/python/tt`.
- Recent updates moved the host artifact to a metadata summary (`main.cpp`) and tightened shard guardrails in every visitor; keep these contracts in mind when editing runtime arguments.

## Build & Test Quickstart
1. Create a virtual environment (`python -m venv .venv && source .venv/bin/activate`).
2. Install development dependencies: `pip install -e ".[dev]"`.
3. For CI-parity mock builds, run `bash maint/scripts/local_build_and_test_tt.sh --skip-deps --jobs 4`.
4. To exercise the real SDK, export `TT_METAL_HOME` and add `--with-metalium` to the script.
5. Standalone tests: `LD_LIBRARY_PATH=build/tvm:$LD_LIBRARY_PATH pytest testing/python/tt -q`.
6. Enforce formatting with `bash format.sh` before staging commits.

## Workflow Expectations
- Always branch from `main` and open PRs against `davorchap/tilelang-tt`; upstream (`tile-ai/tilelang`) is read-only for this effort.
- Keep commits focused, imperative, and documented (note impacted docs/scripts when relevant).
- PR descriptions should summarise behaviour changes, list validation commands, and link documentation updates (e.g., `docs/tenstorrent/CI.md`).
- Coordinate significant interface changes (runtime metadata, pass ordering) with updates to tests and docs in the same PR.

## Reference Documents
- Architecture overview: `docs/tenstorrent/TT_ARCHITECTURE.md`
- Layout-aware roadmap: `docs/tenstorrent/IR_LOWERING_TASKS.md`
- CI and local build parity: `docs/tenstorrent/CI.md`
- SDK setup: `docs/tenstorrent/METALIUM_SETUP_GUIDE.md`
- Contributor tips for Tenstorrent-specific flows: `docs/tenstorrent/README.md`

Review these files when making changes; update them if instructions drift from reality.

