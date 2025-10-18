# CLAUDE.md

Guidelines for Claude Code when collaborating on this repository.

## Repository Snapshot
- Fork `davorchap/tilelang-tt` hosts the Tenstorrent backend effort: layout-aware metadata passes, shard-aware persistent lowering, and IR-driven reader/compute/writer/host generators.
- Python orchestration lives in `tilelang/tenstorrent`, compiler passes in `src/transform/tenstorrent/`, and codegen visitors in `src/target/tenstorrent/`. Tests reside under `testing/python/tenstorrent/`.
- As of 2025-10-10 the host artifact emits a metadata summary (`main.cpp`) aligned with runtime argument schemas; guardrails ensure shard completeness across visitors.

## Build & Test Quickstart
1. Create an isolated environment: `python -m venv .venv && source .venv/bin/activate`.
2. Install Tenstorrent dependencies: `pip install -r requirements-tenstorrent.txt` (uses CPU-only PyTorch, saves ~5GB).
3. Run the mock-mode CI replica: `bash maint/scripts/local_build_and_test_tt.sh --skip-deps --jobs 4`.
4. For SDK-backed runs, export `TT_METAL_HOME` and add `--with-metalium`.
5. Standalone tests:
   ```bash
   # Always activate venv first and set LD_LIBRARY_PATH
   source .venv/bin/activate
   export LD_LIBRARY_PATH=build/tvm:$LD_LIBRARY_PATH
   pytest testing/python/tenstorrent/ -v

   # Quick summary:
   pytest testing/python/tenstorrent/ --tb=no -q

   # With line-level traceback:
   pytest testing/python/tenstorrent/ -v --tb=line
   ```
6. Enforce formatting and lint baselines with `source .venv/bin/activate && bash format.sh` before staging commits. **Important:** Always run `format.sh` from within the virtual environment to avoid dependency issues.

**Note:** Use `requirements-tenstorrent.txt` for TT backend development. It includes CPU-only PyTorch and excludes CUDA dependencies, preventing disk space issues.

## Running Examples
To run TileLang examples with the TT backend:
```bash
# Always activate venv first
source .venv/bin/activate

# Run an example (e.g., GEMM)
python examples/tenstorrent/example_gemm_tt_minimal.py

# Enable IR dumping to debug compiler passes
export TT_DUMP_IR=1
export TT_IR_DUMP_DIR=ir_dumps  # Optional: default is "tt_pass_ir"
python examples/tenstorrent/example_gemm_tt_minimal.py

# IR dumps will be in the specified directory with format:
# 00_initial.tir, 01_infer_tt_layout_v5.tir, 02_propagate_tt_layout_v5.tir, etc.
```

**Artifacts Location:**
- Generated C++ files are cached in `~/.tilelang/cache/<hash>/` with files:
  - `reader.cpp`, `compute.cpp`, `writer.cpp` - Device kernels
  - `main.cpp` - Host program
  - `tt.plan.json` - Runtime execution plan
  - `CMakeLists.txt` - Build configuration

## Workflow Expectations
- Branch from `main` and open PRs against `davorchap/tilelang-tt`; upstream `tile-ai/tilelang` stays read-only for TT backend development.
- Keep commits focused, imperative, and documented—reference any touched docs or scripts when workflows change.
- PR descriptions must summarise behavioural changes, list validation commands (e.g., `pytest testing/python/tenstorrent/ -v`), and link doc updates such as `docs/tenstorrent/CI.md`.
- Coordinate interface changes (runtime metadata, pass ordering) with matching test and documentation updates in the same PR.
- Use the GitHub CLI (`gh`) to draft and submit pull requests when changes are ready.

## Reference Documents
- Architecture & runtime contracts: `docs/tenstorrent/architecture/TT_ARCHITECTURE.md`
- Backend overview & doc index: `docs/tenstorrent/README.md`
- CI/local parity workflows: `docs/tenstorrent/setup/CI.md`
- SDK setup for hardware validation: `docs/tenstorrent/setup/METALIUM_SETUP_GUIDE.md`
- Implementation plan & tasks: `docs/tenstorrent/planning/TT_Implementation_Plan.md`, `docs/tenstorrent/planning/TT_BACKEND_TASKS.md`
- Pass reference & analysis: `docs/tenstorrent/architecture/IR_LOWERING_ANALYSIS.md`, `docs/tenstorrent/reference/PASS_TABLE_*.md`
- Build deep-dive & scripts: `docs/tenstorrent/setup/local_build_guide.md`

Review these files when making changes; update them if instructions drift from reality.
