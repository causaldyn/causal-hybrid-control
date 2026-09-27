#!/usr/bin/env bash
# Build the documentation site into site/, exactly as CI and the Pages deploy do.
#
# Every result a page shows is printed here, by the committed notebook or script the page names: the
# notebooks are executed into docs/tutorials/ and each script's output is captured in docs/output/,
# which the pages include. Both are build artefacts, and git ignores them.
# Needs `uv sync --group docs --group notebooks`.
set -euo pipefail

cd "$(dirname "$0")/.."

# The pages say which precision printed each number, so an x64 flag inherited from the caller's
# shell must not change it; and a pinned CPU keeps JAX's accelerator probe out of the outputs.
unset JAX_ENABLE_X64
export JAX_PLATFORMS=cpu

timeout 3600 uv run --group notebooks jupyter nbconvert --to markdown --execute \
  --ExecutePreprocessor.timeout=1200 --output-dir docs/tutorials notebooks/*.ipynb

mkdir -p docs/output
timeout 1800 uv run python docs/quickstart.py >docs/output/quickstart.txt
for script in flagship_demo epidemic_demo mmm_demo run_marketplace_demo spine_demo \
  run_dynamic_confounding_demo run_benchmark run_causal_bench; do
  timeout 1800 uv run python "scripts/$script.py" >"docs/output/$script.txt"
done

uv run --group docs mkdocs build --strict
