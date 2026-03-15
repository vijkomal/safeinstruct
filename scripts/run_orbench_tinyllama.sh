#!/bin/bash
# Run OR-Bench-Hard-1K for TinyLlama only (base + FT).
# FT: uses --local-ft-path if checkpoint is local; or --ft-model-path if on Modal volume.

set -e
cd "$(dirname "$0")/.."

if [[ ! -f data/benchmarks/orbench/orbench_hard_1k.jsonl ]]; then
  echo "Downloading OR-Bench-Hard-1K..."
  python3 scripts/download_orbench.py
fi

echo "Running OR-Bench-Hard-1K (TinyLlama, 2 models)..."
echo ""

echo "=== 1/2 TinyLlama base ==="
modal run scripts/run_eval_modal.py::orbench \
  --model base \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0

echo ""
echo "=== 2/2 TinyLlama FT ==="
modal run scripts/run_eval_modal.py::orbench \
  --model ft \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --local-ft-path results/checkpoints/checkpoints/tinyllama

echo ""
echo "Done. Results in results/orbench/"
