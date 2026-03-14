#!/bin/bash
# Run OR-Bench-Hard-1K for GPT-Neo only (base + FT).
# Requires: generator_gptneo_ft checkpoint on Modal volume.

set -e
cd "$(dirname "$0")/.."

if [[ ! -f data/benchmarks/orbench/orbench_hard_1k.jsonl ]]; then
  echo "Downloading OR-Bench-Hard-1K..."
  python3 scripts/download_orbench.py
fi

echo "Running OR-Bench-Hard-1K (GPT-Neo, 2 models)..."
echo ""

echo "=== 1/2 GPT-Neo base ==="
modal run scripts/run_eval_modal.py::orbench \
  --model base \
  --base-model EleutherAI/gpt-neo-2.7B

echo ""
echo "=== 2/2 GPT-Neo FT ==="
modal run scripts/run_eval_modal.py::orbench \
  --model ft \
  --base-model EleutherAI/gpt-neo-2.7B \
  --ft-model-path /root/results/checkpoints/generator_gptneo_ft

echo ""
echo "Done. Results in results/orbench/"
