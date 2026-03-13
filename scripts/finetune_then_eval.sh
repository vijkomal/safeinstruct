#!/usr/bin/env bash
# Run Modal fine-tuning, then eval when done.
# Usage: ./scripts/finetune_then_eval.sh

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== Step 1: Fine-tune GPT-Neo on Modal ==="
CONFIG_PATH=configs/gptneo_ft.yaml modal run scripts/modal_finetune.py \
  --use-local-data \
  --local-verified-path results/gptneo_verified_data.jsonl \
  --target-filename gptneo_verified_data.jsonl

echo ""
echo "=== Step 2: Run eval (HarmBench + Alpaca) ==="
modal run scripts/run_eval_modal.py::main \
  --model ft \
  --inputs "data/benchmarks/harmbench/harmbench.jsonl:results/harmbench_gptneo_ft.jsonl,data/benchmarks/alpaca/alpaca.jsonl:results/alpaca_gptneo_ft.jsonl" \
  --limit 400

echo ""
echo "=== Done. Check results/harmbench_gptneo_ft.jsonl, results/alpaca_gptneo_ft.jsonl ==="
