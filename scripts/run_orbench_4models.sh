#!/bin/bash
# Run OR-Bench-Hard-1K for Gemma + GPT-Neo (4 model variants).
# Use this if TinyLlama FT checkpoint is not on Modal. For all 6 models, use run_orbench_all.sh

set -e
cd "$(dirname "$0")/.."

if [[ ! -f data/benchmarks/orbench/orbench_hard_1k.jsonl ]]; then
  echo "Downloading OR-Bench-Hard-1K..."
  python3 scripts/download_orbench.py
fi

echo "Running OR-Bench-Hard-1K (Gemma + GPT-Neo, 4 models)..."
echo ""

for i in 1 2 3 4; do
  case $i in
    1) echo "=== 1/4 Gemma base ==="
       modal run scripts/run_eval_modal.py::orbench --model base --base-model google/gemma-2-2b-it ;;
    2) echo "=== 2/4 Gemma FT ==="
       modal run scripts/run_eval_modal.py::orbench --model ft --base-model google/gemma-2-2b-it --ft-model-path /root/results/checkpoints/generator_single_layer_ft ;;
    3) echo "=== 3/4 GPT-Neo base ==="
       modal run scripts/run_eval_modal.py::orbench --model base --base-model EleutherAI/gpt-neo-2.7B ;;
    4) echo "=== 4/4 GPT-Neo FT ==="
       modal run scripts/run_eval_modal.py::orbench --model ft --base-model EleutherAI/gpt-neo-2.7B --ft-model-path /root/results/checkpoints/generator_gptneo_ft ;;
  esac
  echo ""
done

echo "Done. Results in results/orbench/"
