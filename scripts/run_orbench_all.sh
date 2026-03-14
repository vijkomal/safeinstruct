#!/bin/bash
# Run OR-Bench-Hard-1K for all model variants (~1K prompts × 4–6 models ≈ 4–6K generations).
# Expect a few hours on Modal. Requires: data/benchmarks/orbench/orbench_hard_1k.jsonl (run download_orbench.py first).

set -e
cd "$(dirname "$0")/.."

# Ensure OR-Bench data exists
if [[ ! -f data/benchmarks/orbench/orbench_hard_1k.jsonl ]]; then
  echo "Downloading OR-Bench-Hard-1K..."
  python3 scripts/download_orbench.py
fi

echo "Running OR-Bench-Hard-1K for all model variants..."
echo ""

# 1. Gemma base
echo "=== 1/6 Gemma base ==="
modal run scripts/run_eval_modal.py::orbench \
  --model base \
  --base-model google/gemma-2-2b-it

# 2. Gemma FT
echo ""
echo "=== 2/6 Gemma FT ==="
modal run scripts/run_eval_modal.py::orbench \
  --model ft \
  --base-model google/gemma-2-2b-it \
  --ft-model-path /root/results/checkpoints/generator_single_layer_ft

# 3. GPT-Neo base
echo ""
echo "=== 3/6 GPT-Neo base ==="
modal run scripts/run_eval_modal.py::orbench \
  --model base \
  --base-model EleutherAI/gpt-neo-2.7B

# 4. GPT-Neo FT
echo ""
echo "=== 4/6 GPT-Neo FT ==="
modal run scripts/run_eval_modal.py::orbench \
  --model ft \
  --base-model EleutherAI/gpt-neo-2.7B \
  --ft-model-path /root/results/checkpoints/generator_gptneo_ft

# 5. TinyLlama base
echo ""
echo "=== 5/6 TinyLlama base ==="
modal run scripts/run_eval_modal.py::orbench \
  --model base \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# 6. TinyLlama FT (requires generator_tinyllama_ft on Modal volume; comment out if not available)
echo ""
echo "=== 6/6 TinyLlama FT ==="
modal run scripts/run_eval_modal.py::orbench \
  --model ft \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --ft-model-path /root/results/checkpoints/generator_tinyllama_ft

echo ""
echo "Done. Results in results/orbench/"
