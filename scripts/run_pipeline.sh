#!/usr/bin/env bash
# Full pipeline: Generate → Verify → Finetune → Eval (+ HarmBench + MT-Bench)
# Usage:
#   ./scripts/run_pipeline.sh              # default SEED_LIMIT (0 = all seeds)
#   SEED_LIMIT=25 ./scripts/run_pipeline.sh # scale to 25 seeds for more verified data
#   EVAL_BENCHMARKS=1 ./scripts/run_pipeline.sh  # run HarmBench + MT-Bench after eval

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${SEED_LIMIT:=0}"  # 0 = use all seeds; set 20–30 for faster iteration
: "${OMP_NUM_THREADS:=1}"
: "${EVAL_BENCHMARKS:=0}"  # 1 = run HarmBench + MT-Bench eval

echo "=== Step 1: Generate + Verify ==="
SEED_LIMIT="$SEED_LIMIT" OMP_NUM_THREADS="$OMP_NUM_THREADS" python scripts/run_generate_verify.py

echo ""
echo "=== Step 2: Finetune on verified data ==="
python scripts/run_finetune.py

echo ""
echo "=== Step 3: Eval (base vs fine-tuned) ==="
python scripts/run_eval.py

if [ "$EVAL_BENCHMARKS" = "1" ]; then
  echo ""
  echo "=== Step 4a: HarmBench eval (Attack Success Rate) ==="
  HARMBENCH_LIMIT=50 python scripts/run_harmbench_eval.py

  echo ""
  echo "=== Step 4b: MT-Bench eval (refusal rate on safe tasks) ==="
  MTBENCH_LIMIT=20 python scripts/run_mtbench_eval.py
fi

echo ""
echo "=== Done. Check results/eval_outputs.jsonl, results/harmbench_eval.json, results/mtbench_eval.json ==="
