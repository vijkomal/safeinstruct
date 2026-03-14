#!/usr/bin/env python3
"""Download OR-Bench-Hard-1K from Hugging Face and save as JSONL for eval."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "benchmarks" / "orbench" / "orbench_hard_1k.jsonl"


def main():
    from datasets import load_dataset

    ds = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "id": f"orbench_{i:05d}",
            "prompt": ex["prompt"],
            "Behavior": ex["prompt"],  # run_eval_modal uses Behavior or prompt
            "category": ex["category"],
        })
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} prompts → {OUT_PATH}")


if __name__ == "__main__":
    main()
