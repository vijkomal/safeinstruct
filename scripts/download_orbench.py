#!/usr/bin/env python3
"""Download OR-Bench-Hard-1K from Hugging Face and save as JSONL for eval."""

import csv
import io
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "benchmarks" / "orbench" / "orbench_hard_1k.jsonl"
HF_URL = "https://huggingface.co/datasets/bench-llm/or-bench/resolve/main/or-bench-hard-1k.csv"


def main():
    with urllib.request.urlopen(HF_URL) as resp:
        content = resp.read().decode("utf-8")
    rows = []
    for i, ex in enumerate(csv.DictReader(io.StringIO(content))):
        rows.append({
            "id": f"orbench_{i:05d}",
            "prompt": ex["prompt"],
            "Behavior": ex["prompt"],  # run_eval_modal uses Behavior or prompt
            "category": ex.get("category", ""),
        })
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} prompts → {OUT_PATH}")


if __name__ == "__main__":
    main()
