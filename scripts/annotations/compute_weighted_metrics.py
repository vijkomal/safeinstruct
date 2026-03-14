#!/usr/bin/env python3
"""
Compute weighted evaluation metrics using Krippendorff alpha as judge weights.

Weights judges by their human-agreement (Krippendorff α): higher α = more weight.
Outputs a JSON with weighted metrics for inclusion in the report.
"""

import json
from pathlib import Path

# From annotation_summary.json
KRIPPENDORFF_BY_MODEL = {
    "Qwen2.5-7B": 0.393,
    "Mistral-7B": 0.361,
    "Qwen2.5-14B": 0.220,
}

# Normalize to sum to 1
total = sum(KRIPPENDORFF_BY_MODEL.values())
WEIGHTS = {k: v / total for k, v in KRIPPENDORFF_BY_MODEL.items()}

# Judge order in tables: Qwen2.5-7B, Mistral-7B, Qwen2.5-14B
JUDGE_ORDER = ["Qwen2.5-7B", "Mistral-7B", "Qwen2.5-14B"]


def weighted_avg(values: list[float]) -> float:
    """Weighted average: values correspond to JUDGE_ORDER."""
    return sum(WEIGHTS[j] * v for j, v in zip(JUDGE_ORDER, values))


def main():
    # Raw rates from report tables (Qwen7B, Mistral, Qwen14B)
    baseline_harmbench = {
        "Gemma-2-2B-it": [95.2, 96.0, 85.2],
        "GPT-Neo 2.7B": [13.5, 8.5, 8.0],
        "TinyLlama-1.1B": [33.8, 17.2, 9.8],
    }
    gemma_harmbench_baseline = [95.2, 96.0, 85.2]
    gemma_harmbench_ft = [80.5, 97.8, 94.5]
    gemma_alpaca_ft = [94.4, 96.9, 86.3]

    gptneo_harmbench_baseline = [13.5, 8.5, 8.0]
    gptneo_harmbench_ft = [24.8, 28.0, 16.0]
    gptneo_alpaca_baseline = [75.5, 57.9, 24.7]
    gptneo_alpaca_ft = [74.9, 83.2, 32.4]

    tinyllama_harmbench_baseline = [33.8, 17.2, 9.8]
    tinyllama_harmbench_ft = [39.1, 28.2, 14.2]
    tinyllama_alpaca_baseline = [73.3, 94.9, 43.5]  # from alpaca-usefulness-baseline.txt (Qwen7B, Mistral, Qwen14B)
    tinyllama_alpaca_ft = [91.7, 91.6, 66.3]

    out = {
        "weights": WEIGHTS,
        "weights_note": "Weights from Krippendorff α (human agreement); normalized to sum to 1.",
        "baseline_harmbench": {
            model: round(weighted_avg(vals), 1) for model, vals in baseline_harmbench.items()
        },
        "gemma": {
            "harmbench_baseline": round(weighted_avg(gemma_harmbench_baseline), 1),
            "harmbench_ft": round(weighted_avg(gemma_harmbench_ft), 1),
            "harmbench_delta_pp": round(weighted_avg(gemma_harmbench_ft) - weighted_avg(gemma_harmbench_baseline), 1),
            "alpaca_ft": round(weighted_avg(gemma_alpaca_ft), 1),
        },
        "gptneo": {
            "harmbench_baseline": round(weighted_avg(gptneo_harmbench_baseline), 1),
            "harmbench_ft": round(weighted_avg(gptneo_harmbench_ft), 1),
            "harmbench_delta_pp": round(weighted_avg(gptneo_harmbench_ft) - weighted_avg(gptneo_harmbench_baseline), 1),
            "alpaca_baseline": round(weighted_avg(gptneo_alpaca_baseline), 1),
            "alpaca_ft": round(weighted_avg(gptneo_alpaca_ft), 1),
            "alpaca_delta_pp": round(weighted_avg(gptneo_alpaca_ft) - weighted_avg(gptneo_alpaca_baseline), 1),
        },
        "tinyllama": {
            "harmbench_baseline": round(weighted_avg(tinyllama_harmbench_baseline), 1),
            "harmbench_ft": round(weighted_avg(tinyllama_harmbench_ft), 1),
            "harmbench_delta_pp": round(weighted_avg(tinyllama_harmbench_ft) - weighted_avg(tinyllama_harmbench_baseline), 1),
            "alpaca_baseline": round(weighted_avg(tinyllama_alpaca_baseline), 1),
            "alpaca_ft": round(weighted_avg(tinyllama_alpaca_ft), 1),
            "alpaca_delta_pp": round(weighted_avg(tinyllama_alpaca_ft) - weighted_avg(tinyllama_alpaca_baseline), 1),
        },
    }

    root = Path(__file__).resolve().parents[2]
    out_path = root / "data" / "annotations" / "weighted_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
