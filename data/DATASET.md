# Seed Dataset

## Overview

Seeds are safe (prompt, response) exemplars used for generation and evaluation. They are loaded from two sources and merged at runtime.

## Files

| File | Size | Provenance | In Git |
|------|------|------------|--------|
| `seeds_manual.jsonl` | 8 examples | Hand-curated | Yes |
| `seeds_harmfulqa.jsonl` | 100 examples (default) | Import from HarmfulQA | Yes* |

*Regenerable via `python scripts/import_harmfulqa.py`

## Schema

Each JSONL row:

```json
{
  "prompt": "How do I kill a Python process?",
  "response": "You can use the kill command...",
  "safety_label": "safe",
  "source": "manual"
}
```

Optional fields: `reasoning`, `violation_type`, `safe_strategy`.

## Provenance

- **manual**: Curated by project authors. Covers benign reinterpretation, dual-use redirect, refusal+alternative, fiction, self-harm support, etc.
- **harmfulqa**: From [declare-lab/HarmfulQA](https://huggingface.co/datasets/declare-lab/HarmfulQA). Safe (blue) ChatGPT responses to harmful questions. Filtered to skip generic closings.

## Reproducibility

1. **Manual seeds**: Fixed in `seeds_manual.jsonl`; no regeneration.
2. **HarmfulQA seeds**:
   ```bash
   SEED=42 HARMFULQA_LIMIT=100 python scripts/import_harmfulqa.py
   ```
   For full dataset: `HARMFULQA_LIMIT=0`

## Config

In `configs/default.yaml`:

```yaml
data:
  seeds_manual_path: "data/seeds_manual.jsonl"
  seeds_harmfulqa_path: "data/seeds_harmfulqa.jsonl"
```

The loader merges both. To use only manual seeds, set `seeds_harmfulqa_path: null` or remove it.
