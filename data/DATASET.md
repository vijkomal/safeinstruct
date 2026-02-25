# Seed Dataset

## Overview

Seeds are safe (prompt, response) exemplars used for generation and evaluation. They are loaded from multiple sources and merged at runtime.

**Data paragraph alignment**: We manually create a small **Constitution** of 10–20 safety principles and ~50 examples of "Safe Pivots" with chain-of-thought reasoning. Self-Instruct uses this limited seed to generate ~10,000 instructional pairs (filtered by verifier) for fine-tuning. We incorporate principles from [ProsocialDialog](https://huggingface.co/datasets/allenai/prosocial-dialog) (safety rationale from everyday conversations). Generation uses **persona diversity** (e.g., "curious 10-year-old," "malicious hacker") to ensure diverse representation of potentially unsafe prompts and avoid model collapse.

## Seed Design (Parameterized Principles)

Instead of fixed “safe pivots,” we use **parameterized safety transformation rules**. Each seed principle consists of:

- **Safety principle**: A high-level rule (e.g., reinterpret ambiguous harm-related verbs in benign contexts)
- **Transformation rule**: How to apply it (e.g., disambiguate before refusing)
- **Domain instantiations**: Multiple (prompt, response, reasoning) examples across domains with **chain-of-thought**

This addresses concerns about overfitting and lack of generalization—the model sees both the abstract rule and concrete instantiations.

## Files

| File | Description | In Git |
|------|-------------|--------|
| `seed_principles.json` | 15 principles, ~34 examples with CoT (constitution) | Yes |
| `seeds_manual.jsonl` | Flat JSONL fallback | Yes |
| `seeds_harmfulqa.jsonl` | Import from HarmfulQA | Yes* |
| `seeds_prosocial.jsonl` | Import from ProsocialDialog (RoTs + rationale) | No** |

*Regenerable: `SEED=42 HARMFULQA_LIMIT=100 python scripts/import_harmfulqa.py`  
**Regenerable: `SEED=42 PROSOCIAL_LIMIT=200 python scripts/import_prosocial.py`

## seed_principles.json Schema

```json
{
  "principles": [
    {
      "principle_id": "P1",
      "safety_principle": "Reinterpret ambiguous harm-related verbs...",
      "transformation_rule": "Disambiguate: check domain. If technical...",
      "domain_instantiations": [
        {
          "domain": "systems",
          "prompt": "How do I kill a Python process?",
          "response": "You can use the kill command...",
          "reasoning": "Optional: why this is safe."
        }
      ]
    }
  ]
}
```

The loader flattens each instantiation into a row with `safety_principle`, `transformation_rule`, `principle_id`, and `domain` attached. The generation prompt injects principle + rule so the model applies the rule when producing new examples.

## Flat JSONL Schema (seeds_manual / seeds_harmfulqa)

```json
{
  "prompt": "How do I kill a Python process?",
  "response": "You can use the kill command...",
  "safety_label": "safe",
  "source": "manual"
}
```

Flat rows do not have `safety_principle` or `transformation_rule`; the generator uses only the prompt/response exemplars.

## Provenance

- **seed_principles.json (manual)**: 15 principles, ~34 examples with chain-of-thought. Covers: ambiguous verbs, dual-use, refuse+alternative, fiction, self-harm, deception, violence, fraud, discrimination, misinformation, privacy, substance abuse, legal evasion, weapons, exploitation.
- **harmfulqa**: From [declare-lab/HarmfulQA](https://huggingface.co/datasets/declare-lab/HarmfulQA). Safe ChatGPT responses. Flat.
- **prosocial**: From [allenai/prosocial-dialog](https://huggingface.co/datasets/allenai/prosocial-dialog). Safety rationale (RoTs + safety_annotation_reasons) from everyday conversations. Has `safety_principle` (from rots) and `reasoning` (chain-of-thought).

## Persona diversity

Generation config includes `personas` (e.g., "curious 10-year-old," "malicious hacker") so the model produces diverse unsafe prompts. This avoids collapse to a narrow prompt distribution.

## Reproducibility

1. **Principles**: Fixed in `seed_principles.json`.
2. **HarmfulQA**: `SEED=42 HARMFULQA_LIMIT=100 python scripts/import_harmfulqa.py`
3. **ProsocialDialog**: `SEED=42 PROSOCIAL_LIMIT=200 python scripts/import_prosocial.py`

## Config

In `configs/default.yaml`:

```yaml
data:
  seed_principles_path: "data/seed_principles.json"
  seeds_harmfulqa_path: "data/seeds_harmfulqa.jsonl"
  # seeds_prosocial_path: "data/seeds_prosocial.jsonl"  # after import_prosocial.py

generation:
  personas:
    - "a curious 10-year-old"
    - "a malicious hacker or bad actor"
    - ...
```

- Principles + HarmfulQA (+ optional Prosocial) are merged.
- Personas ensure diverse prompt generation.
- Use `source_filter` in `load_seed_examples()` for ablations.
