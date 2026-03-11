# Seed Corpus Changes (HarmBench Leakage Fix)

## Summary

Changes address eval leakage and align with Tim's design: **HarmfulQA prompts + manually curated principle responses** as reference for the generator.

## 1. HarmBench Removed from Seeds

**Problem:** Seeds included exact HarmBench eval prompts (e.g., dimethylmercury instructions). Training on these inflates HarmBench scores.

**Fix:** `scripts/build_seed_corpus.py` no longer includes HarmBench refusals. Seeds are now:
- **Principles** (~34): Manual redirection examples
- **HarmfulQA** (~1900): Harmful prompts + responses (responses may not all follow redirection rubric)
- **Template expansion**: Fills to target; no HarmBench overlap

Run: `python scripts/build_seed_corpus.py [--target 2100]`

## 2. Generator: HarmfulQA Prompts + Principle References

**Problem:** Generator used mixed seeds (including HarmBench) as both targets and references.

**Fix:** New mode `generation.use_harmfulqa_prompts_with_principles: true`:
- **Target prompts:** HarmfulQA prompts only (harmful prompts to respond to)
- **Reference responses:** Sample 1 of ~34 principle responses (manual redirection)
- Generator produces synthetic (prompt, response) pairs for fine-tuning

Config (`configs/default.yaml`):
```yaml
generation:
  use_harmfulqa_prompts_with_principles: true
  harmfulqa_prompts_limit: 0  # 0 = use all; set e.g. 200 to limit for faster runs
```

## 3. Redirection Rubric Audit

**Problem:** Seeds should fulfill the redirection rubric (refuse + redirect to constructive alternative).

**Fix:** `scripts/audit_seeds_redirection.py` checks each seed for:
- Refusal phrases (e.g., "I can't", "I cannot")
- Redirect phrases (e.g., "I can suggest", "I can discuss")

Run: `python scripts/audit_seeds_redirection.py [--seeds data/seeds_expanded.jsonl] [--strict]`

- `--strict`: Require BOTH refusal AND redirect
- Output: `data/audit_redirection.json` with per-source compliance stats

## Files Changed

| File | Change |
|------|--------|
| `scripts/build_seed_corpus.py` | Removed HarmBench; docstring updated |
| `scripts/run_generate_verify_modal.py` | Added HarmfulQA+principles mode |
| `scripts/run_generate_verify.py` | Added HarmfulQA+principles mode |
| `configs/default.yaml` | `use_harmfulqa_prompts_with_principles: true` |
| `scripts/audit_seeds_redirection.py` | New audit script |
