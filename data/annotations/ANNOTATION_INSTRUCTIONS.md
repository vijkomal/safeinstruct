# Human Annotation — 3-Judge Agreement

**Goal:** Each annotator rates how much they agree (1–5) with each of 3 judge scores. We use this to calibrate our LLM judges and report inter-annotator reliability.

---

## Models & Data Sources

| Model | Baseline | Finetuned (FT) |
|-------|----------|----------------|
| **Gemma-2-2B** | `results/baseline/gemma-2-2b-it/` | `results/harmbench_gemma_ft.jsonl`, `results/alpaca_gemma_ft.jsonl` |
| **TinyLlama-1.1B** | `results/baseline/tiny-llama-1.1b/` | `results/harmbench_tinyllama_ft.jsonl`, `results/alpaca_tinyllama_ft.jsonl` |
| **GPT-Neo-2.7B** | `results/baseline/gpt-neo/` | `results/ft/gpt-neo/` |

---

## What You Need to Do

### 1. Open your file

- **Tim:** `data/annotations/annotator_tim.jsonl` or `annotator_tim.csv`
- **Komal:** `data/annotations/annotator_komal.jsonl` or `annotator_komal.csv`
- **J:** `data/annotations/annotator_j.jsonl` or `annotator_j.csv`

### 2. Rate agreement with each judge (1–5)

You see **3 judge scores** (Judge 1, Judge 2, Judge 3) for each response. Rate how much you **agree** with each judge:

| Score | Meaning |
|-------|---------|
| **5** | Strongly agree with this judge |
| **4** | Agree |
| **3** | Neutral / somewhat agree |
| **2** | Disagree |
| **1** | Strongly disagree with this judge |

**Example:** Judge 1 says 3, Judge 2 says 4, Judge 3 says 5. You think the true score should be 3.
→ Give **5, 4, 3** (agree most with Judge 1, kinda with Judge 2, not really with Judge 3).

**Fill in:** `your_vote_1`, `your_vote_2`, `your_vote_3` for each example.

### 3. Save and run

```bash
cd safeinstruct
python scripts/annotations/process_annotations.py --annotation-dir data/annotations
```

Outputs: Krippendorff's alpha per judge, mean agreement per annotator, mean agreement per judge position.

---

## Exporting the Annotation Pool

```bash
cd safeinstruct
```

**Option A: Baseline only (10 per model)**  
Gemma, TinyLlama, GPT-Neo baselines:
```bash
python scripts/annotations/export_for_annotation.py --mix-models --n-per-model 10
```

**Option B: Baseline + FT (10 per model per variant)**  
Full structure: 60 examples (10 × 3 models × 2 variants):
```bash
python scripts/annotations/export_for_annotation.py --mix-models --include-ft --n-per-model 10
```

**Option C: FT only (recommended for calibration)**  
30 examples (10 per model) — avoids GPT-Neo baseline whitespace/repetition; all responses have substantive content:
```bash
python scripts/annotations/export_for_annotation.py --mix-models --ft-only --n-per-model 10
```

**Option C': FT only, exclude GPT-Neo**  
20 examples (10 per model × Gemma + TinyLlama):
```bash
python scripts/annotations/export_for_annotation.py --mix-models --ft-only --exclude-gptneo --n-per-model 10
```

**Option D: Custom sources**  
Use specific files with all 3 judges (overrides --mix-models):
```bash
python scripts/annotations/export_for_annotation.py \
  --three-judge-sources results/baseline/alpaca-results-baseline.jsonl --n-per-model 10
```

**Source:** All 3 judges must have scored each example. Judge identity stored in `judge_mapping.json` (internal only).

---

## File Structure

```
data/annotations/
├── ANNOTATION_INSTRUCTIONS.md   # This file
├── README.md                    # Quick task for annotators
├── annotation_pool.jsonl        # All examples (no judge identity)
├── judge_mapping.json           # Internal: Judge 1/2/3 → Qwen7/Mistral/Qwen14
├── annotator_tim.jsonl          # Tim's batch
├── annotator_tim.csv
├── annotator_komal.jsonl
├── annotator_komal.csv
├── annotator_j.jsonl
├── annotator_j.csv
└── annotation_summary.json      # Output after process_annotations.py
```

---

## Checklist

- [ ] Tim: Fill `your_vote_1`, `your_vote_2`, `your_vote_3` (1–5 each) for all examples
- [ ] Komal: Fill `your_vote_1`, `your_vote_2`, `your_vote_3` for all examples
- [ ] J: Fill `your_vote_1`, `your_vote_2`, `your_vote_3` for all examples
- [ ] Run `process_annotations.py --annotation-dir data/annotations` and share `annotation_summary.json`
