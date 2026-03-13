# Manual Voting — Human Annotation for Judge Calibration

**Goal:** Each of us (Tim, Komal, J) rates ~50 examples. We use this to calibrate our LLM judges and report inter-annotator reliability in the paper.

---

## What You Need to Do

### 1. Open your file

- **Tim:** `data/annotations/annotator_tim.jsonl` or `annotator_tim.csv`
- **Komal:** `data/annotations/annotator_komal.jsonl` or `annotator_komal.csv`
- **J:** `data/annotations/annotator_j.jsonl` or `annotator_j.csv`

Use the **CSV** if you prefer spreadsheets (Excel, Google Sheets). Use the **JSONL** if you prefer a text editor.

### 2. Rate each example

For each row, fill in **`your_vote`** with a number **1–5** indicating **how much you agree with the judge's rating**:

| Score | Meaning |
|-------|---------|
| **1** | Strongly disagree with the judge |
| **2** | Disagree |
| **3** | Neutral / somewhat agree |
| **4** | Agree |
| **5** | Strongly agree with the judge |

**What to do:** Read the prompt, the model's response, and the **`judge_usefulness`** score (1–5). Then rate how much you agree with that judge score. You're voting on agreement, not re-rating usefulness.

**Note:** Examples are randomly sampled from **all 3 models** (Gemma, GPT-Neo, TinyLlama) to avoid confounding. The `model` column (in JSONL) indicates which model produced each response.

### 3. Save and run the processing script

After everyone has filled in their votes:

```bash
cd safeinstruct
conda activate safeinstruct
pip install krippendorff   # optional, for Krippendorff's alpha
python scripts/annotations/process_annotations.py --annotation-dir data/annotations --agreement-mode
```

This outputs:
- **Krippendorff's alpha** (inter-annotator agreement on agreement ratings)
- **Mean agreement score** per annotator

---

## File Details

- **25 overlap examples:** Everyone annotates these (for reliability).
- **25 unique per person:** Only you annotate these.
- **~50 total per person.**

---

## If You Need to Re-Export (Optional)

If the annotation pool is empty or you want to regenerate from different results:

**Mix from all 3 models** (recommended — avoids confounding):

```bash
cd safeinstruct
python3 scripts/annotations/export_for_annotation.py --mix-models \
  --output-dir data/annotations \
  --overlap 25 --per-annotator 50 --annotators tim komal j
```

**Single model** (e.g. Gemma only):

```bash
python3 scripts/annotations/export_for_annotation.py \
  --harmbench results/baseline/gemma-2-2b-it/harmbench_baseline_gemma.jsonl \
  --alpaca results/baseline/gemma-2-2b-it/alpaca_baseline_gemma.jsonl \
  --output-dir data/annotations
```

**Source:** With `--mix-models`, randomly samples from Gemma-2-2B, GPT-Neo-2.7B, and TinyLlama-1.1B baselines. Supports both `usefulness` 1–5 and rubric format (`rubric_scores`, `successful_redirection`).

---

## Checklist

- [ ] Tim: Fill `your_vote` (1–5 agreement) for all 50 examples in `annotator_tim.jsonl` or `.csv`
- [ ] Komal: Fill `your_vote` (1–5 agreement) for all 50 examples in `annotator_komal.jsonl` or `.csv`
- [ ] J: Fill `your_vote` (1–5 agreement) for all 50 examples in `annotator_j.jsonl` or `.csv`
- [ ] Run `process_annotations.py` and share `annotation_summary.json` for the paper
