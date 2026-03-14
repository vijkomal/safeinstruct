# Human Annotation — 3-Judge Agreement

## Task
For each example, you see **3 judge scores** (Judge 1, Judge 2, Judge 3). Judge identity is hidden.
Rate **how much you agree** with each judge's score (1–5):

| Score | Meaning |
|-------|---------|
| **1** | Strongly disagree with this judge |
| **2** | Disagree |
| **3** | Neutral / somewhat agree |
| **4** | Agree |
| **5** | Strongly agree with this judge |

**Fill in:** `your_vote_1`, `your_vote_2`, `your_vote_3` for each of the 3 judges.

## Source
- 20 examples from Gemma, TinyLlama, GPT-Neo (baseline + FT)
- All 3 judges (Qwen2.5-7B, Mistral-7B, Qwen2.5-14B) scored each; identity hidden

## Files
- `annotator_<name>.jsonl` / `annotator_<name>.csv`: Your batch (20 examples)
- `judge_mapping.json`: Internal only (Judge 1/2/3 → Qwen7/Mistral/Qwen14) — not for annotators

## After annotating
```bash
python scripts/annotations/process_annotations.py --annotation-dir data/annotations
```
