# Human Annotation — 3-Judge Agreement

## Task
You see **3 judge scores** (Judge 1, Judge 2, Judge 3) for each response. Rate how much you **agree** with each judge (1–5):

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

## Source
- 20 examples from Gemma, TinyLlama (FT)

## Files
- `annotator_<name>.jsonl` / `annotator_<name>.csv`: Your batch (20 examples)

## After annotating
```bash
python scripts/annotations/process_annotations.py --annotation-dir data/annotations
```
