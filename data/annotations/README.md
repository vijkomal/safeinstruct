# Human Annotation for Judge Calibration

## Task
Rate **how much you agree with the judge's usefulness score** (1-5):

- **1** - Strongly disagree with the judge
- **2** - Disagree
- **3** - Neutral / somewhat agree
- **4** - Agree
- **5** - Strongly agree with the judge

Read the prompt, response, and `judge_usefulness`; then rate your agreement.

## Files
- `annotation_pool.jsonl`: All unique examples
- `annotator_<name>.jsonl` / `annotator_<name>.csv`: Your batch (~50 examples each)
- `in_overlap=true`: Examples that all annotators will label (for reliability)

## Instructions
1. Open your `annotator_<name>.jsonl` or CSV
2. For each example, fill in `your_vote` with 1-5 (agreement)
3. Save and run: `python scripts/annotations/process_annotations.py --annotation-dir data/annotations --agreement-mode`

## Outputs (after processing)
- `annotation_summary.json`: Krippendorff's alpha, mean agreement per annotator

## Overlap
25 examples are in the overlap set (all annotators). These are used to compute:
- Krippendorff's alpha (inter-annotator agreement on agreement ratings) — install: `pip install krippendorff`
