# Human Annotation Workflow for Judge Calibration

This document describes the human annotation setup for calibrating the LLM judge usefulness scores.

## Quick Start (for annotators)

**→ See [data/annotations/ANNOTATION_INSTRUCTIONS.md](annotations/ANNOTATION_INSTRUCTIONS.md) for step-by-step instructions.**

## Overview

- **Goal:** Validate judge scores with human votes; compute inter-annotator reliability and calibration weight.
- **Design:** Each annotator labels ~50 examples; ~25 overlap (all annotators) for Krippendorff's alpha.
- **Output:** `annotation_summary.json` with Krippendorff's alpha, per-annotator judge agreement, calibration weight.

## Workflow

### 1. Export annotation batches (already done; pool exists)

```bash
python3 scripts/annotations/export_for_annotation.py \
  --harmbench results/baseline/gemma-2-2b-it/harmbench_baseline_gemma.jsonl \
  --alpaca results/baseline/gemma-2-2b-it/alpaca_baseline_gemma.jsonl \
  --output-dir data/annotations \
  --overlap 25 \
  --per-annotator 50 \
  --annotators tim komal j
```

Creates: `annotation_pool.jsonl`, `annotator_<name>.jsonl`, `annotator_<name>.csv`

### 2. Annotate

Each annotator fills `your_vote` (1–5) in their file. Scale: 1 = refused/irrelevant, 5 = fully fulfilled.

### 3. Process annotations

```bash
pip install krippendorff  # optional
python scripts/annotations/process_annotations.py --annotation-dir data/annotations
```

Outputs: Krippendorff's alpha, per-annotator judge agreement, calibration weight, `annotation_summary.json`

### 4. Use in paper

Report Krippendorff's alpha and calibration weight when discussing judge-weighted usefulness.
