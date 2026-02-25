# safeinstruct

CS224N final project: safe instruction-following with self-instruct-style data generation and an **external verifier**.

**Data pipeline:** (1) Small seed set (~50 safe pivot examples) → (2) model generates more examples → (3) verifier filters unsafe examples → (4) fine-tune on the verified dataset.

**Where you indicate dataset(s):** All paths are set in **`configs/default.yaml`** (no fixed external dataset). The code in `src/data/` reads these keys:

| What | Config key |
|------|------------|
| Seed pivot examples (~50) | `data.seed_pivots_path` |
| Constitution(s) | `data.constitution_path` |
| Prompts for generation (optional) | `data.prompt_source` |
| Verified dataset (output of 2+3; used for training) | `data.verified_path` |

## Setup

```bash
conda env create -f environment.yml
conda activate safeinstruct
```

For Gemma models (e.g. `google/gemma-2-2b`), first accept access on the model page, then login:

1. Open `https://huggingface.co/google/gemma-2-2b`
2. Click **Agree and access repository** (or equivalent access button)
3. Login locally:

```bash
hf auth login
```

## Run

- Generate + verify loop (minimal):
  - `python scripts/run_generate_verify.py`
  - Uses `model.generator_name` from `configs/default.yaml` to generate responses
  - Uses `data.prompt_source` prompts (e.g. `hf://declare-lab/HarmfulQA:train`) when provided
  - Uses `data/seed_pivots.jsonl` response style as safe pivot exemplars
  - With `generation.external_prompt_streaming: true`, prompts are streamed from HF and randomly shuffled via buffer sampling (not just first N prompts)
  - Saves raw generated pairs (pre-verifier) to `results/generated_candidates.jsonl`
  - Saves rejected pairs + verifier outputs to `results/rejected_candidates.jsonl`
  - Writes verified examples to `results/verified_data.jsonl`
  - Optional: set `generation.verify_only_from_generated_candidates: true` to skip generation and run verifier only on `results/generated_candidates.jsonl`

## Layout

- `configs/` — experiment config (data paths, model names, hyperparams).
- `src/data` — data loaders and preprocessing.
- `src/model` — generator and verifier.
- `src/eval` — metrics and evaluation.
- `scripts/` — `run_train.py`, `run_eval.py`, etc.
- `results/` — checkpoints, predictions, result tables.
