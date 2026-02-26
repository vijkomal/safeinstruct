# safeinstruct

CS224N final project: safe instruction-following with self-instruct-style data generation and an **external verifier**.

**Data pipeline:** (1) Small seed set (~50 safe pivot examples) → (2) model generates more examples → (3) verifier filters unsafe examples → (4) fine-tune on the verified dataset.

**Where you indicate dataset(s):** All paths are set in **`configs/default.yaml`**. The loader merges seeds from multiple sources at runtime:

| What | Config key | Path | Provenance |
|------|------------|------|------------|
| Principle-based manual seeds (primary) | `data.seed_principles_path` | `data/seed_principles.json` | In git |
| Flat manual seeds (fallback) | `data.seeds_manual_path` | `data/seeds_manual.jsonl` | In git |
| HarmfulQA seeds (regenerable) | `data.seeds_harmfulqa_path` | `data/seeds_harmfulqa.jsonl` | Run `import_harmfulqa.py` |
| Prompts for generation (optional) | `data.prompt_source` | — | — |
| Verified dataset (output of generation) | `data.verified_path` | `results/verified_data.jsonl` | Generated |

## Setup

```bash
conda env create -f environment.yml
conda activate safeinstruct
```

For Gemma models (e.g. `google/gemma-2-2b`), first accept access on the model page, then login:

1. Open `https://huggingface.co/google/gemma-2-2b` and **accept the model's terms and conditions** (click "Agree and access repository").
2. Login locally in the CLI:

```bash
huggingface-cli login
```

(If using `hf` alias: `hf auth login`)

**Important:** If you created a *fine-grained* token, it must have **"Access gated repos"** enabled. Go to https://huggingface.co/settings/tokens, edit your token, and turn on that permission. Or create a *classic* token (Read or Write)—those have gated access by default.

### Troubleshooting

**OMP / shared memory error** (`Can't open SHM2`): Common in Cursor, Docker, or sandboxed environments. Run with:

```bash
OMP_NUM_THREADS=1 python scripts/run_generate_verify.py
```

**Quick debugging** (fewer seeds to isolate bugs): Limit seeds with `SEED_LIMIT`:

```bash
SEED_LIMIT=5 OMP_NUM_THREADS=1 python scripts/run_generate_verify.py
```

**Ungated model for local testing**: If Gemma access fails, use an ungated model without editing config:

```bash
GENERATOR_MODEL=HuggingFaceTB/SmolLM-360M-Instruct SEED_LIMIT=3 OMP_NUM_THREADS=1 python scripts/run_generate_verify.py
```

(When `GENERATOR_MODEL` is set, the verifier uses the same model by default, so you only load one ungated model.)

## Seed Data

Seeds are loaded from two sources and merged at runtime:

1. **`data/seeds_manual.jsonl`** — Hand-curated examples (in git). Add your own prompt/response pairs here; include `"source": "manual"` for provenance.
2. **`data/seeds_harmfulqa.jsonl`** — Import from [HarmfulQA](https://huggingface.co/datasets/declare-lab/HarmfulQA). Regenerate with:

```bash
conda activate safeinstruct
python scripts/import_harmfulqa.py
```

By default 100 examples; `HARMFULQA_LIMIT=0` imports all ~1,960. If HuggingFace cache fails: `HF_HOME=./.cache/huggingface python scripts/import_harmfulqa.py`.

**Schema (both sources):** `{"prompt": "...", "response": "...", "safety_label": "safe", "source": "manual"|"harmfulqa"}`. Optional: `reasoning`, `violation_type`, `safe_strategy`.

**Reproducibility:** Manual seeds are fixed in git. For HarmfulQA, run the import script with `SEED=42` and `HARMFULQA_LIMIT=100` (default) to match reported experiments.

## Run

- Generate + verify loop (minimal):
  - `python scripts/run_generate_verify.py`
  - Uses `model.generator_name` from `configs/default.yaml` to generate responses
  - Uses `data.prompt_source` prompts (e.g. `hf://declare-lab/HarmfulQA:train`) when provided
  - HF prompt sources are always loaded with streaming + random shuffle
  - Set `data.prompt_source: null` to use only seed prompts
  - Uses merged seeds (`seed_principles` + `seeds_harmfulqa` + optional prosocial) as safe pivot exemplars
  - `generation.max_seeds`: 0 = use all; N = randomly sample N seeds (uses config seed for reproducibility)
  - For principle-based seeds, the top-level `instruction` is injected into generation prompt context
  - Prompt context toggles:
    - `generation.use_persona_context`
    - `generation.use_instruction_context`
    - `generation.use_principle_context`
    - Set any of these to `false` to disable that context block
  - Generator output is strict JSON-only (`{"prompt": "...", "response": "..."}`); non-JSON outputs are dropped
  - Responses are intentionally short (prompt asks for <=2 sentences); `generation.max_new_tokens` controls hard length cap
  - Verifier is strict-label: only `PASS` or `NO_PASS` accepted
  - `model.load_in_4bit: true` — 4-bit quantization (faster, less memory; requires CUDA; ignored on CPU/MPS)
  - Saves raw generated pairs (pre-verifier) to `results/generated_candidates.jsonl`
  - Saves rejected pairs + verifier outputs to `results/rejected_candidates.jsonl`
  - Writes verified examples to `results/verified_data.jsonl`
  - Optional: set `generation.verify_only_from_generated_candidates: true` to skip generation and run verifier only on `results/generated_candidates.jsonl`

- Naive single-layer fine-tuning (generator):
  - `python scripts/run_finetune.py`
  - Trains only `lm_head` (all other model params frozen)
  - Uses verified examples from `data.verified_path`
  - Saves model to `train.output_dir/train.save_name`

- Simple eval (sanity check):
  - `python scripts/run_eval.py`
  - Uses seed prompts as evaluation inputs
  - Runs both models: base `model.generator_name` and fine-tuned checkpoint at `train.output_dir/train.save_name`
  - Writes side-by-side generations to `results/eval_outputs.jsonl`

## Layout

- `configs/` — experiment config (data paths, model names, hyperparams).
- `src/data` — data loaders and preprocessing.
- `src/model` — generator and verifier.
- `src/eval` — metrics and evaluation.
- `scripts/` — `run_generate_verify.py`, `run_finetune.py`, `run_eval.py`, etc.
- `results/` — checkpoints, predictions, result tables.
