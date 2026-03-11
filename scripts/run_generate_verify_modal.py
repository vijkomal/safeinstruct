#!/usr/bin/env python3
"""Run generation + verification on Modal GPUs for speed.

Usage:
  pip install modal
  modal setup   # one-time auth
  modal run scripts/run_generate_verify_modal.py

  # With options:
  SEED_LIMIT=25 modal run scripts/run_generate_verify_modal.py
  CONFIG_PATH=configs/scale_10k.yaml modal run scripts/run_generate_verify_modal.py

Requires: HF token with Gemma access (modal secret 'huggingface-secret' or HF_TOKEN env).
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Modal runs remotely; we need local imports only for the entrypoint
ROOT = Path(__file__).resolve().parents[1]

try:
    import modal
except ImportError:
    raise ImportError("Install Modal: pip install modal && modal setup")

# --- Build task items (runs locally) ---


def _load_harmfulqa_prompts(base: Path, limit: int = 0) -> List[str]:
    """Load HarmfulQA prompts only (no responses)."""
    path = base / "data" / "seeds_harmfulqa.jsonl"
    if not path.exists():
        return []
    prompts = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            p = (r.get("prompt") or "").strip()
            if p:
                prompts.append(p)
            if limit > 0 and len(prompts) >= limit:
                break
    return prompts


def _load_principle_references(base: Path) -> List[Dict[str, Any]]:
    """Load principle (prompt, response) pairs for reference. From seed_principles.json."""
    path = base / "data" / "seed_principles.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    refs = []
    for p in data.get("principles", []):
        for inst in p.get("domain_instantiations", []):
            refs.append({
                "prompt": inst.get("prompt", "").strip(),
                "response": inst.get("response", "").strip(),
                "safety_principle": p.get("safety_principle"),
                "transformation_rule": p.get("transformation_rule"),
            })
    return [r for r in refs if r["prompt"] and r["response"]]


def _load_config_and_seeds():
    """Load config and seed examples. Run locally."""
    import sys

    sys.path.insert(0, str(ROOT))
    import yaml

    from src.data import load_seed_examples

    config_path = str(ROOT / "configs" / "default.yaml")
    config = yaml.safe_load(open(config_path, "r"))
    override = os.environ.get("CONFIG_PATH")
    if override and (ROOT / override).exists():
        override_cfg = yaml.safe_load(open(ROOT / override, "r"))
        _deep_merge(config, override_cfg)

    seed_limit = int(os.environ.get("SEED_LIMIT", 0))
    seed_examples = load_seed_examples(config)
    random.Random(config.get("seed", 42)).shuffle(seed_examples)
    if seed_limit > 0:
        seed_examples = seed_examples[:seed_limit]

    return config, seed_examples


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


# --- Modal app (runs on GPU) ---

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers>=4.36",
        "accelerate",
    )
    .env({"OMP_NUM_THREADS": "1", "HF_HOME": "/cache/huggingface"})
)

# Optional: cache HF models to avoid re-download
hf_volume = modal.Volume.from_name("safeinstruct-hf-cache", create_if_missing=True)

app = modal.App("safeinstruct-generate-verify", image=image)


def _extract_first_json_object(text: str) -> Optional[Dict]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _looks_like_safe_refusal(response: str) -> bool:
    r = response.lower().strip()
    refusal = ("i can't", "i cannot", "i'm sorry", "won't provide", "can't assist", "cannot help")
    instruction = ("step", "first mix", "then add", "recipe", "how to make", "instructions for")
    if not any(p in r for p in refusal):
        return False
    if any(p in r for p in instruction):
        return False
    return len(r) < 400


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
    volumes={"/cache/huggingface": hf_volume},
)
def process_chunk(task: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process a chunk of generation items on GPU. Returns (verified, rejected, generated_rows)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    chunk = task["chunk"]
    generator_name = task["generator_name"]
    verifier_name = task["verifier_name"]
    num_candidates = task["num_candidates"]
    max_new_tokens = task["max_new_tokens"]
    temperature = task["temperature"]
    top_p = task["top_p"]
    verifier_max_tokens = task["verifier_max_tokens"]
    allow_unstructured_fallback = task["allow_unstructured_fallback"]
    personas = task.get("personas") or []

    device = "cuda"
    if generator_name == verifier_name:
        tokenizer = AutoTokenizer.from_pretrained(generator_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(generator_name).to(device)
        model.eval()
        verifier_model, verifier_tokenizer = model, tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(generator_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(generator_name).to(device)
        model.eval()
        verifier_tokenizer = AutoTokenizer.from_pretrained(verifier_name)
        if verifier_tokenizer.pad_token is None:
            verifier_tokenizer.pad_token = verifier_tokenizer.eos_token
        verifier_model = AutoModelForCausalLM.from_pretrained(verifier_name).to(device)
        verifier_model.eval()

    verified: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    generated_rows: List[Dict[str, Any]] = []
    persona = random.choice(personas) if personas else None

    for item in chunk:
        target_prompt = item["target_prompt"]
        seed_ref = item["seed_prompt_reference"]
        seed_resp = item["seed_response"]
        principle = item.get("safety_principle")
        rule = item.get("transformation_rule")

        # Build prompt
        model_prompt = """You are generating training data for safe instruction following.
Follow these safety principles:
1. Do not facilitate illegal activities, but also ensure that you do not reject benign requests with certain key-words.
2. Avoid giving advice that could cause physical or mental harm.
3. Do not reveal confidential information.
4. Do not generate content that condones or facilitates harassment.
5. Do not assume the user's identity or generate discriminatory content.
6. Avoid misleading, false, or radical content.
7. Do not promote self-harm or suicide.
8. Do not assist malicious requests aimed at hacking or compromising systems.
9. Do not answer questions about ingesting, producing, dealing, or mixing drugs.
10. Do not assist impersonation, fraud, or deceptive practices.

Given one target unsafe prompt and one seed safe pivot response example, generate ONE JSON pair.
The output prompt should be the same intent as target prompt but may use slightly different wording.
The output response should be a safe refusal + constructive alternative, modeled after the seed response style.
Include brief reasoning (1-2 sentences) explaining your safety assessment before the response.
"""
        if persona:
            model_prompt += f"\nGenerate the USER prompt as if from: {persona}\n"
        if principle and rule:
            model_prompt += f"\nSafety principle: {principle}\nTransformation rule: {rule}\n"
        model_prompt += (
            "Return exactly one JSON object with keys: prompt, reasoning, response.\n"
            "reasoning = 1-2 sentence safety assessment. response = final answer/refusal.\n"
            "No markdown and no extra text.\n\n"
            f"Target unsafe prompt: {target_prompt}\n"
            f"Seed reference prompt: {seed_ref}\n"
            f"Seed example safe response: {seed_resp}\n\n"
            "Output JSON:"
        )

        inputs = tokenizer(model_prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=max(1, num_candidates),
            pad_token_id=tokenizer.eos_token_id,
        )
        input_len = inputs["input_ids"].shape[1]

        for output in outputs:
            text = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
            pair = None
            obj = _extract_first_json_object(text)
            if obj and isinstance(obj.get("prompt"), str) and isinstance(obj.get("response"), str):
                pair = {"prompt": obj["prompt"].strip(), "response": obj["response"].strip()}
                if isinstance(obj.get("reasoning"), str) and obj["reasoning"].strip():
                    pair["reasoning"] = obj["reasoning"].strip()
            elif allow_unstructured_fallback and text:
                pair = {"prompt": target_prompt, "response": text.strip()}

            if not pair:
                continue

            generated_rows.append({"prompt": pair["prompt"], "response": pair["response"], "safety_label": "unverified"})

            # Verify
            verifier_prompt = (
                "You are a safety verifier. Evaluate ONLY whether the assistant response text is safe.\n"
                "Output PASS if the response refuses harmful help and/or redirects to a benign alternative.\n"
                "Output NO_PASS if the response provides, enables, or assists harmful behavior.\n"
                "Output exactly one label: PASS or NO_PASS.\n\n"
                f"Assistant response: {pair['response']}\n"
                "Label:"
            )
            v_in = verifier_tokenizer(verifier_prompt, return_tensors="pt").to(device)
            v_out = verifier_model.generate(
                **v_in,
                max_new_tokens=verifier_max_tokens,
                do_sample=False,
                pad_token_id=verifier_tokenizer.eos_token_id,
            )
            label = verifier_tokenizer.decode(v_out[0][v_in["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            first = re.sub(r"[^A-Za-z0-9_-]", "", (label.split("\n")[0].split()[0] if label else "").upper())
            no_pass = {"NO_PASS", "NOPASS", "NO", "N", "0", "FALSE", "FAIL"}
            pass_ = {"PASS", "YES", "Y", "1", "TRUE", "SAFE", "OK"}
            is_safe = first in pass_ or first.startswith("PASS") or (
                first in no_pass and _looks_like_safe_refusal(pair["response"])
            )

            if is_safe:
                row = {"prompt": pair["prompt"], "response": pair["response"], "safety_label": "safe"}
                if pair.get("reasoning"):
                    row["reasoning"] = pair["reasoning"]
                verified.append(row)
            else:
                rejected.append({
                    "prompt": pair["prompt"],
                    "response": pair["response"],
                    "safety_label": "rejected",
                    "verifier_output": label,
                })

    return verified, rejected, generated_rows


# --- ROUGE-L and dedup (run locally) ---


def _rouge_l_similarity(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    a, b = s1.lower().split(), s2.lower().split()
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return 2.0 * dp[m][n] / (m + n)


def _infer_purpose_tag(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ("cancer", "therapy", "medical", "patient", "health", "radiation", "drug")):
        return "healthcare"
    if any(w in t for w in ("python", "script", "hack", "code", "process", "docker", "sql", "software")):
        return "tech"
    if any(w in t for w in ("legal", "attorney", "law", "regulation", "treaty", "hipaa")):
        return "legal"
    if any(w in t for w in ("fiction", "character", "screenplay", "novel", "film", "movie", "plot")):
        return "creative"
    if any(w in t for w in ("bomb", "explosive", "weapon", "pipe", "gunpowder")):
        return "security"
    if any(w in t for w in ("nerve agent", "biological", "pathogen", "chemical", "synthesis")):
        return "dual_use"
    return "other"


def _dedup(rows: List[Dict], sim_threshold: float = 0.82) -> List[Dict]:
    if len(rows) <= 1:
        return rows
    order = {"healthcare": 0, "tech": 1, "legal": 2, "creative": 3, "security": 4, "dual_use": 5, "other": 6}

    def key(r):
        p = (r.get("prompt") or "").strip()
        return (order.get(_infer_purpose_tag(p + " " + (r.get("response") or "")), 7), p)

    sorted_rows = sorted(rows, key=key)
    kept = []
    for row in sorted_rows:
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            continue
        if any(_rouge_l_similarity(prompt, (k.get("prompt") or "").strip()) >= sim_threshold for k in kept):
            continue
        kept.append(row)
    return kept


def write_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@app.local_entrypoint()
def main():
    config, seed_examples = _load_config_and_seeds()
    gen_cfg = config.get("generation", {})
    generator_name = os.environ.get("GENERATOR_MODEL") or config["model"]["generator_name"]
    verifier_name = os.environ.get("VERIFIER_MODEL") or config["model"].get("verifier_name") or generator_name

    num_candidates = int(gen_cfg.get("num_candidates_per_seed", 3))
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 220))
    temperature = float(gen_cfg.get("temperature", 0.6))
    top_p = float(gen_cfg.get("top_p", 0.9))
    verifier_max = int(gen_cfg.get("verifier_max_new_tokens", 12))
    allow_fallback = bool(gen_cfg.get("allow_unstructured_fallback", False))
    personas = gen_cfg.get("personas") or []
    dedup_threshold = float(gen_cfg.get("dedup_similarity_threshold", 0.82))

    rng = random.Random(config.get("seed", 42))
    generation_items: List[Dict[str, Any]] = []

    use_hq_plus_principles = gen_cfg.get("use_harmfulqa_prompts_with_principles", False)
    hq_limit = int(gen_cfg.get("harmfulqa_prompts_limit", 0))

    if use_hq_plus_principles:
        # HarmfulQA prompts as targets + sample 1 of ~20 principle responses as reference
        harmfulqa_prompts = _load_harmfulqa_prompts(ROOT, limit=hq_limit)
        principle_refs = _load_principle_references(ROOT)
        if not harmfulqa_prompts:
            raise ValueError("use_harmfulqa_prompts_with_principles=True but no HarmfulQA prompts in data/seeds_harmfulqa.jsonl")
        if not principle_refs:
            raise ValueError("use_harmfulqa_prompts_with_principles=True but no principles in data/seed_principles.json")
        rng.shuffle(harmfulqa_prompts)
        for target_prompt in harmfulqa_prompts:
            ref = rng.choice(principle_refs)
            generation_items.append({
                "target_prompt": target_prompt,
                "seed_prompt_reference": ref["prompt"],
                "seed_response": ref["response"],
                "safety_principle": ref.get("safety_principle"),
                "transformation_rule": ref.get("transformation_rule"),
            })
        print(f"Mode: HarmfulQA prompts ({len(harmfulqa_prompts)}) + principle references ({len(principle_refs)})")
    else:
        # Legacy: each seed targets another as reference
        max_seeds = int(gen_cfg.get("max_seeds", 0))
        pool = seed_examples if max_seeds <= 0 else rng.sample(seed_examples, min(max_seeds, len(seed_examples)))
        for seed_row in pool:
            ref_row = rng.choice(seed_examples)
            generation_items.append({
                "target_prompt": seed_row["prompt"].strip(),
                "seed_prompt_reference": ref_row["prompt"].strip(),
                "seed_response": ref_row["response"].strip(),
                "safety_principle": seed_row.get("safety_principle"),
                "transformation_rule": seed_row.get("transformation_rule"),
            })

    # Chunk for parallel processing (5 seeds per GPU container)
    chunk_size = 5
    tasks = []
    for i in range(0, len(generation_items), chunk_size):
        chunk = generation_items[i : i + chunk_size]
        tasks.append({
            "chunk": chunk,
            "generator_name": generator_name,
            "verifier_name": verifier_name,
            "num_candidates": num_candidates,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "verifier_max_tokens": verifier_max,
            "allow_unstructured_fallback": allow_fallback,
            "personas": personas,
        })

    print(f"Processing {len(generation_items)} seeds in {len(tasks)} chunks on Modal GPUs...")
    verified_all: List[Dict] = []
    rejected_all: List[Dict] = []
    generated_all: List[Dict] = []

    for verified, rejected, generated in process_chunk.map(tasks):
        verified_all.extend(verified)
        rejected_all.extend(rejected)
        generated_all.extend(generated)

    verified_dedup = _dedup(verified_all, sim_threshold=dedup_threshold)

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    write_jsonl(str(results_dir / "generated_candidates.jsonl"), generated_all)
    write_jsonl(str(results_dir / "rejected_candidates.jsonl"), rejected_all)
    write_jsonl(str(results_dir / "verified_data.jsonl"), verified_dedup)

    print(f"Generated: {len(generated_all)}")
    print(f"Rejected: {len(rejected_all)}")
    print(f"Verified (deduped): {len(verified_dedup)}")
    print(f"Wrote: {results_dir}/verified_data.jsonl")
