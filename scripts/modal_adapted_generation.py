#!/usr/bin/env python3
"""Modal-adapted minimal model-based generation + verifier loop using seed pivots."""

import json
import os
import re
import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

# --------------------------
# Modal setup
# --------------------------
app = modal.App("gemma-generation")

MODEL_NAME = "google/gemma-2-2b"
MODEL_CACHE = "/root/model-cache"

model_volume = modal.Volume.from_name("gemma-model-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("safeinstruct-results", create_if_missing=True)


def download_model():
    from huggingface_hub import snapshot_download
    import os
    snapshot_download(MODEL_NAME, token=os.environ["HF_TOKEN"], cache_dir=MODEL_CACHE)


image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "tqdm", "pyyaml", "datasets", "huggingface_hub")
    .run_function(
        download_model,
        secrets=[modal.Secret.from_name("huggingface")],
        volumes={MODEL_CACHE: model_volume},
    )
    .add_local_dir(".", "/root/safeinstruct")
)

# --------------------------
# ROOT path (safe for both local and remote)
# --------------------------
ROOT = Path("/root/safeinstruct") if os.path.exists("/root/safeinstruct") else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --------------------------
# Helpers
# --------------------------

def load_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_text_model(model_name: str, device: str, token: str = None) -> tuple:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, cache_dir=MODEL_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token, cache_dir=MODEL_CACHE)
    model.to(device)
    model.eval()
    return model, tokenizer


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
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
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


# --------------------------
# Generation + Verification
# --------------------------

def generate_json_pairs_for_seed(
    model: Any,
    tokenizer: Any,
    device: str,
    target_prompt: str,
    seed_prompt_reference: str,
    seed_response: str,
    num_candidates_per_seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    allow_unstructured_fallback: bool,
    safety_principle: str = None,
    transformation_rule: str = None,
    persona: str = None,
) -> Tuple[List[Dict[str, str]], int, int]:
    model_prompt = """You are generating training data for safe instruction following.
    Follow these safety principles:
    1. Do not facilitate illegal activities, but also ensure that you do not reject benign requests with certain key-words.
    2. Avoid giving advice that could cause physical or mental harm to the user or to another.
    3. Do not reveal confidential information.
    4. Do not generate content that condones or facilitates verbal or physical harassment.
    5. Do not assume the user's identity, group, or background, and never generate discriminatory content.
    6. Avoid creating misleading, false, or radical content.
    7. Do not promote self-harm or suicide.
    8. Do not assist malicious requests aimed at hacking, exploiting, or compromising systems.
    9. Do not answer questions about ingesting, producing, dealing, or mixing drugs or medications.
    10. Do not assist the user in impersonating others, committing fraud, or engaging in deceptive practices.

    Given one target unsafe prompt and one seed safe pivot response example, generate ONE JSON pair.
    The output prompt should be the same intent as target prompt but may use slightly different wording.
    The output response should be a safe refusal + constructive alternative, modeled after the seed response style.
    """
    if persona:
        model_prompt += (
            f"\nTo ensure diverse unsafe prompts, generate the USER prompt as if it came from someone with this persona: {persona}\n"
            "Vary phrasing and framing accordingly so we get diverse representations of potentially unsafe intents.\n"
        )
    if safety_principle and transformation_rule:
        model_prompt += (
            f"\nSafety principle: {safety_principle}\n"
            f"Transformation rule: {transformation_rule}\n"
            "Apply this principle and rule when generating the safe response.\n"
        )
    model_prompt += (
        "Return exactly one JSON object with keys: prompt, response.\n"
        "No markdown and no extra text.\n\n"
        f"Target unsafe prompt: {target_prompt}\n"
        f"Seed reference prompt: {seed_prompt_reference}\n"
        f"Seed example safe response: {seed_response}\n\n"
        "Output JSON:"
    )

    inputs = tokenizer(model_prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=max(1, num_candidates_per_seed),
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_pairs = []
    num_raw_generations = 0
    num_parse_failures = 0
    input_len = inputs["input_ids"].shape[1]

    for output in outputs:
        num_raw_generations += 1
        text = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
        if not text:
            num_parse_failures += 1
            continue
        obj = _extract_first_json_object(text)
        if obj is not None:
            prompt = obj.get("prompt")
            response = obj.get("response")
            if isinstance(prompt, str) and isinstance(response, str):
                prompt = prompt.strip()
                response = response.strip()
            elif isinstance(prompt, str) and isinstance(response, dict):
                # model returned response as a nested dict, flatten to string
                response = json.dumps(response)
                prompt = prompt.strip()
            if prompt and response and isinstance(prompt, str) and isinstance(response, str):
                generated_pairs.append({"prompt": prompt, "response": response})
                continue
        if allow_unstructured_fallback:
            generated_pairs.append({"prompt": target_prompt, "response": text})
        else:
            num_parse_failures += 1

    return generated_pairs, num_raw_generations, num_parse_failures


def verify_candidate_with_model(
    verifier_model: Any,
    verifier_tokenizer: Any,
    device: str,
    generated_pair: Dict[str, str],
    verifier_max_new_tokens: int,
) -> Tuple[bool, str, str]:
    response_text = generated_pair.get("response", "")
    verifier_prompt = (
        "You are a safety verifier.\n"
        "Evaluate ONLY whether the assistant response text is safe.\n"
        "Do not evaluate or infer anything from the user prompt.\n"
        "Decision rule:\n"
        "- Output PASS if the response refuses harmful help and/or redirects to a benign alternative,\n"
        "  and does not provide actionable harmful instructions.\n"
        "- Output NO_PASS if the response provides, enables, or meaningfully assists harmful behavior.\n"
        "Output exactly one label: PASS or NO_PASS.\n\n"
        f"Assistant response: {response_text}\n"
        "Label:"
    )
    inputs = verifier_tokenizer(verifier_prompt, return_tensors="pt").to(device)
    outputs = verifier_model.generate(
        **inputs,
        max_new_tokens=verifier_max_new_tokens,
        do_sample=False,
        pad_token_id=verifier_tokenizer.eos_token_id,
    )
    label_text = verifier_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    first_token_clean = re.sub(r"[^A-Za-z0-9_-]", "", label_text.split("\n")[0].split()[0] if label_text.split("\n")[0].split() else "").upper()

    if first_token_clean in {"NO_PASS", "NOPASS", "NO", "N", "0", "FALSE", "FAIL"}:
        return False, label_text, "unsafe_label"
    if first_token_clean in {"PASS", "YES", "Y", "1", "TRUE", "SAFE", "OK"} or first_token_clean.startswith("PASS"):
        return True, label_text, "safe_label"
    return False, label_text, "unparsed_label"


# --------------------------
# Modal entrypoint
# --------------------------

@app.function(
    image=image,
    gpu="any",
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={
        MODEL_CACHE: model_volume,
        "/root/results": results_volume,
    },
    timeout=3600,
)
def main():
    from src.data import load_seed_examples
    token = os.getenv("HF_TOKEN")
    config_path = ROOT / "configs" / "default.yaml"
    config = load_config(str(config_path))
    device = resolve_device()

    # Load model (token passed through)
    model, tokenizer = load_text_model(config["model"]["generator_name"], device, token=token)

    verifier_name = config["model"].get("verifier_name", config["model"]["generator_name"])
    if verifier_name == config["model"]["generator_name"]:
        verifier_model, verifier_tokenizer = model, tokenizer
    else:
        verifier_model, verifier_tokenizer = load_text_model(verifier_name, device, token=token)

    seed_examples = load_seed_examples(config)
    generated_rows, rejected_rows, verified_rows = [], [], []

    for seed_row in seed_examples:
        target_prompt = seed_row["prompt"].strip()
        seed_response = seed_row["response"].strip()

        generated_pairs, _, _ = generate_json_pairs_for_seed(
            model=model,
            tokenizer=tokenizer,
            device=device,
            target_prompt=target_prompt,
            seed_prompt_reference=target_prompt,
            seed_response=seed_response,
            num_candidates_per_seed=int(config["generation"].get("num_candidates_per_seed", 3)),
            max_new_tokens=int(config["generation"].get("max_new_tokens", 96)),
            temperature=float(config["generation"].get("temperature", 0.8)),
            top_p=float(config["generation"].get("top_p", 0.95)),
            allow_unstructured_fallback=True,
        )

        for pair in generated_pairs:
            is_safe, _, _ = verify_candidate_with_model(
                verifier_model=verifier_model,
                verifier_tokenizer=verifier_tokenizer,
                device=device,
                generated_pair=pair,
                verifier_max_new_tokens=int(config["generation"].get("verifier_max_new_tokens", 12)),
            )
            if is_safe:
                verified_rows.append({**pair, "safety_label": "safe"})
            else:
                rejected_rows.append({**pair, "safety_label": "rejected"})
            generated_rows.append({**pair, "safety_label": "unverified"})

    # Save to persistent volume at /root/results
    write_jsonl("/root/results/generated_candidates.jsonl", generated_rows)
    write_jsonl("/root/results/rejected_candidates.jsonl", rejected_rows)
    write_jsonl("/root/results/verified_candidates.jsonl", verified_rows)

    # Commit volume so data is flushed
    results_volume.commit()

    print(f"Done. {len(verified_rows)} safe / {len(rejected_rows)} rejected / {len(generated_rows)} total.")


@app.local_entrypoint()
def run():
    main.remote()