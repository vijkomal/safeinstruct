#!/usr/bin/env python3
"""Minimal model-based generation + verifier loop using seed pivots."""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_seed_examples


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(config: Dict[str, Any]) -> str:
    device_cfg = config.get("device")
    if device_cfg:
        return str(device_cfg)
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_text_model(model_name: str, device: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


def load_generator(config: Dict[str, Any]) -> tuple[Any, Any, str]:
    model_name = config["model"]["generator_name"]
    device = resolve_device(config)
    model, tokenizer = load_text_model(model_name=model_name, device=device)
    return model, tokenizer, device


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
    if not isinstance(obj, dict):
        return None
    return obj


def _extract_pair_with_regex(text: str) -> Optional[Dict[str, str]]:
    prompt_match = re.search(r'"prompt"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
    response_match = re.search(r'"response"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
    if prompt_match and response_match:
        prompt = prompt_match.group(1).strip()
        response = response_match.group(1).strip()
        if prompt and response:
            return {"prompt": prompt, "response": response}
    return None


def generate_json_pairs_for_seed(
    model: Any,
    tokenizer: Any,
    device: str,
    seed_prompt: str,
    seed_response: str,
    num_candidates_per_seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    allow_unstructured_fallback: bool,
) -> Tuple[List[Dict[str, str]], int, int]:
    model_prompt = (
        "You are generating training data for safe instruction following.\n"
        "Given one unsafe prompt and a safe pivot response example, generate ONE NEW "
        "unsafe prompt with its safe pivoted response.\n"
        "Return exactly one JSON object with keys: prompt, response.\n"
        "No markdown and no extra text.\n\n"
        f"Seed example prompt: {seed_prompt}\n"
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

    generated_pairs: List[Dict[str, str]] = []
    num_raw_generations = 0
    num_parse_failures = 0
    input_len = inputs["input_ids"].shape[1]
    for output in outputs:
        num_raw_generations += 1
        completion_ids = output[input_len:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
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
                if prompt and response:
                    generated_pairs.append({"prompt": prompt, "response": response})
                    continue

        regex_pair = _extract_pair_with_regex(text)
        if regex_pair is not None:
            generated_pairs.append(regex_pair)
            continue

        if allow_unstructured_fallback:
            # Dev fallback: still produce a pair even if model ignored strict JSON format.
            generated_pairs.append(
                {"prompt": seed_prompt, "response": text.strip()}
            )
        else:
            num_parse_failures += 1

    return generated_pairs, num_raw_generations, num_parse_failures


def verify_candidate_with_model(
    verifier_model: Any,
    verifier_tokenizer: Any,
    device: str,
    generated_pair: Dict[str, str],
) -> bool:
    pair_json = json.dumps(generated_pair, ensure_ascii=True)
    verifier_prompt = (
        "You are a safety verifier.\n"
        "Given a generated JSON object with keys prompt and response, decide whether:\n"
        "1) The prompt is unsafe/harmful, and\n"
        "2) The response safely refuses or redirects and does NOT provide harmful instructions.\n"
        "Output exactly one label: SAFE or UNSAFE.\n\n"
        f"Generated JSON: {pair_json}\n"
        "Label:"
    )
    inputs = verifier_tokenizer(verifier_prompt, return_tensors="pt").to(device)
    outputs = verifier_model.generate(
        **inputs,
        max_new_tokens=4,
        do_sample=False,
        pad_token_id=verifier_tokenizer.eos_token_id,
    )
    input_len = inputs["input_ids"].shape[1]
    label_text = verifier_tokenizer.decode(
        outputs[0][input_len:],
        skip_special_tokens=True,
    ).strip()
    label_upper = label_text.upper()
    if "UNSAFE" in label_upper:
        return False
    if "SAFE" in label_upper:
        return True
    return False


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    config = load_config(str(ROOT / "configs" / "default.yaml"))

    model, tokenizer, device = load_generator(config)
    generator_name = config["model"]["generator_name"]
    verifier_name = config["model"].get("verifier_name") or generator_name
    if verifier_name == generator_name:
        verifier_model, verifier_tokenizer = model, tokenizer
    else:
        verifier_model, verifier_tokenizer = load_text_model(
            model_name=verifier_name,
            device=device,
        )

    num_candidates = int(config.get("generation", {}).get("num_candidates_per_seed", 3))
    max_new_tokens = int(config.get("generation", {}).get("max_new_tokens", 96))
    temperature = float(config.get("generation", {}).get("temperature", 0.8))
    top_p = float(config.get("generation", {}).get("top_p", 0.95))
    allow_unstructured_fallback = bool(
        config.get("generation", {}).get("allow_unstructured_fallback", True)
    )
    generated_candidates_path = str(
        ROOT
        / config.get("generation", {}).get(
            "generated_candidates_path", "results/generated_candidates.jsonl"
        )
    )
    verified_path = str(ROOT / config["data"]["verified_path"])

    seed_examples = load_seed_examples(config)
    generated_rows: List[Dict[str, Any]] = []
    verified_rows: List[Dict[str, Any]] = []
    num_generated = 0
    num_parse_failures = 0
    for seed_row in tqdm(seed_examples, desc="Seeds", unit="seed"):
        seed_prompt = seed_row["prompt"].strip()
        seed_response = seed_row["response"].strip()
        generated_pairs, raw_count, parse_failures = generate_json_pairs_for_seed(
            model=model,
            tokenizer=tokenizer,
            device=device,
            seed_prompt=seed_prompt,
            seed_response=seed_response,
            num_candidates_per_seed=num_candidates,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            allow_unstructured_fallback=allow_unstructured_fallback,
        )
        num_generated += raw_count
        num_parse_failures += parse_failures

        for generated_pair in tqdm(
            generated_pairs,
            desc="Verify",
            unit="pair",
            leave=False,
        ):
            generated_rows.append(
                {
                    "prompt": generated_pair["prompt"],
                    "response": generated_pair["response"],
                    "safety_label": "unverified",
                }
            )
            if verify_candidate_with_model(
                verifier_model=verifier_model,
                verifier_tokenizer=verifier_tokenizer,
                device=device,
                generated_pair=generated_pair,
            ):
                verified_rows.append(
                    {
                        "prompt": generated_pair["prompt"],
                        "response": generated_pair["response"],
                        "safety_label": "safe",
                    }
                )

    write_jsonl(generated_candidates_path, generated_rows)
    write_jsonl(verified_path, verified_rows)

    print(f"Seed examples: {len(seed_examples)}")
    print(f"Generated candidates: {num_generated}")
    print(f"Parse failures: {num_parse_failures}")
    print(f"Saved generated (pre-verifier): {len(generated_rows)}")
    print(f"Wrote generated to: {generated_candidates_path}")
    print(f"Verified safe examples: {len(verified_rows)}")
    print(f"Wrote: {verified_path}")
    print(f"Generator model: {generator_name} on {device}")
    print(f"Verifier model: {verifier_name} on {device}")


if __name__ == "__main__":
    main()
