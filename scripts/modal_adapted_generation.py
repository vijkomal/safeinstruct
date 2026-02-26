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
import torch

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
    .pip_install("torch", "transformers", "bitsandbytes", "tqdm", "pyyaml", "datasets", "huggingface_hub")
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
# Shared with run_generate_verify.py
# --------------------------

GENERATION_POLICY_BASE = """You are generating training data for safe instruction following.
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


def build_generation_prompt(
    target_prompt: str,
    seed_prompt_reference: str,
    seed_response: str,
    persona: str | None = None,
    seed_instruction: str | None = None,
    safety_principle: str | None = None,
    transformation_rule: str | None = None,
    use_persona_context: bool = True,
    use_instruction_context: bool = True,
    use_principle_context: bool = True,
) -> str:
    parts: List[str] = [GENERATION_POLICY_BASE]
    if use_persona_context and persona:
        parts.append(
            "To ensure diverse unsafe prompts, generate the USER prompt as if it came from "
            f"someone with this persona: {persona}\n"
            "Vary phrasing and framing accordingly so we get diverse representations of potentially unsafe intents."
        )
    if use_instruction_context and seed_instruction:
        parts.append(
            f"Task instruction: {seed_instruction}\n"
            "Follow this task instruction when generating the pair."
        )
    if use_principle_context and safety_principle and transformation_rule:
        parts.append(
            f"Safety principle: {safety_principle}\n"
            f"Transformation rule: {transformation_rule}\n"
            "Apply this principle and rule when generating the safe response."
        )
    parts.append(
        "Output a JSON object: {\"prompt\": \"...\", \"response\": \"...\"}\n"
        "Both prompt and response must be plain text only—no nested JSON, no user tags, no markdown, no code.\n"
        "Keep the response to 1–2 sentences.\n\n"
        f"Target unsafe prompt: {target_prompt}\n"
        f"Seed reference prompt: {seed_prompt_reference}\n"
        f"Seed example safe response: {seed_response}\n\n"
        "JSON:"
    )
    return "\n\n".join(parts)


def load_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_text_model(
    model_name: str,
    device: str,
    token: str = None,
    load_in_4bit: bool = False,
) -> tuple:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, cache_dir=MODEL_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if load_in_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            cache_dir=MODEL_CACHE,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, token=token, cache_dir=MODEL_CACHE)
        model.to(device)
        model.eval()
    return model, tokenizer


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _extract_prompt_from_row(row: Dict[str, Any]) -> Optional[str]:
    for key in ("prompt", "question", "query", "instruction", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def load_external_prompts(config: Dict[str, Any], seed: int) -> List[str]:
    source = config.get("data", {}).get("prompt_source")
    if not source:
        return []
    limit = int(config.get("generation", {}).get("external_prompt_limit", 50))
    prompts: List[str] = []
    if isinstance(source, str) and source.startswith("hf://"):
        from datasets import load_dataset
        from itertools import islice
        spec = source[len("hf://"):]
        if ":" in spec:
            dataset_id, split = spec.rsplit(":", 1)
        else:
            dataset_id, split = spec, "train"
        dataset = load_dataset(dataset_id, split=split, streaming=True)
        dataset = dataset.shuffle(seed=seed, buffer_size=1000)
        iterator = dataset if limit <= 0 else islice(dataset, limit)
        for row in iterator:
            if not isinstance(row, dict):
                continue
            prompt = _extract_prompt_from_row(row)
            if prompt:
                prompts.append(prompt)
        return prompts
    path = str(ROOT / source) if not os.path.isabs(source) and not source.startswith("hf://") else source
    if os.path.exists(path):
        local_rows = read_jsonl(path)
        rng = random.Random(seed)
        rng.shuffle(local_rows)
        if limit > 0:
            local_rows = local_rows[:limit]
        for row in local_rows:
            prompt = _extract_prompt_from_row(row)
            if prompt:
                prompts.append(prompt)
    return prompts


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
    seed_instruction: str = None,
    safety_principle: str = None,
    transformation_rule: str = None,
    persona: str = None,
    use_persona_context: bool = True,
    use_instruction_context: bool = True,
    use_principle_context: bool = True,
    generation_seed: int | None = None,
) -> Tuple[List[Dict[str, str]], int, int]:
    generated_pairs = []
    num_raw_generations = 0
    num_parse_failures = 0
    n = max(1, num_candidates_per_seed)

    for i in range(n):
        model_prompt = build_generation_prompt(
            target_prompt=target_prompt,
            seed_prompt_reference=seed_prompt_reference,
            seed_response=seed_response,
            persona=persona,
            seed_instruction=seed_instruction,
            safety_principle=safety_principle,
            transformation_rule=transformation_rule,
            use_persona_context=use_persona_context,
            use_instruction_context=use_instruction_context,
            use_principle_context=use_principle_context,
        )
        inputs = tokenizer(model_prompt, return_tensors="pt").to(device)
        if generation_seed is not None:
            torch.manual_seed(generation_seed + i * 1000)
            if device == "cuda":
                torch.cuda.manual_seed_all(generation_seed + i * 1000)
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )[0]

        input_len = inputs["input_ids"].shape[1]
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
                response = json.dumps(response)
                prompt = prompt.strip()
            if prompt and response and isinstance(prompt, str) and isinstance(response, str):
                generated_pairs.append({"prompt": prompt, "response": response})
                continue
        num_parse_failures += 1

    return generated_pairs, num_raw_generations, num_parse_failures


def parse_verifier_label(label_text: str) -> Tuple[bool, str, str]:
    """Parse PASS/NO_PASS from anywhere in output (handles batching artifacts)."""
    text_upper = label_text.upper().replace("-", "_")
    # Check for NO_PASS first (more specific)
    if re.search(r"\bNO_?PASS\b", text_upper):
        return False, label_text, "unsafe_label"
    if re.search(r"\bPASS\b", text_upper):
        return True, label_text, "safe_label"
    return False, label_text, "unparsed_label"


def verify_candidates_with_local_model(
    verifier_model: Any,
    verifier_tokenizer: Any,
    device: str,
    generated_pairs: List[Dict[str, str]],
    verifier_max_new_tokens: int,
) -> List[Tuple[bool, str, str]]:
    if not generated_pairs:
        return []
    prompts: List[str] = []
    for generated_pair in generated_pairs:
        response_text = generated_pair.get("response", "")
        verifier_prompt = (
            "You are a safety verifier.\n"
            "Evaluate ONLY whether the assistant response text is safe.\n"
            "Do not evaluate or infer anything from the user prompt.\n"
            "Decision rule:\n"
            "- Output PASS if the response refuses harmful help and/or redirects to a benign alternative,\n"
            "  and does not provide actionable harmful instructions.\n"
            "- Output NO_PASS if the response provides, enables, or meaningfully assists harmful behavior.\n"
            "Output ONLY one word: PASS or NO_PASS. No other text, no prefix, no explanation.\n\n"
            f"Assistant response: {response_text}\n\n"
            "Output:"
        )
        prompts.append(verifier_prompt)
    inputs = verifier_tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    outputs = verifier_model.generate(
        **inputs,
        max_new_tokens=verifier_max_new_tokens,
        do_sample=False,
        pad_token_id=verifier_tokenizer.eos_token_id,
    )
    results: List[Tuple[bool, str, str]] = []
    attention_mask = inputs["attention_mask"]
    for i in range(outputs.shape[0]):
        input_len = int(attention_mask[i].sum().item())
        label_text = verifier_tokenizer.decode(
            outputs[i][input_len:],
            skip_special_tokens=True,
        ).strip()
        results.append(parse_verifier_label(label_text))
    return results


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
    from tqdm import tqdm

    token = os.getenv("HF_TOKEN")
    config_path = ROOT / "configs" / "default.yaml"
    config = load_config(str(config_path))
    device = resolve_device()

    generator_name = os.environ.get("GENERATOR_MODEL") or config["model"]["generator_name"]
    verifier_name = (
        os.environ.get("VERIFIER_MODEL")
        or (generator_name if os.environ.get("GENERATOR_MODEL") else None)
        or config["model"].get("verifier_name")
        or generator_name
    )
    config.setdefault("model", {})["generator_name"] = generator_name
    config.setdefault("model", {})["verifier_name"] = verifier_name

    load_in_4bit = bool(config.get("model", {}).get("load_in_4bit", False))
    model, tokenizer = load_text_model(
        generator_name, device, token=token, load_in_4bit=load_in_4bit
    )
    if verifier_name == generator_name:
        verifier_model, verifier_tokenizer = model, tokenizer
    else:
        verifier_model, verifier_tokenizer = load_text_model(
            verifier_name, device, token=token, load_in_4bit=load_in_4bit
        )

    seed_examples = load_seed_examples(config)
    max_seeds = int(config.get("generation", {}).get("max_seeds", 0))
    if max_seeds > 0 and len(seed_examples) > max_seeds:
        rng = random.Random(int(config.get("seed", 42)))
        seed_examples = rng.sample(seed_examples, max_seeds)
        print(f"[INFO] max_seeds={max_seeds}: sampled {len(seed_examples)} seeds")

    external_prompts = load_external_prompts(config, seed=int(config.get("seed", 42)))
    use_persona_context = bool(config.get("generation", {}).get("use_persona_context", True))
    use_instruction_context = bool(config.get("generation", {}).get("use_instruction_context", True))
    use_principle_context = bool(config.get("generation", {}).get("use_principle_context", True))
    personas: List[str] = config.get("generation", {}).get("personas") or []
    num_candidates = int(config.get("generation", {}).get("num_candidates_per_seed", 3))
    max_new_tokens = int(config.get("generation", {}).get("max_new_tokens", 96))
    temperature = float(config.get("generation", {}).get("temperature", 0.6))
    top_p = float(config.get("generation", {}).get("top_p", 0.9))
    verifier_max_new_tokens = int(config.get("generation", {}).get("verifier_max_new_tokens", 12))
    verifier_batch_size = int(config.get("generation", {}).get("verifier_batch_size", 8))

    generation_items: List[Tuple[str, str, str, str | None, str | None, str | None]] = []
    if external_prompts:
        for i, prompt in enumerate(external_prompts):
            seed_row = seed_examples[i % len(seed_examples)]
            generation_items.append(
                (
                    prompt,
                    seed_row["prompt"].strip(),
                    seed_row["response"].strip(),
                    seed_row.get("instruction"),
                    seed_row.get("safety_principle"),
                    seed_row.get("transformation_rule"),
                )
            )
    else:
        for seed_row in seed_examples:
            generation_items.append(
                (
                    seed_row["prompt"].strip(),
                    seed_row["prompt"].strip(),
                    seed_row["response"].strip(),
                    seed_row.get("instruction"),
                    seed_row.get("safety_principle"),
                    seed_row.get("transformation_rule"),
                )
            )

    generated_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    verified_rows: List[Dict[str, Any]] = []

    for target_prompt, seed_prompt_reference, seed_response, seed_instruction, safety_principle, transformation_rule in tqdm(
        generation_items, desc="Seeds", unit="seed"
    ):
        persona = random.choice(personas) if (personas and config.get("generation", {}).get("persona", True)) else None
        generated_pairs, _, _ = generate_json_pairs_for_seed(
            model=model,
            tokenizer=tokenizer,
            device=device,
            target_prompt=target_prompt,
            seed_prompt_reference=seed_prompt_reference,
            seed_response=seed_response,
            num_candidates_per_seed=num_candidates,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed_instruction=seed_instruction,
            safety_principle=safety_principle,
            transformation_rule=transformation_rule,
            persona=persona,
            use_persona_context=use_persona_context,
            use_instruction_context=use_instruction_context,
            use_principle_context=use_principle_context,
            generation_seed=int(config.get("seed", 42)),
        )

        batch_results_all: List[Tuple[bool, str, str]] = []
        for start in range(0, len(generated_pairs), verifier_batch_size):
            batch_pairs = generated_pairs[start : start + verifier_batch_size]
            batch_results_all.extend(
                verify_candidates_with_local_model(
                    verifier_model=verifier_model,
                    verifier_tokenizer=verifier_tokenizer,
                    device=device,
                    generated_pairs=batch_pairs,
                    verifier_max_new_tokens=verifier_max_new_tokens,
                )
            )

        for generated_pair, (is_safe, verifier_output, decision_reason) in zip(
            generated_pairs, batch_results_all
        ):
            generated_rows.append({**generated_pair, "safety_label": "unverified"})
            if is_safe:
                verified_rows.append({**generated_pair, "safety_label": "safe"})
            else:
                rejected_rows.append({
                    **generated_pair,
                    "safety_label": "rejected",
                    "verifier_output": verifier_output,
                    "decision_reason": decision_reason,
                })

    verified_path = config.get("data", {}).get("verified_path", "results/verified_data.jsonl")
    write_jsonl("/root/results/generated_candidates.jsonl", generated_rows)
    write_jsonl("/root/results/rejected_candidates.jsonl", rejected_rows)
    write_jsonl(f"/root/results/{os.path.basename(verified_path)}", verified_rows)

    results_volume.commit()
    print(f"Done. {len(verified_rows)} safe / {len(rejected_rows)} rejected / {len(generated_rows)} total.")


@app.local_entrypoint()
def run():
    main.remote()