#!/usr/bin/env python3
"""Minimal model-based generation + verifier loop using seed pivots."""

import os

# Avoid OpenMP SHM errors in sandboxed/Docker environments
os.environ.setdefault("OMP_NUM_THREADS", "8")

import json
import re
import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_seed_examples

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
        "Keep response to 1–2 sentences.\n\n"
        f"Target unsafe prompt: {target_prompt}\n"
        f"Seed reference prompt: {seed_prompt_reference}\n"
        f"Seed example safe response: {seed_response}\n\n"
        "JSON:"
    )
    return "\n\n".join(parts)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(config: Dict[str, Any]) -> str:
    device_cfg = config.get("device")
    if device_cfg:
        return str(device_cfg)
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_text_model(
    model_name: str,
    device: str,
    load_in_4bit: bool = False,
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
            quantization_config=bnb_config,
            device_map="auto",
        )
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        model.eval()
    return model, tokenizer


def load_generator(config: Dict[str, Any]) -> tuple[Any, Any, str]:
    model_name = config["model"]["generator_name"]
    device = resolve_device(config)
    load_in_4bit = bool(config.get("model", {}).get("load_in_4bit", False))
    model, tokenizer = load_text_model(
        model_name=model_name,
        device=device,
        load_in_4bit=load_in_4bit,
    )
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


def _extract_prompt_from_row(row: Dict[str, Any]) -> Optional[str]:
    for key in ("prompt", "question", "query", "instruction", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def load_external_prompts(
    config: Dict[str, Any],
    seed: int,
) -> List[str]:
    source = config.get("data", {}).get("prompt_source")
    if not source:
        return []

    limit = int(config.get("generation", {}).get("external_prompt_limit", 50))
    prompts: List[str] = []
    if isinstance(source, str) and source.startswith("hf://"):
        # Format: hf://dataset_id:split  (split optional; defaults to train)
        from datasets import load_dataset
        from itertools import islice

        spec = source[len("hf://") :]
        if ":" in spec:
            dataset_id, split = spec.rsplit(":", 1)
        else:
            dataset_id, split = spec, "train"
        # Always stream + shuffle for HF sources.
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

    # Local file source (.jsonl expected)
    if os.path.exists(source):
        local_rows = read_jsonl(source)
        import random

        rng = random.Random(seed)
        rng.shuffle(local_rows)
        if limit > 0:
            local_rows = local_rows[:limit]
        for row in local_rows:
            prompt = _extract_prompt_from_row(row)
            if prompt:
                prompts.append(prompt)
    return prompts


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
    seed_instruction: str | None = None,
    safety_principle: str | None = None,
    transformation_rule: str | None = None,
    persona: str | None = None,
    use_persona_context: bool = True,
    use_instruction_context: bool = True,
    use_principle_context: bool = True,
    generation_seed: int | None = None,
) -> Tuple[List[Dict[str, str]], int, int]:
    generated_pairs: List[Dict[str, str]] = []
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
        num_parse_failures += 1

    return generated_pairs, num_raw_generations, num_parse_failures


def verify_candidate_with_local_model(
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
        "Output ONLY one word: PASS or NO_PASS. No other text, no prefix, no explanation.\n\n"
        f"Assistant response: {response_text}\n\n"
        "Output:"
    )
    inputs = verifier_tokenizer(verifier_prompt, return_tensors="pt").to(device)
    outputs = verifier_model.generate(
        **inputs,
        max_new_tokens=verifier_max_new_tokens,
        do_sample=False,
        pad_token_id=verifier_tokenizer.eos_token_id,
    )
    input_len = inputs["input_ids"].shape[1]
    label_text = verifier_tokenizer.decode(
        outputs[0][input_len:],
        skip_special_tokens=True,
    ).strip()
    return parse_verifier_label(label_text)


def parse_verifier_label(label_text: str) -> Tuple[bool, str, str]:
    """Parse PASS/NO_PASS from anywhere in output (handles batching artifacts)."""
    text_upper = label_text.upper().replace("-", "_")
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


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


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


def main() -> None:

    config = load_config(str(ROOT / "configs" / "default.yaml"))

    device = resolve_device(config)
    # Override models via env for quick testing with ungated models (e.g. SmolLM)
    generator_name = os.environ.get("GENERATOR_MODEL") or config["model"]["generator_name"]
    verifier_name = (
        os.environ.get("VERIFIER_MODEL")
        or (generator_name if os.environ.get("GENERATOR_MODEL") else None)
        or config["model"].get("verifier_name")
        or generator_name
    )
    # Patch config so load_generator / verifier use overridden models
    config.setdefault("model", {})["generator_name"] = generator_name
    config.setdefault("model", {})["verifier_name"] = verifier_name

    num_candidates = int(config.get("generation", {}).get("num_candidates_per_seed", 3))
    max_new_tokens = int(config.get("generation", {}).get("max_new_tokens", 96))
    temperature = float(config.get("generation", {}).get("temperature", 0.8))
    top_p = float(config.get("generation", {}).get("top_p", 0.95))
    verifier_max_new_tokens = int(
        config.get("generation", {}).get("verifier_max_new_tokens", 12)
    )
    verifier_batch_size = int(config.get("generation", {}).get("verifier_batch_size", 8))
    use_persona_context = bool(config.get("generation", {}).get("use_persona_context", True))
    use_instruction_context = bool(
        config.get("generation", {}).get("use_instruction_context", True)
    )
    use_principle_context = bool(
        config.get("generation", {}).get("use_principle_context", True)
    )
    verify_only_mode = bool(
        config.get("generation", {}).get("verify_only_from_generated_candidates", False)
    )
    generated_candidates_path = str(
        ROOT
        / config.get("generation", {}).get(
            "generated_candidates_path", "results/generated_candidates.jsonl"
        )
    )
    rejected_candidates_path = str(
        ROOT
        / config.get("generation", {}).get(
            "rejected_candidates_path", "results/rejected_candidates.jsonl"
        )
    )
    verified_path = str(ROOT / config["data"]["verified_path"])
    personas: List[str] = config.get("generation", {}).get("personas") or []
    load_in_4bit = bool(config.get("model", {}).get("load_in_4bit", False))

    if verify_only_mode:
        verifier_model, verifier_tokenizer = load_text_model(
            model_name=verifier_name,
            device=device,
            load_in_4bit=load_in_4bit,
        )
        model = None
        tokenizer = None
    else:
        model, tokenizer, _ = load_generator(config)
        if verifier_name == generator_name:
            verifier_model, verifier_tokenizer = model, tokenizer
        else:
            verifier_model, verifier_tokenizer = load_text_model(
                model_name=verifier_name,
                device=device,
                load_in_4bit=load_in_4bit,
            )

    seed_examples = [] if verify_only_mode else load_seed_examples(config)

    max_seeds = int(config.get("generation", {}).get("max_seeds", 0))
    if max_seeds > 0 and len(seed_examples) > max_seeds:
        rng = random.Random(int(config.get("seed", 42)))
        seed_examples = rng.sample(seed_examples, max_seeds)
        print(f"[INFO] max_seeds={max_seeds}: sampled {len(seed_examples)} seeds")
    # Env override for quick debugging (slices first N)
    try:
        seed_limit = int(os.environ.get("SEED_LIMIT", 0))
    except (ValueError, TypeError):
        seed_limit = 0
    if seed_limit > 0:
        seed_examples = seed_examples[:seed_limit]
        print(f"[DEBUG] SEED_LIMIT={seed_limit}: using {len(seed_examples)} seeds")

    external_prompts = [] if verify_only_mode else load_external_prompts(config, seed=int(config.get("seed", 42)))
    generated_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    verified_rows: List[Dict[str, Any]] = []
    num_generated = 0
    num_parse_failures = 0
    if verify_only_mode:
        input_rows = read_jsonl(generated_candidates_path)
        generated_pairs: List[Dict[str, str]] = []
        for row in input_rows:
            prompt = row.get("prompt")
            response = row.get("response")
            if isinstance(prompt, str) and isinstance(response, str):
                generated_pairs.append({"prompt": prompt, "response": response})
                generated_rows.append(
                    {
                        "prompt": prompt,
                        "response": response,
                        "safety_label": "unverified",
                    }
                )
        num_generated = len(generated_pairs)

        for start in tqdm(range(0, len(generated_pairs), verifier_batch_size), desc="Verify", unit="batch"):
            batch_pairs = generated_pairs[start : start + verifier_batch_size]
            batch_results = verify_candidates_with_local_model(
                verifier_model=verifier_model,
                verifier_tokenizer=verifier_tokenizer,
                device=device,
                generated_pairs=batch_pairs,
                verifier_max_new_tokens=verifier_max_new_tokens,
            )
            for generated_pair, (is_safe, verifier_output, decision_reason) in zip(batch_pairs, batch_results):
                if is_safe:
                    verified_rows.append(
                        {
                            "prompt": generated_pair["prompt"],
                            "response": generated_pair["response"],
                            "safety_label": "safe",
                        }
                    )
                else:
                    rejected_rows.append(
                        {
                            "prompt": generated_pair["prompt"],
                            "response": generated_pair["response"],
                            "safety_label": "rejected",
                            "verifier_output": verifier_output,
                            "decision_reason": decision_reason,
                        }
                    )
    else:
        generation_items: List[Tuple[str, str, str, str | None, str | None, str | None]] = []
        if external_prompts:
            # Use external prompts for diversity while modeling response style on seeds.
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

        for idx, (target_prompt, seed_prompt_reference, seed_response, seed_instruction, safety_principle, transformation_rule) in enumerate(
            tqdm(generation_items, desc="Seeds", unit="seed")
        ):
            persona = random.choice(personas) if (personas and config.get("generation", {}).get("persona", True)) else None
            generated_pairs, raw_count, parse_failures = generate_json_pairs_for_seed(
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
            num_generated += raw_count
            num_parse_failures += parse_failures

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

            for generated_pair, (is_safe, verifier_output, decision_reason) in tqdm(
                zip(generated_pairs, batch_results_all),
                total=len(generated_pairs),
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
                if is_safe:
                    verified_rows.append(
                        {
                            "prompt": generated_pair["prompt"],
                            "response": generated_pair["response"],
                            "safety_label": "safe",
                        }
                    )
                else:
                    rejected_rows.append(
                        {
                            "prompt": generated_pair["prompt"],
                            "response": generated_pair["response"],
                            "safety_label": "rejected",
                            "verifier_output": verifier_output,
                            "decision_reason": decision_reason,
                        }
                    )

    write_jsonl(generated_candidates_path, generated_rows)
    write_jsonl(rejected_candidates_path, rejected_rows)
    write_jsonl(verified_path, verified_rows)

    print(f"Mode: {'verify-only' if verify_only_mode else 'generate+verify'}")
    print(f"Seed examples: {len(seed_examples)}")
    print(f"External prompts: {len(external_prompts)}")
    print(f"Generated candidates: {num_generated}")
    print(f"Parse failures: {num_parse_failures}")
    print(f"Saved generated (pre-verifier): {len(generated_rows)}")
    print(f"Wrote generated to: {generated_candidates_path}")
    print(f"Rejected examples: {len(rejected_rows)}")
    print(f"Wrote rejected to: {rejected_candidates_path}")
    print(f"Verified safe examples: {len(verified_rows)}")
    print(f"Wrote: {verified_path}")
    print(f"Generator model: {generator_name} on {device}")
    print(f"Verifier model: {verifier_name} on {device}")


if __name__ == "__main__":
    main()
