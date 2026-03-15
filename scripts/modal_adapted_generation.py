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

try:
    import transformers  # noqa: F401
except ImportError:
    pass

app = modal.App("safeinstruct-generation")

MODEL_CACHE = "/root/model-cache"

model_volume = modal.Volume.from_name("gemma-model-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("safeinstruct-results", create_if_missing=True)


def ensure_model_cached(model_name: str, token: Optional[str] = None) -> None:
    """Download model to cache if not already present. Uses HF_TOKEN for gated models (e.g. Gemma)."""
    from huggingface_hub import snapshot_download
    snapshot_download(model_name, token=token, cache_dir=MODEL_CACHE)


image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers[torch]>=4.46.0",  # Gemma 2 requires 4.46+; [torch] satisfies lm-format-enforcer check
        "accelerate",
        "bitsandbytes>=0.43.2",  # fixes .to() not supported for 4-bit models
        "tqdm",
        "pyyaml",
        "datasets",
        "huggingface_hub",
        "pydantic",
        "lm-format-enforcer",
    )
    .run_commands(
        "python -c \"import transformers; print('transformers', transformers.__version__)\""
    )
    .add_local_dir(
        ".",
        "/root/safeinstruct",
        ignore=[
            ".git",
            ".venv",
            "venv",
            "env",
            "__pycache__",
            "*.pyc",
            ".cursor",
            "*.safetensors",
            "*.bin",
            "*.pt",
            "*.ckpt",
            "results/",
            ".idea",
            ".vscode",
            "*.log",
        ],
    )
)

ROOT = Path("/root/safeinstruct") if os.path.exists("/root/safeinstruct") else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def build_generation_prompt(
    target_prompt: str,
    seed_response: str,
    seed_reasoning: str | None = None,
    safety_principle: str | None = None,
    transformation_rule: str | None = None,
    use_principle_context: bool = True,
    use_example: bool = False,
    second_example: tuple[str, str, str | None] | None = None,
) -> str:
    """Build the generator prompt that requests strict JSON output.

    TinyLlama legacy prompt (kept for reference; not active in current flow):
    ------------------------------------------------------------------------
    if use_example:
        ex = {"prompt": "<seed_prompt_reference>", "response": seed_response}
        if seed_reasoning:
            ex["reasoning"] = seed_reasoning
        parts.append("Example:\n" + json.dumps(ex, ensure_ascii=False))
        if second_example:
            ex2_prompt, ex2_resp, ex2_reason = second_example
            ex2 = {"prompt": ex2_prompt, "response": ex2_resp}
            if ex2_reason:
                ex2["reasoning"] = ex2_reason
            parts.append(json.dumps(ex2, ensure_ascii=False))
        parts.append("")

    parts.append(
        f"Target: {target_prompt}\n\n"
        "Reason and respond to the target above. Reason why the request is harmful, "
        "refuse that specific request, then offer a safe alternative. "
        "Output JSON with response and reasoning:"
    )
    """
    parts: List[str] = []
    if use_example:
        ex: Dict[str, str] = {"response": seed_response, "reasoning": seed_reasoning or ""}
        parts.append("Example output:\n" + json.dumps(ex, ensure_ascii=False))
        if second_example:
            _, ex2_resp, ex2_reason = second_example
            ex2: Dict[str, str] = {"response": ex2_resp, "reasoning": ex2_reason or ""}
            parts.append(json.dumps(ex2, ensure_ascii=False))
        parts.append("")
    if use_principle_context and safety_principle and transformation_rule:
        parts.append(f"Principle: {safety_principle}\nRule: {transformation_rule}")
    parts.append(
        f"Target request: {target_prompt}\n\n"
        "Refuse this request and offer a safe alternative. "
        "Output ONLY valid JSON with exactly two keys: \"response\" (your safe answer) and \"reasoning\" (why the request is harmful). "
        "Example format: {\"response\": \"...\", \"reasoning\": \"...\"}"
    )
    return "\n".join(parts)


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
            device_map="cuda:0",
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


def load_principle_mapping(
    principles_path: str,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[Dict[str, Any]]]]:
    """Load principle_id -> safety_principle, principle_id -> transformation_rule, and principle_id -> domain_instantiations."""
    full_path = ROOT / principles_path if not os.path.isabs(principles_path) else Path(principles_path)
    if not full_path.exists():
        return {}, {}, {}
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    id_to_principle: Dict[str, str] = {}
    id_to_rule: Dict[str, str] = {}
    id_to_instantiations: Dict[str, List[Dict[str, Any]]] = {}
    for p in data.get("principles", []):
        pid = p.get("principle_id", "")
        if not pid:
            continue
        id_to_principle[pid] = p.get("safety_principle", "")
        id_to_rule[pid] = p.get("transformation_rule", "")
        id_to_instantiations[pid] = list(p.get("domain_instantiations", []))
    return id_to_principle, id_to_rule, id_to_instantiations


def write_principle_id_mapping(id_to_principle: Dict[str, str], output_path: str) -> None:
    """Write simple principle_id -> safety_principle mapping for easy reference."""
    out = ROOT / output_path if not os.path.isabs(output_path) else Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(id_to_principle, f, indent=2, ensure_ascii=False)


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


PRINCIPLE_CLASSIFICATION_PROMPT = """You are a safety classifier. Given a user prompt that may be harmful or ambiguous, choose the single safety principle that best applies.

Safety principles (choose exactly one by ID):
{principles_list}

User prompt: {target_prompt}

Output ONLY the principle ID (e.g., P1, P2, P3, ...). No other text."""


def classify_principle_for_prompt(
    model: Any,
    tokenizer: Any,
    device: str,
    target_prompt: str,
    id_to_principle: Dict[str, str],
    max_new_tokens: int = 8,
) -> Optional[str]:
    """Query model to determine which safety principle best aligns with the target prompt. Returns principle_id or None."""
    lines = [f"{pid}: {princ}" for pid, princ in id_to_principle.items() if pid and princ]
    principles_list = "\n".join(lines)
    prompt_text = PRINCIPLE_CLASSIFICATION_PROMPT.format(
        principles_list=principles_list,
        target_prompt=target_prompt,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )[0]
    input_len = inputs["input_ids"].shape[1]
    text = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
    match = re.search(r"\b(P\d+)\b", text, re.IGNORECASE)
    if match and match.group(1).upper() in id_to_principle:
        return match.group(1).upper()
    return None


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract first valid JSON object, handling markdown, preamble, single-quotes, arrays."""
    raw = text.strip()
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            if "{" in part:
                obj = _extract_first_json_object(part)
                if obj is not None:
                    return obj
        raw = "".join(p for p in parts if p.strip())  # fallback: concat non-empty
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw)
    start = raw.find("{")
    if start == -1:
        arr_start = raw.find("[{")
        if arr_start != -1:
            start = arr_start + 1
        else:
            return None
    raw = raw[start:]

    def _try_parse(s: str) -> Optional[Dict[str, Any]]:
        for candidate in [s, s.rstrip() + "}" * (s.count("{") - s.count("}"))]:
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
                if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                    return obj[0]
            except json.JSONDecodeError:
                pass
        return None

    depth = 0
    i = 0
    in_string = False
    escape = False
    quote_char = None
    while i < len(raw):
        c = raw[i]
        if escape:
            escape = False
            i += 1
            continue
        if c == "\\" and in_string:
            escape = True
            i += 1
            continue
        if c in ('"', "'") and not escape:
            if not in_string:
                in_string = True
                quote_char = c
            elif c == quote_char:
                in_string = False
            i += 1
            continue
        if in_string:
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                obj = _try_parse(raw[: i + 1])
                if obj is not None:
                    return obj
        i += 1

    for end in range(len(raw), 0, -1):
        obj = _try_parse(raw[:end])
        if obj is not None:
            return obj

    try:
        import ast
        obj = ast.literal_eval(raw)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
    except (ValueError, SyntaxError):
        pass
    return None


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def generate_json_pairs_for_seed(
    model: Any,
    tokenizer: Any,
    device: str,
    target_prompt: str,
    seed_response: str,
    num_candidates_per_seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    seed_reasoning: str | None = None,
    safety_principle: str = None,
    transformation_rule: str = None,
    use_principle_context: bool = True,
    use_example: bool = False,
    second_example: tuple[str, str, str | None] | None = None,
    generation_seed: int | None = None,
    seed_idx: int = 0,
    parse_failures_out: Optional[List[Dict[str, Any]]] = None,
    use_grammar_constraint: bool = False,
    response_max_length: int = 800,
    reasoning_max_length: int = 500,
    generator_name: str = "",
) -> Tuple[List[Dict[str, Any]], int, int]:
    generated_pairs = []
    num_raw_generations = 0
    num_parse_failures = 0
    n = max(1, num_candidates_per_seed)

    def _log_parse_failure(reason: str, raw_text: str = "", parsed: Optional[Dict] = None) -> None:
        if parse_failures_out is not None:
            parse_failures_out.append({
                "target_prompt": target_prompt,
                "reason": reason,
                "raw_output": raw_text[:2000] if raw_text else "",
                "parsed": parsed,
            })

    for i in range(n):
        model_prompt = build_generation_prompt(
            target_prompt=target_prompt,
            seed_response=seed_response,
            seed_reasoning=seed_reasoning,
            safety_principle=safety_principle,
            transformation_rule=transformation_rule,
            use_principle_context=use_principle_context,
            use_example=use_example,
            second_example=second_example,
        )
        if getattr(tokenizer, "chat_template", None):
            model_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": model_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = tokenizer(model_prompt, return_tensors="pt").to(device)
        if generation_seed is not None:
            s = generation_seed + seed_idx * 10000 + i * 1000
            torch.manual_seed(s)
            if device == "cuda":
                torch.cuda.manual_seed_all(s)
            try:
                import numpy as np
                np.random.seed(s)
            except ImportError:
                pass
        gen_kwargs: Dict[str, Any] = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
        )
        if no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        use_grammar = use_grammar_constraint and "gemma" not in (generator_name or "").lower()
        if use_grammar:
            try:
                output = _generate_with_grammar_constraint(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    inputs=dict(inputs),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    pad_token_id=tokenizer.eos_token_id,
                    response_max_length=response_max_length,
                    reasoning_max_length=reasoning_max_length,
                    generator_name=generator_name,
                )
            except Exception as e:
                print(f"[WARN] Grammar constraint failed ({e}), falling back to unconstrained")
                output = model.generate(**gen_kwargs)[0]
        else:
            output = model.generate(**gen_kwargs)[0]

        input_len = inputs["input_ids"].shape[1]
        num_raw_generations += 1
        text = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
        if use_grammar_constraint and text and not text.lstrip().startswith("{"):
            print(f"[WARN] Grammar constraint may not have applied: output does not start with {{ (got {repr(text[:50])}...)")
        if not text:
            _log_parse_failure("empty_output", raw_text=text)
            num_parse_failures += 1
            continue
        obj = _extract_first_json_object(text)
        if obj is not None:
            response = obj.get("response") or obj.get("output")
            reasoning = obj.get("reasoning") or obj.get("reason")
            if isinstance(response, dict):
                response = response.get("text") or response.get("response") or json.dumps(response)
            if isinstance(response, str):
                response = response.strip()
            elif isinstance(response, dict):
                response = json.dumps(response).strip()
            if response and isinstance(response, str):
                reasoning_str = (reasoning or "").strip() if isinstance(reasoning, str) else ""
                placeholder_patterns = (
                    "[your prompt here]", "[your response here]", "[your reasoning here]",
                    "[insert reasoning here]", "[Insert reasoning here]",
                    "your 2-3 sentence safe answer", "one short sentence",
                    "Your 2-3 sentence safe answer", "Your response to the prompt",
                    "a safe response", "a short sentence",
                    "<your safe answer>", "<one short sentence>", "<target prompt",
                    "[insert ", "[replace ", "[add ", "[fill ", "[write ",
                    "<prompt>", "<response>", "<reasoning>",
                )
                meta_phrases = (
                    "i refuse the request", "i'll explain why", "refuse and redirect",
                    "no placeholders", "offers 1-3 safe alternatives", "offers 1–3 safe alternatives",
                    "what approach you'll take", "refuses request;",
                    '"reasoning":', '"response":',
                )
                resp_lower = response.lower()
                rsn_lower = reasoning_str.lower()
                if any(ph in resp_lower or ph in rsn_lower for ph in placeholder_patterns):
                    _log_parse_failure("placeholder_detected", raw_text=text, parsed=obj)
                    num_parse_failures += 1
                    continue
                if any(m in resp_lower for m in meta_phrases):
                    _log_parse_failure("meta_description_in_response", raw_text=text, parsed=obj)
                    num_parse_failures += 1
                    continue
                if response.strip().startswith("{") or '"reasoning"' in response or '"response"' in response:
                    _log_parse_failure("json_in_response_field", raw_text=text, parsed=obj)
                    num_parse_failures += 1
                    continue
                if len(response) < 15:
                    _log_parse_failure("length_too_short", raw_text=text, parsed=obj)
                    num_parse_failures += 1
                    continue
                pair: Dict[str, Any] = {"prompt": target_prompt, "response": response}
                if not reasoning_str or len(reasoning_str) < 10:
                    _log_parse_failure("reasoning_empty_or_too_short", raw_text=text, parsed=obj)
                    num_parse_failures += 1
                    continue
                pair["reasoning"] = reasoning_str
                generated_pairs.append(pair)
                break
        _log_parse_failure("json_parse_failed" if obj is None else "missing_response_or_reasoning", raw_text=text, parsed=obj)
        num_parse_failures += 1

    return generated_pairs, num_raw_generations, num_parse_failures


def _build_json_parser(
    response_max_length: int = 800,
    reasoning_max_length: int = 500,
) -> Optional[Any]:
    """Build JsonSchemaParser for grammar-constrained JSON. Returns None if lm-format-enforcer unavailable."""
    try:
        from pydantic import BaseModel, Field
        from lmformatenforcer import JsonSchemaParser

        class _GenerationOutput(BaseModel):
            response: str = Field(max_length=response_max_length)
            reasoning: str = Field(max_length=reasoning_max_length)

        schema = getattr(_GenerationOutput, "model_json_schema", lambda: _GenerationOutput.schema())()
        return JsonSchemaParser(schema)
    except ImportError:
        return None


def _generate_with_grammar_constraint(
    model: Any,
    tokenizer: Any,
    device: str,
    inputs: Dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    pad_token_id: int,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    response_max_length: int = 800,
    reasoning_max_length: int = 500,
    generator_name: str = "",
) -> Any:
    """Use lm-format-enforcer: prefix_allowed_tokens_fn for Gemma (transformers 4.46+), generate_enforced for TinyLlama/GPT-Neo."""
    parser = _build_json_parser(
        response_max_length=response_max_length,
        reasoning_max_length=reasoning_max_length,
    )
    if parser is None:
        raise RuntimeError("lm-format-enforcer not available for grammar constraint")

    use_prefix_allowed = "gemma" in (generator_name or "").lower()

    if use_prefix_allowed:
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

        prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
        gen_kwargs: Dict[str, Any] = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            pad_token_id=pad_token_id,
            repetition_penalty=repetition_penalty,
            prefix_allowed_tokens_fn=prefix_fn,
        )
    else:
        from lmformatenforcer.integrations.transformers import generate_enforced

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            pad_token_id=pad_token_id,
            repetition_penalty=repetition_penalty,
        )
        if no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        output = generate_enforced(model, tokenizer, parser, **gen_kwargs)
        if hasattr(output, "sequences"):
            return output.sequences[0]
        return output[0] if hasattr(output, "__getitem__") else output

    if no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    return model.generate(**gen_kwargs)[0]


def parse_verifier_label(
    label_text: str,
    unparsed_as_pass: bool = True,
    lenient: bool = False,
) -> Tuple[bool, str, str]:
    """Parse PASS/NO_PASS from anywhere in output (handles batching artifacts)."""
    text_upper = label_text.upper().replace("-", "_")
    if re.search(r"\bNO_?PASS\b", text_upper):
        if lenient:
            return True, label_text, "unsafe_label_overridden"
        return False, label_text, "unsafe_label"
    if re.search(r"\bPASS\b", text_upper):
        return True, label_text, "safe_label"
    return (True, label_text, "unparsed_as_pass") if unparsed_as_pass else (False, label_text, "unparsed_label")


def verify_candidates_with_local_model(
    verifier_model: Any,
    verifier_tokenizer: Any,
    device: str,
    generated_pairs: List[Dict[str, str]],
    verifier_max_new_tokens: int,
    unparsed_as_pass: bool = True,
    verifier_lenient: bool = False,
) -> List[Tuple[bool, str, str]]:
    if not generated_pairs:
        return []
    prompts: List[str] = []
    for generated_pair in generated_pairs:
        user_prompt = generated_pair.get("prompt", "")
        response_text = generated_pair.get("response", "")
        reasoning_text = generated_pair.get("reasoning", "").strip()
        if reasoning_text:
            verifier_prompt = (
                "User asked: " + user_prompt + "\n"
                "Assistant: " + response_text + "\n\n"
                "Does the assistant refuse/redirect (PASS) or comply with the harmful request (NO_PASS)? Reply with PASS or NO_PASS:"
            )
        else:
            verifier_prompt = (
                "User asked: " + user_prompt + "\n"
                "Assistant: " + response_text + "\n\n"
                "Does the assistant refuse/redirect (PASS) or comply (NO_PASS)? Reply with PASS or NO_PASS:"
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
        results.append(parse_verifier_label(
            label_text, unparsed_as_pass=unparsed_as_pass, lenient=verifier_lenient
        ))
    return results


CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "50"))  # prompts per container


@app.function(
    image=image,
    gpu="any",
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={
        MODEL_CACHE: model_volume,
        "/root/results": results_volume,
    },
    timeout=7200,  # 2h per chunk (50 prompts ~25-40 min)
)
def process_chunk(chunk_id: int, target_prompts: List[str]):
    """Process a chunk of target prompts: classify+sample, generate, verify. Writes to chunk_{id}_*.jsonl."""
    from tqdm import tqdm

    if not target_prompts:
        return

    results_volume.reload()  # get latest from previous runs for resume check
    results_dir = "/root/results"
    chunk_prefix = f"{results_dir}/chunk_{chunk_id}"
    gen_path = f"{chunk_prefix}_generated.jsonl"
    existing = read_jsonl(gen_path)
    if len(existing) >= len(target_prompts) * 0.95:
        print(f"[CHUNK {chunk_id}] Already complete ({len(existing)}/{len(target_prompts)}), skipping")
        return

    token = os.getenv("HF_TOKEN")
    config_path = ROOT / "configs" / "default.yaml"
    config = load_config(str(config_path))
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    generation_cfg = config.get("generation", {})
    device = resolve_device()
    rng = random.Random(int(config.get("seed", 42)) + chunk_id * 1000)

    generator_name = os.environ.get("GENERATOR_MODEL") or model_cfg["generator_name"]
    verifier_name = (
        os.environ.get("VERIFIER_MODEL")
        or (generator_name if os.environ.get("GENERATOR_MODEL") else None)
        or model_cfg.get("verifier_name")
        or generator_name
    )

    ensure_model_cached(generator_name, token=token)
    if verifier_name != generator_name:
        ensure_model_cached(verifier_name, token=token)

    load_in_4bit = bool(model_cfg.get("load_in_4bit", False))
    model, tokenizer = load_text_model(
        generator_name, device, token=token, load_in_4bit=load_in_4bit
    )
    if verifier_name == generator_name:
        verifier_model, verifier_tokenizer = model, tokenizer
    else:
        verifier_model, verifier_tokenizer = load_text_model(
            verifier_name, device, token=token, load_in_4bit=load_in_4bit
        )

    principles_path = data_cfg.get("seed_principles_path", "data/seed_principles.json")
    id_to_principle, id_to_rule, id_to_instantiations = load_principle_mapping(principles_path)
    if chunk_id == 0:
        mapping_out = data_cfg.get("principle_id_mapping_path", "data/principle_id_mapping.json")
        write_principle_id_mapping(id_to_principle, mapping_out)

    principle_ids = [p for p in id_to_principle.keys() if id_to_instantiations.get(p)]

    use_principle_context = bool(generation_cfg.get("use_principle_context", True))
    use_example = bool(generation_cfg.get("use_example", False))
    use_two_examples = bool(generation_cfg.get("use_two_examples", False))
    num_candidates = int(generation_cfg.get("num_candidates_per_seed", 3))
    response_max_length = int(generation_cfg.get("response_max_length", 800))
    reasoning_max_length = int(generation_cfg.get("reasoning_max_length", 500))
    max_new_tokens = (response_max_length + reasoning_max_length + 80) // 3
    temperature = float(generation_cfg.get("temperature", 0.6))
    top_p = float(generation_cfg.get("top_p", 0.9))
    repetition_penalty = float(generation_cfg.get("repetition_penalty", 1.0))
    no_repeat_ngram_size = int(generation_cfg.get("no_repeat_ngram_size", 0))
    verifier_max_new_tokens = int(generation_cfg.get("verifier_max_new_tokens", 12))
    verifier_batch_size = int(generation_cfg.get("verifier_batch_size", 8))
    verifier_unparsed_as_pass = bool(generation_cfg.get("verifier_unparsed_as_pass", True))
    verifier_lenient = bool(generation_cfg.get("verifier_lenient", False))
    use_grammar_constraint = bool(generation_cfg.get("use_grammar_constraint", False))
    if use_grammar_constraint:
        if _build_json_parser(response_max_length=response_max_length, reasoning_max_length=reasoning_max_length) is None:
            print("[WARN] use_grammar_constraint=True but lm-format-enforcer unavailable; disabling")
            use_grammar_constraint = False
        else:
            print("[INFO] Grammar-constrained decoding enabled (lm-format-enforcer generate_enforced)")

    generation_items: List[Dict[str, Any]] = []
    for target_prompt in tqdm(target_prompts, desc=f"Chunk{chunk_id} classify", unit="prompt"):
        principle_id = classify_principle_for_prompt(
            model=model,
            tokenizer=tokenizer,
            device=device,
            target_prompt=target_prompt,
            id_to_principle=id_to_principle,
        )
        if principle_id is None or not id_to_instantiations.get(principle_id):
            principle_id = rng.choice(principle_ids) if principle_ids else None
        if principle_id is None:
            continue
        instantiations = id_to_instantiations[principle_id]
        ref = rng.choice(instantiations)
        second_ref = None
        if use_two_examples and len(instantiations) >= 2:
            others = [i for i in instantiations if i.get("prompt") != ref.get("prompt")]
            if others:
                second_ref = rng.choice(others)
        safety_principle = id_to_principle.get(principle_id)
        transformation_rule = id_to_rule.get(principle_id)
        second_example_tuple = None
        if second_ref:
            second_example_tuple = (
                second_ref.get("prompt", "").strip(),
                second_ref.get("response", "").strip(),
                second_ref.get("reasoning"),
            )
        generation_items.append({
            "target_prompt": target_prompt,
            "seed_response": ref.get("response", "").strip(),
            "safety_principle": safety_principle,
            "transformation_rule": transformation_rule,
            "seed_reasoning": ref.get("reasoning"),
            "second_example": second_example_tuple,
        })

    generated_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    verified_rows: List[Dict[str, Any]] = []
    parse_failures: List[Dict[str, Any]] = []

    gen_seed_cfg = generation_cfg.get("generation_seed")
    generation_seed = int(gen_seed_cfg) if isinstance(gen_seed_cfg, (int, float)) else None

    for seed_idx, item in enumerate(tqdm(generation_items, desc=f"Chunk{chunk_id} seeds", unit="seed")):
        generated_pairs, _, _ = generate_json_pairs_for_seed(
            model=model,
            tokenizer=tokenizer,
            device=device,
            target_prompt=item["target_prompt"],
            seed_response=item["seed_response"],
            num_candidates_per_seed=num_candidates,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            seed_reasoning=item["seed_reasoning"],
            safety_principle=item["safety_principle"],
            transformation_rule=item["transformation_rule"],
            use_principle_context=use_principle_context,
            use_example=use_example,
            second_example=item["second_example"],
            generation_seed=generation_seed,
            seed_idx=seed_idx,
            parse_failures_out=parse_failures,
            use_grammar_constraint=use_grammar_constraint,
            response_max_length=response_max_length,
            reasoning_max_length=reasoning_max_length,
            generator_name=generator_name,
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
                    unparsed_as_pass=verifier_unparsed_as_pass,
                    verifier_lenient=verifier_lenient,
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

    write_jsonl(gen_path, generated_rows)
    write_jsonl(f"{chunk_prefix}_rejected.jsonl", rejected_rows)
    write_jsonl(f"{chunk_prefix}_verified.jsonl", verified_rows)
    write_jsonl(f"{chunk_prefix}_parse_failures.jsonl", parse_failures)
    results_volume.commit()
    print(f"[CHUNK {chunk_id}] Done. {len(verified_rows)} safe / {len(rejected_rows)} rejected / {len(generated_rows)} total. {len(parse_failures)} parse failures")


@app.function(
    image=image,
    volumes={"/root/results": results_volume},
    timeout=600,
)
def merge_chunks(num_chunks: int):
    """Merge chunk_*_*.jsonl files into final output files."""
    results_dir = "/root/results"
    results_volume.reload()

    generated_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    verified_rows: List[Dict[str, Any]] = []
    parse_failures: List[Dict[str, Any]] = []

    for i in range(num_chunks):
        for suffix, lst in [
            ("_generated.jsonl", generated_rows),
            ("_rejected.jsonl", rejected_rows),
            ("_verified.jsonl", verified_rows),
            ("_parse_failures.jsonl", parse_failures),
        ]:
            path = f"{results_dir}/chunk_{i}{suffix}"
            if os.path.exists(path):
                lst.extend(read_jsonl(path))

    verified_path = "results/verified_data.jsonl"
    parse_failures_path = "results/parse_failures.jsonl"
    write_jsonl(f"{results_dir}/generated_candidates.jsonl", generated_rows)
    write_jsonl(f"{results_dir}/rejected_candidates.jsonl", rejected_rows)
    write_jsonl(f"{results_dir}/{os.path.basename(verified_path)}", verified_rows)
    write_jsonl(f"{results_dir}/{os.path.basename(parse_failures_path)}", parse_failures)

    for i in range(num_chunks):
        for suffix in ["_generated.jsonl", "_rejected.jsonl", "_verified.jsonl", "_parse_failures.jsonl"]:
            path = f"{results_dir}/chunk_{i}{suffix}"
            if os.path.exists(path):
                os.remove(path)

    results_volume.commit()
    print(f"[MERGE] Done. {len(verified_rows)} verified / {len(rejected_rows)} rejected / {len(generated_rows)} generated / {len(parse_failures)} parse failures")


@app.local_entrypoint()
def run():
    """Load prompts, split into chunks, run in parallel, merge results."""
    import yaml

    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    harmfulqa_path = config.get("data", {}).get("seeds_harmfulqa_path", "data/seeds_harmfulqa.jsonl")
    base = Path(__file__).resolve().parents[1]
    full_path = base / harmfulqa_path if not str(harmfulqa_path).startswith("/") else Path(harmfulqa_path)
    harmfulqa_rows = read_jsonl(str(full_path))

    target_prompts: List[str] = []
    for row in harmfulqa_rows:
        p = _extract_prompt_from_row(row)
        if p:
            target_prompts.append(p)

    max_seeds = int(config.get("generation", {}).get("max_seeds", 0))
    if max_seeds > 0 and len(target_prompts) > max_seeds:
        rng = random.Random(int(config.get("seed", 42)))
        target_prompts = rng.sample(target_prompts, max_seeds)
        print(f"[INFO] max_seeds={max_seeds}: sampled {len(target_prompts)} target prompts")
    else:
        print(f"[INFO] Loaded {len(target_prompts)} target prompts")

    chunk_size = int(os.environ.get("CHUNK_SIZE") or config.get("generation", {}).get("chunk_size", CHUNK_SIZE))
    chunks: List[Tuple[int, List[str]]] = []
    for i in range(0, len(target_prompts), chunk_size):
        chunk_prompts = target_prompts[i : i + chunk_size]
        chunks.append((len(chunks), chunk_prompts))

    print(f"[INFO] Split into {len(chunks)} chunks (CHUNK_SIZE={chunk_size})")
    if not chunks:
        print("[INFO] No prompts to process")
        return

    list(process_chunk.starmap(chunks))

    merge_chunks.remote(len(chunks))