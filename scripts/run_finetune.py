#!/usr/bin/env python3
"""Fine-tuning for generator model on verified data. Supports response-only loss and full/LoRA tuning."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HF_TOKEN = os.environ.get("HF_TOKEN")

from src.data import load_verified_examples


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(config: Dict[str, Any]) -> str:
    requested = config.get("device")
    if requested:
        return str(requested)
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def format_example(prompt: str, response: str) -> Tuple[str, str]:
    """Return (full_text, prompt_prefix) for loss masking. prompt_prefix is everything before response."""
    prefix = f"Prompt: {prompt}\nResponse: "
    full = prefix + response
    return full, prefix


def batchify(items: List[Dict], batch_size: int) -> List[List[Dict]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def main() -> None:
    config = load_config(str(ROOT / "configs" / "default.yaml"))
    device = resolve_device(config)

    model_name = config["model"]["generator_name"]
    train_cfg = config.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 2))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    learning_rate = float(train_cfg.get("learning_rate", 2e-5))
    num_epochs = int(train_cfg.get("num_epochs", 3))
    max_length = int(train_cfg.get("max_length", 512))
    output_dir = str(ROOT / train_cfg.get("output_dir", "results/checkpoints"))
    save_name = str(train_cfg.get("save_name", "generator_single_layer_ft"))
    train_on_response_only = bool(train_cfg.get("train_on_response_only", True))
    full_finetune = bool(train_cfg.get("full_finetune", False))
    use_lora = bool(train_cfg.get("lora", False))
    lora_r = int(train_cfg.get("lora_r", 8))
    lora_alpha = int(train_cfg.get("lora_alpha", 16))
    lora_dropout = float(train_cfg.get("lora_dropout", 0.05))
    val_ratio = float(config.get("data", {}).get("val_ratio", 0.2))

    rows = load_verified_examples(config)
    if not rows:
        raise ValueError(
            "No verified examples found. Run generation+verification first to populate "
            "data.verified_path."
        )

    examples = [
        {"prompt": r["prompt"], "response": r["response"]}
        for r in rows
        if isinstance(r.get("prompt"), str) and isinstance(r.get("response"), str)
    ]

    # Train/val split
    n_val = max(0, int(len(examples) * val_ratio))
    rng = __import__("random").Random(int(config.get("seed", 42)))
    rng.shuffle(examples)
    train_examples = examples[n_val:]
    val_examples = examples[:n_val]

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN)
    model.to(device)

    if use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Gemma/Llama attn
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    elif full_finetune:
        model.train()
        print("Trainable: full model")
    else:
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "lm_head"):
            for param in model.lm_head.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Model does not expose lm_head; use full_finetune: true or lora: true")
        model.train()
        print("Trainable: lm_head only")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate)

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Train examples: {len(train_examples)}, Val: {len(val_examples)}")
    print(f"Response-only loss: {train_on_response_only}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batches = batchify(train_examples, batch_size=batch_size)
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(batches, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")):
            full_texts = []
            prompt_prefixes = []
            for ex in batch:
                full, prefix = format_example(ex["prompt"], ex["response"])
                full_texts.append(full)
                prompt_prefixes.append(prefix)

            enc = tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100

            if train_on_response_only:
                for i in range(len(batch)):
                    prefix_ids = tokenizer(prompt_prefixes[i], add_special_tokens=False)["input_ids"]
                    prefix_len = len(prefix_ids)
                    if prefix_len < labels.shape[1]:
                        labels[i, :prefix_len] = -100

            outputs = model(**enc, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += float(outputs.loss.item())

        avg_loss = running_loss / max(1, len(batches))
        print(f"Epoch {epoch + 1} train loss: {avg_loss:.4f}")

        if val_examples:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in batchify(val_examples, batch_size=batch_size):
                    full_texts = [format_example(ex["prompt"], ex["response"])[0] for ex in batch]
                    enc = tokenizer(
                        full_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                    ).to(device)
                    labels = enc["input_ids"].clone()
                    labels[enc["attention_mask"] == 0] = -100
                    if train_on_response_only:
                        for i, ex in enumerate(batch):
                            _, prefix = format_example(ex["prompt"], ex["response"])
                            prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
                            if prefix_len < labels.shape[1]:
                                labels[i, :prefix_len] = -100
                    out = model(**enc, labels=labels)
                    val_loss += float(out.loss.item())
            val_loss /= max(1, len(batchify(val_examples, batch_size)))
            print(f"Epoch {epoch + 1} val loss: {val_loss:.4f}")

    save_path = os.path.join(output_dir, save_name)
    os.makedirs(save_path, exist_ok=True)
    if use_lora:
        model.save_pretrained(save_path)  # saves adapter only
        tokenizer.save_pretrained(save_path)
        print(f"Saved LoRA adapter to: {save_path}")
        print("Load with: model = PeftModel.from_pretrained(base_model, save_path)")
    else:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
