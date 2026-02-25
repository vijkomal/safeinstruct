#!/usr/bin/env python3
"""Naive single-layer fine-tuning for generator model on verified data."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


def format_example(prompt: str, response: str) -> str:
    return f"User: {prompt}\nAssistant: {response}"


def batchify(items: List[Dict[str, str]], batch_size: int) -> List[List[Dict[str, str]]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def main() -> None:
    config = load_config(str(ROOT / "configs" / "default.yaml"))
    device = resolve_device(config)

    model_name = config["model"]["generator_name"]
    train_cfg = config.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 2))
    learning_rate = float(train_cfg.get("learning_rate", 1e-4))
    num_epochs = int(train_cfg.get("num_epochs", 1))
    max_length = int(train_cfg.get("max_length", 256))
    output_dir = str(ROOT / train_cfg.get("output_dir", "results/checkpoints"))
    save_name = str(train_cfg.get("save_name", "generator_single_layer_ft"))

    rows = load_verified_examples(config)
    if not rows:
        raise ValueError(
            "No verified examples found. Run generation+verification first to populate "
            "data.verified_path."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.train()

    # Freeze everything, then train exactly one layer: lm_head.
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "lm_head"):
        for param in model.lm_head.parameters():
            param.requires_grad = True
    else:
        raise ValueError("Model does not expose lm_head; cannot do single-layer tuning.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate)

    examples = [
        {"text": format_example(r["prompt"], r["response"])}
        for r in rows
        if isinstance(r.get("prompt"), str) and isinstance(r.get("response"), str)
    ]

    if not examples:
        raise ValueError("Verified dataset is empty after formatting.")

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Examples: {len(examples)}")
    print("Trainable layer: lm_head only")

    for epoch in range(num_epochs):
        running_loss = 0.0
        batches = batchify(examples, batch_size=batch_size)
        for batch in tqdm(batches, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            texts = [x["text"] for x in batch]
            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100

            outputs = model(**enc, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        avg_loss = running_loss / max(1, len(batches))
        print(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

    save_path = os.path.join(output_dir, save_name)
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved fine-tuned model to: {save_path}")


if __name__ == "__main__":
    main()
