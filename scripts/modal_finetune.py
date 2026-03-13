#!/usr/bin/env python3
"""Modal-based fine-tuning for generator model on verified data."""

import contextlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import modal

app = modal.App("safeinstruct-finetune")

MODEL_CACHE = "/root/model-cache"
results_volume = modal.Volume.from_name("safeinstruct-results", create_if_missing=True)
model_volume = modal.Volume.from_name("safeinstruct-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers[torch]>=4.46.0",  # Gemma 2 requires 4.46+
        "accelerate",
        "peft",
        "tqdm",
        "pyyaml",
        "huggingface_hub",
        "matplotlib",
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

ROOT = Path("/root/safeinstruct")
sys.path.insert(0, str(ROOT))


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL without relying on src.data (avoids base_dir logic)."""
    import json

    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append({"prompt": item["prompt"], "response": item["response"]})
    return rows


def format_example(prompt: str, response: str, tokenizer=None) -> Tuple[str, str]:
    """Return (full_text, prompt_prefix) for loss masking. Use chat template when available (Gemma-it, etc)."""
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prefix = f"Prompt: {prompt}\nResponse: "
    return prefix + response, prefix


def batchify(items: List[Dict], batch_size: int) -> List[List[Dict]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


@app.function(
    image=image,
    gpu="T4",
    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={
        "/root/safeinstruct/results": results_volume,
        MODEL_CACHE: model_volume,
    },
)
def finetune(verified_path_override: str = "") -> None:
    """Run fine-tuning on Modal GPU."""
    import torch
    import yaml
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HF_TOKEN = os.environ.get("HF_TOKEN")

    with open(ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    verified_path = verified_path_override if verified_path_override else config.get("data", {}).get("verified_path", "results/verified_data.jsonl")
    data_path = ROOT / verified_path if not os.path.isabs(verified_path) else Path(verified_path)

    results_volume.reload()
    rows = _read_jsonl(str(data_path))
    if not rows:
        raise ValueError(
            f"No verified examples at {data_path}. Run generation+merge on Modal first, "
            "or use --local-data to upload local results/verified_data.jsonl"
        )

    examples = [r for r in rows if isinstance(r.get("prompt"), str) and isinstance(r.get("response"), str)]

    model_name = config["model"]["generator_name"]
    train_cfg = config.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 4))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 4))
    learning_rate = float(train_cfg.get("learning_rate", 2e-4))
    num_epochs = int(train_cfg.get("num_epochs", 5))
    max_length = int(train_cfg.get("max_length", 512))
    save_name = str(train_cfg.get("save_name", "generator_single_layer_ft"))
    train_on_response_only = bool(train_cfg.get("train_on_response_only", True))
    full_finetune = bool(train_cfg.get("full_finetune", False))
    use_lora = bool(train_cfg.get("lora", False))
    lora_r = int(train_cfg.get("lora_r", 8))
    lora_alpha = int(train_cfg.get("lora_alpha", 16))
    lora_dropout = float(train_cfg.get("lora_dropout", 0.05))
    val_ratio = float(config.get("data", {}).get("val_ratio", 0.1))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))
    use_bf16 = bool(train_cfg.get("use_bf16", True))
    use_fp16 = bool(train_cfg.get("use_fp16", False))
    grad_checkpointing = bool(train_cfg.get("grad_checkpointing", False))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 2))
    save_best = bool(train_cfg.get("save_best", True))
    save_plots = bool(train_cfg.get("save_plots", True))
    seed = int(config.get("seed", 42))

    output_dir = str(ROOT / train_cfg.get("output_dir", "results/checkpoints"))
    save_path = os.path.join(output_dir, save_name)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = __import__("random").Random(seed)

    n_val = max(0, int(len(examples) * val_ratio))
    rng.shuffle(examples)
    train_examples = examples[n_val:]
    val_examples = examples[:n_val]

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, cache_dir=MODEL_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN, cache_dir=MODEL_CACHE)
    if grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.to("cuda")

    if use_lora:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    elif full_finetune:
        model.train()
    else:
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "lm_head"):
            for param in model.lm_head.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Model does not expose lm_head; use full_finetune: true or lora: true")
        model.train()

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

    num_batches = (len(train_examples) + batch_size - 1) // batch_size
    steps_per_epoch = max(1, num_batches // grad_accum_steps)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = SequentialLR(
        optimizer,
        [
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=learning_rate * 0.1),
        ],
        milestones=[warmup_steps],
    )

    use_amp = use_bf16 or use_fp16
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else None)
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    print(f"Model: {model_name}")
    print(f"Finetuning on {len(train_examples)} train examples, {len(val_examples)} val examples ({len(examples)} total)")
    print(f"Effective batch: {batch_size * grad_accum_steps}")

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(num_epochs):
        model.train()
        rng.shuffle(train_examples)
        batches = batchify(train_examples, batch_size=batch_size)
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(batches, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")):
            full_texts = []
            prompt_prefixes = []
            for ex in batch:
                full, prefix = format_example(ex["prompt"], ex["response"], tokenizer=tokenizer)
                full_texts.append(full)
                prompt_prefixes.append(prefix)

            enc = tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to("cuda")

            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100

            if train_on_response_only:
                for i in range(len(batch)):
                    prefix_ids = tokenizer(prompt_prefixes[i], add_special_tokens=False)["input_ids"]
                    prefix_len = len(prefix_ids)
                    if prefix_len < labels.shape[1]:
                        labels[i, :prefix_len] = -100

            with (torch.cuda.amp.autocast(dtype=amp_dtype) if use_amp else contextlib.nullcontext()):
                outputs = model(**enc, labels=labels)
                loss = outputs.loss / grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += float(outputs.loss.item())

            if (step + 1) % grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = running_loss / max(1, len(batches))
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1} train loss: {avg_loss:.4f}")

        val_loss = float("inf")
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
                    ).to("cuda")
                    labels = enc["input_ids"].clone()
                    labels[enc["attention_mask"] == 0] = -100
                    if train_on_response_only:
                        for i, ex in enumerate(batch):
                            _, prefix = format_example(ex["prompt"], ex["response"])
                            prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
                            if prefix_len < labels.shape[1]:
                                labels[i, :prefix_len] = -100
                    with (torch.cuda.amp.autocast(dtype=amp_dtype) if use_amp else contextlib.nullcontext()):
                        out = model(**enc, labels=labels)
                    val_loss += float(out.loss.item())
            val_batches = len(batchify(val_examples, batch_size))
            val_loss /= max(1, val_batches)
            val_losses.append(val_loss)
            print(f"Epoch {epoch + 1} val loss: {val_loss:.4f}")
        else:
            val_losses.append(float("nan"))

        if save_best and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(save_path, exist_ok=True)
            if use_lora:
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
            else:
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
            print(f"Saved best (val_loss={val_loss:.4f}) to: {save_path}")
        else:
            patience_counter += 1

        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    os.makedirs(save_path, exist_ok=True)
    if use_lora:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    else:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    if save_plots and train_losses:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        epochs_x = list(range(1, len(train_losses) + 1))
        ax.plot(epochs_x, train_losses, "b-o", label="Train", markersize=4)
        if val_losses and any(v == v for v in val_losses):
            valid_val = [(e, v) for e, v in zip(epochs_x, val_losses) if v == v]
            if valid_val:
                ex, vx = zip(*valid_val)
                ax.plot(ex, vx, "r-s", label="Val", markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.set_title(f"Training curves ({model_name})")
        fig.tight_layout()
        plot_path = os.path.join(output_dir, "training_curves.png")
        fig.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved training curves to: {plot_path}")

    results_volume.commit()
    print(f"Checkpoints saved to volume (results/checkpoints/{save_name})")


@app.function(
    image=image,
    volumes={"/root/safeinstruct/results": results_volume},
)
def upload_verified_data(jsonl_content: str, target_path: str) -> None:
    """Upload local verified data to the results volume."""
    path = f"/root/safeinstruct/{target_path}" if not target_path.startswith("/") else target_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(jsonl_content)
    results_volume.commit()
    print(f"Uploaded {len(jsonl_content.splitlines())} lines to volume")


@app.local_entrypoint()
def run(use_local_data: bool = False, verified_path: str = "results/verified_data.jsonl"):
    """Run fine-tuning on Modal.

    Use --verified-path to specify data file (e.g. results/gptneo_verified_data.jsonl).
    Use --use-local-data to upload from local before running.
    """
    project_root = Path(__file__).resolve().parents[1]
    local_path = project_root / verified_path

    if use_local_data:
        if not local_path.exists():
            raise FileNotFoundError(f"Local {local_path} not found.")
        content = local_path.read_text(encoding="utf-8")
        upload_verified_data.remote(content, verified_path)

    finetune.remote(verified_path)
