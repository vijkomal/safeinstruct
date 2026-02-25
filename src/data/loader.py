"""
Data loaders for safeinstruct.

Expected JSONL row:
{"prompt": "...", "response": "...", "safety_label": "...", "source": "manual"|"harmfulqa" (optional)}
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]


class SafeInstructDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


def _read_jsonl(path: str, base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if base_dir:
        path = str(base_dir / path) if not os.path.isabs(path) else path
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            row = {
                "prompt": item["prompt"],
                "response": item["response"],
                "safety_label": item.get("safety_label"),
            }
            if "source" in item:
                row["source"] = item["source"]
            rows.append(row)
    return rows


def _split_train_val(
    rows: List[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    n_val = max(1, int(len(rows) * val_ratio))
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    return shuffled[n_val:], shuffled[:n_val]


def load_seed_examples(
    config: Dict[str, Any],
    source_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load and merge seeds from manual and HarmfulQA sources.

    Config keys:
      - data.seeds_manual_path: path to curated seeds
      - data.seeds_harmfulqa_path: path to HarmfulQA import (optional)

    If source_filter is set ("manual" or "harmfulqa"), only that source is loaded.
    """
    data_cfg = config.get("data", {})
    base = Path(data_cfg.get("base_dir", ROOT))

    manual_path = data_cfg.get("seeds_manual_path")
    harmfulqa_path = data_cfg.get("seeds_harmfulqa_path")

    if not manual_path and not harmfulqa_path:
        raise ValueError(
            "No seed source configured. Set data.seeds_manual_path and/or data.seeds_harmfulqa_path."
        )

    rows: List[Dict[str, Any]] = []
    if manual_path and source_filter in (None, "manual"):
        rows.extend(_read_jsonl(manual_path, base))
    if harmfulqa_path and source_filter in (None, "harmfulqa"):
        rows.extend(_read_jsonl(harmfulqa_path, base))
    return rows


def load_verified_examples(
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    data_cfg = config.get("data", {})
    base = Path(data_cfg.get("base_dir", ROOT))
    return _read_jsonl(data_cfg["verified_path"], base)


def get_seed_dataloader(
    config: Dict[str, Any],
    batch_size: Optional[int] = None,
    shuffle: bool = False,
) -> DataLoader:
    rows = load_seed_examples(config)
    if batch_size is None:
        batch_size = int(config["train"]["batch_size"])
    return DataLoader(
        SafeInstructDataset(rows),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def get_finetune_dataloaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader]:
    val_ratio = float(config["data"].get("val_ratio"))
    seed = int(config.get("seed"))
    batch_size = int(config["train"]["batch_size"])
    rows = load_verified_examples(config)
    train_rows, val_rows = _split_train_val(rows, val_ratio=val_ratio, seed=seed)

    train_loader = DataLoader(
        SafeInstructDataset(train_rows),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SafeInstructDataset(val_rows),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader
