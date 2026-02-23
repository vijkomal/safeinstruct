"""
Data loaders for safeinstruct.

Expected JSONL row:
{"prompt": "...", "response": "...", "safety_label": "..."}
"""

import json
import random
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset


class SafeInstructDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append(
                {
                    "prompt": item["prompt"],
                    "response": item["response"],
                    "safety_label": item.get("safety_label"),
                }
            )
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


def load_seed_examples(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _read_jsonl(config["data"]["seed_pivots_path"])


def load_verified_examples(
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return _read_jsonl(config["data"]["verified_path"])


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
