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
            for key in ("source", "safety_principle", "transformation_rule", "reasoning"):
                if key in item and item[key]:
                    row[key] = item[key]
            rows.append(row)
    return rows


def _load_seed_principles(path: str, base_dir: Path) -> List[Dict[str, Any]]:
    """Load structured principles and flatten to (prompt, response) rows with principle metadata."""
    full_path = base_dir / path if not os.path.isabs(path) else Path(path)
    if not full_path.exists():
        return []
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows: List[Dict[str, Any]] = []
    for p in data.get("principles", []):
        principle_id = p.get("principle_id", "")
        instruction = p.get("instruction", "")
        safety_principle = p.get("safety_principle", "")
        transformation_rule = p.get("transformation_rule", "")
        for inst in p.get("domain_instantiations", []):
            row = {
                "prompt": inst["prompt"],
                "response": inst["response"],
                "safety_label": "safe",
                "source": "manual",
                "principle_id": principle_id,
                "instruction": instruction,
                "safety_principle": safety_principle,
                "transformation_rule": transformation_rule,
                "domain": inst.get("domain", ""),
            }
            if "reasoning" in inst:
                row["reasoning"] = inst["reasoning"]
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
    """Load and merge seeds from manual (principles or flat) and HarmfulQA sources.

    Config keys:
      - data.seeds_expanded_path: 2000+ merged seeds (principles + HarmfulQA + HarmBench) - preferred
      - data.seed_principles_path: parameterized principles (used when seeds_expanded not set)
      - data.seeds_manual_path: flat JSONL fallback
      - data.seeds_harmfulqa_path: path to HarmfulQA import (optional)
      - data.seeds_prosocial_path: ProsocialDialog import (optional)

    When seeds_expanded_path is set, it is loaded as the primary source (contains merged corpus).
    Otherwise, principles + harmfulqa + prosocial are merged.
    """
    data_cfg = config.get("data", {})
    base = Path(data_cfg.get("base_dir", ROOT))

    expanded_path = data_cfg.get("seeds_expanded_path")
    principles_path = data_cfg.get("seed_principles_path")
    manual_path = data_cfg.get("seeds_manual_path")
    harmfulqa_path = data_cfg.get("seeds_harmfulqa_path")
    prosocial_path = data_cfg.get("seeds_prosocial_path")

    rows: List[Dict[str, Any]] = []
    if expanded_path and (base / expanded_path).exists():
        rows.extend(_read_jsonl(expanded_path, base))
    else:
        if not principles_path and not manual_path and not harmfulqa_path and not prosocial_path:
            raise ValueError(
                "No seed source configured. Set data.seeds_expanded_path, data.seed_principles_path, "
                "data.seeds_manual_path, data.seeds_harmfulqa_path, and/or data.seeds_prosocial_path."
            )
        if source_filter in (None, "manual"):
            if principles_path:
                rows.extend(_load_seed_principles(principles_path, base))
            elif manual_path:
                rows.extend(_read_jsonl(manual_path, base))
        if harmfulqa_path and source_filter in (None, "harmfulqa"):
            rows.extend(_read_jsonl(harmfulqa_path, base))
        if prosocial_path and source_filter in (None, "prosocial"):
            rows.extend(_read_jsonl(prosocial_path, base))
    # Deduplicate by (prompt, response) - keep first occurrence
    seen: set[tuple[str, str]] = set()
    deduped: List[Dict[str, Any]] = []
    for r in rows:
        key = (r.get("prompt", ""), r.get("response", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


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
