#!/usr/bin/env python3
"""Import HarmfulQA prompts and safe responses into seeds_harmfulqa.jsonl.

HarmfulQA (https://huggingface.co/datasets/declare-lab/HarmfulQA) provides:
- question: harmful question (becomes prompt)
- blue_conversations: safe ChatGPT responses; we use the last GPT message as response

Extracts (question, safe_response) pairs and writes to data/seeds_harmfulqa.jsonl.
The loader merges this with data/seeds_manual.jsonl at runtime.
"""

import json
import os
import random
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]


def _get_last_safe_response(blue_conversations) -> Optional[str]:
    """Extract the last substantive GPT response from the first non-null blue conversation.

    Skips generic closings (e.g. 'You're welcome') and favors refusal/redirect content.
    """
    GENERIC_CLOSINGS = (
        "you're welcome",
        "glad i could help",
        "let me know",
        "i'm glad",
        "it was my pleasure",
    )
    if not blue_conversations or not isinstance(blue_conversations, dict):
        return None
    for qid in sorted(blue_conversations.keys(), key=str):
        conv = blue_conversations.get(qid)
        if not conv or not isinstance(conv, list):
            continue
        candidates = []
        for msg in reversed(conv):
            if isinstance(msg, dict) and msg.get("from") == "gpt":
                val = msg.get("value")
                if isinstance(val, str) and len(val.strip()) >= 30:
                    lower = val.strip().lower()
                    if not any(lower.startswith(g) for g in GENERIC_CLOSINGS):
                        candidates.append(val.strip())
        if candidates:
            return candidates[0]
    return None


def main() -> None:
    limit = int(os.environ.get("HARMFULQA_LIMIT", "100"))  # 0 = use all ~1960
    seed = int(os.environ.get("SEED", "42"))
    output_path = ROOT / "data" / "seeds_harmfulqa.jsonl"

    from datasets import load_dataset

    print("Loading HarmfulQA from HuggingFace...")
    ds = load_dataset("declare-lab/HarmfulQA", split="train")

    rows: list[dict] = []
    for ex in ds:
        if not isinstance(ex, dict):
            continue
        question = ex.get("question")
        if not isinstance(question, str) or not question.strip():
            continue
        response = _get_last_safe_response(ex.get("blue_conversations"))
        if not response:
            continue
        rows.append({
            "prompt": question.strip(),
            "response": response,
            "safety_label": "safe",
            "source": "harmfulqa",
        })

    random.Random(seed).shuffle(rows)
    if limit > 0:
        rows = rows[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            out = {
                "prompt": r["prompt"],
                "response": r["response"],
                "safety_label": r.get("safety_label", "safe"),
                "source": "harmfulqa",
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} HarmfulQA examples to seeds_harmfulqa.jsonl.")
    print(f"Written to: {output_path}")


if __name__ == "__main__":
    main()
