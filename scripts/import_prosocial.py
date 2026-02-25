#!/usr/bin/env python3
"""Import ProsocialDialog (allenai/prosocial-dialog) into seed format.

ProsocialDialog includes safety rationale reasoning from everyday conversations:
- context: potentially unsafe utterance → prompt
- response: prosocial/safe response → response
- rots: rules-of-thumb (social norms) → safety_principle
- safety_annotation_reasons: free-form rationales → reasoning (chain-of-thought)

Exports to data/seeds_prosocial.jsonl for use as seed examples with principle + CoT.
Env: PROSOCIAL_LIMIT (default 200), SEED (default 42).
"""

import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _first_rot(rots) -> str | None:
    if not rots or not isinstance(rots, list):
        return None
    for r in rots:
        if isinstance(r, str) and r.strip():
            return r.strip()
    return None


def _first_reason(reasons) -> str | None:
    if not reasons or not isinstance(reasons, list):
        return None
    for r in reasons:
        if isinstance(r, str) and r.strip():
            return r.strip()
    return None


def main() -> None:
    limit = int(os.environ.get("PROSOCIAL_LIMIT", "200"))  # 0 = use all ~165k
    seed = int(os.environ.get("SEED", "42"))
    output_path = ROOT / "data" / "seeds_prosocial.jsonl"

    from datasets import load_dataset

    print("Loading ProsocialDialog from HuggingFace...")
    ds = load_dataset("allenai/prosocial-dialog", split="train")

    rows: list[dict] = []
    for ex in ds:
        if not isinstance(ex, dict):
            continue
        context = ex.get("context")
        response = ex.get("response")
        if not isinstance(context, str) or not context.strip():
            continue
        if not isinstance(response, str) or not response.strip() or len(response.strip()) < 20:
            continue
        rot = _first_rot(ex.get("rots"))
        reason = _first_reason(ex.get("safety_annotation_reasons"))
        rows.append({
            "prompt": context.strip(),
            "response": response.strip(),
            "safety_label": "safe",
            "source": "prosocial",
            "safety_principle": rot or "",
            "reasoning": reason or "",
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
                "source": r.get("source", "prosocial"),
            }
            if r.get("safety_principle"):
                out["safety_principle"] = r["safety_principle"]
            if r.get("reasoning"):
                out["reasoning"] = r["reasoning"]
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} ProsocialDialog examples to seeds_prosocial.jsonl.")
    print(f"Written to: {output_path}")


if __name__ == "__main__":
    main()
