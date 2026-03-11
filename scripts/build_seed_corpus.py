#!/usr/bin/env python3
"""Build seed corpus by merging: seed_principles, HarmfulQA, and template expansion.

HarmBench is EXCLUDED to avoid eval leakage (training on HarmBench prompts inflates eval scores).

Output: data/seeds_expanded.jsonl
Usage:
  python scripts/build_seed_corpus.py [--target 2100]
  # Requires: HarmfulQA import (HARMFULQA_LIMIT=0) via scripts/import_harmfulqa.py
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]


def load_seed_principles(base: Path) -> List[Dict]:
    """Load and flatten seed_principles.json."""
    path = base / "data" / "seed_principles.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    rows = []
    for p in data.get("principles", []):
        for inst in p.get("domain_instantiations", []):
            rows.append({
                "prompt": inst["prompt"],
                "response": inst["response"],
                "reasoning": inst.get("reasoning", ""),
                "safety_label": "safe",
                "source": "principles",
                "principle_id": p.get("principle_id", ""),
                "safety_principle": p.get("safety_principle", ""),
                "transformation_rule": p.get("transformation_rule", ""),
            })
    return rows


def load_harmfulqa(base: Path, limit: int = 0) -> List[Dict]:
    """Load HarmfulQA seeds. Run: HARMFULQA_LIMIT=0 python scripts/import_harmfulqa.py first."""
    path = base / "data" / "seeds_harmfulqa.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            r["source"] = "harmfulqa"
            rows.append(r)
    if limit > 0:
        rows = rows[:limit]
    return rows


def run_template_expansion(base: Path, target: int, rng: random.Random) -> List[Dict]:
    """Run the template-based expansion (from generate_seed_expansion)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gen_expansion", base / "scripts" / "generate_seed_expansion.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    principles = gen.load_principles()
    per = max(50, target // 15)
    rows = []
    rows.extend(gen.generate_p1(rng, principles, per))
    rows.extend(gen.generate_p2(rng, principles, per))
    rows.extend(gen.generate_p3(rng, principles, per))
    rows.extend(gen.generate_p4(rng, principles, per))
    rows.extend(gen.generate_p5(rng, principles, per))
    rows.extend(gen.generate_p6(rng, principles, per))
    rows.extend(gen.generate_p7_to_p15(rng, principles, "P7", gen.P7_TEMPLATES, gen.P7_SLOTS, per))
    rows.extend(gen.generate_p7_to_p15(rng, principles, "P8", gen.P8_TEMPLATES, gen.P8_SLOTS, per))
    rows.extend(gen.generate_p7_to_p15(rng, principles, "P9", gen.P9_TEMPLATES, gen.P9_SLOTS, per))
    rows.extend(gen.generate_p7_to_p15(rng, principles, "P10", gen.P10_TEMPLATES, gen.P10_SLOTS, per))
    rows.extend(gen.generate_p7_to_p15(rng, principles, "P11", gen.P11_TEMPLATES, gen.P11_SLOTS, per))
    rows.extend(gen.generate_p7_to_p15(rng, principles, "P12", gen.P12_TEMPLATES, gen.P12_SLOTS, per))
    rows.extend(gen.generate_p7_to_p15(rng, principles, "P13", gen.P13_TEMPLATES, gen.P13_SLOTS, per))
    rows.extend(gen.generate_p7_to_p15(rng, principles, "P14", gen.P14_TEMPLATES, gen.P14_SLOTS, per))
    rows.extend(gen.generate_p7_to_p15(rng, principles, "P15", gen.P15_TEMPLATES, gen.P15_SLOTS, per))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=2100)
    parser.add_argument("--output", type=str, default="data/seeds_expanded.jsonl")
    parser.add_argument("--harmfulqa-limit", type=int, default=0, help="0 = use all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base = ROOT

    all_rows: List[Dict] = []

    # 1. Seed principles (~34)
    p_rows = load_seed_principles(base)
    all_rows.extend(p_rows)
    print(f"Loaded {len(p_rows)} from seed_principles.json")

    # 2. HarmfulQA
    hq_rows = load_harmfulqa(base, limit=args.harmfulqa_limit)
    all_rows.extend(hq_rows)
    print(f"Loaded {len(hq_rows)} from seeds_harmfulqa.jsonl")

    # 3. Template expansion (fill to target) — HarmBench excluded to avoid eval leakage
    need = max(0, args.target - len(all_rows))
    if need > 0:
        exp_rows = run_template_expansion(base, need, rng)
        all_rows.extend(exp_rows)
        print(f"Generated {len(exp_rows)} from template expansion")

    # Deduplicate by prompt
    seen = set()
    deduped = []
    for r in all_rows:
        key = r.get("prompt", "")
        if key and key not in seen:
            seen.add(key)
            deduped.append(r)

    rng.shuffle(deduped)

    out_path = base / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in deduped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(deduped)} unique seed samples (target {args.target})")
    print(f"Written to {out_path}")

    by_source = {}
    for r in deduped:
        src = r.get("source", "?")
        by_source[src] = by_source.get(src, 0) + 1
    for src in sorted(by_source):
        print(f"  {src}: {by_source[src]}")


if __name__ == "__main__":
    main()
