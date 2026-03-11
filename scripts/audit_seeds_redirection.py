#!/usr/bin/env python3
"""Audit seed corpus for redirection rubric compliance.

Redirection rubric: responses should (1) refuse harmful requests and (2) redirect to a
constructive alternative. Seeds that don't fulfill this may need to be pruned or rewritten.

Usage:
  python scripts/audit_seeds_redirection.py [--seeds data/seeds_expanded.jsonl] [--output data/audit_redirection.json]
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]

# Refusal indicators (lowercase)
REFUSAL_PHRASES = [
    "i can't", "i cannot", "i won't", "i'm sorry", "won't provide", "can't assist",
    "cannot help", "can't help", "i'm not able", "i don't provide", "i won't provide",
    "refuse", "decline", "unable to", "not able to", "don't provide",
]

# Redirect indicators (offer alternative)
REDIRECT_PHRASES = [
    "i can", "i can suggest", "i can discuss", "i can help", "i can point",
    "i can explain", "i can offer", "i can provide", "instead", "alternative",
    "however", "you could", "consider", "if you're interested", "if that's helpful",
    "i'd be happy to", "happy to discuss", "can discuss", "can explain",
]


def has_refusal(text: str) -> bool:
    """Check if response contains refusal language."""
    t = text.lower().strip()
    return any(p in t for p in REFUSAL_PHRASES)


def has_redirect(text: str) -> bool:
    """Check if response offers a constructive alternative."""
    t = text.lower().strip()
    return any(p in t for p in REDIRECT_PHRASES)


def audit_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Audit a single seed row. Returns audit result."""
    prompt = (row.get("prompt") or "").strip()
    response = (row.get("response") or "").strip()
    source = row.get("source", "?")

    refusal = has_refusal(response)
    redirect = has_redirect(response)

    # For harmful prompts, we expect refuse + redirect. For benign (e.g. P1 disambiguation), redirect may be the answer itself.
    # Simple heuristic: if it's from principles or harmfulqa with safety_label safe, check both
    compliant = refusal or redirect  # At minimum, one of these
    if refusal and redirect:
        compliant = True
    elif not refusal and not redirect:
        compliant = False
    # refusal-only: might be okay for clear refusals
    # redirect-only: might be benign answer (P1) or evasive (HarmfulQA sometimes)

    return {
        "idx": idx,
        "source": source,
        "prompt_preview": prompt[:80] + "..." if len(prompt) > 80 else prompt,
        "has_refusal": refusal,
        "has_redirect": redirect,
        "compliant": compliant,
        "needs_review": not compliant,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="data/seeds_expanded.jsonl", help="Seed corpus JSONL")
    parser.add_argument("--output", default="data/audit_redirection.json", help="Output audit JSON")
    parser.add_argument("--strict", action="store_true", help="Strict: require BOTH refusal AND redirect")
    args = parser.parse_args()

    path = ROOT / args.seeds
    if not path.exists():
        print(f"Seeds file not found: {path}")
        return

    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    results = []
    for i, row in enumerate(rows):
        r = audit_row(row, i)
        if args.strict:
            r["compliant"] = r["has_refusal"] and r["has_redirect"]
            r["needs_review"] = not r["compliant"]
        results.append(r)

    n_compliant = sum(1 for r in results if r["compliant"])
    n_refusal_only = sum(1 for r in results if r["has_refusal"] and not r["has_redirect"])
    n_redirect_only = sum(1 for r in results if r["has_redirect"] and not r["has_refusal"])
    n_neither = sum(1 for r in results if not r["has_refusal"] and not r["has_redirect"])

    by_source = {}
    for r in results:
        s = r["source"]
        by_source[s] = by_source.get(s, {"total": 0, "compliant": 0})
        by_source[s]["total"] += 1
        if r["compliant"]:
            by_source[s]["compliant"] += 1

    needs_review = [r for r in results if r["needs_review"]]

    out = {
        "summary": {
            "total": len(rows),
            "compliant": n_compliant,
            "needs_review": len(needs_review),
            "refusal_only": n_refusal_only,
            "redirect_only": n_redirect_only,
            "neither": n_neither,
            "by_source": by_source,
        },
        "needs_review_samples": needs_review[:50],
        "all_results": results,
    }

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Audit: {path}")
    print(f"  Total: {len(rows)} | Compliant: {n_compliant} | Needs review: {len(needs_review)}")
    print(f"  Refusal-only: {n_refusal_only} | Redirect-only: {n_redirect_only} | Neither: {n_neither}")
    for src, stats in sorted(by_source.items()):
        print(f"  {src}: {stats['compliant']}/{stats['total']} compliant")
    print(f"  Wrote: {out_path}")


if __name__ == "__main__":
    main()
