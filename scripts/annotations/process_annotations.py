#!/usr/bin/env python3
"""
Process human annotations: 3-judge agreement mode.

Annotators rate agreement (1-5) with each of 3 judges. We compute:
- Krippendorff's alpha: inter-annotator agreement per judge
- Mean agreement per annotator and per LLM judge

Usage:
  pip install krippendorff
  python scripts/annotations/process_annotations.py --annotation-dir data/annotations
"""

import json
import argparse
from pathlib import Path


def _parse_vote(v) -> int | None:
    if v is None or v == "":
        return None
    try:
        x = int(v) if isinstance(v, str) else int(v)
        return x if 1 <= x <= 5 else None
    except (ValueError, TypeError):
        return None


def load_annotations(
    annotation_dir: Path, annotators: list[str], three_judges: bool = True
) -> dict:
    """Load annotations. Returns {annotator: {id: (v1,v2,v3)}} for 3-judge mode."""
    import csv
    out = {}
    vote_keys = ["your_vote_1", "your_vote_2", "your_vote_3"] if three_judges else ["your_rating"]

    for name in annotators:
        votes = {}
        jsonl_path = annotation_dir / f"annotator_{name}.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    vid = row.get("id")
                    if not vid:
                        continue
                    if three_judges:
                        vals = tuple(_parse_vote(row.get(k)) for k in vote_keys)
                        if any(v is not None for v in vals):
                            votes[vid] = vals
                    else:
                        v = _parse_vote(row.get("your_rating"))
                        if v is not None:
                            votes[vid] = v
        csv_path = annotation_dir / f"annotator_{name}.csv"
        if not votes and csv_path.exists():
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vid = row.get("id", "").strip('"')
                    if not vid:
                        continue
                    if three_judges:
                        vals = tuple(_parse_vote(row.get(k, "").strip()) for k in vote_keys)
                        if any(v is not None for v in vals):
                            votes[vid] = vals
                    else:
                        v = _parse_vote(row.get("your_rating", "").strip())
                        if v is not None:
                            votes[vid] = v
        if not jsonl_path.exists() and not csv_path.exists():
            print(f"Warning: no file for {name}, skipping")
        elif votes:
            out[name] = votes
    return out


def _extract_votes_for_judge(annotations: dict, example_ids: list, judge_idx: int) -> dict:
    """Extract single vote per annotator per example for a given judge (0, 1, or 2)."""
    out = {}
    for name, votes in annotations.items():
        out[name] = {}
        for eid in example_ids:
            v = votes.get(eid)
            if v is not None and isinstance(v, tuple) and judge_idx < len(v) and v[judge_idx] is not None:
                out[name][eid] = v[judge_idx]
            elif v is not None and not isinstance(v, tuple) and judge_idx == 0:
                out[name][eid] = v
        if out[name]:
            out[name] = {k: v for k, v in out[name].items() if v is not None}
    return {n: v for n, v in out.items() if v}


def krippendorff_alpha_matrix(annotations: dict, example_ids: list, judge_idx: int | None = None) -> float:
    """Compute Krippendorff's alpha. For 3-judge mode, pass judge_idx 0/1/2."""
    try:
        import krippendorff
    except ImportError:
        return _fallback_agreement(annotations, example_ids, judge_idx)

    ann = _extract_votes_for_judge(annotations, example_ids, judge_idx) if judge_idx is not None else annotations
    annotator_names = list(ann.keys())
    n_coders = len(annotator_names)
    n_units = len(example_ids)
    id_to_col = {eid: i for i, eid in enumerate(example_ids)}

    mat = [[float("nan")] * n_units for _ in range(n_coders)]
    for c, name in enumerate(annotator_names):
        votes = ann[name]
        for eid in example_ids:
            if eid in votes:
                v = votes[eid]
                if isinstance(v, (int, float)):
                    mat[c][id_to_col[eid]] = float(v)
                elif isinstance(v, tuple) and len(v) > 0 and v[0] is not None:
                    mat[c][id_to_col[eid]] = float(v[0])

    alpha = krippendorff.alpha(
        reliability_data=mat,
        level_of_measurement="ordinal",
    )
    return alpha


def _fallback_agreement(annotations: dict, example_ids: list, judge_idx: int | None = None) -> float:
    """Simple pairwise agreement when krippendorff not installed."""
    ann = _extract_votes_for_judge(annotations, example_ids, judge_idx) if judge_idx is not None else annotations
    names = list(ann.keys())
    if len(names) < 2:
        return float("nan")
    pairs = 0
    agreed = 0
    for eid in example_ids:
        vals = []
        for n in names:
            v = ann[n].get(eid)
            if v is not None:
                vals.append(v if isinstance(v, (int, float)) else v[0] if isinstance(v, tuple) else None)
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            continue
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                pairs += 1
                if vals[i] == vals[j]:
                    agreed += 1
    return agreed / pairs if pairs > 0 else float("nan")


def _pearson_corr(x: list[float], y: list[float]) -> float:
    """Pearson correlation. Returns nan if insufficient data."""
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def main():
    parser = argparse.ArgumentParser(description="Process human annotations (direct rating + judge calibration)")
    parser.add_argument(
        "--annotation-dir",
        default="data/annotations",
        help="Directory with annotation files",
    )
    parser.add_argument(
        "--annotators",
        nargs="+",
        default=["tim", "komal", "j"],
        help="Annotator names",
    )
    parser.add_argument(
        "--overlap-ids",
        help="Optional: path to file with overlap example IDs (one per line).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    ann_dir = Path(args.annotation_dir)
    if not ann_dir.is_absolute():
        ann_dir = root / ann_dir

    if not ann_dir.exists():
        print(f"Annotation dir not found: {ann_dir}")
        return

    annotations = load_annotations(ann_dir, args.annotators, three_judges=True)

    if not annotations:
        print("No annotations found.")
        return

    # Infer overlap IDs from first annotator file
    overlap_ids = []
    for name in args.annotators:
        path = ann_dir / f"annotator_{name}.jsonl"
        if path.exists():
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("in_overlap"):
                        overlap_ids.append(row["id"])
            break
    overlap_ids = list(dict.fromkeys(overlap_ids))

    if args.overlap_ids:
        overlap_path = Path(args.overlap_ids)
        if not overlap_path.is_absolute():
            overlap_path = root / overlap_path
        if overlap_path.exists():
            overlap_ids = [x.strip() for x in open(overlap_path) if x.strip()]

    overlap_with_votes = [
        eid for eid in overlap_ids
        if sum(1 for a in annotations.values() if eid in a) >= 2
    ]

    print("=" * 60)
    print("HUMAN ANNOTATION ANALYSIS (3-Judge Agreement)")
    print("=" * 60)

    # Krippendorff per judge
    try:
        import krippendorff as _k
        alpha_label = "Krippendorff's alpha"
    except ImportError:
        alpha_label = "Pairwise agreement (install krippendorff for Krippendorff's alpha)"

    alphas = {}
    if len(overlap_with_votes) >= 5:
        for j in range(3):
            alpha = krippendorff_alpha_matrix(annotations, overlap_with_votes, judge_idx=j)
            alphas[f"Judge_{j+1}"] = alpha
            print(f"\n{alpha_label} (Judge {j+1}, n={len(overlap_with_votes)}): {alpha:.3f}")
            try:
                import krippendorff
                if alpha < 0.2:
                    print("  → Slight agreement")
                elif alpha < 0.4:
                    print("  → Fair agreement")
                elif alpha < 0.6:
                    print("  → Moderate agreement")
                elif alpha < 0.8:
                    print("  → Substantial agreement")
                else:
                    print("  → Near-perfect agreement")
            except ImportError:
                pass
    else:
        print(f"\nKrippendorff: skipped (need ≥5 overlap examples with ≥2 votes, got {len(overlap_with_votes)})")

    # Mean agreement per annotator
    agg = {}
    print("\n--- Mean agreement per annotator (1-5 scale) ---")
    for name, votes in annotations.items():
        vals = []
        for v in votes.values():
            if isinstance(v, tuple):
                vals.extend(x for x in v if x is not None and 1 <= x <= 5)
            elif isinstance(v, (int, float)) and 1 <= v <= 5:
                vals.append(v)
        mean_agr = sum(vals) / len(vals) if vals else float("nan")
        agg[name] = {"mean_agreement": mean_agr, "n": len(vals)}
        print(f"  {name}: mean={mean_agr:.2f} (n={len(vals)})")
    all_vals = []
    for votes in annotations.values():
        for v in votes.values():
            if isinstance(v, tuple):
                all_vals.extend(x for x in v if x is not None and 1 <= x <= 5)
            elif isinstance(v, (int, float)) and 1 <= v <= 5:
                all_vals.append(v)
    if all_vals:
        print(f"\nOverall mean agreement: {sum(all_vals)/len(all_vals):.2f}")

    # Mean agreement per judge position (Judge 1/2/3; identity varies per example)
    all_example_ids = list(set().union(*(set(a.keys()) for a in annotations.values())))
    print("\n--- Mean agreement per judge position ---")
    for j in range(3):
        ann_j = _extract_votes_for_judge(annotations, all_example_ids, j)
        vals = []
        for v in ann_j.values():
            vals.extend(x for x in v.values() if x is not None and 1 <= x <= 5)
        mean_j = sum(vals) / len(vals) if vals else float("nan")
        agg[f"judge_{j+1}"] = {"mean_agreement": mean_j, "n": len(vals)}
        print(f"  Judge {j+1}: mean={mean_j:.2f} (n={len(vals)})")

    # Save summary
    summary = {
        "krippendorff_alpha": alphas if alphas else None,
        "overlap_n": len(overlap_with_votes),
        "mean_agreement_per_annotator": agg,
    }
    summary_path = ann_dir / "annotation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
