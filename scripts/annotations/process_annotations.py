#!/usr/bin/env python3
"""
Process human annotations: Krippendorff's alpha, judge agreement, weighted scoring.

- Krippendorff's alpha: inter-annotator agreement on overlap set
- Per-annotator agreement with judge: used as weights
- Weighted usefulness: when aggregating, weight by judge agreement
  (annotators who agree more with judge get higher weight when we use judge scores)

Usage:
  pip install krippendorff
  python scripts/annotations/process_annotations.py --annotation-dir data/annotations
"""

import json
import argparse
from pathlib import Path


def load_annotations(annotation_dir: Path, annotators: list[str]) -> dict:
    """Load annotations from annotator_<name>.jsonl or .csv. Returns {annotator: {id: vote}}."""
    import csv
    out = {}
    for name in annotators:
        votes = {}
        # Try JSONL first
        jsonl_path = annotation_dir / f"annotator_{name}.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    vid = row.get("id")
                    v = row.get("your_vote")
                    if vid and v is not None and v != "":
                        try:
                            votes[vid] = int(v)
                        except (ValueError, TypeError):
                            pass
        # Fallback: CSV (if JSONL has no votes or doesn't exist)
        csv_path = annotation_dir / f"annotator_{name}.csv"
        if not votes and csv_path.exists():
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vid = row.get("id", "").strip('"')
                    v = row.get("your_vote", "").strip()
                    if vid and v:
                        try:
                            votes[vid] = int(v)
                        except (ValueError, TypeError):
                            pass
        if not jsonl_path.exists() and not csv_path.exists():
            print(f"Warning: no file for {name}, skipping")
        elif votes:
            out[name] = votes
    return out


def load_pool(annotation_dir: Path) -> dict:
    """Load annotation pool for judge scores. Returns {id: judge_usefulness}."""
    path = annotation_dir / "annotation_pool.jsonl"
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            vid = row.get("id")
            j = row.get("judge_usefulness")
            if vid and j is not None and 1 <= j <= 5:
                out[vid] = int(j)
    return out


def krippendorff_alpha_matrix(annotations: dict, example_ids: list) -> float:
    """
    Compute Krippendorff's alpha on overlap examples.
    reliability_data[coder_idx][unit_idx] = rating or nan
    """
    try:
        import krippendorff
    except ImportError:
        # Fallback: pairwise percent agreement (not alpha, but indicative)
        return _fallback_agreement(annotations, example_ids)

    annotator_names = list(annotations.keys())
    n_coders = len(annotator_names)
    n_units = len(example_ids)
    id_to_col = {eid: i for i, eid in enumerate(example_ids)}

    mat = [[float("nan")] * n_units for _ in range(n_coders)]
    for c, name in enumerate(annotator_names):
        votes = annotations[name]
        for eid in example_ids:
            if eid in votes:
                mat[c][id_to_col[eid]] = float(votes[eid])

    alpha = krippendorff.alpha(
        reliability_data=mat,
        level_of_measurement="ordinal",
    )
    return alpha


def _fallback_agreement(annotations: dict, example_ids: list) -> float:
    """Simple pairwise agreement when krippendorff not installed."""
    names = list(annotations.keys())
    if len(names) < 2:
        return float("nan")
    pairs = 0
    agreed = 0
    for eid in example_ids:
        vals = [annotations[n].get(eid) for n in names]
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            continue
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                pairs += 1
                if vals[i] == vals[j]:
                    agreed += 1
    return agreed / pairs if pairs > 0 else float("nan")


def agreement_with_judge(annotations: dict, judge_scores: dict) -> dict[str, float]:
    """Per-annotator agreement with judge: % exact match (and optionally within ±1)."""
    out = {}
    for name, votes in annotations.items():
        matches_exact = 0
        matches_near = 0
        n = 0
        for eid, human_v in votes.items():
            judge_v = judge_scores.get(eid)
            if judge_v is None or judge_v < 1 or judge_v > 5:
                continue
            n += 1
            if human_v == judge_v:
                matches_exact += 1
            if abs(human_v - judge_v) <= 1:
                matches_near += 1
        if n > 0:
            out[name] = {
                "exact": matches_exact / n,
                "within_1": matches_near / n,
                "n": n,
            }
        else:
            out[name] = {"exact": 0.0, "within_1": 0.0, "n": 0}
    return out


def weighted_human_score_per_example(
    annotations: dict, annotator_agreement: dict, example_ids: list
) -> dict[str, float]:
    """
    For each example, compute weighted average of human votes, weighted by
    annotator–judge agreement. Annotators who agree more with the judge get higher weight.
    """
    out = {}
    for eid in example_ids:
        votes = []
        weights = []
        for name, avotes in annotations.items():
            if eid not in avotes:
                continue
            agg = annotator_agreement.get(name, {})
            w = agg.get("exact", 0.5)  # fallback 0.5 if missing
            votes.append(avotes[eid])
            weights.append(w)
        if votes:
            total_w = sum(weights)
            if total_w > 0:
                out[eid] = sum(v * w for v, w in zip(votes, weights)) / total_w
            else:
                out[eid] = sum(votes) / len(votes)
    return out


def weighted_judge_score(judge_scores: dict, annotator_agreement: dict) -> dict:
    """
    Compute calibration metrics: mean annotator–judge agreement.
    Use when reporting judge-weighted usefulness in the paper.
    """
    if not annotator_agreement:
        return {}

    mean_exact = sum(a["exact"] for a in annotator_agreement.values()) / len(annotator_agreement)
    return {
        "mean_annotator_judge_agreement": mean_exact,
        "calibration_weight": mean_exact,
    }


def main():
    parser = argparse.ArgumentParser(description="Process human annotations")
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
        help="Optional: path to file with overlap example IDs (one per line). If not set, inferred from first annotator's in_overlap=True.",
    )
    parser.add_argument(
        "--agreement-mode",
        action="store_true",
        help="Annotators rated agreement (1-5) with judge, not usefulness. Report mean agreement instead of agreement-with-judge.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    ann_dir = Path(args.annotation_dir)
    if not ann_dir.is_absolute():
        ann_dir = root / ann_dir

    if not ann_dir.exists():
        print(f"Annotation dir not found: {ann_dir}")
        return

    annotations = load_annotations(ann_dir, args.annotators)
    judge_scores = load_pool(ann_dir)

    if not annotations:
        print("No annotations found.")
        return

    # Infer overlap IDs from first annotator file (examples with in_overlap=True)
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
    overlap_ids = list(dict.fromkeys(overlap_ids))  # dedup preserve order

    if args.overlap_ids:
        overlap_path = Path(args.overlap_ids)
        if not overlap_path.is_absolute():
            overlap_path = root / overlap_path
        if overlap_path.exists():
            overlap_ids = [x.strip() for x in open(overlap_path) if x.strip()]

    # Filter to examples that have at least 2 annotations (for Krippendorff)
    overlap_with_votes = [
        eid for eid in overlap_ids
        if sum(1 for a in annotations.values() if eid in a) >= 2
    ]

    print("=" * 60)
    print("HUMAN ANNOTATION ANALYSIS")
    print("=" * 60)

    # Krippendorff (or fallback pairwise agreement)
    try:
        import krippendorff as _k
        alpha_label = "Krippendorff's alpha"
    except ImportError:
        alpha_label = "Pairwise agreement (install krippendorff for Krippendorff's alpha)"
    if len(overlap_with_votes) >= 5:
        alpha = krippendorff_alpha_matrix(annotations, overlap_with_votes)
        print(f"\n{alpha_label} (ordinal, overlap n={len(overlap_with_votes)}): {alpha:.3f}")
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
            print("  (interpretation applies to Krippendorff's alpha; pairwise % differs)")
    else:
        print(f"\nKrippendorff: skipped (need ≥5 overlap examples with ≥2 votes, got {len(overlap_with_votes)})")

    agg = None
    cal = None
    weighted_scores = {}

    if args.agreement_mode:
        # Agreement mode: annotators rated 1-5 how much they agree with judge
        print("\n--- Mean agreement score per annotator (1-5 scale) ---")
        agg = {}
        for name, votes in annotations.items():
            vals = [v for v in votes.values() if 1 <= v <= 5]
            mean_agr = sum(vals) / len(vals) if vals else float("nan")
            agg[name] = {"mean_agreement": mean_agr, "n": len(vals)}
            print(f"  {name}: mean={mean_agr:.2f} (n={len(vals)})")
        all_vals = [v for votes in annotations.values() for v in votes.values() if 1 <= v <= 5]
        if all_vals:
            print(f"\nOverall mean agreement: {sum(all_vals)/len(all_vals):.2f}")
    else:
        # Usefulness mode: compare annotator votes to judge
        print("\n--- Per-annotator agreement with judge ---")
        agg = agreement_with_judge(annotations, judge_scores)
        for name, stats in agg.items():
            print(f"  {name}: exact={stats['exact']:.2%}, within±1={stats['within_1']:.2%} (n={stats['n']})")

        # Weighted calibration
        cal = weighted_judge_score(judge_scores, agg)
        if cal and "mean_annotator_judge_agreement" in cal:
            print(f"\nMean annotator–judge agreement: {cal['mean_annotator_judge_agreement']:.2%}")
            print("Use this as calibration weight when reporting judge-weighted usefulness.")

        # Weighted human scores (for annotated examples)
        all_annotated_ids = set()
        for v in annotations.values():
            all_annotated_ids.update(v.keys())
        weighted_scores = weighted_human_score_per_example(annotations, agg, list(all_annotated_ids))
        if weighted_scores:
            avg_weighted = sum(weighted_scores.values()) / len(weighted_scores)
            print(f"\nWeighted human score (avg over {len(weighted_scores)} annotated): {avg_weighted:.2f}")

    # Save summary
    summary = {
        "krippendorff_alpha": krippendorff_alpha_matrix(annotations, overlap_with_votes) if len(overlap_with_votes) >= 5 else None,
        "overlap_n": len(overlap_with_votes),
        "agreement_mode": args.agreement_mode,
        "annotator_judge_agreement": agg,
        "calibration": cal,
        "weighted_human_scores": {k: round(v, 2) for k, v in weighted_scores.items()} if weighted_scores else {},
    }
    summary_path = ann_dir / "annotation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
