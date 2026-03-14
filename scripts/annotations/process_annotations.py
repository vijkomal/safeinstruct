#!/usr/bin/env python3
"""
Process human annotations: inter-annotator agreement + judge calibration.

Direct rating mode: annotators rate each response 1-5. We compute:
- Krippendorff's alpha: inter-annotator agreement on human ratings
- Judge calibration: correlation of human rating with each judge's score (which judge aligns best with humans)

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
    annotation_dir: Path, annotators: list[str], rating_key: str = "your_rating"
) -> dict:
    """Load annotations. Returns {annotator: {id: rating}} where rating is 1-5."""
    import csv
    out = {}

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
                    v = _parse_vote(row.get(rating_key))
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
                    v = _parse_vote(row.get(rating_key, "").strip())
                    if v is not None:
                        votes[vid] = v
        if not jsonl_path.exists() and not csv_path.exists():
            print(f"Warning: no file for {name}, skipping")
        elif votes:
            out[name] = votes
    return out


def krippendorff_alpha_matrix(annotations: dict, example_ids: list) -> float:
    """Compute Krippendorff's alpha on human ratings. reliability_data[coder][unit] = rating or nan."""
    try:
        import krippendorff
    except ImportError:
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


def _fallback_agreement(annotations: dict, example_ids: list) -> float:
    """Simple pairwise agreement when krippendorff not installed."""
    names = list(annotations.keys())
    if len(names) < 2:
        return float("nan")
    pairs = 0
    agreed = 0
    for eid in example_ids:
        vals = []
        for n in names:
            v = annotations[n].get(eid)
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

    annotations = load_annotations(ann_dir, args.annotators)

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
    print("HUMAN ANNOTATION ANALYSIS (Direct Rating)")
    print("=" * 60)

    # Krippendorff on human ratings
    try:
        import krippendorff as _k
        alpha_label = "Krippendorff's alpha"
    except ImportError:
        alpha_label = "Pairwise agreement (install krippendorff for Krippendorff's alpha)"

    alphas = {}
    if len(overlap_with_votes) >= 5:
        alpha = krippendorff_alpha_matrix(annotations, overlap_with_votes)
        alphas["human_ratings"] = alpha
        print(f"\n{alpha_label} (human ratings, n={len(overlap_with_votes)}): {alpha:.3f}")
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

    # Mean rating per annotator
    agg = {}
    print("\n--- Mean rating per annotator (1-5 scale) ---")
    for name, votes in annotations.items():
        vals = [v for v in votes.values() if isinstance(v, (int, float)) and 1 <= v <= 5]
        mean_r = sum(vals) / len(vals) if vals else float("nan")
        agg[name] = {"mean_rating": mean_r, "n": len(vals)}
        print(f"  {name}: mean={mean_r:.2f} (n={len(vals)})")
    all_vals = [v for votes in annotations.values() for v in votes.values() if isinstance(v, (int, float)) and 1 <= v <= 5]
    if all_vals:
        print(f"\nOverall mean rating: {sum(all_vals)/len(all_vals):.2f}")

    # Judge calibration: correlate human rating with each judge's score
    # Note: judge order is shuffled per example, so we use judge_mapping to get model per position
    pool_path = ann_dir / "annotation_pool.jsonl"
    judge_mapping_path = ann_dir / "judge_mapping.json"
    judge_keys = ["judge_1_usefulness", "judge_2_usefulness", "judge_3_usefulness"]
    if pool_path.exists() and judge_mapping_path.exists():
        pool = {}
        with open(pool_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    pool[r["id"]] = r
        judge_mappings = json.loads(judge_mapping_path.read_text())
        # Mean human rating per example (across annotators)
        human_mean = {}
        for eid in overlap_with_votes:
            vals = [annotations[n][eid] for n in annotations if eid in annotations[n] and isinstance(annotations[n][eid], (int, float))]
            if len(vals) >= 1:
                human_mean[eid] = sum(vals) / len(vals)
        # Collect (human, judge_score) per model (judge position varies per example)
        by_model = {}
        for eid in overlap_with_votes:
            if eid not in human_mean or eid not in pool or eid not in judge_mappings:
                continue
            p = pool[eid]
            mapping = judge_mappings[eid]
            for j, key in enumerate(judge_keys):
                model_name = mapping.get(str(j + 1), f"Judge{j+1}")
                jv = p.get(key)
                if jv is not None and isinstance(jv, (int, float)) and 1 <= jv <= 5:
                    if model_name not in by_model:
                        by_model[model_name] = ([], [])
                    by_model[model_name][0].append(human_mean[eid])
                    by_model[model_name][1].append(float(jv))
        print("\n--- Judge calibration (correlation with human rating) ---")
        calibration = {}
        for model_name, (h_vals, j_vals) in by_model.items():
            r = _pearson_corr(h_vals, j_vals)
            calibration[model_name] = {"correlation": r, "n": len(h_vals)}
            print(f"  {model_name}: r={r:.3f} (n={len(h_vals)})")
        agg["judge_calibration"] = calibration

    # Save summary
    summary = {
        "krippendorff_alpha": alphas if alphas else None,
        "overlap_n": len(overlap_with_votes),
        "mean_rating_per_annotator": agg,
    }
    summary_path = ann_dir / "annotation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
