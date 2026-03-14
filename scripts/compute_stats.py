import json
import os
import glob

# Standard schema (Gemma, TinyLlama): rubric_scores, rubric_scores_3, rubric_scores_4
JUDGES_STANDARD = [
    ("rubric_scores",   "successful_redirection",   "Qwen2.5-7B  (J1)", "rubric_scores"),
    ("rubric_scores_3", "successful_redirection_3", "Mistral-7B  (J3)", "rubric_scores_3"),
    ("rubric_scores_4", "successful_redirection_4", "Qwen2.5-14B (J4)", "rubric_scores_4"),
]
# GPT-Neo uses model-specific keys; map to standard for aggregate
JUDGES_GPTNEO = [
    ("rubric_scores_qwen2.5-7b",   "successful_redirection_qwen2.5-7b",   "Qwen2.5-7B  (J1)", "rubric_scores"),
    ("rubric_scores_mistral-7b",   "successful_redirection_mistral-7b",   "Mistral-7B  (J3)", "rubric_scores_3"),
    ("rubric_scores_qwen2.5-14b", "successful_redirection_qwen2.5-14b", "Qwen2.5-14B (J4)", "rubric_scores_4"),
]

INPUT_PATHS = [
    "results/baseline/gemma-2-2b-it/harmbench_baseline_gemma.jsonl",
    "results/baseline/gpt-neo/harmbench_baseline_gpt-neo.jsonl",
    "results/baseline/tiny-llama-1.1b/harmbench_baseline_tinyllama_1.1B.jsonl",
]

METRIC_LABELS = ["Redirected %", "N redirected", "Avg safety", "Avg pivot", "Avg acknowledgment", "Avg tone", "Parse failures"]

def compute_stats(data, scores_key, redir_key):
    valid = [r for r in data if r.get(scores_key, [-1])[0] != -1]
    if not valid:
        return None
    n_r   = sum(1 for r in valid if r.get(redir_key))
    pct   = 100 * n_r / len(valid)
    avgs  = [sum(r[scores_key][i] for r in valid) / len(valid) for i in range(4)]
    fails = len(data) - len(valid)
    return {
        "redirected_pct": pct,
        "n_redirected":   n_r,
        "n_valid":        len(valid),
        "avg_safety":     avgs[0],
        "avg_pivot":      avgs[1],
        "avg_acknowledgment": avgs[2],
        "avg_tone":       avgs[3],
        "parse_failures": fails,
    }

def print_table(title, rows_by_judge):
    col_w = 22
    print(f"\n=== Results: {title} ===")

    # header
    header = f"{'Metric':<30}" + "".join(f"{lbl:>{col_w}}" for lbl, _ in rows_by_judge)
    print(header)
    print("-" * len(header))

    metrics = [
        ("Redirected (%)",     lambda s: f"{s['n_redirected']}/{s['n_valid']} ({s['redirected_pct']:.1f}%)" if s else "n/a"),
        ("Avg safety",         lambda s: f"{s['avg_safety']:.2f}"         if s else "n/a"),
        ("Avg pivot",          lambda s: f"{s['avg_pivot']:.2f}"          if s else "n/a"),
        ("Avg acknowledgment", lambda s: f"{s['avg_acknowledgment']:.2f}" if s else "n/a"),
        ("Avg tone",           lambda s: f"{s['avg_tone']:.2f}"           if s else "n/a"),
        ("Parse failures",     lambda s: str(s['parse_failures'])         if s else "n/a"),
    ]

    for label, fn in metrics:
        line = f"{label:<30}"
        for _, s in rows_by_judge:
            val = fn(s)
            line += f"{val:>{col_w}}"
        print(line)

# ── Per-file tables ────────────────────────────────────────────────────────
all_data_by_judge = {sk: [] for sk, _, _, _ in JUDGES_STANDARD}

for path in INPUT_PATHS:
    if not os.path.exists(path):
        print(f"WARNING: file not found — {path}")
        continue

    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # GPT-Neo uses different schema
    judges = JUDGES_GPTNEO if "gpt-neo" in path else JUDGES_STANDARD

    rows = []
    for judge_tuple in judges:
        scores_key = judge_tuple[0]
        redir_key = judge_tuple[1]
        label = judge_tuple[2]
        agg_key = judge_tuple[3] if len(judge_tuple) > 3 else scores_key
        s = compute_stats(data, scores_key, redir_key)
        rows.append((label, s))
        if s and agg_key in all_data_by_judge:
            valid_rows = [r for r in data if r.get(scores_key, [-1])[0] != -1]
            # Normalize GPT-Neo rows to standard keys for aggregate
            if "gpt-neo" in path:
                std_scores = {"rubric_scores": "rubric_scores", "rubric_scores_3": "rubric_scores_3", "rubric_scores_4": "rubric_scores_4"}
                std_redir = {"rubric_scores": "successful_redirection", "rubric_scores_3": "successful_redirection_3", "rubric_scores_4": "successful_redirection_4"}
                valid_rows = [
                    {**r, std_scores[agg_key]: r[scores_key], std_redir[agg_key]: r[redir_key]}
                    for r in valid_rows
                ]
            all_data_by_judge[agg_key].extend(valid_rows)

    print_table(os.path.basename(path), rows)

# ── Overall aggregate table ────────────────────────────────────────────────
overall_rows = []
for scores_key, redir_key, label, _ in JUDGES_STANDARD:
    combined = all_data_by_judge[scores_key]
    if not combined:
        overall_rows.append((label, None))
        continue
    n_r  = sum(1 for r in combined if r.get(redir_key))
    pct  = 100 * n_r / len(combined)
    avgs = [sum(r[scores_key][i] for r in combined) / len(combined) for i in range(4)]
    overall_rows.append((label, {
        "redirected_pct": pct,
        "n_redirected":   n_r,
        "n_valid":        len(combined),
        "avg_safety":     avgs[0],
        "avg_pivot":      avgs[1],
        "avg_acknowledgment": avgs[2],
        "avg_tone":       avgs[3],
        "parse_failures": 0,
    }))

print_table("OVERALL (all files combined)", overall_rows)
print()