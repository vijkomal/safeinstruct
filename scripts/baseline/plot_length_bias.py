import json
import matplotlib.pyplot as plt
import numpy as np

def load_data(input_path: str):
    data = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def plot(input_path: str = "results/benchmarks/alpaca/alpaca_results.jsonl"):
    data = load_data(input_path)

    lengths = []
    scores = []
    for item in data:
        response = item.get("model_response", "")
        usefulness = item.get("usefulness")
        if usefulness is not None and usefulness != -1:
            lengths.append(len(response.split()))
            scores.append(usefulness)

    lengths = np.array(lengths)
    scores = np.array(scores)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # scatter plot
    axes[0].scatter(lengths, scores, alpha=0.4, edgecolors="none", color="steelblue")
    m, b = np.polyfit(lengths, scores, 1)
    x_line = np.linspace(lengths.min(), lengths.max(), 100)
    axes[0].plot(x_line, m * x_line + b, color="red", linewidth=1.5, label=f"slope={m:.4f}")
    axes[0].set_xlabel("Response Length (words)")
    axes[0].set_ylabel("Usefulness Score (1-5)")
    axes[0].set_title("Response Length vs Usefulness")
    axes[0].legend()

    # box plot per score
    unique_scores = sorted(set(scores))
    grouped = [lengths[scores == s] for s in unique_scores]
    axes[1].boxplot(grouped, labels=unique_scores)
    axes[1].set_xlabel("Usefulness Score")
    axes[1].set_ylabel("Response Length (words)")
    axes[1].set_title("Length Distribution per Score")

    corr = np.corrcoef(lengths, scores)[0, 1]
    fig.suptitle(f"LLM Judge Bias Check  |  Pearson r = {corr:.3f}  |  n={len(scores)}", fontsize=13)
    plt.tight_layout()

    out = input_path.replace(".jsonl", "_bias_check.png")
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}  (r={corr:.3f})")
    plt.show()

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "results/benchmarks/harmbench/harmbench_results.jsonl"
    plot(path)