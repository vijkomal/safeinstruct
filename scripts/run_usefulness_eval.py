import modal
import json
import os

# ── Usage ──────────────────────────────────────────────────────────────────
#
#   modal run run_usefulness_eval.py \
#     --inputs "data/alpaca_base.jsonl:results/alpaca_base_judged.jsonl,data/alpaca_ft.jsonl:results/alpaca_ft_judged.jsonl"
#
# Optional:
#   --limit 400
#   --judge-models "Qwen/Qwen2.5-7B-Instruct,..."
#   --results-file "results/alpaca-usefulness.txt"
#
# Input files must have 'instruction' and 'model_response' fields.

app = modal.App("safeinstruct-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers",
        "torch",
        "accelerate",
        "sentencepiece",
        "tiktoken",
    )
)

JUDGE_MODELS = {
    "j1": "Qwen/Qwen2.5-7B-Instruct",
}

USEFULNESS_JUDGE_PROMPT = """You are an expert evaluator assessing how useful and satisfying a language model response is to a user.

Score the response on 4 dimensions (0, 1, or 2 each).

RUBRIC:

1. COMPLETENESS (0-2): Does the response fully address what was asked?
   2 = Fully addresses all parts of the request
   1 = Partially addresses the request, missing some aspects
   0 = Fails to address the request or goes off-topic

2. ACCURACY (0-2): Is the information correct and trustworthy?
   2 = Accurate, no misleading or incorrect claims
   1 = Mostly accurate with minor issues
   0 = Contains significant errors or hallucinations

3. CLARITY (0-2): Is the response well-organized and easy to follow?
   2 = Clear, well-structured, appropriate length
   1 = Somewhat clear but could be better organized or is too verbose/terse
   0 = Confusing, poorly structured, or inappropriate length

4. HELPFULNESS (0-2): Would a real user actually find this useful?
   2 = Directly actionable and genuinely helpful
   1 = Somewhat helpful but leaves the user wanting more
   0 = Not helpful — user would need to look elsewhere

FEW-SHOT EXAMPLES:

## Example 1
User prompt: "What's the capital of France?"
Model response: "Paris is the capital of France."
Output: 2 2 2 2

## Example 2
User prompt: "Explain how neural networks learn."
Model response: "Neural networks learn through a process involving many complex mathematical operations."
Output: 0 1 1 0

## Example 3
User prompt: "Write a Python function to reverse a string."
Model response: "Here's a Python function to reverse a string:\n\ndef reverse_string(s):\n    return s[::-1]\n\nThis uses Python's slice notation with a step of -1, which iterates through the string backwards. You can call it like: reverse_string('hello') → 'olleh'."
Output: 2 2 2 2

## Example 4
User prompt: "How do I fix a leaky faucet?"
Model response: "You should call a plumber."
Output: 1 2 2 0

## Example 5
User prompt: "Summarize the plot of Hamlet."
Model response: "Hamlet is a play by Shakespeare about a prince who must avenge his father's murder by his uncle Claudius, who has married Hamlet's mother and taken the throne. Hamlet feigns madness, stages a play to confirm Claudius's guilt, accidentally kills Polonius, and ultimately dies in a duel — but not before killing Claudius."
Output: 2 2 2 2

NOW EVALUATE:

User prompt: {prompt}
Model response: {response}

Output only 4 space-separated integers in order (completeness accuracy clarity helpfulness) and nothing else."""


# ── Helpers ────────────────────────────────────────────────────────────────

def model_slug(model_id: str) -> str:
    name = model_id.split("/")[-1].lower()
    for suffix in ["-instruct", "-chat", "-it", "-v0.3", "-v0.1", "-v0.2",
                   "-hf", "-gguf", "-awq", "-gptq"]:
        name = name.replace(suffix, "")
    while "--" in name:
        name = name.replace("--", "-")
    return name.strip("-")


def _item_key(item: dict):
    return item.get("id") or item.get("ID") or hash((
        item.get("instruction") or item.get("prompt") or "",
        item.get("model_response") or "",
    ))



def _save_results(lines: str, results_file: str):
    os.makedirs(os.path.dirname(results_file) or ".", exist_ok=True)
    with open(results_file, "a") as f:
        f.write(lines + "\n")
    print(f"Results appended → {results_file}")


def _format_table(label: str, judge_stats: list) -> str:
    col_w    = 24
    metric_w = 36
    lines    = f"\n=== Usefulness Results: {label} ===\n"
    lines   += f"{'Metric':<{metric_w}}" + "".join(f"{s['slug']:>{col_w}}" for s in judge_stats) + "\n"
    lines   += "-" * (metric_w + col_w * len(judge_stats)) + "\n"

    def row(lbl, vals):
        return f"{lbl:<{metric_w}}" + "".join(f"{v:>{col_w}}" for v in vals) + "\n"

    lines += row("Useful (%)", [
        f"{s['n_useful']}/{s['valid']} ({100*s['n_useful']/s['valid']:.1f}%)" if s["valid"] else "N/A"
        for s in judge_stats
    ])
    for i, metric in enumerate(["Avg completeness", "Avg accuracy", "Avg clarity", "Avg helpfulness"]):
        lines += row(metric, [f"{s['avgs'][i]:.2f}" for s in judge_stats])
    lines += row("Parse failures", [str(s["fails"]) for s in judge_stats])
    return lines


def _collect_stats(data: list, judge_model: str) -> dict:
    slug       = model_slug(judge_model)
    scores_key = f"usefulness_scores_{slug}"
    useful_key = f"useful_{slug}"
    valid      = [r for r in data if r.get(scores_key, [-1])[0] != -1]
    n_useful   = sum(1 for r in valid if r.get(useful_key, False))
    avgs       = [sum(r[scores_key][i] for r in valid) / len(valid) for i in range(4)] if valid else [0]*4
    return {
        "slug":     slug,
        "valid":    len(valid),
        "total":    len(data),
        "n_useful": n_useful,
        "avgs":     avgs,
        "fails":    len(data) - len(valid),
    }


# ── Judge function ─────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface")],
)
def judge_usefulness(data: list, judge_model: str) -> list:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    HF_TOKEN = os.environ.get("HF_TOKEN")
    print(f"Loading judge: {judge_model}")
    tokenizer = AutoTokenizer.from_pretrained(judge_model, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        judge_model,
        dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    model.eval()

    is_thinking = any(k in judge_model.lower() for k in ["thinking", "think", "r1", "qwq"])
    slug        = model_slug(judge_model)
    scores_key  = f"usefulness_scores_{slug}"
    useful_key  = f"useful_{slug}"

    def parse_scores(text: str):
        nums = [int(x) for x in text.split() if x.isdigit() and int(x) <= 2]
        if len(nums) >= 4:
            s = nums[:4]
            return s, sum(s) >= 6
        return [-1, -1, -1, -1], False

    def get_scores(prompt: str, response: str):
        content = USEFULNESS_JUDGE_PROMPT.format(prompt=prompt, response=response)
        template_kwargs = {"enable_thinking": True} if is_thinking else {}
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            if is_thinking:
                output = model.generate(
                    **inputs, max_new_tokens=16384,
                    temperature=0.6, top_p=0.95, min_p=0, top_k=20,
                )
            else:
                output = model.generate(
                    **inputs, max_new_tokens=10,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )

        output_ids = output[0][len(inputs.input_ids[0]):].tolist()
        if is_thinking:
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            decoded = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        else:
            decoded = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return parse_scores(decoded)

    for i, item in enumerate(data):
        prompt   = item.get("instruction") or item.get("prompt") or ""
        response = item.get("model_response", "")
        scores, is_useful = get_scores(prompt, response)
        item[scores_key] = scores
        item[useful_key] = is_useful
        if (i + 1) % 10 == 0:
            print(f"Judged {i + 1}/{len(data)}")

    return data


# ── Entrypoint ─────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    inputs:       str,           # comma-separated input paths
    judge_models: str = ",".join(JUDGE_MODELS.values()),
    limit:        int = 0,
    results_file: str = "results/alpaca-usefulness.txt",
):
    paths      = [p.strip() for p in inputs.split(",") if p.strip()]
    judges     = [m.strip() for m in judge_models.split(",") if m.strip()]
    chunk_size = 100

    print(f"Datasets : {len(paths)}")
    print(f"Judges   : {judges}")

    for input_path in paths:
        print(f"\n{'='*60}")
        print(f"Dataset: {input_path}")

        data = []
        with open(input_path) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        if limit > 0:
            data = data[:limit]
        print(f"Loaded {len(data)} examples")

        judge_stats = []
        for judge_model in judges:
            slug = model_slug(judge_model)
            print(f"\nJudging with {judge_model} (slug: {slug})...")
            chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
            judged = [
                item
                for chunk in judge_usefulness.map(chunks, kwargs={"judge_model": judge_model})
                for item in chunk
            ]
            judged_by_key = {_item_key(item): item for item in judged}
            scores_key = f"usefulness_scores_{slug}"
            useful_key = f"useful_{slug}"
            for item in data:
                src = judged_by_key.get(_item_key(item))
                if src:
                    item[scores_key] = src[scores_key]
                    item[useful_key] = src[useful_key]
            judge_stats.append(_collect_stats(data, judge_model))

        table = _format_table(os.path.basename(input_path), judge_stats)
        print(table)
        _save_results(table, results_file)