import os
import json
import argparse
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from specdecode.simulator import NGramDrafter, GreedyVerifier, PlaybackMetrics, SpeculativePlayback
from specdecode.datasets import get_dataset, REGISTRY

# This script lives in <repo>/scripts/, so anchor experiments/ and configs/ to the
# repo root — the benchmark then works regardless of the current working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_experiment_config(dataset: str) -> dict:
    """Load config from experiments/<dataset>/config.json, fall back to configs/simulator_config.json."""
    experiment_config = os.path.join(PROJECT_ROOT, "experiments", dataset, "config.json")
    if os.path.exists(experiment_config):
        with open(experiment_config, "r", encoding="utf-8") as f:
            return json.load(f)

    fallback = os.path.join(PROJECT_ROOT, "configs", "simulator_config.json")
    if os.path.exists(fallback):
        with open(fallback, "r", encoding="utf-8") as f:
            return json.load(f)

    return {}


def run_benchmark(
    dataset: str, tokenizer_name: str, n_gram_size: int, max_draft: int, sample_index: int
):
    artifacts_dir = os.path.join(PROJECT_ROOT, "experiments", dataset, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    print("=" * 70)
    print(f"Speculative Decoding Simulator — dataset: {dataset}")
    print(f"Tokenizer:   {tokenizer_name}")
    print(
        f"N-gram Size: {n_gram_size}-gram  |  Max Draft: K={max_draft}  |  Sample: #{sample_index}"
    )
    print(f"Artifacts:   {artifacts_dir}")
    print("=" * 70)

    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Warning: could not load {tokenizer_name}, falling back to gpt2: {e}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    corpus_text, target_text = get_dataset(dataset, index=sample_index)

    corpus_tokens = tokenizer.encode(corpus_text)
    target_tokens = tokenizer.encode(target_text)

    print(f"Corpus tokens (drafter knowledge): {len(corpus_tokens)}")
    print(f"Target tokens (generation target): {len(target_tokens)}")
    print("-" * 70)

    draft_sizes = list(range(1, max_draft + 1))
    speedups = []
    avg_accepted = []
    sweep_results = []

    # Baseline — normal token-by-token
    baseline_metrics = PlaybackMetrics()
    SpeculativePlayback(
        tokenizer=tokenizer,
        drafter=None,  # type: ignore
        verifier=GreedyVerifier(),
        metrics=baseline_metrics,
    ).run_playback(target_text, use_drafter=False)
    print(f"Baseline (no drafter): {baseline_metrics.speculative_steps} steps, 1.0x")

    # Sweep K = 1 … max_draft
    for k in draft_sizes:
        metrics = PlaybackMetrics()
        playback = SpeculativePlayback(
            tokenizer=tokenizer,
            drafter=NGramDrafter(corpus_tokens=corpus_tokens, n=n_gram_size, draft_size=k),
            verifier=GreedyVerifier(),
            metrics=metrics,
        )
        reconstructed = playback.run_playback(target_text, use_drafter=True)
        summary = metrics.get_summary()

        total_drafted = summary["accepted_tokens"] + summary["rejected_tokens"]
        acceptance_rate = (
            round(summary["accepted_tokens"] / total_drafted, 4) if total_drafted > 0 else 0.0
        )

        speedups.append(summary["speedup_ratio"])
        avg_accepted.append(summary["average_accepted_per_step"])
        sweep_results.append(
            {
                "k": k,
                "speedup": summary["speedup_ratio"],
                "avg_accepted": summary["average_accepted_per_step"],
                "accepted_tokens": summary["accepted_tokens"],
                "rejected_tokens": summary["rejected_tokens"],
                "steps": summary["speculative_steps"],
                "acceptance_rate": acceptance_rate,
                "drafter_calls": summary["drafter_calls"],
                "drafter_wall_time_ms": summary["drafter_wall_time_ms"],
                "average_drafter_wall_time_ms": summary["average_drafter_wall_time_ms"],
            }
        )

        print(
            f"K={k}: steps={summary['speculative_steps']} | "
            f"avg_accept={summary['average_accepted_per_step']} | "
            f"accept_rate={acceptance_rate:.1%} | "
            f"speedup={summary['speedup_ratio']}x | "
            f"drafter={summary['average_drafter_wall_time_ms']} ms/call"
        )
        assert reconstructed == target_text, f"Reconstruction mismatch at K={k}!"

    # Save raw metrics as JSON for analyze.py
    results_data = {
        "dataset": dataset,
        "sample_index": sample_index,
        "tokenizer": tokenizer_name,
        "n_gram_size": n_gram_size,
        "corpus_token_count": len(corpus_tokens),
        "target_token_count": len(target_tokens),
        "baseline_steps": baseline_metrics.speculative_steps,
        "sweep": sweep_results,
    }
    json_path = os.path.join(artifacts_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"Metrics saved → {json_path}")

    print("-" * 70)
    print("Generating chart...")

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("ggplot")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    primary, secondary = "#3b82f6", "#8b5cf6"

    ax1.plot(
        draft_sizes,
        speedups,
        marker="o",
        color=primary,
        linewidth=2.5,
        markersize=8,
        label="Speedup Ratio",
    )
    ax1.axhline(1.0, color="#ef4444", linestyle="--", alpha=0.7, label="Baseline (1.0x)")
    ax1.set_title("Inference Acceleration (Speedup Ratio)", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("Speculative Draft Size (K)", fontsize=12, labelpad=10)
    ax1.set_ylabel("Speedup Multiplier (x)", fontsize=12, labelpad=10)
    ax1.set_xticks(draft_sizes)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")

    ax2.bar(draft_sizes, avg_accepted, color=secondary, alpha=0.85, edgecolor="#6d28d9", width=0.5)
    ax2.set_title(
        "Average Speculated Tokens Accepted per Step", fontsize=14, fontweight="bold", pad=15
    )
    ax2.set_xlabel("Speculative Draft Size (K)", fontsize=12, labelpad=10)
    ax2.set_ylabel("Average Tokens Accepted", fontsize=12, labelpad=10)
    ax2.set_xticks(draft_sizes)
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")

    plt.suptitle(
        f"Speculative Decoding — {dataset} (sample #{sample_index})\n"
        f"{n_gram_size}-gram drafter · {tokenizer_name}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    chart_path = os.path.join(artifacts_dir, "speedup_benchmark.png")
    plt.savefig(chart_path, dpi=300)
    print(f"Chart saved → {chart_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative Decoding Simulator")
    parser.add_argument(
        "--dataset",
        type=str,
        default="wiki_demo",
        choices=list(REGISTRY.keys()),
        help="Dataset to run (default: wiki_demo)",
    )
    parser.add_argument("--tokenizer", type=str, default=None, help="Override tokenizer name")
    parser.add_argument("--n", type=int, default=None, help="Override n-gram size")
    parser.add_argument("--max_draft", type=int, default=None, help="Override max draft size K")
    parser.add_argument("--index", type=int, default=None, help="Override sample index")

    args = parser.parse_args()

    # Load per-dataset config, then apply CLI overrides
    cfg = load_experiment_config(args.dataset)
    tokenizer_name = args.tokenizer or cfg.get("tokenizer_name", "Qwen/Qwen2.5-0.5B-Instruct")
    n_gram_size = args.n or cfg.get("n_gram_size", 3)
    max_draft = args.max_draft or cfg.get("max_draft", 6)
    sample_index = args.index if args.index is not None else cfg.get("sample_index", 0)

    run_benchmark(args.dataset, tokenizer_name, n_gram_size, max_draft, sample_index)
