"""Figures for the RQ2 factorial.

Four claims, one figure each:

1. the 2x2 itself -- the two factors are separable, and their lines are parallel
2. why the efficiency factor exists -- retrieval cost scales with the datastore
3. what it costs -- speedup by datastore size, including the cell that drops below 1
4. where the budget optimum sits -- accepted tokens rise, verification cost rises faster

    python scripts/plot_rq2.py
"""

import argparse
import json
import math
import os
import textwrap
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Validated categorical slots 1-4 (see dataviz reference palette). Three of these
# sit below 3:1 on a light surface, so every series carries a direct label.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE, GRID = "#fcfcfb", "#e6e5e0"

LINE_W, MARKER = 2.0, 6.5


def style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, fontweight="bold", loc="left", pad=12)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=0)


def label_end(ax: plt.Axes, x: float, y: float, text: str, color: str, dx: float = 6) -> None:
    """Direct label at the series end. Identity is never colour-alone."""
    ax.annotate(
        text, xy=(x, y), xytext=(dx, 0), textcoords="offset points",
        color=color, fontsize=8.5, fontweight="bold", va="center",
    )


def legend(ax: plt.Axes) -> None:
    """A legend accompanies the direct labels; identity never rests on colour alone."""
    leg = ax.legend(loc="best", frameon=True, fontsize=8.5, labelcolor=INK_2,
                    facecolor=SURFACE, edgecolor=GRID, borderpad=0.6)
    leg.get_frame().set_linewidth(0.8)


def caption(fig: plt.Figure, text: str, width: int = 128) -> None:
    fig.text(0.012, 0.012, textwrap.fill(text, width), color=INK_2, fontsize=8.5,
             ha="left", va="bottom", linespacing=1.5)


def cell(cells: Dict, dataset: str, variant: str, store: int, budget: int) -> Dict:
    return cells[f"{dataset}|{variant}|{store}|B{budget}|S1"]


# --------------------------------------------------------------------------- #

def fig_interaction(cells: Dict, stores: Sequence[int], budget: int, out: str) -> None:
    """Slope chart: does moving one factor change what the other factor buys?"""
    fig, axes = plt.subplots(1, len(stores), figsize=(5.6 * len(stores), 5.0), sharey=False)
    fig.patch.set_facecolor(SURFACE)
    axes = [axes] if len(stores) == 1 else list(axes)

    for ax, store in zip(axes, stores):
        xs = [0, 1]
        for variant, colour, name in (
            ("self_relevant", BLUE, "high-spec"),
            ("pooled", ORANGE, "low-spec"),
        ):
            arms = cell(cells, "cnn", variant, store, budget)["arms"]
            ys = [
                arms["torch_scan"]["speedup_excluding_build"],
                arms["torch_index"]["speedup_excluding_build"],
            ]
            ax.plot(xs, ys, color=colour, linewidth=LINE_W, marker="o",
                    markersize=MARKER, zorder=3, label=name)
            label_end(ax, xs[-1], ys[-1], f"  {name}\n  {ys[-1]:.3f}x", colour)
            ax.annotate(f"{ys[0]:.3f}x", xy=(xs[0], ys[0]), xytext=(-8, 0),
                        textcoords="offset points", color=colour, fontsize=8.5,
                        fontweight="bold", ha="right", va="center")

        ax.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate("no speedup", xy=(0.02, 1.0), xytext=(0, 4),
                    textcoords="offset points", color=INK_MUTED, fontsize=7.5)
        ax.set_xlim(-0.45, 1.95)
        ax.set_xticks(xs)
        ax.set_xticklabels(["on-the-fly", "precomputed"])
        # Speedup is a ratio, so "no interaction" means equal *ratios*, which reads
        # as parallel only on a log axis. On a linear axis equal ratios look like
        # diverging lines and the figure would contradict its own caption.
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:.2f}x"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        legend(ax)
        style(ax, f"datastore {store:,} tokens", "", "modeled speedup (log)")

    fig.suptitle(
        "Both factors move speedup, and the lines stay parallel",
        color=INK, fontsize=13, fontweight="bold", x=0.012, ha="left", y=0.99,
    )
    caption(fig, "CNN/DailyMail, budget B=4, depth drafting. Log y-axis, so parallel "
        "means equal ratios. Precomputing buys 1.147x at 1M tokens for both workloads, "
        "and the workload factor buys ~1.17x under either retrieval strategy -- the two "
        "factors compose without interaction, which is what the cost model predicts.")
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_latency(cells: Dict, stores: Sequence[int], budget: int, out: str) -> None:
    """Why an efficiency factor exists at all: one arm scales, the other does not."""
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    fig.patch.set_facecolor(SURFACE)

    series: List[Tuple[str, str, str]] = [
        ("torch_scan", BLUE, "on-the-fly (tensor scan)"),
        ("python_scan", ORANGE, "on-the-fly (python scan)"),
        ("torch_index", AQUA, "precomputed (tensor index)"),
        ("python_index", YELLOW, "precomputed (dict index)"),
    ]
    # p95, not median: the python scan returns on its first match, so a larger
    # datastore can find one *sooner* and its median falls -- which reads as "bigger
    # is cheaper" and is exactly backwards. The tail, where no match exists and the
    # scan runs to the end of the corpus, is what actually grows with the datastore.
    for arm, colour, name in series:
        ys = [
            cell(cells, "cnn", "self_relevant", s, budget)["arms"][arm]["latency"][
                "p95_ns"
            ]
            / 1e3
            for s in stores
        ]
        ax.plot(stores, ys, color=colour, linewidth=LINE_W, marker="o",
                markersize=MARKER, zorder=3, label=name)
        label_end(ax, stores[-1], ys[-1], f"  {name}\n  {ys[-1]:,.0f} us", colour)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(stores[0] * 0.8, stores[-1] * 26)
    ax.set_xticks(list(stores))
    ax.set_xticklabels([f"{s // 1000}K" if s < 1_000_000 else "1M" for s in stores])
    style(
        ax,
        "Worst-case retrieval cost, by datastore size",
        "datastore tokens",
        "p95 latency per generate_draft (us, log)",
    )
    legend(ax)
    caption(fig, "p95, not median. The python scan returns on its first match, so a "
        "larger datastore often finds one sooner and its median falls -- reading as "
        "'bigger is cheaper', which is backwards. In the tail no match exists and the "
        "scan runs to the end of the corpus, so every arm grows. The tensor index "
        "grows too (24 to 348 us): buckets are uncapped here for width-safety, so a "
        "frequent gram is a long position list. The index still wins at every size, "
        "but by 8-17x on the tail, not the 206x the median suggests.")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_speedup_by_store(cells: Dict, stores: Sequence[int], budget: int, out: str) -> None:
    """The consequence: on a large datastore, scanning erases the whole benefit."""
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    fig.patch.set_facecolor(SURFACE)

    combos = [
        ("self_relevant", "torch_index", BLUE, "high-spec + precomputed"),
        ("self_relevant", "torch_scan", ORANGE, "high-spec + on-the-fly"),
        ("pooled", "torch_index", AQUA, "low-spec + precomputed"),
        ("pooled", "torch_scan", YELLOW, "low-spec + on-the-fly"),
    ]
    for variant, arm, colour, name in combos:
        ys = [
            cell(cells, "cnn", variant, s, budget)["arms"][arm]["speedup_excluding_build"]
            for s in stores
        ]
        ax.plot(stores, ys, color=colour, linewidth=LINE_W, marker="o",
                markersize=MARKER, zorder=3, label=name)
        label_end(ax, stores[-1], ys[-1], f"  {name}\n  {ys[-1]:.3f}x", colour)

    ax.axhline(1.0, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("break-even: slower than plain decoding below this line",
                xy=(stores[0], 1.0), xytext=(2, -12), textcoords="offset points",
                color=INK_MUTED, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlim(stores[0] * 0.85, stores[-1] * 24)
    ax.set_xticks(list(stores))
    ax.set_xticklabels([f"{s // 1000}K" if s < 1_000_000 else "1M" for s in stores])
    style(ax, "Speedup by datastore size, all four cells", "datastore tokens",
          "modeled speedup")
    legend(ax)
    caption(fig, "A bigger datastore raises acceptance but also raises retrieval cost. "
        "For the low-speculatability workload the scanning arm has already crossed "
        "below 1.0 by 500K tokens and keeps falling: speculating costs more than it saves.")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_budget(cells: Dict, store: int, budgets: Sequence[int], out: str) -> None:
    """Longer drafts accept more tokens but cost more to verify; the peak is the design point."""
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    fig.patch.set_facecolor(SURFACE)

    for variant, colour, name in (
        ("self_relevant", BLUE, "high-spec"),
        ("pooled", ORANGE, "low-spec"),
    ):
        ys = [
            cell(cells, "cnn", variant, store, b)["arms"]["torch_index"][
                "speedup_excluding_build"
            ]
            for b in budgets
        ]
        ax.plot(budgets, ys, color=colour, linewidth=LINE_W, marker="o",
                markersize=MARKER, zorder=3, label=name)
        peak = max(range(len(ys)), key=lambda i: ys[i])
        ax.plot([budgets[peak]], [ys[peak]], marker="o", markersize=MARKER + 5,
                markerfacecolor="none", markeredgecolor=colour, markeredgewidth=2,
                zorder=4)
        label_end(ax, budgets[-1], ys[-1], f"  {name}", colour)
        ax.annotate(f"peak B={budgets[peak]}", xy=(budgets[peak], ys[peak]),
                    xytext=(0, 12), textcoords="offset points", color=colour,
                    fontsize=8, fontweight="bold", ha="center")

    ax.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xlim(budgets[0] - 0.4, budgets[-1] + 3.4)
    ax.set_xticks(list(budgets))
    style(ax, f"Draft budget sweep, precomputed arm, datastore {store:,}",
          "draft budget B (tokens proposed per step)", "modeled speedup")
    legend(ax)
    caption(fig, "Acceptance decays geometrically with draft length while verification cost "
        "grows linearly, so speedup peaks and then falls. The peak is workload-dependent: "
        "the high-speculatability workload pays for a 4-token draft, the low-speculatability "
        "one stops earning past 2 -- so a single tuned budget is wrong for both.")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def report(cells: Dict, stores: Sequence[int], budget: int) -> None:
    """Print the three contrasts per cell.

    ``joint / (efficiency x workload)`` is the multiplicative form of the
    log-interaction: 1.000 means the two factors simply compose.
    """
    datasets = sorted({key.split("|")[0] for key in cells})
    for dataset in datasets:
        print(f"\nfactorial contrasts ({dataset}, B={budget}, excluding build):")
        for store in stores:
            try:
                arms_lo = cell(cells, dataset, "pooled", store, budget)["arms"]
                arms_hi = cell(cells, dataset, "self_relevant", store, budget)["arms"]
            except KeyError:
                continue
            a = arms_lo["torch_scan"]["speedup_excluding_build"]
            b = arms_lo["torch_index"]["speedup_excluding_build"]
            c = arms_hi["torch_scan"]["speedup_excluding_build"]
            dd = arms_hi["torch_index"]["speedup_excluding_build"]
            inter = (math.log(dd) - math.log(c)) - (math.log(b) - math.log(a))
            print(
                f"  {store:>9,}  A={a:.4f} B={b:.4f} C={c:.4f} D={dd:.4f} | "
                f"efficiency {b / a:.3f}x  workload {c / a:.3f}x  "
                f"joint {dd / a:.3f}x (predicted {(b / a) * (c / a):.3f}x, "
                f"ratio {(dd / a) / ((b / a) * (c / a)):.4f}) | "
                f"log-interaction {inter:+.5f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default=os.path.join(PROJECT_ROOT, "experiments", "rq2_factorial", "results.json"),
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(PROJECT_ROOT, "experiments", "rq2_factorial", "artifacts"),
    )
    parser.add_argument("--budget", type=int, default=4)
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as fh:
        cells = json.load(fh)["cells"]
    cells = {k: v for k, v in cells.items() if "error" not in v}

    stores = sorted({v["store_tokens"] for v in cells.values()})
    budgets = sorted({v["budget"] for v in cells.values()})
    os.makedirs(args.outdir, exist_ok=True)

    fig_interaction(cells, [stores[1], stores[-1]], args.budget,
                    os.path.join(args.outdir, "rq2_1_interaction.png"))
    fig_latency(cells, stores, args.budget,
                os.path.join(args.outdir, "rq2_2_retrieval_cost.png"))
    fig_speedup_by_store(cells, stores, args.budget,
                         os.path.join(args.outdir, "rq2_3_speedup_by_datastore.png"))
    fig_budget(cells, stores[-1], budgets,
               os.path.join(args.outdir, "rq2_4_budget_sweep.png"))

    report(cells, stores, args.budget)
    print(f"\nfigures written to {args.outdir}")


if __name__ == "__main__":
    main()
