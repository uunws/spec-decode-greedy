"""Figures for RQ3.

Four figures, one claim each. A figure that does not correspond to a claim is not
drawn -- the pilot shipped one whose caption said an axis was rho when no axis was.

    1  relevance isolated   each treatment beside its size-matched control
    2  movement             where scoping moves a workload on the support/structure plane
    3  sweet spot           modelled speedup against scope width, build cost included
    4  negative control     the narrow(N) ladder, which cuts on recency and loses

Palette: the first three slots of the validated categorical set, which clear the
all-pairs colour-vision floors. Several slots fall below 3:1 against the surface, so
every series carries a visible direct label rather than relying on its colour --
which is what a paper figure wants regardless.

    python scripts/plot_rq3.py
    python scripts/plot_rq3.py --results experiments/rq3_scoping/results.json
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
YELLOW, MAGENTA, VIOLET = "#eda100", "#e87ba4", "#4a3aa7"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

SCOPE_COLOR = {"global": BLUE, "group": ORANGE, "local": AQUA}
SOURCE_COLORS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET]

SCOPE_LABEL = {
    "global": "global",
    "group": "group",
    "local": "local",
    "narrow1": "narrow(1)",
    "narrow5": "narrow(5)",
    "narrow10": "narrow(10)",
    "narrow50": "narrow(50)",
}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 9,
        "axes.titlesize": 10,
        "lines.linewidth": 2.0,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load(path: str) -> Tuple[Dict, Dict]:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    cells = {
        k: v
        for k, v in payload.get("cells", {}).items()
        if "error" not in v and k.count("|") == 2
    }
    return payload.get("config", {}), cells


def split_id(cell_id: str) -> Tuple[str, int, str]:
    source, n_part, scope = cell_id.split("|", 2)
    return source, int(n_part[1:]), scope


def sources_in(cells: Dict) -> List[str]:
    return sorted({split_id(k)[0] for k in cells})


def pick(cells: Dict, source: str, n_gram: int, scope: str) -> Optional[Dict]:
    return cells.get(f"{source}|n{n_gram}|{scope}")


def default_n(cells: Dict) -> int:
    orders = sorted({split_id(k)[1] for k in cells})
    return 3 if 3 in orders else (orders[0] if orders else 3)


def outcome_key(config: Dict, source: str) -> str:
    """L/S where targets are long enough for it to mean anything, else per-token.

    A three-token SQuAD answer allows at most one speculative step, so its L/S is
    quantised into a handful of values. Reporting it beside a 500-token chat turn
    would be comparing a coin flip with a measurement.
    """
    stream = (config.get("streams") or {}).get(source, {})
    return "token_speedup" if (stream.get("median_target_tokens") or 0) >= 32 else "accepted_per_token"


def outcome_label(key: str) -> str:
    return "accepted length  L/S" if key == "token_speedup" else "accepted tokens per target token"


def fig_relevance(config: Dict, cells: Dict, n_gram: int, out: str) -> None:
    """Figure 1. Each treatment beside a random datastore of the same size.

    The primary result. The pair differs only in *which* tokens are in the
    datastore, so the gap is relevance and nothing else. Without it a scope
    comparison cannot be read at all, because narrowing changes relevance and corpus
    size together and the two oppose each other.
    """
    by_metric: Dict[str, List[Tuple[str, str, float, float]]] = {}
    for source in sources_in(cells):
        key = outcome_key(config, source)
        for scope in ("global", "group", "local"):
            treatment = pick(cells, source, n_gram, scope)
            control = pick(cells, source, n_gram, f"control_{scope}")
            if not treatment or not control:
                continue
            by_metric.setdefault(key, []).append(
                (source, scope, treatment["overall"][key], control["overall"][key])
            )

    if not by_metric:
        print("  figure 1 skipped: no treatment/control pairs")
        return

    # One panel per metric. L/S sits around 1 to 2 and per-token acceptance around
    # 0 to 0.6; putting both on one axis would make the short-target datasets look
    # like failures when they are simply measured in a different unit.
    keys = sorted(by_metric, key=lambda k: -len(by_metric[k]))
    widths = [max(2.2, 0.62 * len(by_metric[k])) for k in keys]
    fig, axes = plt.subplots(
        1, len(keys), figsize=(sum(widths) + 2.0, 4.6),
        gridspec_kw={"width_ratios": widths}, squeeze=False,
    )

    bar_w = 0.38
    for panel, key in enumerate(keys):
        ax = axes[0][panel]
        rows = by_metric[key]
        for i, (_, scope, treat, ctrl) in enumerate(rows):
            ax.bar(i - bar_w / 2, treat, bar_w, color=SCOPE_COLOR.get(scope, BLUE),
                   edgecolor="white", linewidth=2.0, zorder=3)
            ax.bar(i + bar_w / 2, ctrl, bar_w, color=GRID, edgecolor="white",
                   linewidth=2.0, zorder=3)
            lift = treat / ctrl if ctrl > 0 else float("inf")
            ax.text(i, max(treat, ctrl) * 1.02,
                    f"{lift:.2f}x" if math.isfinite(lift) else "n/a",
                    ha="center", va="bottom", fontsize=8, color=INK, zorder=4)

        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(
            [f"{s}\n{SCOPE_LABEL.get(sc, sc)}" for s, sc, _, _ in rows],
            fontsize=7.5, rotation=45, ha="right",
        )
        ax.set_ylabel(outcome_label(key) + "\n(grey = same size, random content)", fontsize=8.5)
        ax.set_ylim(bottom=0, top=max(max(r[2], r[3]) for r in rows) * 1.18)

    fig.suptitle("Relevance isolated: each scope against a size-matched random datastore",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_a_scope_movement(config: Dict, cells: Dict, n_gram: int, out: str) -> None:
    """Figure A. Change the scope, measure the two axes, look at where it goes.

    This is the whole of RQ3 as the question is written. x is Structure
    (1 - normalized branching entropy), y is History (mean support, log scale).
    Speculatability rises towards the top right.

    Nothing about speed appears here, deliberately. Defining a high-speculatability
    workload as "the one that ran fast" would make the finding circular, so the
    plane is measured from the text alone and the outcome is reported separately.
    ``coverage`` is printed beside each point because mean support is averaged over
    covered positions only and is misleading without it.
    """
    names = sources_in(cells)
    if not names:
        return
    cols = min(3, len(names))
    rows_n = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.3 * cols, 3.8 * rows_n), squeeze=False)

    for idx, source in enumerate(names):
        ax = axes[idx // cols][idx % cols]
        points: List[Tuple[str, float, float, float]] = []
        for scope in ("global", "group", "local"):
            cell = pick(cells, source, n_gram, scope)
            if not cell or cell["overall"]["coverage"] <= 0:
                continue
            o = cell["overall"]
            points.append((scope, o["structure"], max(o["mean_support"], 1e-3), o["coverage"]))

        for a, b in zip(points, points[1:]):
            ax.annotate("", xy=(b[1], b[2]), xytext=(a[1], a[2]),
                        arrowprops={"arrowstyle": "->", "color": MUTED,
                                    "linewidth": 1.4, "shrinkA": 10, "shrinkB": 10})
        for scope, structure, support, coverage in points:
            ax.scatter([structure], [support], s=150, color=SCOPE_COLOR[scope],
                       edgecolor="white", linewidth=2.0, zorder=4)
            ax.annotate(f"{SCOPE_LABEL[scope]}\ncov {coverage:.2f}", (structure, support),
                        textcoords="offset points", xytext=(10, -4), fontsize=8, color=INK)

        ax.set_yscale("log")
        ax.set_title(source, loc="left", color=INK)
        ax.set_xlabel("Structure  =  1 - normalized branching entropy")
        ax.set_ylabel("History  =  mean support")

    for idx in range(len(names), rows_n * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle(
        "A. Narrowing the scope moves every workload down and to the right: "
        "support traded for structure",
        x=0.01, ha="left", fontsize=11, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_b_relevance_plane(config: Dict, cells: Dict, n_gram: int, out: str) -> None:
    """Figure B. Hold the corpus size fixed and change only which tokens are in it.

    Figure A cannot separate two things: a narrower scope is both more relevant and
    smaller, and support falls with size on its own. Each treatment is therefore
    paired with a random datastore of the same token count, and the arrow runs from
    the control to the treatment. Whatever that arrow shows is relevance, because
    size, reuse structure and scope key are all identical across the pair.

    Same axes as figure A so the two read together.
    """
    names = sources_in(cells)
    if not names:
        return
    cols = min(3, len(names))
    rows_n = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.3 * cols, 3.8 * rows_n), squeeze=False)
    drawn = 0

    for idx, source in enumerate(names):
        ax = axes[idx // cols][idx % cols]
        for scope in ("group", "local"):
            treat = pick(cells, source, n_gram, scope)
            ctrl = pick(cells, source, n_gram, f"control_{scope}")
            if not treat or not ctrl or treat["overall"]["coverage"] <= 0:
                continue
            t, c = treat["overall"], ctrl["overall"]
            tx, ty = t["structure"], max(t["mean_support"], 1e-3)
            cx, cy = c["structure"], max(c["mean_support"], 1e-3)
            ax.annotate("", xy=(tx, ty), xytext=(cx, cy),
                        arrowprops={"arrowstyle": "->", "color": SCOPE_COLOR[scope],
                                    "linewidth": 1.6, "shrinkA": 9, "shrinkB": 9})
            ax.scatter([cx], [cy], s=110, facecolor="white",
                       edgecolor=SCOPE_COLOR[scope], linewidth=2.0, zorder=4)
            ax.scatter([tx], [ty], s=150, color=SCOPE_COLOR[scope],
                       edgecolor="white", linewidth=2.0, zorder=5)
            ax.annotate(f"{SCOPE_LABEL[scope]}\ncov {c['coverage']:.2f} to {t['coverage']:.2f}",
                        (tx, ty), textcoords="offset points", xytext=(10, -4),
                        fontsize=8, color=INK)
            drawn += 1

        ax.set_yscale("log")
        ax.set_title(source, loc="left", color=INK)
        ax.set_xlabel("Structure  =  1 - normalized branching entropy")
        ax.set_ylabel("History  =  mean support")

    for idx in range(len(names), rows_n * cols):
        axes[idx // cols][idx % cols].axis("off")

    if not drawn:
        plt.close(fig)
        print("  figure B skipped: no treatment/control pairs")
        return

    fig.suptitle(
        "B. At the same corpus size, relevance alone moves the workload to the right "
        "(hollow = random content)",
        x=0.01, ha="left", fontsize=11, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_sweet_spot(config: Dict, cells: Dict, n_gram: int, out: str) -> None:
    """Figure 3. Modelled speedup against scope width, index build cost included.

    Build cost is what makes this a trade-off rather than a preference: a narrow
    scope builds a small index but rebuilds it constantly, so its amortization
    factor R is small. Excluding build would flatter the narrow end for free.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ladder = ("local", "narrow1", "narrow5", "narrow10", "narrow50", "group", "global")
    drawn = 0

    for i, source in enumerate(sources_in(cells)):
        xs: List[float] = []
        ys: List[float] = []
        for scope in ladder:
            cell = pick(cells, source, n_gram, scope)
            if not cell or cell["mean_corpus_tokens"] <= 0:
                continue
            xs.append(cell["mean_corpus_tokens"])
            ys.append(cell["overall"]["speedup_including_build"])
        if len(xs) < 2:
            continue
        pairs = sorted(zip(xs, ys))
        xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
        color = SOURCE_COLORS[i % len(SOURCE_COLORS)]
        ax.plot(xs, ys, marker="o", markersize=8, color=color,
                markeredgecolor="white", markeredgewidth=2.0, zorder=3)
        ax.annotate(source, (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(9, 0), fontsize=9, color=INK, va="center")
        drawn += 1

    if not drawn:
        plt.close(fig)
        print("  figure 3 skipped: no scope ladders")
        return

    ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.text(ax.get_xlim()[0], 1.0, " break-even", fontsize=8, color=MUTED, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("datastore size (tokens, log scale)")
    ax.set_ylabel("modelled speedup, index build cost included")
    ax.set_title("Sweet spot: how wide the retained history should be", loc="left", color=INK)
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_negative_control(config: Dict, cells: Dict, n_gram: int, out: str) -> None:
    """Figure 4. Cutting on recency instead of relevance.

    narrow(N) keeps the N most recent documents of the request's own group -- narrow
    in exactly the way the naive hypothesis means. If it loses while request-local,
    at a fraction of the size, sits above it, then narrow and relevant are not the
    same thing. This is the figure that says so.
    """
    ladder = ("narrow1", "narrow5", "narrow10", "narrow50")
    series: Dict[str, List[Tuple[str, List[float], List[float], Optional[Tuple[float, float]]]]] = {}

    for source in sources_in(cells):
        key = outcome_key(config, source)
        xs, ys = [], []
        for scope in ladder:
            cell = pick(cells, source, n_gram, scope)
            if not cell or cell["mean_corpus_tokens"] <= 0:
                continue
            xs.append(cell["mean_corpus_tokens"])
            ys.append(cell["overall"][key])
        if len(xs) < 2:
            continue
        local = pick(cells, source, n_gram, "local")
        point = None
        if local and local["mean_corpus_tokens"] > 0:
            point = (local["mean_corpus_tokens"], local["overall"][key])
        series.setdefault(key, []).append((source, xs, ys, point))

    if not series:
        print("  figure 4 skipped: no narrow ladder in results")
        return

    # Same faceting rule as figure 1: L/S and per-token acceptance are different
    # units and must not share a y axis.
    keys = sorted(series, key=lambda k: -len(series[k]))
    fig, axes = plt.subplots(1, len(keys), figsize=(5.4 * len(keys), 4.4), squeeze=False)

    for panel, key in enumerate(keys):
        ax = axes[0][panel]
        for i, (source, xs, ys, point) in enumerate(series[key]):
            color = SOURCE_COLORS[i % len(SOURCE_COLORS)]
            ax.plot(xs, ys, marker="o", markersize=8, color=color,
                    markeredgecolor="white", markeredgewidth=2.0, zorder=3)
            ax.annotate(f"{source} narrow(N)", (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(9, 0), fontsize=8, color=INK, va="center")
            if point is not None:
                ax.scatter([point[0]], [point[1]], s=150, marker="D", color=color,
                           edgecolor="white", linewidth=2.0, zorder=4)
                ax.annotate(f"{source} local", point, textcoords="offset points",
                            xytext=(9, 7), fontsize=8, color=INK)
        ax.set_xscale("log")
        ax.set_xlabel("datastore size (tokens, log scale)")
        ax.set_ylabel(outcome_label(key))

    fig.suptitle(
        "Negative control: the narrow(N) ladder rises with size alone, not with relevance",
        x=0.01, ha="left", fontsize=11, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def report(config: Dict, cells: Dict, n_gram: int) -> None:
    """The numbers the figures are drawn from, so any claim can be checked by eye."""
    header = (f"\n{'cell':<36} {'corpus':>10} {'support':>9} {'struct':>7} {'cov':>6} "
              f"{'L/S':>7} {'acc/tok':>8} {'R':>6} {'speedup':>8} {'+build':>8}")
    print(header)
    print("-" * 118)
    for source in sources_in(cells):
        for scope in ("global", "group", "narrow50", "narrow10", "narrow5", "narrow1",
                      "local", "control_global", "control_group", "control_local"):
            cell = pick(cells, source, n_gram, scope)
            if not cell:
                continue
            o = cell["overall"]
            print(f"{source + '|' + scope:<36} {cell['mean_corpus_tokens']:>10,.0f} "
                  f"{o['mean_support']:>9.1f} {o['structure']:>7.3f} {o['coverage']:>6.3f} "
                  f"{o['token_speedup']:>7.3f} {o['accepted_per_token']:>8.3f} "
                  f"{o['reuse']:>6.0f} {o['speedup_excluding_build']:>8.4f} "
                  f"{o['speedup_including_build']:>8.4f}")
        print()

    print("relevance lift (treatment / size-matched control):")
    for source in sources_in(cells):
        key = outcome_key(config, source)
        for scope in ("global", "group", "local"):
            treatment = pick(cells, source, n_gram, scope)
            control = pick(cells, source, n_gram, f"control_{scope}")
            if not treatment or not control:
                continue
            t, c = treatment["overall"][key], control["overall"][key]
            lift = t / c if c else float("nan")
            print(f"  {source:<12} {scope:<8} {outcome_label(key):<34} "
                  f"{t:.4f} / {c:.4f} = {lift:.3f}x")

    warnings: Sequence[str] = config.get("size_match_warnings") or []
    if warnings:
        print(f"\n{len(warnings)} size-match warning(s) -- the lift above is not clean for these:")
        for line in warnings:
            print(f"  {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default=os.path.join(PROJECT_ROOT, "experiments", "rq3_scoping", "results.json"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(PROJECT_ROOT, "experiments", "rq3_scoping", "artifacts"),
    )
    parser.add_argument("--n-gram", type=int, default=0, help="0 = use 3 if present")
    args = parser.parse_args()

    config, cells = load(args.results)
    if not cells:
        print("no successful cells in results")
        return
    n_gram = args.n_gram or default_n(cells)
    os.makedirs(args.out_dir, exist_ok=True)
    style()

    print(f"figures for n={n_gram}, {len(cells)} successful cells")
    # The two primary figures answer RQ3 on its own terms: the plane, measured.
    fig_a_scope_movement(config, cells, n_gram,
                         os.path.join(args.out_dir, "rq3_A_scope_movement.png"))
    fig_b_relevance_plane(config, cells, n_gram,
                          os.path.join(args.out_dir, "rq3_B_relevance_fixed_size.png"))
    # Appendix: outcome and cost, which belong to RQ2's axis rather than this one.
    fig_relevance(config, cells, n_gram,
                  os.path.join(args.out_dir, "rq3_apx1_outcome_bars.png"))
    fig_sweet_spot(config, cells, n_gram,
                   os.path.join(args.out_dir, "rq3_apx2_sweet_spot.png"))
    fig_negative_control(config, cells, n_gram,
                         os.path.join(args.out_dir, "rq3_apx3_negative_control.png"))
    report(config, cells, n_gram)


if __name__ == "__main__":
    main()
