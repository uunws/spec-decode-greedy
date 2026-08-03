"""Generate the RQ3 write-up as a standalone HTML page.

Every number in the page is read out of ``results.json`` at generation time rather
than typed by hand. The pilot's write-up went stale the moment a claim was walked
back, because the prose and the results were two separate documents; here they are
one command apart.

    python scripts/report_rq3.py
    python scripts/report_rq3.py --results experiments/rq3_scoping/results.json
"""

import argparse
import html
import json
import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCOPE_ORDER = ("global", "group", "narrow50", "narrow10", "narrow5", "narrow1", "local")
SCOPE_TH = {
    "global": "global",
    "group": "group",
    "local": "local",
    "narrow1": "narrow(1)",
    "narrow5": "narrow(5)",
    "narrow10": "narrow(10)",
    "narrow50": "narrow(50)",
}
SOURCE_TH = {
    "squad": "SQuAD ถามตอบแบบดึงคำตอบจากย่อหน้า จับกลุ่มด้วย title",
    "spider": "Spider แปลงคำถามเป็น SQL จับกลุ่มด้วย db_id",
    "codesearch": "CodeSearchNet เขียนฟังก์ชันจาก docstring จับกลุ่มด้วย repository",
    "samsum": "SAMSum สรุปบทสนทนา ไม่มี group key",
    "wildchat": "WildChat แชทจริงของผู้ใช้ จับกลุ่มด้วย hashed_ip",
    "swetraj": "SWE-agent trajectories ร่องรอยการทำงานของ agent จับกลุ่มด้วย repository",
}


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
    stream = (config.get("streams") or {}).get(source, {})
    return "token_speedup" if (stream.get("median_target_tokens") or 0) >= 32 else "accepted_per_token"



FIGURES = {
    "rq3_A_scope_movement.png":
        "รูป A  เปลี่ยน scope แล้ววัดสองแกน ลูกศรคือ global ไป group ไป local",
    "rq3_B_relevance_fixed_size.png":
        "รูป B  ตรึงขนาด corpus แล้วเปลี่ยนแค่เนื้อหา วงกลมกลวงคือคู่เทียบสุ่ม",
    "rq3_apx1_outcome_bars.png":
        "รูป ผนวก 1  ผลลัพธ์จริงของแต่ละ scope เทียบคู่เทียบ",
    "rq3_apx2_sweet_spot.png":
        "รูป ผนวก 2  speedup เทียบกับความกว้างของ scope",
    "rq3_apx3_negative_control.png":
        "รูป ผนวก 3  บันได narrow(N) ที่หั่นด้วยความใหม่",
}

FIG_TOKEN = re.compile(r"\{\{FIG:([A-Za-z0-9_.]+)\}\}")


def _fill_figures(fragment: str, *, embed: bool, artifacts: str = "artifacts") -> str:
    """Replace {{FIG:name}} with a real image, or with a paste marker for Docs.

    The markers live inline in the prose rather than in a figure appendix, so the
    document reads top to bottom and each image lands beside the claim it supports.
    """
    def swap(match: "re.Match[str]") -> str:
        name = match.group(1)
        caption = FIGURES.get(name, name)
        if embed:
            return (f'<figure><img src="{artifacts}/{esc(name)}" alt="{esc(caption)}">'
                    f"<figcaption>{esc(caption)}</figcaption></figure>")
        return (f'<p style="border:1px dashed #888;padding:10px;background:#f7f7f4;">'
                f"<b>[ แทรกรูปตรงนี้ ]</b> &nbsp; "
                f'<span style="font-family:Consolas,monospace;font-size:10pt;">'
                f"experiments/rq3_scoping/artifacts/{esc(name)}</span><br>"
                f"<i>{esc(caption)}</i></p>")

    return FIG_TOKEN.sub(swap, fragment)


def esc(text: object) -> str:
    return html.escape(str(text))


def table(headers: Sequence[str], rows: Sequence[Sequence[str]], classes: str = "") -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    return (f'<div class="scroll"><table class="{classes}"><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


def ladder_table(config: Dict, cells: Dict, source: str, n_gram: int) -> str:
    rows: List[List[str]] = []
    for scope in SCOPE_ORDER:
        cell = pick(cells, source, n_gram, scope)
        if not cell:
            continue
        o = cell["overall"]
        rows.append([
            f'<span class="scope">{esc(SCOPE_TH.get(scope, scope))}</span>',
            f"{cell['mean_corpus_tokens']:,.0f}",
            f"{o['mean_support']:.1f}",
            f"{o['structure']:.3f}",
            f"{o['coverage']:.3f}",
            f"{o['token_speedup']:.3f}",
            f"{o['accepted_per_token']:.3f}",
            f"{o['reuse']:.0f}",
            f"{o['speedup_including_build']:.3f}",
        ])
    return table(
        ["scope", "corpus (tok)", "support", "structure", "coverage",
         "L/S", "acc/tok", "R", "speedup รวม build"],
        rows,
    )


def lift_rows(config: Dict, cells: Dict, n_gram: int) -> List[Tuple[str, str, float, float, float]]:
    out: List[Tuple[str, str, float, float, float]] = []
    for source in sources_in(cells):
        key = outcome_key(config, source)
        for scope in ("global", "group", "local"):
            treatment = pick(cells, source, n_gram, scope)
            control = pick(cells, source, n_gram, f"control_{scope}")
            if not treatment or not control:
                continue
            t = treatment["overall"][key]
            c = control["overall"][key]
            out.append((source, scope, t, c, t / c if c else float("nan")))
    return out


def lift_table(config: Dict, cells: Dict, n_gram: int) -> str:
    rows = []
    for source, scope, t, c, lift in lift_rows(config, cells, n_gram):
        if not math.isfinite(lift):
            # Both sides are zero: the source has no grouping key, so the group
            # scope resolves to an empty datastore. That is a result, not a gap.
            verdict = "<b>ไม่มี group key</b>"
        else:
            highlight = ' class="hot"' if lift >= 1.10 else ""
            verdict = f"<b{highlight}>{lift:.2f}x</b>"
        rows.append([
            esc(source),
            f'<span class="scope">{esc(SCOPE_TH.get(scope, scope))}</span>',
            f"{t:.3f}",
            f"{c:.3f}",
            verdict,
        ])
    return table(
        ["dataset", "scope", "ของจริง", "คู่เทียบสุ่ม (ขนาดเท่ากัน)", "ส่วนต่างจาก relevance"],
        rows,
    )


def stream_table(config: Dict) -> str:
    rows = []
    for source, info in (config.get("streams") or {}).items():
        if "error" in info:
            rows.append([esc(source), '<span class="warn">โหลดไม่สำเร็จ</span>',
                         "", "", "", esc(info["error"])[:80]])
            continue
        rows.append([
            esc(source),
            esc(info.get("grouping", "")),
            f"{info.get('n_groups', 0):,}",
            f"{info.get('n_requests', 0):,}",
            f"{info.get('history_tokens', 0):,}",
            f"{info.get('median_target_tokens', 0):.0f}",
        ])
    return table(
        ["dataset", "group key", "จำนวน group", "จำนวน request", "history (tok)", "target กลาง (tok)"],
        rows,
    )



def plane_table(config: Dict, cells: Dict, n_gram: int) -> str:
    """Treatment against its size-matched control, on the two axes only.

    No speed appears in this table by design. The claim it supports is that
    relevance moves the workload on the plane, and mixing an outcome column in
    would invite exactly the circular reading the advisor warned about.
    """
    rows: List[List[str]] = []
    for source in sources_in(cells):
        for scope in ("group", "local"):
            t_cell = pick(cells, source, n_gram, scope)
            c_cell = pick(cells, source, n_gram, f"control_{scope}")
            if not t_cell or not c_cell or t_cell["overall"]["coverage"] <= 0:
                continue
            t, c = t_cell["overall"], c_cell["overall"]

            def pair(key: str) -> str:
                up = t[key] > c[key]
                mark = "ขึ้น" if up else "ลง"
                weight = " style=\"font-weight:600;\"" if up else ""
                return f"<span{weight}>{c[key]:.3f} → {t[key]:.3f}  {mark}</span>"

            rows.append([
                esc(source),
                f'<span class="scope">{esc(SCOPE_TH.get(scope, scope))}</span>',
                f"{t_cell['mean_corpus_tokens']:,.0f}",
                pair("mean_support"),
                pair("structure"),
                pair("coverage"),
            ])
    return table(
        ["dataset", "scope", "corpus (tok)", "support", "structure", "coverage"],
        rows,
    )


def build_html(config: Dict, cells: Dict, n_gram: int, prose: Dict[str, str]) -> str:
    prose = {k: _fill_figures(v, embed=True) for k, v in prose.items()}
    sources = sources_in(cells)
    ladders = "".join(
        f'<h3>{esc(SOURCE_TH.get(s, s))}</h3>{ladder_table(config, cells, s, n_gram)}'
        for s in sources
    )
    warnings = config.get("size_match_warnings") or []
    warn_block = ""
    if warnings:
        items = "".join(f"<li>{esc(w)}</li>" for w in warnings)
        warn_block = f'<div class="callout warn"><b>size-match ไม่ผ่าน</b><ul>{items}</ul></div>'

    return f"""<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RQ3 History Scoping</title>
<style>
  :root {{
    --ink: #14140f; --muted: #55544d; --line: #e2e1da; --bg: #fbfbf8;
    --card: #ffffff; --blue: #2a78d6; --orange: #eb6834; --aqua: #1baf7a;
    --warn: #e34948;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 40px 24px 80px; background: var(--bg); color: var(--ink);
    font: 15px/1.75 "IBM Plex Sans Thai", "Noto Sans Thai", -apple-system,
          BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  .wrap {{ max-width: 940px; margin: 0 auto; }}
  h1 {{ font-size: 30px; line-height: 1.3; margin: 0 0 6px; letter-spacing: -0.01em; }}
  .sub {{ color: var(--muted); margin: 0 0 40px; }}
  h2 {{
    font-size: 21px; margin: 52px 0 14px; padding-top: 22px;
    border-top: 1px solid var(--line);
  }}
  h3 {{ font-size: 16px; margin: 30px 0 10px; }}
  h4 {{ font-size: 14px; margin: 22px 0 8px; color: var(--muted);
       text-transform: uppercase; letter-spacing: 0.06em; }}
  p {{ margin: 0 0 14px; }}
  ul, ol {{ margin: 0 0 16px; padding-left: 22px; }}
  li {{ margin: 0 0 7px; }}
  code {{
    font: 13px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace;
    background: #f2f1ec; padding: 1px 5px; border-radius: 4px;
  }}
  pre {{
    background: #f7f6f2; border: 1px solid var(--line); border-radius: 8px;
    padding: 14px 16px; overflow-x: auto;
    font: 13px/1.65 ui-monospace, SFMono-Regular, Menlo, monospace;
  }}
  pre code {{ background: none; padding: 0; }}
  .scroll {{ overflow-x: auto; margin: 0 0 20px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
  th, td {{
    text-align: right; padding: 7px 11px; border-bottom: 1px solid var(--line);
    white-space: nowrap;
  }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  thead th {{
    color: var(--muted); font-weight: 600; font-size: 12px;
    text-transform: uppercase; letter-spacing: 0.04em;
  }}
  tbody tr:hover {{ background: #f7f6f2; }}
  .scope {{ font: 12.5px ui-monospace, Menlo, monospace; color: var(--muted); }}
  .hot {{ color: var(--aqua); }}
  .warn {{ color: var(--warn); }}
  .callout {{
    background: var(--card); border: 1px solid var(--line);
    border-left: 3px solid var(--blue); border-radius: 6px;
    padding: 16px 18px; margin: 0 0 20px;
  }}
  .callout.warn {{ border-left-color: var(--warn); }}
  .callout.key {{ border-left-color: var(--aqua); }}
  .callout p:last-child, .callout ul:last-child {{ margin-bottom: 0; }}
  figure {{ margin: 0 0 26px; }}
  figure img {{
    width: 100%; height: auto; border: 1px solid var(--line);
    border-radius: 8px; background: #fff;
  }}
  figcaption {{ color: var(--muted); font-size: 13px; margin-top: 8px; }}
  .meta {{ color: var(--muted); font-size: 13px; }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --ink: #f2f1ea; --muted: #a5a49a; --line: #33322c; --bg: #161613;
      --card: #1e1e1a; --blue: #3987e5; --aqua: #199e70; --warn: #e66767;
    }}
    code {{ background: #26251f; }}
    pre {{ background: #1c1b17; }}
    tbody tr:hover {{ background: #1e1e1a; }}
    figure img {{ background: #fff; }}
  }}
</style>
</head>
<body>
<div class="wrap">

<h1>RQ3 · History Scoping</h1>
<p class="sub">การจำกัดขอบเขต history ยกระดับ speculatability ได้ไหม โดยไม่แตะอัลกอริทึม</p>

{prose["intro"]}

<h2>1. วิธีทำ</h2>
{prose["method"]}

<h2>2. ข้อมูลที่ใช้</h2>
{prose["data"]}
{stream_table(config)}

<h2>3. ผลหลัก</h2>
{prose["result_a"]}
{prose["result_b"]}
{plane_table(config, cells, n_gram)}
{prose["result_after"]}

<h2>4. ตารางเต็ม</h2>
<p class="meta">n-gram = {n_gram}</p>
{ladders}

<h2>5. ภาคผนวก ผลลัพธ์และความเร็ว</h2>
{prose["appendix"]}
{lift_table(config, cells, n_gram)}
{warn_block}

<h2>6. ข้อจำกัดและคำสั่ง</h2>
{prose["limits"]}
{prose["commands"]}

</div>
</body>
</html>
"""



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    base = os.path.join(PROJECT_ROOT, "experiments", "rq3_scoping")
    parser.add_argument("--results", default=os.path.join(base, "results.json"))
    parser.add_argument("--prose", default=os.path.join(base, "prose.json"))
    parser.add_argument("--out", default=os.path.join(base, "RESULTS_TH.html"))
    parser.add_argument(
        "--advisor",
        default=os.path.join(base, "RESULTS_ADVISOR.html"),
        help="only what answers RQ3: the plane, measured; no speed anywhere",
    )
    parser.add_argument(
        "--advisor-prose",
        default=os.path.join(base, "prose_advisor.json"),
    )
    parser.add_argument("--n-gram", type=int, default=0)
    parser.add_argument(
        "--gdocs",
        default=os.path.join(base, "RESULTS_GDOCS.html"),
        help="paste-ready copy for Google Docs; written alongside the styled page",
    )
    args = parser.parse_args()

    config, cells = load(args.results)
    if not cells:
        print("no successful cells")
        return
    n_gram = args.n_gram or default_n(cells)

    with open(args.prose, encoding="utf-8") as fh:
        prose = json.load(fh)


    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(build_html(config, cells, n_gram, prose))
    print(f"wrote {args.out}")

    with open(args.gdocs, "w", encoding="utf-8") as fh:
        fh.write(build_gdocs(config, cells, n_gram, prose))
    print(f"wrote {args.gdocs}")

    if os.path.exists(args.advisor_prose):
        with open(args.advisor_prose, encoding="utf-8") as fh:
            lean = json.load(fh)
        with open(args.advisor, "w", encoding="utf-8") as fh:
            fh.write(build_advisor(config, cells, n_gram, lean))
        print(f"wrote {args.advisor}")



# ---------------------------------------------------------------------------
# Google Docs flavour
# ---------------------------------------------------------------------------
#
# Google Docs keeps headings, bold, lists, links and table *attributes* when HTML
# is pasted, and throws away stylesheets. So the styled page above pastes as an
# unreadable wall. This flavour emits the same content with presentation carried
# inline on the elements Docs actually honours, callouts turned into ordinary
# paragraphs, and figures left as named placeholders because a local <img src>
# does not survive a paste either.

CALLOUT_LEAD = {
    "callout key": "ข้อสรุปสำคัญ",
    "callout warn": "ข้อควรระวัง",
    "callout": "หมายเหตุ",
}

TABLE_STYLE = 'border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;"'


def _to_gdocs(fragment: str) -> str:
    """Rewrite a styled fragment into something a Docs paste survives."""
    out = fragment
    for cls, lead in CALLOUT_LEAD.items():
        out = out.replace(
            f'<div class="{cls}">', f'<div><p><b>[ {lead} ]</b></p>'
        )
    out = out.replace('<div class="scroll">', "<div>")
    out = out.replace(
        "<table class=\"\">",
        f"<table {TABLE_STYLE}>",
    )
    out = out.replace("<table>", f"<table {TABLE_STYLE}>")
    out = out.replace('<span class="scope">', "<span>")
    out = out.replace('<b class="hot">', "<b>")
    out = out.replace('<span class="warn">', "<span>")
    out = out.replace('<p class="meta">', "<p>")
    out = out.replace(
        "<pre><code>",
        '<pre style="font-family:Consolas,monospace;font-size:10pt;'
        'background:#f4f4f0;padding:8px;">',
    )
    out = out.replace("</code></pre>", "</pre>")
    out = out.replace(
        "<code>", '<span style="font-family:Consolas,monospace;font-size:10pt;">'
    )
    out = out.replace("</code>", "</span>")
    return out




def build_gdocs(config: Dict, cells: Dict, n_gram: int, prose: Dict[str, str]) -> str:
    prose = {k: _fill_figures(v, embed=False) for k, v in prose.items()}
    sources = sources_in(cells)
    ladders = "".join(
        f"<h3>{esc(SOURCE_TH.get(s, s))}</h3>{_to_gdocs(ladder_table(config, cells, s, n_gram))}"
        for s in sources
    )
    warnings = config.get("size_match_warnings") or []
    warn_block = ""
    if warnings:
        items = "".join(f"<li>{esc(w)}</li>" for w in warnings)
        warn_block = f"<p><b>[ ข้อควรระวัง ] size-match ไม่ผ่าน</b></p><ul>{items}</ul>"

    body = f"""<h1>RQ3 History Scoping</h1>
<p><i>การจำกัดขอบเขต history ยกระดับ speculatability ได้ไหม โดยไม่แตะอัลกอริทึม</i></p>
{_to_gdocs(prose["intro"])}
<h2>1. วิธีทำ</h2>
{_to_gdocs(prose["method"])}
<h2>2. ข้อมูลที่ใช้</h2>
{_to_gdocs(prose["data"])}
{_to_gdocs(stream_table(config))}
<h2>3. ผลหลัก</h2>
{_to_gdocs(prose["result_a"])}
{_to_gdocs(prose["result_b"])}
{_to_gdocs(plane_table(config, cells, n_gram))}
{_to_gdocs(prose["result_after"])}
<h2>4. ตารางเต็ม</h2>
<p>n-gram = {n_gram}</p>
{ladders}
<h2>5. ภาคผนวก ผลลัพธ์และความเร็ว</h2>
{_to_gdocs(prose["appendix"])}
{_to_gdocs(lift_table(config, cells, n_gram))}
{warn_block}
<h2>6. ข้อจำกัดและคำสั่ง</h2>
{_to_gdocs(prose["limits"])}
{_to_gdocs(prose["commands"])}
"""
    return (
        '<!doctype html>\n<html lang="th"><head><meta charset="utf-8">'
        "<title>RQ3 History Scoping</title></head>"
        '<body style="font-family:\'Sarabun\',\'Noto Sans Thai\',Arial,sans-serif;'
        'font-size:11pt;line-height:1.6;color:#111;">'
        + body
        + "</body></html>"
    )


# ---------------------------------------------------------------------------
# Advisor flavour: only what answers the question
# ---------------------------------------------------------------------------
#
# RQ3 asks about speculatability. Everything that measures speed -- L/S, the cost
# model, R, build cost, the narrow ladder -- is deliberately absent here, not
# hidden but out of scope: defining a high-speculatability workload by how fast it
# ran is the circularity the whole design exists to avoid. Those numbers live in
# the full write-up instead.


def plane_ladder_table(config: Dict, cells: Dict, n_gram: int) -> str:
    """The scope sweep on the two axes only, so figure A is checkable by hand."""
    rows: List[List[str]] = []
    for source in sources_in(cells):
        prev = None
        for scope in ("global", "group", "local"):
            cell = pick(cells, source, n_gram, scope)
            if not cell or cell["overall"]["coverage"] <= 0:
                continue
            o = cell["overall"]
            move = ""
            if prev is not None:
                sup = "ลง" if o["mean_support"] < prev["mean_support"] else "ขึ้น"
                st = "ขวา" if o["structure"] > prev["structure"] else "ซ้าย"
                move = f"support {sup} · structure {st}"
            rows.append([
                esc(source),
                f'<span class="scope">{esc(SCOPE_TH.get(scope, scope))}</span>',
                f"{cell['mean_corpus_tokens']:,.0f}",
                f"{o['mean_support']:.1f}",
                f"{o['structure']:.3f}",
                f"{o['coverage']:.3f}",
                move,
            ])
            prev = o
    return table(
        ["dataset", "scope", "corpus (tok)", "support", "structure", "coverage", "ขยับไปทาง"],
        rows,
    )


def build_advisor(config: Dict, cells: Dict, n_gram: int, prose: Dict[str, str]) -> str:
    prose = {k: _fill_figures(v, embed=False) for k, v in prose.items()}
    body = f"""<h1>RQ3 History Scoping</h1>
<p><i>การจำกัดขอบเขต history ยกระดับ workload speculatability ได้ไหม
โดยไม่เปลี่ยนอัลกอริทึม speculative decoding</i></p>
{_to_gdocs(prose["question"])}
<h2>1. วิธีวัด</h2>
{_to_gdocs(prose["method"])}
<h2>2. ข้อมูล</h2>
{_to_gdocs(prose["data"])}
{_to_gdocs(stream_table(config))}
<h2>3. ผล A เปลี่ยน scope แล้ว workload ขยับไปทางไหน</h2>
{_to_gdocs(prose["result_a"])}
{_to_gdocs(plane_ladder_table(config, cells, n_gram))}
{_to_gdocs(prose["result_a_after"])}
<h2>4. ผล B ตรึงขนาด เปลี่ยนแค่ความเกี่ยวข้อง</h2>
{_to_gdocs(prose["result_b"])}
{_to_gdocs(plane_table(config, cells, n_gram))}
{_to_gdocs(prose["result_b_after"])}
<h2>5. ข้อสรุป</h2>
{_to_gdocs(prose["conclusion"])}
<h2>6. ข้อจำกัด</h2>
{_to_gdocs(prose["limits"])}
"""
    return (
        '<!doctype html>\n<html lang="th"><head><meta charset="utf-8">'
        "<title>RQ3 History Scoping</title></head>"
        '<body style="font-family:\'Sarabun\',\'Noto Sans Thai\',Arial,sans-serif;'
        'font-size:11pt;line-height:1.6;color:#111;">'
        + body
        + "</body></html>"
    )


if __name__ == "__main__":
    main()
