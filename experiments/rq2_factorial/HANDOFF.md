# RQ2 Handoff Context

Paste this to another agent to continue. Written 2026-07-28.

---

## Project

Pre-print: *"Rethinking Speculative Decoding as Workload-Technique Co-Design"*.
Faith (4th-year CompEng, Chulalongkorn) owns **RQ2** and RQ3. Paper writing deferred.

**RQ2:** What are the independent and joint effects of workload speculatability and
speculation efficiency on inference speedup?

Repo: `/Users/faith/Chula/LLM/spec-decode-greedy`, branch `feat/rq2-experiment-harness`.
**Nothing is committed yet** — Faith commits himself, wants commands only.

---

## The core problem this work solves

The existing simulator measures `speedup = normal_steps / speculative_steps` (call it
`L/S`). That treats the drafter as free. Two retrieval strategies that produce
identical drafts get identical scores even if one is 206× slower. The efficiency axis
is invisible to it.

Fix: a cost model layered on top.

```
speedup = (L/S) / (1 + ρ + β·B)

L, S  — from simulator, exact, no assumptions
t_d   — drafter wall-clock, measured on CPU
ρ     — t_d / t_v(1)          ← the entire efficiency axis lives here
t_v(1)— ASSUMED 20 ms (7B-class model, consumer GPU, from memory-bandwidth reasoning)
β     — ASSUMED 0.01 (never measured; pure guess)
B     — draft budget = num_sequences × draft_depth
R     — requests sharing one index build (amortization; bridge to RQ3)
```

Every result records `source="assumed"`, `model_name="assumed-7b"`. No real model is run.

---

## Experimental design

**2×2 factorial**

|  | on-the-fly | precomputed |
|---|---|---|
| low-spec workload | A | B |
| high-spec workload | C | D |

**Workload axis** — datastore *relevance*, not dataset identity. Two variants, identical
token counts, differing only in which documents are inside:
- `pooled` — the target's source document is NOT in the datastore
- `self_relevant` — it IS

Validated by `oracle_accept_rate` (median, 200K store): CNN 0.1005 → 0.2663 (2.65×),
SAMSum 0.0905 → 0.1558 (1.72×), non-overlapping Q1–Q3. XSum only 1.15× so it was
dropped. Two earlier attempts failed: within-dataset request splitting (1.17×) and
cross-dataset task type (1.07×).

**Efficiency axis** — 4 arms, all producing **bit-identical drafts** (asserted on 244
prompts per cell before any timing; run aborts on divergence):
`torch_scan`, `python_scan` (on-the-fly) vs `torch_index`, `python_index` (precomputed).
Main pair is `torch_scan` ↔ `torch_index`.

**Conditions:** Qwen2.5-0.5B-Instruct tokenizer, 200 requests/cell × 200 tokens,
stores 50K/200K/500K/1M, B ∈ {1,2,3,4,6,8}, depth drafting (`num_sequences=1`),
1 thread, warm-up discarded, median + p95 reported.

---

## THE FOUR RESULTS — with honest status

This is the most important section. Several claims were walked back during review.

### Result 1 — "interaction ≈ 1.000" ❌ NOT A FINDING

`joint = efficiency × workload` to within 0.05% across all 8 cells.

**This is algebraically forced.** Workload only moves `L/S` (numerator); efficiency only
moves `ρ` (denominator). A ratio of the two composes multiplicatively by construction.
The result is knowable without running anything.

The only empirical content: the tiny deviation from 1.000 comes entirely from ρ differing
between workloads (0.152443 vs 0.151858 = 0.4%). So what it actually tests is
**"does drafter cost depend on workload content?" — answer: no.**

**Action: demote out of Results into Methods as a validity check.**

### Result 2 — retrieval cost vs datastore size ✅ STRONGEST (pure measurement)

CNN, per `generate_draft` call:

| store | scan median | index median | scan p95 | index p95 |
|---|---|---|---|---|
| 50K | 171 µs | 11 µs | 187 µs | 24 µs |
| 200K | 652 µs | 12 µs | 709 µs | 41 µs |
| 500K | 1,595 µs | 12 µs | 1,747 µs | 114 µs |
| 1M | 3,068 µs | 15 µs | 3,460 µs | 348 µs |

Datastore ×20 → scan ×17.9, index ×1.3 (median).

**Two corrections made during review:**
1. The figure originally plotted **median**, where `python_scan` *falls* from 500K→1M
   (1,059 → 764 µs). Cause found in code: `_collect_candidates` returns on the first
   match ([tensorNGramDrafter.py:161-162](../../src/specdecode/simulator/drafter/tensorNGramDrafter.py#L161-L162)),
   so a larger datastore finds a match *sooner*. Meanwhile p95 (no match → full scan)
   grows 2,350 → 33,993 µs. The figure now plots **p95** and every line is monotonic.
   (An earlier guess that this was a measurement-budget artifact was **wrong** —
   call counts are identical to `torch_scan` at 732.)
2. "Index lookups stay flat" is **false at p95** — tensor index grows 24 → 348 µs (14.5×),
   because `cap_positions = len(corpus)` (set for width-safety) leaves buckets uncapped.
   Only `python_index` (dict) is genuinely flat at 2 µs.
3. The headline "206× gap" is a **median** number. At p95 the gap is 7.8× / 17.2× / 15.3× /
   9.9× — not growing. **Report both or the claim is overstated.**

### Result 3 — speedup by datastore size ⚠️ MIXED

CNN, B=4, excluding build:

| store | A lo+fly | B lo+pre | C hi+fly | D hi+pre |
|---|---|---|---|---|
| 50K | 1.0291 | 1.0371 | 1.1050 | 1.1135 |
| 200K | 1.0179 | 1.0491 | 1.2114 | 1.2486 |
| 500K | **0.9817** | 1.0568 | 1.1631 | 1.2515 |
| 1M | **0.9316** | 1.0687 | 1.0937 | 1.2541 |

**Assumption-free core (this is the real finding):** datastore ×20 raises benefit only
+3.1% (low-spec L/S 1.0684→1.1012) and +12.6% (high-spec 1.1471→1.2922), while
on-the-fly cost rises +1,690%. **Benefit saturates; cost does not.**

**Assumption-dependent part:** "cell A drops below 1.0" depends on the assumed
t_v(1)=20 ms. Stronger phrasing that avoids the assumption:

> on-the-fly at 1M tokens only pays off if the target model exceeds **50.3 ms/step**
> (low-spec) or **12.2 ms/step** (high-spec); precomputed pays off at every value.

**Also assumption-free and important:** A and B have *identical* L/S (1.1012 at 1M) by
construction of draft equivalence — so the token-accounting metric declares them equal
despite a 206× retrieval gap. This is the methodological argument for the cost model.

### Result 4 — budget sweep ⚠️ NOT YET AUDITED

Store 1M, precomputed arm:

| B | CNN low | CNN high | SAMSum low | SAMSum high |
|---|---|---|---|---|
| 1 | 1.0792 | 1.1755 | **1.0706** | 1.1376 |
| 2 | **1.0831** | 1.2290 | 1.0660 | 1.1591 |
| 3 | 1.0774 | 1.2474 | 1.0563 | **1.1605** |
| 4 | 1.0687 | **1.2541** | 1.0461 | 1.1536 |
| 6 | 1.0492 | 1.2467 | 1.0264 | 1.1363 |
| 8 | 1.0301 | 1.2329 | 1.0077 | 1.1160 |

**Known issue not yet worked through:** raw `L/S` is **monotonically increasing** in B
(CNN high: 1.1763 → 1.3192 from B=1 to B=8). So the peak comes **entirely from the
assumed `β·B` penalty**. The exact peak positions therefore depend on β=0.01, which was
never measured. The claim likely to survive is the *qualitative* one — the high-spec
workload's optimal budget is larger than the low-spec one's, because its L/S curve is
steeper (+12.1% vs +2.1% across B) — but this needs to be verified before publishing.

**This is the next task.**

---

## How to answer RQ2 (the phrasing landed on after several rejected drafts)

Report effect sizes (2×2 standard) **and** the mechanism. Neither alone is enough.
Faith explicitly rejected a bare range of numbers as unscientific ("มันก็ต้องต่างตาม
dataset อยู่แล้ว").

The framing that worked:

- **speculatability** lives in the numerator → property of the **data**; transfers across
  hardware; sets the **ceiling** of achievable speedup; engineering cannot move it.
- **efficiency** lives in the denominator → property of the **system**; transfers across
  datasets; determines **how close you get to that ceiling**; hard-capped at
  `1 + ρ/(1+βB)` — measure ρ alone and the effect is predictable without experiments
  (predicted 1.147×, measured 1.147×).
- **joint** = 1.346× and multiplicative — **but state plainly that this composition is
  imposed by the cost model, not measured.** What was measured is the condition that
  makes the composition valid: ρ is workload-independent (0.4%).

Numbers at 1M, B=4: speculatability 1.17× (CNN) / 1.10× (SAMSum); efficiency 1.147×;
joint 1.346×.

Practical rule that falls out: **measure ρ first.** ρ≈0 → retrieval is already free, go
fix the datastore. ρ≈0.15 → ~13% is still on the table.

---

## Limitations (agreed list)

1. **ρ spans devices** — t_d measured on CPU, t_v(1) assumed GPU. The scanning arm is
   the one that would benefit most from a GPU (it is a parallel tensor comparison; an
   index lookup is pointer-chasing and gains nothing), so this biases **against**
   on-the-fly. All biases in this work point the same direction — they make the results
   look worse, never better.
2. **t_v(1) and β are assumed.** t_v(1)=20 ms has a defensible derivation
   (7B fp16 ≈ 14 GB / ~700 GB/s). **β=0.01 is a pure guess** and is cheap to measure —
   forward passes at B=1..8 on Qwen2.5-0.5B, a few minutes on Faith's RTX 5060.
3. **Cost model never validated end-to-end.** The *form* `1 step = drafter + verify` is
   unchecked; KV-cache handling, transfers, sync could break it. Recommended gate:
   one real workload, measured vs modeled within 20%.
4. **Ground truth used in place of target-model output.** Real greedy decoding is more
   repetitive than human reference text, which an n-gram drafter matches more easily, so
   reported L/S is likely a **lower bound**.
5. **Depth drafting only** (`num_sequences=1`). Depth-vs-width is an appendix, not run.

Also noted but unresolved: `t_v(B) = t_v(1)(1+βB)` is off by one — at B=1 it yields
`1.01·t_v(1)` instead of `t_v(1)`. Correct form is `(1+β(B−1))`. **Does not affect any
ratio** (identical βB across all four cells cancels), affects raw speedups by ~1% and
may shift Result 4's peaks slightly. Not yet fixed.

---

## Files

| what | where |
|---|---|
| raw results | `experiments/rq2_factorial/results.json` |
| speculatability probe | `experiments/rq2_workload_probe/probe.json` |
| figures | `experiments/rq2_factorial/artifacts/*.png` |
| Thai write-up (**STALE**) | `experiments/rq2_factorial/RESULTS_TH.md` + `.html` |
| runner | `scripts/run_rq2.py` |
| plotting + effect report | `scripts/plot_rq2.py` (`report()` prints all contrasts) |
| cost model | `src/specdecode/costmodel/` |
| speculatability metrics | `src/specdecode/speculatability/` |
| equivalence gate | `src/specdecode/experiments/equivalence.py` |
| timing helpers | `src/specdecode/experiments/timing.py` |
| workload construction | `src/specdecode/experiments/workloads.py` |

`ruff check . && pyright && python -m pytest tests/ -q` — 130 tests, all passing.

---

## Open tasks, in order

1. **Audit Result 4** — establish which part of the budget-sweep claim survives given
   that the peak is created by the assumed β.
2. **Rewrite `RESULTS_TH.md`** — it still contains the walked-back claims: Result 1 framed
   as a falsifiable test, "index lookups stay flat", the bare "206×" headline, "ρ is a
   graph axis" (**no figure uses ρ as an axis — that claim is simply false**), and the
   wrong `python_scan` dip explanation (says budget cap; it is early-exit).
3. Optionally measure β (cheap, converts one assumption into a measurement).
4. Commit. Nothing is staged; Faith runs the commands.
5. Later: RQ3 history scoping (design parked in memory).

---

## Working with Faith

Short answers, one step at a time, lead with the number. Long structured documents get
"งง โคตร" and stall the work. Prefer a worked example with real numbers over a definition.
Use English technical terms transliterated, not translated ("independent effect", not
"ผลอิสระ"). He pushes back hard and correctly on overstated claims — several of the
corrections above came from his questions, not from self-review. He runs long jobs
himself; give him the command.
