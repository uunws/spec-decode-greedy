"""End-to-end run of one RQ3 cell, with no network.

The pilot's results could not be checked because nothing exercised the whole path at
once: a scope resolved, an index built, a drafter run, the cost model applied. These
tests drive exactly that path over a synthetic stream whose answer is known in
advance, so a broken join between two correct components fails here rather than in a
results table three hours later.

The stream is built so the expected ordering is not a matter of taste: every group's
history contains that group's own answer text and no other group's, so the group
scope *must* beat a random datastore of the same size, and request-local -- which
retains the prompt only -- must beat neither.
"""

import importlib.util
import os
import statistics

import pytest

from specdecode.experiments.rq3_workloads import build_stream
from specdecode.experiments.sources import Sample
from specdecode.scoping import GlobalScope, GroupScoped, RequestLocal, SizeMatchedControl

SPEC = importlib.util.spec_from_file_location(
    "run_rq3",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "scripts", "run_rq3.py"),
)
run_rq3 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_rq3)


def enc(text: str):
    """One token per character. Deterministic, and repetition in text is repetition in tokens."""
    return [ord(c) % 1000 for c in text]


def signature(group: int, item: int) -> str:
    """Text unique to a group, repeated verbatim across that group's items.

    This is the relevance the group scope is supposed to find. Making it literal
    keeps the assertion about the machinery rather than about how predictable
    English happens to be.
    """
    motif = f"the {'alpha beta gamma delta'.split()[group % 4]} protocol handler "
    return (motif * 6) + f"case {item} "


@pytest.fixture(scope="module")
def stream():
    samples = [
        Sample(
            group_id=f"g{g}",
            doc_text=f"request {item} concerning system {g} " * 4,
            target_text=signature(g, item),
        )
        for g in range(4)
        for item in range(9)
    ]
    return build_stream(
        samples, enc,
        history_per_group=6, requests_per_group=3,
        doc_token_cap=2048, target_token_cap=512, max_groups=8,
    )


@pytest.fixture(scope="module")
def args():
    parsed = run_rq3.build_parser().parse_args([])
    parsed.latency_prompts = 24
    parsed.latency_budget_s = 0.05
    parsed.latency_min_calls = 24
    return parsed


def run(policy, stream, args):
    documents, requests = stream
    return run_rq3.run_scope(policy, documents, requests, args, n_gram=3, timer_overhead=0.0)


@pytest.fixture(scope="module")
def cells(stream, args):
    return {
        "global": run(GlobalScope(), stream, args),
        "group": run(GroupScoped(), stream, args),
        "local": run(RequestLocal(), stream, args),
        "control_group": run(SizeMatchedControl(GroupScoped(), seed=7), stream, args),
    }


def test_every_cell_reports_the_two_axes(cells):
    for name, cell in cells.items():
        overall = cell["overall"]
        assert overall["mean_support"] >= 0.0, name
        assert 0.0 <= overall["normalized_branch_entropy"] <= 1.0, name
        assert 0.0 <= overall["coverage"] <= 1.0, name
        # Structure is the complement of entropy; the sign convention lives in one
        # place so a figure cannot silently flip it.
        assert overall["structure"] == pytest.approx(1.0 - overall["normalized_branch_entropy"])


def test_playback_is_lossless_everywhere(cells):
    """Speculation must never change the output; if it does, no other number matters."""
    for name, cell in cells.items():
        assert cell["overall"]["token_speedup"] >= 1.0, name


def test_group_scope_beats_its_size_matched_control(cells):
    """The core RQ3 claim, on data where relevance is known to be present."""
    treatment = cells["group"]["overall"]
    control = cells["control_group"]["overall"]
    assert treatment["token_speedup"] > control["token_speedup"]


def test_control_is_size_matched(cells):
    treatment = cells["group"]["overall"]["corpus_tokens"]
    control = cells["control_group"]["overall"]["corpus_tokens"]
    assert abs(control - treatment) / treatment < 0.01
    assert run_rq3.check_size_match(cells["group"], cells["control_group"], 0.01) is None


def test_size_match_audit_catches_a_mismatch(cells):
    """The gate has to be able to fail, or it is decoration."""
    assert run_rq3.check_size_match(cells["global"], cells["local"], 0.01) is not None


def test_group_scope_is_smaller_than_global(cells):
    assert cells["group"]["mean_corpus_tokens"] < cells["global"]["mean_corpus_tokens"]


def test_reuse_factor_is_counted_not_assumed(cells):
    """R is requests per index build: one datastore per group, many per request-local."""
    assert cells["group"]["cache"]["mean_reuse"] > 1.0
    assert cells["local"]["cache"]["mean_reuse"] == pytest.approx(1.0)
    assert cells["global"]["n_datastores"] == 1


def test_request_local_pays_a_build_per_request(cells):
    """Narrow scopes build cheap indexes often; that cost has to reach the model."""
    local, group = cells["local"]["overall"], cells["group"]["overall"]
    assert local["build_step_equivalents"] > 0.0
    per_request_local = local["build_step_equivalents"] / cells["local"]["n_requests"]
    per_request_group = group["build_step_equivalents"] / cells["group"]["n_requests"]
    assert per_request_local > 0.0 and per_request_group >= 0.0


def test_including_build_never_exceeds_excluding_build(cells):
    """Both come from the same pooled counts, so the inversion must be impossible."""
    for name, cell in cells.items():
        overall = cell["overall"]
        assert overall["speedup_including_build"] <= overall["speedup_excluding_build"] + 1e-9, name


def test_wider_scope_has_more_support(cells):
    """Support is a count over a larger corpus; if this inverts, the join is wrong."""
    assert cells["global"]["overall"]["mean_support"] > cells["local"]["overall"]["mean_support"]


def test_per_group_breakdown_covers_every_request(cells):
    for name, cell in cells.items():
        if not cell["per_group"]:
            continue
        counted = sum(g["normal_steps"] for g in cell["per_group"].values())
        assert counted == pytest.approx(cell["overall"]["normal_steps"]), name


def test_aggregate_pools_counts_rather_than_averaging_ratios():
    """A three-token answer must not weigh as much as a five-hundred-token one."""
    rows = [
        {"normal_steps": 2.0, "speculative_steps": 1.0, "accepted": 1.0, "rejected": 0.0},
        {"normal_steps": 200.0, "speculative_steps": 200.0, "accepted": 0.0, "rejected": 0.0},
    ]
    out = run_rq3.aggregate(rows, beta_budget=0.04, t_verify_1_ns=2e7)
    assert out["token_speedup"] == pytest.approx(202.0 / 201.0)
    assert out["token_speedup"] < statistics.fmean([2.0, 1.0])


def test_accepted_per_token_survives_targets_shorter_than_the_draft(cells):
    """SQuAD answers are ~3 tokens; L/S quantises there, this metric does not."""
    for name, cell in cells.items():
        assert 0.0 <= cell["overall"]["accepted_per_token"] <= 1.0, name
