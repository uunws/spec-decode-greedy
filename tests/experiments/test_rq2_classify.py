"""The low/high labelling rule.

This is the piece the whole of RQ2 rests on. If a dataset could be called
high-speculatability because it ran fast, the headline claim would be true by
definition, so the rule has to be checkable on its own, without any timing.
"""

import importlib.util
import os

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "classify_rq2",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "scripts", "classify_rq2.py"),
)
assert _SPEC and _SPEC.loader
classify_rq2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(classify_rq2)


def _row(support: float, structure: float):
    return {"mean_support": support, "structure": structure}


def test_a_pair_needs_both_axes_not_just_one():
    """One axis winning is exactly the situation the median split mishandled."""
    rows = {
        "top_right": _row(10.0, 0.9),
        "bottom_left": _row(1.0, 0.1),
        "findable_but_ambiguous": _row(10.0, 0.1),
        "unambiguous_but_rare": _row(1.0, 0.9),
    }
    pairs = {(p["high"], p["low"]) for p in classify_rq2.dominance_pairs(rows)}
    assert ("top_right", "bottom_left") in pairs
    assert ("findable_but_ambiguous", "bottom_left") not in pairs
    assert ("top_right", "findable_but_ambiguous") not in pairs


def test_a_gap_smaller_than_the_margin_is_not_a_pair():
    """Two datasets a percent apart are not a low/high contrast."""
    rows = {"a": _row(10.0, 0.500), "b": _row(9.9, 0.499)}
    assert classify_rq2.dominance_pairs(rows, margin=0.05) == []
    assert len(classify_rq2.dominance_pairs(rows, margin=0.0)) == 1


def test_pairing_never_reads_a_speedup():
    """The regression that would make the headline claim true by definition."""
    rows = {
        "a": {"mean_support": 10.0, "structure": 0.9, "token_speedup": 0.01},
        "b": {"mean_support": 1.0, "structure": 0.1, "token_speedup": 99.0},
    }
    pairs = classify_rq2.dominance_pairs(rows)
    assert [(p["high"], p["low"]) for p in pairs] == [("a", "b")]


def test_pooling_weights_positions_not_requests():
    """A 200-token request must count for more than a 70-token one."""
    long_target = list(range(1, 60)) * 4
    datastore = long_target * 3
    pooled = classify_rq2.measure(datastore, [long_target, [1, 2, 3]], n=3)
    assert pooled["n_positions"] == (len(long_target) - 1) + 2
    assert pooled["coverage"] == pytest.approx(1.0, abs=0.1)


def test_an_empty_datastore_reports_zero_structure_rather_than_crashing():
    out = classify_rq2.measure([], [[1, 2, 3, 4]], n=3)
    assert out["coverage"] == 0.0
    assert out["structure"] == 0.0
