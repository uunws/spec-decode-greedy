"""Construction rules for the RQ2 workloads.

RQ2's factor is the dataset, so anything that differs between datasets other than
their content is a confound. These pin the invariants that keep that true: equal
datastore sizes, held-out targets, no padding, and a candidate rule fixed before
any measurement.
"""

import pytest

from specdecode.experiments import rq2_workloads
from specdecode.experiments.rq2_workloads import (
    MIN_TARGET_TOKENS,
    NotEnoughData,
    build_workload,
)
from specdecode.experiments.sources import Sample


def _samples(count: int, doc_len: int = 40, target_len: int = 80):
    """Distinct token ranges per sample, so leakage is visible rather than plausible."""
    out = []
    for i in range(count):
        base = 1000 * (i + 1)
        out.append(
            Sample(
                group_id=f"g{i}",
                doc_text=" ".join(str(base + j) for j in range(doc_len)),
                target_text=" ".join(str(base + 500 + j) for j in range(target_len)),
            )
        )
    return out


def _enc(text: str):
    return [int(tok) for tok in text.split()]


@pytest.fixture
def patched(monkeypatch):
    def install(samples):
        monkeypatch.setattr(
            rq2_workloads, "load_samples", lambda dataset, limit: samples[:limit]
        )
    return install


def test_datastore_is_exactly_the_requested_size(patched):
    patched(_samples(200))
    workload = build_workload("fake", _enc, 2000, 5, 200)
    assert len(workload.datastore) == 2000


def test_a_targets_own_document_is_never_in_the_datastore(patched):
    """The holdout is the difference between measuring a workload and a planted copy."""
    patched(_samples(200))
    workload = build_workload("fake", _enc, 2000, 5, 200)

    store = set(workload.datastore)
    for target in workload.targets:
        assert not (set(target) & store), "a request's own tokens leaked into the datastore"


def test_targets_are_truncated_but_never_padded(patched):
    """Concatenating short outputs to hit a quota invents requests nobody issues."""
    patched(_samples(200, target_len=80))
    workload = build_workload("fake", _enc, 2000, 5, target_tokens=50)
    assert all(len(t) == 50 for t in workload.targets)

    workload = build_workload("fake", _enc, 2000, 5, target_tokens=500)
    assert all(len(t) == 80 for t in workload.targets)


def test_short_output_datasets_are_excluded_by_the_rule_not_by_their_score(patched):
    """SQuAD-shaped data cannot carry an L/S comparison, and says so up front."""
    patched(_samples(200, target_len=MIN_TARGET_TOKENS - 1))
    with pytest.raises(NotEnoughData, match="reach"):
        build_workload("fake", _enc, 2000, 5, 200)


def test_a_thin_source_fails_loudly_rather_than_shrinking_the_datastore(patched):
    """An undersized store would make one dataset look more structured for free."""
    patched(_samples(6))
    with pytest.raises(NotEnoughData, match="datastore reached"):
        build_workload("fake", _enc, 1_000_000, 5, 200)


def test_every_candidate_is_a_known_source():
    from specdecode.experiments.sources import ITERATORS

    for name in rq2_workloads.CANDIDATES:
        assert name in ITERATORS
