"""Tests for the runtime draft-equivalence gate."""

import numpy as np
import pytest

from specdecode.experiments.equivalence import (
    DraftDivergence,
    assert_draft_equivalent,
    diagnose_settings,
    equivalence_prompts,
    sample_prompts,
    saturated_prompts,
)
from specdecode.simulator.drafter.precomputeTensorNGramDrafter import (
    PrecomputeTensorNGramDrafter,
)
from specdecode.simulator.drafter.vectorizeTensorNGramDrafter import (
    VectorizeTensorNGramDrafter,
)

CORPUS = [1, 2, 3, 4, 1, 2, 5, 6, 1, 2, 3, 7, 8, 1, 2, 5, 9, 1, 2, 3]


def _arms(*, num_sequences: int, cap_positions: int, size_limit=None):
    return {
        "on_the_fly": VectorizeTensorNGramDrafter(
            corpus_tokens=CORPUS, n=3, num_sequences=num_sequences,
            draft_depth=3, size_limit=size_limit,
        ),
        "precomputed": PrecomputeTensorNGramDrafter(
            corpus_tokens=CORPUS, n=3, num_sequences=num_sequences,
            draft_depth=3, cap_positions=cap_positions, size_limit=size_limit,
        ),
    }


def test_sample_prompts_draws_from_corpus_within_prefix_bounds() -> None:
    rng = np.random.default_rng(0)
    prompts = sample_prompts(CORPUS, max_prefix=2, count=50, rng=rng)

    assert len(prompts) == 50
    for prompt in prompts:
        assert 1 <= len(prompt) <= 2
        joined = ",".join(map(str, prompt))
        assert joined in ",".join(map(str, CORPUS))


def test_sample_prompts_is_deterministic_for_a_seed() -> None:
    a = sample_prompts(CORPUS, max_prefix=2, count=20, rng=np.random.default_rng(7))
    b = sample_prompts(CORPUS, max_prefix=2, count=20, rng=np.random.default_rng(7))
    assert a == b


def test_passes_for_equivalent_arms() -> None:
    rng = np.random.default_rng(1)
    prompts = sample_prompts(CORPUS, max_prefix=2, count=200, rng=rng)

    report = assert_draft_equivalent(
        _arms(num_sequences=1, cap_positions=len(CORPUS)), prompts, context="depth"
    )

    assert report.arm_ids == ("on_the_fly", "precomputed")
    assert report.prompts_checked == 200
    assert report.corpus_tokens == len(CORPUS)
    assert report.context == "depth"


def test_raises_with_actionable_message_when_cap_binds() -> None:
    rng = np.random.default_rng(2)
    prompts = sample_prompts(CORPUS, max_prefix=2, count=200, rng=rng)

    with pytest.raises(DraftDivergence) as excinfo:
        assert_draft_equivalent(
            _arms(num_sequences=3, cap_positions=1), prompts, context="cell=A k=3"
        )

    message = str(excinfo.value)
    assert "on_the_fly" in message and "precomputed" in message
    assert "cell=A k=3" in message
    assert "cap_positions" in message  # the diagnosis names the setting to change


def test_diagnose_flags_mismatched_size_limit() -> None:
    arms = {
        "on_the_fly": VectorizeTensorNGramDrafter(corpus_tokens=CORPUS, n=3, size_limit=5),
        "precomputed": PrecomputeTensorNGramDrafter(corpus_tokens=CORPUS, n=3, size_limit=10),
    }
    notes = diagnose_settings(arms)
    assert any("size_limit differs" in note for note in notes)


def test_diagnose_flags_index_max_k_too_small() -> None:
    drafter = PrecomputeTensorNGramDrafter(corpus_tokens=CORPUS, n=3)
    drafter.n = 4  # index was built for n=3, so max_k=2 < n-1=3
    notes = diagnose_settings({"a": drafter, "b": drafter})
    assert any("max_k" in note for note in notes)


def test_diagnose_is_silent_for_a_correct_setup() -> None:
    assert diagnose_settings(_arms(num_sequences=3, cap_positions=len(CORPUS))) == []


def test_saturated_prompts_empty_when_cap_never_binds() -> None:
    """The strong guarantee: nothing truncated means width drafting cannot diverge."""
    arms = _arms(num_sequences=3, cap_positions=len(CORPUS))
    assert saturated_prompts(arms["precomputed"].index) == []


def test_saturated_prompts_finds_the_truncated_grams() -> None:
    arms = _arms(num_sequences=3, cap_positions=1)
    prompts = saturated_prompts(arms["precomputed"].index)

    assert prompts, "grams were truncated but none were reported"
    # [1, 2] occurs 4 times in CORPUS, so a cap of 1 truncates it.
    assert [1, 2] in prompts


def test_equivalence_prompts_puts_saturated_grams_first() -> None:
    arms = _arms(num_sequences=3, cap_positions=3)
    rng = np.random.default_rng(3)
    prompts = equivalence_prompts(arms, max_prefix=2, count=50, rng=rng)

    targeted = saturated_prompts(arms["precomputed"].index)
    assert targeted, "expected at least one truncated gram at cap_positions=3"
    assert prompts[: len(targeted)] == targeted
    assert len(prompts) > len(targeted)  # random coverage still added
    assert len({tuple(p) for p in prompts}) == len(prompts)  # no duplicates


def test_equivalence_prompts_are_purely_random_when_nothing_truncated() -> None:
    arms = _arms(num_sequences=3, cap_positions=len(CORPUS))
    rng = np.random.default_rng(5)
    prompts = equivalence_prompts(arms, max_prefix=2, count=20, rng=rng)

    assert saturated_prompts(arms["precomputed"].index) == []
    assert prompts
    assert len({tuple(p) for p in prompts}) == len(prompts)


def test_equivalence_prompts_catch_the_cap_divergence() -> None:
    arms = _arms(num_sequences=3, cap_positions=1)
    rng = np.random.default_rng(4)
    prompts = equivalence_prompts(arms, max_prefix=2, count=10, rng=rng)

    with pytest.raises(DraftDivergence):
        assert_draft_equivalent(arms, prompts)


def test_requires_at_least_two_arms() -> None:
    with pytest.raises(ValueError):
        assert_draft_equivalent({"only": VectorizeTensorNGramDrafter(CORPUS)}, [[1, 2]])
