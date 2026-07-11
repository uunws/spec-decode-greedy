"""Tests for PrecomputeTensorNGramDrafter."""

import torch

from specdecode.simulator.drafter.precomputeTensorNGramDrafter import (
    PrecomputeTensorNGramDrafter,
)


def test_precompute_tensor_ngram_preserves_tensor_contract() -> None:
    drafter = PrecomputeTensorNGramDrafter(
        corpus_tokens=[1, 2, 3, 4, 5, 6],
        n=3,
        num_sequences=1,
        draft_depth=2,
    )

    draft = drafter.generate_draft([2, 3])

    assert isinstance(draft, torch.Tensor)
    assert draft.dtype == torch.long
    assert draft.shape == (1, 2)
    assert draft.tolist() == [[4, 5]]


def test_precompute_tensor_ngram_width_returns_distinct_branches() -> None:
    drafter = PrecomputeTensorNGramDrafter(
        corpus_tokens=[9, 1, 0, 9, 2, 0, 9, 3, 0],
        n=2,
        num_sequences=3,
        draft_depth=2,
    )

    draft = drafter.generate_draft([9])

    assert draft.shape == (3, 2)
    assert sorted(int(row[0]) for row in draft) == [1, 2, 3]


def test_precompute_tensor_ngram_size_limit_pads_at_limit() -> None:
    drafter = PrecomputeTensorNGramDrafter(
        corpus_tokens=[1, 2, 3, 4, 5, 6],
        n=3,
        num_sequences=1,
        draft_depth=4,
        size_limit=5,
    )

    draft = drafter.generate_draft([2, 3])

    assert draft.shape == (1, 4)
    assert draft.tolist() == [[4, 5, -1, -1]]
