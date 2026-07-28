"""Tests for VectorizeTensorNGramDrafter."""

import torch

from specdecode.simulator.drafter.vectorizeTensorNGramDrafter import VectorizeTensorNGramDrafter


def test_vectorize_tensor_ngram_matches_tensor_contract() -> None:
    drafter = VectorizeTensorNGramDrafter(
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


def test_vectorize_tensor_ngram_full_budget_single_sequence() -> None:
    drafter = VectorizeTensorNGramDrafter(
        corpus_tokens=list(range(1, 21)),
        n=3,
        num_sequences=1,
        draft_depth=10,
    )

    draft = drafter.generate_draft([2, 3])

    assert draft.shape == (1, 10)
    assert draft.tolist() == [[4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]


def test_vectorize_tensor_ngram_backoff() -> None:
    drafter = VectorizeTensorNGramDrafter(
        corpus_tokens=[1, 2, 4, 5, 6, 7],
        n=3,
        num_sequences=1,
        draft_depth=2,
    )

    draft = drafter.generate_draft([3, 4])

    assert draft.shape == (1, 2)
    assert draft.tolist() == [[5, 6]]


def test_vectorize_tensor_ngram_no_match_returns_empty() -> None:
    drafter = VectorizeTensorNGramDrafter(
        corpus_tokens=[1, 2, 3],
        n=3,
        num_sequences=1,
        draft_depth=3,
    )

    draft = drafter.generate_draft([99, 100])

    assert draft.shape == (0, 0)
    assert draft.numel() == 0


def test_vectorize_tensor_ngram_padding_near_corpus_end() -> None:
    drafter = VectorizeTensorNGramDrafter(
        corpus_tokens=[1, 2, 3, 4, 5],
        n=3,
        num_sequences=1,
        draft_depth=4,
    )

    draft = drafter.generate_draft([2, 3])

    assert draft.shape == (1, 4)
    assert draft.tolist() == [[4, 5, -1, -1]]


def test_vectorize_tensor_ngram_width_returns_distinct_branches() -> None:
    drafter = VectorizeTensorNGramDrafter(
        corpus_tokens=[9, 1, 0, 9, 2, 0, 9, 3, 0],
        n=2,
        num_sequences=3,
        draft_depth=2,
    )

    draft = drafter.generate_draft([9])

    assert draft.shape == (3, 2)
    assert sorted(int(row[0]) for row in draft) == [1, 2, 3]


def test_vectorize_tensor_ngram_width_dedupes_identical_continuations() -> None:
    drafter = VectorizeTensorNGramDrafter(
        corpus_tokens=[9, 1, 1, 0, 9, 1, 1, 0, 9, 2, 2],
        n=2,
        num_sequences=3,
        draft_depth=2,
    )

    draft = drafter.generate_draft([9])
    rows = [tuple(int(token) for token in row) for row in draft]

    assert (1, 1) in rows
    assert (2, 2) in rows
    assert len(rows) == len(set(rows))


def test_vectorize_tensor_ngram_width_fewer_candidates_than_requested() -> None:
    drafter = VectorizeTensorNGramDrafter(
        corpus_tokens=[9, 1, 2, 0, 9, 1, 2],
        n=2,
        num_sequences=4,
        draft_depth=2,
    )

    draft = drafter.generate_draft([9])

    assert draft.shape == (1, 2)
    assert draft.tolist() == [[1, 2]]


def test_vectorize_tensor_ngram_width_pads_short_branch() -> None:
    drafter = VectorizeTensorNGramDrafter(
        corpus_tokens=[9, 1, 1, 1, 5, 9, 2],
        n=2,
        num_sequences=2,
        draft_depth=3,
    )

    draft = drafter.generate_draft([9])
    rows = [tuple(int(token) for token in row) for row in draft]

    assert draft.shape == (2, 3)
    assert (1, 1, 1) in rows
    assert (2, -1, -1) in rows
