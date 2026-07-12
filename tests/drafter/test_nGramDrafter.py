import torch

from specdecode.simulator.drafter.nGramDrafter import NGramDrafter
from specdecode.simulator.drafter.tensorNGramDrafter import PAD_ID, TensorNGramDrafter

# =============================================================================
# NGramDrafter
# =============================================================================


def test_ngram_drafter_perfect_match():
    corpus = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    drafter = NGramDrafter(corpus_tokens=corpus, n=3, draft_size=3)
    assert drafter.generate_draft([99, 100, 3, 4]) == [5, 6, 7]


def test_ngram_drafter_backoff():
    corpus = [1, 2, 4, 5, 6, 7]
    drafter = NGramDrafter(corpus_tokens=corpus, n=3, draft_size=2)
    assert drafter.generate_draft([3, 4]) == [5, 6]


def test_ngram_drafter_no_match():
    corpus = [1, 2, 3]
    drafter = NGramDrafter(corpus_tokens=corpus, n=3, draft_size=3)
    assert drafter.generate_draft([99, 100]) == []


def test_ngram_drafter_empty_corpus():
    drafter = NGramDrafter(corpus_tokens=[], n=3, draft_size=3)
    assert drafter.generate_draft([1, 2, 3]) == []


def test_ngram_drafter_empty_prompt():
    drafter = NGramDrafter(corpus_tokens=[1, 2, 3, 4, 5], n=3, draft_size=3)
    assert drafter.generate_draft([]) == []


def test_ngram_drafter_prompt_shorter_than_n():
    corpus = [1, 2, 3, 4]
    drafter = NGramDrafter(corpus_tokens=corpus, n=3, draft_size=1)
    assert drafter.generate_draft([2]) == [3]


def test_ngram_drafter_draft_size_exceeds_remaining_corpus():
    corpus = [1, 2, 3, 4, 5]
    drafter = NGramDrafter(corpus_tokens=corpus, n=2, draft_size=100)
    result = drafter.generate_draft([1, 2, 3])
    assert len(result) <= 2


def test_ngram_propose_corpus_match():
    corpus = [3, 4, 2, 5]
    drafter = NGramDrafter(corpus_tokens=corpus, n=2, draft_size=1)
    assert drafter.generate_draft([3]) == [4]


def test_ngram_propose_no_corpus_match():
    corpus = [3, 4, 2, 5]
    drafter = NGramDrafter(corpus_tokens=corpus, n=2, draft_size=1)
    assert drafter.generate_draft([10]) == []


# =============================================================================
# TensorNGramDrafter — depth mode (S=1)
# =============================================================================


def test_depth_drafter_perfect_match_shape_and_values():
    corpus = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    drafter = TensorNGramDrafter(corpus, n=3, num_sequences=1, draft_depth=3)
    draft = drafter.generate_draft([99, 100, 3, 4])
    assert isinstance(draft, torch.Tensor)
    assert draft.dtype == torch.long
    assert draft.shape == (1, 3)
    assert draft.tolist() == [[5, 6, 7]]


def test_depth_drafter_full_budget_single_sequence():
    corpus = list(range(1, 21))
    drafter = TensorNGramDrafter(corpus, n=3, num_sequences=1, draft_depth=10)
    draft = drafter.generate_draft([2, 3])
    assert draft.shape == (1, 10)
    assert draft.tolist() == [[4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]


def test_depth_drafter_backoff():
    corpus = [1, 2, 4, 5, 6, 7]
    drafter = TensorNGramDrafter(corpus, n=3, num_sequences=1, draft_depth=2)
    draft = drafter.generate_draft([3, 4])
    assert draft.shape == (1, 2)
    assert draft.tolist() == [[5, 6]]


def test_depth_drafter_no_match_returns_empty():
    corpus = [1, 2, 3]
    drafter = TensorNGramDrafter(corpus, n=3, num_sequences=1, draft_depth=3)
    draft = drafter.generate_draft([99, 100])
    assert draft.shape == (0, 0)
    assert draft.numel() == 0


def test_depth_drafter_padding_near_corpus_end():
    corpus = [1, 2, 3, 4, 5]
    drafter = TensorNGramDrafter(corpus, n=3, num_sequences=1, draft_depth=4)
    draft = drafter.generate_draft([2, 3])
    assert draft.shape == (1, 4)
    assert draft.tolist() == [[4, 5, PAD_ID, PAD_ID]]


# =============================================================================
# TensorNGramDrafter — width mode (S>1)
# =============================================================================


def test_width_drafter_multiple_distinct_branches():
    corpus = [9, 1, 0, 9, 2, 0, 9, 3, 0]
    drafter = TensorNGramDrafter(corpus, n=2, num_sequences=3, draft_depth=2)
    draft = drafter.generate_draft([9])
    assert draft.shape == (3, 2)
    first_tokens = sorted(int(row[0]) for row in draft)
    assert first_tokens == [1, 2, 3]


def test_width_drafter_dedupes_identical_continuations():
    corpus = [9, 1, 1, 0, 9, 1, 1, 0, 9, 2, 2]
    drafter = TensorNGramDrafter(corpus, n=2, num_sequences=3, draft_depth=2)
    draft = drafter.generate_draft([9])
    rows = [tuple(int(t) for t in row) for row in draft]
    assert (1, 1) in rows
    assert (2, 2) in rows
    assert len(rows) == len(set(rows))


def test_width_drafter_fewer_candidates_than_requested():
    corpus = [9, 1, 2, 0, 9, 1, 2]
    drafter = TensorNGramDrafter(corpus, n=2, num_sequences=4, draft_depth=2)
    draft = drafter.generate_draft([9])
    assert draft.shape[0] == 1
    assert draft.shape[1] == 2
    assert draft.tolist() == [[1, 2]]


def test_width_drafter_pads_short_branch():
    corpus = [9, 1, 1, 1, 5, 9, 2]
    drafter = TensorNGramDrafter(corpus, n=2, num_sequences=2, draft_depth=3)
    draft = drafter.generate_draft([9])
    assert draft.shape == (2, 3)
    rows = [tuple(int(t) for t in row) for row in draft]
    assert (1, 1, 1) in rows
    assert (2, PAD_ID, PAD_ID) in rows
