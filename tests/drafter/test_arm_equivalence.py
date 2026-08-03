"""Draft-equivalence between the on-the-fly and precomputed retrieval arms.

The RQ2 factorial varies retrieval *cost* while holding draft *quality* fixed.
That only works if both arms emit bitwise-identical drafts for the same prompt.
These tests are the control: if one fails, any speedup difference measured
between the arms is confounded with draft quality and cannot be interpreted.
"""

from typing import List, Sequence

import torch

from specdecode.interface.abstractTensorDrafter import AbstractTensorDrafter
from specdecode.simulator.drafter.precomputeTensorNGramDrafter import (
    PrecomputeTensorNGramDrafter,
)
from specdecode.simulator.drafter.tensorNGramDrafter import (
    IndexedTensorNGramDrafter,
    NGramIndex,
    TensorNGramDrafter,
)
from specdecode.simulator.drafter.vectorizeTensorNGramDrafter import (
    VectorizeTensorNGramDrafter,
)

# Repeated grams with several distinct continuations, so backoff and width
# selection are both exercised rather than trivially agreeing.
CORPUS = [1, 2, 3, 4, 1, 2, 5, 6, 1, 2, 3, 7, 8, 1, 2, 5, 9, 1, 2, 3]


def _prompts(corpus: Sequence[int], max_len: int) -> List[List[int]]:
    """Every corpus suffix window up to max_len -- what a playback loop queries."""
    return [
        list(corpus[end - k : end])
        for end in range(1, len(corpus) + 1)
        for k in range(1, max_len + 1)
        if end - k >= 0
    ]


def _assert_equivalent(
    on_the_fly: AbstractTensorDrafter,
    precomputed: AbstractTensorDrafter,
    prompts: Sequence[List[int]],
) -> None:
    for prompt in prompts:
        a = on_the_fly.generate_draft(prompt)
        b = precomputed.generate_draft(prompt)
        assert a.shape == b.shape, f"shape differs for {prompt}: {a.shape} vs {b.shape}"
        assert torch.equal(a, b), f"draft differs for {prompt}: {a.tolist()} vs {b.tolist()}"
        assert on_the_fly.last_n_used == precomputed.last_n_used, (  # type: ignore[attr-defined]
            f"last_n_used differs for {prompt}"
        )


# --- pair 2: python scan (TensorNGramDrafter) vs dict index (Indexed...) ---


def test_pair2_depth_mode_equivalent_across_size_limits() -> None:
    prompts = _prompts(CORPUS, max_len=3)
    index = NGramIndex(corpus_tokens=CORPUS, max_k=2, cap_positions=len(CORPUS))
    for size_limit in range(1, len(CORPUS) + 1):
        on_the_fly = TensorNGramDrafter(
            corpus_tokens=CORPUS, n=3, num_sequences=1, draft_depth=4,
            size_limit=size_limit,
        )
        precomputed = IndexedTensorNGramDrafter(
            index=index, n=3, num_sequences=1, draft_depth=4, size_limit=size_limit
        )
        _assert_equivalent(on_the_fly, precomputed, prompts)


def test_pair2_width_mode_equivalent_when_cap_does_not_bind() -> None:
    prompts = _prompts(CORPUS, max_len=3)
    index = NGramIndex(corpus_tokens=CORPUS, max_k=2, cap_positions=len(CORPUS))
    for size_limit in (5, 12, len(CORPUS)):
        on_the_fly = TensorNGramDrafter(
            corpus_tokens=CORPUS, n=3, num_sequences=3, draft_depth=3,
            size_limit=size_limit,
        )
        precomputed = IndexedTensorNGramDrafter(
            index=index, n=3, num_sequences=3, draft_depth=3, size_limit=size_limit
        )
        _assert_equivalent(on_the_fly, precomputed, prompts)


def test_depth_mode_equivalent_across_size_limits() -> None:
    """The corpus-size sweep must not make the two arms diverge."""
    prompts = _prompts(CORPUS, max_len=3)
    for size_limit in range(1, len(CORPUS) + 1):
        on_the_fly = VectorizeTensorNGramDrafter(
            corpus_tokens=CORPUS, n=3, num_sequences=1, draft_depth=4,
            size_limit=size_limit,
        )
        precomputed = PrecomputeTensorNGramDrafter(
            corpus_tokens=CORPUS, n=3, num_sequences=1, draft_depth=4,
            cap_positions=len(CORPUS), size_limit=size_limit,
        )
        _assert_equivalent(on_the_fly, precomputed, prompts)


def test_depth_mode_equivalent_with_default_size_limit() -> None:
    prompts = _prompts(CORPUS, max_len=3)
    on_the_fly = VectorizeTensorNGramDrafter(
        corpus_tokens=CORPUS, n=4, num_sequences=1, draft_depth=3
    )
    precomputed = PrecomputeTensorNGramDrafter(
        corpus_tokens=CORPUS, n=4, num_sequences=1, draft_depth=3,
        cap_positions=len(CORPUS),
    )
    _assert_equivalent(on_the_fly, precomputed, prompts)


def test_width_mode_equivalent_when_cap_does_not_bind() -> None:
    """Width drafting agrees as long as the index keeps every match position."""
    prompts = _prompts(CORPUS, max_len=3)
    for size_limit in (5, 12, len(CORPUS)):
        on_the_fly = VectorizeTensorNGramDrafter(
            corpus_tokens=CORPUS, n=3, num_sequences=3, draft_depth=3,
            size_limit=size_limit,
        )
        precomputed = PrecomputeTensorNGramDrafter(
            corpus_tokens=CORPUS, n=3, num_sequences=3, draft_depth=3,
            cap_positions=len(CORPUS), size_limit=size_limit,
        )
        _assert_equivalent(on_the_fly, precomputed, prompts)


def test_width_mode_diverges_when_cap_positions_binds() -> None:
    """Known limitation, pinned deliberately.

    NGramIndex keeps only the first `cap_positions` occurrences of a gram, so the
    precomputed arm can see fewer distinct branches than the scanning arm. Width
    experiments must therefore run with cap_positions large enough not to bind.
    """
    corpus = [9, 1, 9, 2, 9, 3]
    on_the_fly = VectorizeTensorNGramDrafter(
        corpus_tokens=corpus, n=2, num_sequences=3, draft_depth=1
    )
    precomputed = PrecomputeTensorNGramDrafter(
        corpus_tokens=corpus, n=2, num_sequences=3, draft_depth=1, cap_positions=2
    )

    scanned = on_the_fly.generate_draft([9])
    indexed = precomputed.generate_draft([9])

    assert sorted(int(row[0]) for row in scanned) == [1, 2, 3]
    assert sorted(int(row[0]) for row in indexed) == [1, 2]
    assert not torch.equal(scanned, indexed)
