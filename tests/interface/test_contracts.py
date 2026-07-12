"""
Interface-level contract tests for speculative decoding abstractions.

These tests validate implementations against the abstract interface contracts
(AbstractDrafter, AbstractVerifier, AbstractTensorDrafter, AbstractTensorVerifier)
so that any future drafter/verifier automatically gets coverage by being added
to the parametrize lists below.
"""

from typing import List

import pytest
import torch

from specdecode.interface.abstractDrafter import AbstractDrafter
from specdecode.interface.abstractPlayback import AbstractPlayback
from specdecode.interface.abstractTensorDrafter import AbstractTensorDrafter
from specdecode.interface.abstractTensorVerifier import AbstractTensorVerifier
from specdecode.interface.abstractVerifier import AbstractVerifier
from specdecode.simulator.drafter.nGramDrafter import NGramDrafter
from specdecode.simulator.drafter.tensorNGramDrafter import PAD_ID, TensorNGramDrafter
from specdecode.simulator.metrics.playbackMetrics import PlaybackMetrics
from specdecode.simulator.playback.speculativePlayback import SpeculativePlayback
from specdecode.simulator.verifier.greedyVerifier import GreedyVerifier
from specdecode.simulator.verifier.tensorGreedyVerifier import TensorGreedyVerifier

# =============================================================================
# ABC instantiation prevention
# =============================================================================


def test_cannot_instantiate_abstract_drafter() -> None:
    with pytest.raises(TypeError) as excinfo:
        AbstractDrafter()  # type: ignore
    assert "Can't instantiate abstract class" in str(
        excinfo.value
    ) or "Can't instantiate class" in str(excinfo.value)


def test_cannot_instantiate_abstract_verifier() -> None:
    with pytest.raises(TypeError) as excinfo:
        AbstractVerifier()  # type: ignore
    assert "Can't instantiate abstract class" in str(
        excinfo.value
    ) or "Can't instantiate class" in str(excinfo.value)


def test_cannot_instantiate_abstract_playback() -> None:
    with pytest.raises(TypeError) as excinfo:
        AbstractPlayback(tokenizer="dummy", drafter=None, verifier=None, metrics=None)  # type: ignore
    assert "Can't instantiate abstract class" in str(
        excinfo.value
    ) or "Can't instantiate class" in str(excinfo.value)


def test_cannot_instantiate_abstract_tensor_drafter() -> None:
    with pytest.raises(TypeError) as excinfo:
        AbstractTensorDrafter()  # type: ignore
    assert "Can't instantiate abstract class" in str(
        excinfo.value
    ) or "Can't instantiate class" in str(excinfo.value)


def test_cannot_instantiate_abstract_tensor_verifier() -> None:
    with pytest.raises(TypeError) as excinfo:
        AbstractTensorVerifier()  # type: ignore
    assert "Can't instantiate abstract class" in str(
        excinfo.value
    ) or "Can't instantiate class" in str(excinfo.value)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_corpus() -> List[int]:
    return [1, 2, 3, 4, 5, 6, 7, 8, 9]


@pytest.fixture
def ngram_drafter(simple_corpus: List[int]) -> AbstractDrafter:
    return NGramDrafter(corpus_tokens=simple_corpus, n=3, draft_size=3)


@pytest.fixture
def greedy_verifier() -> AbstractVerifier:
    return GreedyVerifier()


@pytest.fixture
def tensor_ngram_drafter(simple_corpus: List[int]) -> AbstractTensorDrafter:
    return TensorNGramDrafter(simple_corpus, n=3, num_sequences=1, draft_depth=3)


@pytest.fixture
def tensor_verifier() -> AbstractTensorVerifier:
    return TensorGreedyVerifier()


# =============================================================================
# 1. AbstractDrafter contract
# =============================================================================

DRAFTER_FACTORIES = [
    pytest.param(
        lambda corpus: NGramDrafter(corpus_tokens=corpus, n=3, draft_size=3), id="NGramDrafter"
    ),
]


@pytest.mark.parametrize("factory", DRAFTER_FACTORIES)
def test_drafter_contract_returns_list(factory) -> None:
    drafter: AbstractDrafter = factory([1, 2, 3, 4, 5])
    result = drafter.generate_draft([1, 2])
    assert isinstance(result, list)
    assert all(isinstance(t, int) for t in result)


@pytest.mark.parametrize("factory", DRAFTER_FACTORIES)
def test_drafter_contract_empty_corpus_returns_empty(factory) -> None:
    drafter: AbstractDrafter = factory([])
    assert drafter.generate_draft([1, 2, 3]) == []


@pytest.mark.parametrize("factory", DRAFTER_FACTORIES)
def test_drafter_contract_empty_prompt_returns_empty(factory) -> None:
    drafter: AbstractDrafter = factory([1, 2, 3, 4, 5])
    assert drafter.generate_draft([]) == []


@pytest.mark.parametrize("factory", DRAFTER_FACTORIES)
def test_drafter_contract_propose_logic(factory) -> None:
    drafter_small = NGramDrafter(corpus_tokens=[3, 4, 2, 5], n=2, draft_size=1)
    assert drafter_small.generate_draft([3]) == [4]


@pytest.mark.parametrize("factory", DRAFTER_FACTORIES)
def test_drafter_contract_no_match_returns_empty(factory) -> None:
    drafter_small = NGramDrafter(corpus_tokens=[3, 4, 2, 5], n=2, draft_size=1)
    assert drafter_small.generate_draft([10]) == []


# =============================================================================
# 2. AbstractVerifier contract
# =============================================================================

VERIFIER_FACTORIES = [pytest.param(lambda: GreedyVerifier(), id="GreedyVerifier")]
REQUIRED_VERIFIER_KEYS = {"accepted_tokens", "accepted_count", "rejected_count"}


@pytest.mark.parametrize("factory", VERIFIER_FACTORIES)
def test_verifier_contract_returns_required_keys(factory) -> None:
    verifier: AbstractVerifier = factory()
    result = verifier.verify([4, 5], [1, 2, 3], [1, 2, 3, 4, 5, 6])
    assert REQUIRED_VERIFIER_KEYS.issubset(result.keys())


@pytest.mark.parametrize("factory", VERIFIER_FACTORIES)
def test_verifier_contract_accepted_tokens_is_list_of_int(factory) -> None:
    verifier: AbstractVerifier = factory()
    result = verifier.verify([4], [1, 2, 3], [1, 2, 3, 4, 5])
    tokens = result["accepted_tokens"]
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)  # type: ignore[union-attr]


@pytest.mark.parametrize("factory", VERIFIER_FACTORIES)
def test_verifier_contract_counts_are_non_negative_ints(factory) -> None:
    verifier: AbstractVerifier = factory()
    result = verifier.verify([4, 5, 99], [1, 2, 3], [1, 2, 3, 4, 5, 6])
    assert isinstance(result["accepted_count"], int)
    assert isinstance(result["rejected_count"], int)
    assert result["accepted_count"] >= 0  # type: ignore[operator]
    assert result["rejected_count"] >= 0  # type: ignore[operator]


@pytest.mark.parametrize("factory", VERIFIER_FACTORIES)
def test_verifier_contract_counts_are_consistent(factory) -> None:
    verifier: AbstractVerifier = factory()
    draft = [4, 5, 99, 100]
    result = verifier.verify(draft, [1, 2, 3], [1, 2, 3, 4, 5, 6, 7])
    total = result["accepted_count"] + result["rejected_count"]  # type: ignore[operator]
    assert total <= len(draft)


@pytest.mark.parametrize("factory", VERIFIER_FACTORIES)
def test_verifier_contract_empty_draft_gives_recovery_only(factory) -> None:
    verifier: AbstractVerifier = factory()
    result = verifier.verify([], [1, 2, 3], [1, 2, 3, 4, 5])
    assert result["accepted_count"] == 0
    assert result["rejected_count"] == 0
    assert result["accepted_tokens"] == [4]


# =============================================================================
# 3. AbstractTensorDrafter contract
# =============================================================================

TENSOR_DRAFTER_FACTORIES = [
    pytest.param(
        lambda corpus: TensorNGramDrafter(corpus, n=3, num_sequences=1, draft_depth=3),
        id="TensorNGramDrafter",
    ),
]


@pytest.mark.parametrize("factory", TENSOR_DRAFTER_FACTORIES)
def test_tensor_drafter_contract_returns_2d_long_tensor(factory) -> None:
    drafter: AbstractTensorDrafter = factory([1, 2, 3, 4, 5, 6, 7])
    result = drafter.generate_draft([1, 2])
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.long
    assert result.dim() == 2


@pytest.mark.parametrize("factory", TENSOR_DRAFTER_FACTORIES)
def test_tensor_drafter_contract_no_match_returns_0x0(factory) -> None:
    drafter: AbstractTensorDrafter = factory([1, 2, 3])
    result = drafter.generate_draft([99, 100])
    assert result.shape == (0, 0)
    assert result.numel() == 0


@pytest.mark.parametrize("factory", TENSOR_DRAFTER_FACTORIES)
def test_tensor_drafter_contract_empty_prompt_returns_0x0(factory) -> None:
    drafter: AbstractTensorDrafter = factory([1, 2, 3, 4, 5])
    result = drafter.generate_draft([])
    assert result.shape == (0, 0)


# =============================================================================
# 4. AbstractTensorVerifier contract
# =============================================================================

TENSOR_VERIFIER_FACTORIES = [
    pytest.param(lambda: TensorGreedyVerifier(), id="TensorGreedyVerifier")
]
REQUIRED_TENSOR_VERIFIER_KEYS = {
    "accepted_tokens",
    "accepted_count",
    "rejected_count",
    "chosen_sequence",
}


@pytest.mark.parametrize("factory", TENSOR_VERIFIER_FACTORIES)
def test_tensor_verifier_contract_returns_required_keys(factory) -> None:
    verifier: AbstractTensorVerifier = factory()
    draft = torch.tensor([[4, 5, 6]], dtype=torch.long)
    result = verifier.verify(draft, [1, 2, 3], [1, 2, 3, 4, 5, 6, 7])
    assert REQUIRED_TENSOR_VERIFIER_KEYS.issubset(result.keys())


@pytest.mark.parametrize("factory", TENSOR_VERIFIER_FACTORIES)
def test_tensor_verifier_contract_chosen_sequence_is_int(factory) -> None:
    verifier: AbstractTensorVerifier = factory()
    draft = torch.empty((0, 0), dtype=torch.long)
    result = verifier.verify(draft, [1, 2, 3], [1, 2, 3, 4, 5])
    assert isinstance(result["chosen_sequence"], int)
    assert result["chosen_sequence"] == -1


@pytest.mark.parametrize("factory", TENSOR_VERIFIER_FACTORIES)
def test_tensor_verifier_contract_counts_consistent(factory) -> None:
    verifier: AbstractTensorVerifier = factory()
    draft = torch.tensor([[4, 5, 99, 100]], dtype=torch.long)
    result = verifier.verify(draft, [1, 2, 3], [1, 2, 3, 4, 5, 6, 7])
    total = result["accepted_count"] + result["rejected_count"]  # type: ignore[operator]
    real_tokens = int((draft[0] != PAD_ID).sum())
    assert total <= real_tokens


# =============================================================================
# 5. AbstractPlayback contract
# =============================================================================


class MockTokenizer:
    def encode(self, text: str) -> List[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join(chr(t) for t in tokens)


def _make_playback(text: str) -> AbstractPlayback:
    tokenizer = MockTokenizer()
    corpus = tokenizer.encode(text)
    drafter = NGramDrafter(corpus_tokens=corpus, n=3, draft_size=3)
    verifier = GreedyVerifier()
    return SpeculativePlayback(tokenizer=tokenizer, drafter=drafter, verifier=verifier)


def test_playback_contract_returns_string() -> None:
    playback: AbstractPlayback = _make_playback("hello world")
    result = playback.run_playback("hello world", use_drafter=True)
    assert isinstance(result, str)


def test_playback_contract_reconstructs_input() -> None:
    text = "hello world"
    playback: AbstractPlayback = _make_playback(text)
    assert playback.run_playback(text, use_drafter=True) == text


def test_playback_contract_drafter_off_also_reconstructs() -> None:
    text = "simple test"
    playback: AbstractPlayback = _make_playback(text)
    assert playback.run_playback(text, use_drafter=False) == text


def test_playback_contract_empty_input_returns_empty_string() -> None:
    playback: AbstractPlayback = _make_playback("anything")
    assert playback.run_playback("") == ""


def test_playback_contract_speculative_faster_than_normal() -> None:
    text = "abcdefghij"
    tokenizer = MockTokenizer()
    corpus = tokenizer.encode(text)
    drafter = NGramDrafter(corpus_tokens=corpus, n=3, draft_size=3)
    verifier = GreedyVerifier()
    metrics = PlaybackMetrics()
    playback = SpeculativePlayback(
        tokenizer=tokenizer, drafter=drafter, verifier=verifier, metrics=metrics
    )
    playback.run_playback(text, use_drafter=True)
    summary = metrics.get_summary()
    assert summary["speculative_steps"] < summary["normal_steps"]
    assert summary["speedup_ratio"] > 1.0  # type: ignore[operator]
