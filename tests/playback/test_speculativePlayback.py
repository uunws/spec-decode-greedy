from typing import List


from specdecode.simulator.drafter.nGramDrafter import NGramDrafter
from specdecode.simulator.drafter.tensorNGramDrafter import TensorNGramDrafter
from specdecode.simulator.verifier.greedyVerifier import GreedyVerifier
from specdecode.simulator.verifier.tensorGreedyVerifier import TensorGreedyVerifier
from specdecode.simulator.metrics.playbackMetrics import PlaybackMetrics
from specdecode.simulator.playback.speculativePlayback import SpeculativePlayback


class MockTokenizer:
    def encode(self, text: str) -> List[int]:
        return [ord(char) for char in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join(chr(t) for t in tokens)


# =============================================================================
# PlaybackMetrics
# =============================================================================


def test_playback_metrics_calculations():
    metrics = PlaybackMetrics()
    metrics.normal_steps = 10
    metrics.record_step(accepted_count=3, rejected_count=1)
    metrics.record_step(accepted_count=1, rejected_count=3)
    summary = metrics.get_summary()
    assert summary["accepted_tokens"] == 4
    assert summary["rejected_tokens"] == 4
    assert summary["max_accepted_in_single_step"] == 3
    assert summary["speculative_steps"] == 2
    assert summary["average_accepted_per_step"] == 2.0
    assert summary["speedup_ratio"] == 5.0


def test_playback_metrics_initial_state():
    metrics = PlaybackMetrics()
    summary = metrics.get_summary()
    assert summary["accepted_tokens"] == 0
    assert summary["rejected_tokens"] == 0
    assert summary["speculative_steps"] == 0
    assert summary["average_accepted_per_step"] == 0.0
    assert summary["speedup_ratio"] == 1.0


def test_playback_metrics_max_accepted_tracked_correctly():
    metrics = PlaybackMetrics()
    metrics.record_step(accepted_count=1, rejected_count=2)
    metrics.record_step(accepted_count=5, rejected_count=0)
    metrics.record_step(accepted_count=3, rejected_count=1)
    assert metrics.get_summary()["max_accepted_in_single_step"] == 5


def test_playback_metrics_drafter_timing():
    metrics = PlaybackMetrics()
    metrics.record_drafter_time(1_000_000)
    metrics.record_drafter_time(3_000_000)

    summary = metrics.get_summary()
    assert summary["drafter_calls"] == 2
    assert summary["drafter_wall_time_ms"] == 4.0
    assert summary["average_drafter_wall_time_ms"] == 2.0
    assert summary["min_drafter_wall_time_ms"] == 1.0
    assert summary["max_drafter_wall_time_ms"] == 3.0


def test_playback_metrics_playback_timing():
    metrics = PlaybackMetrics()
    metrics.record_playback_time(3_000_000, excluded_ns=1_000_000)
    metrics.record_playback_time(6_000_000, excluded_ns=2_000_000)

    summary = metrics.get_summary()
    assert summary["playback_runs"] == 2
    assert summary["playback_wall_time_ms"] == 6.0
    assert summary["average_playback_wall_time_ms"] == 3.0
    assert summary["min_playback_wall_time_ms"] == 2.0
    assert summary["max_playback_wall_time_ms"] == 4.0


# =============================================================================
# SpeculativePlayback — end-to-end
# =============================================================================


def test_end_to_end_speculative_playback_with_drafter():
    tokenizer = MockTokenizer()
    text = "hello world speculative decoding"
    tokens = tokenizer.encode(text)
    drafter = NGramDrafter(corpus_tokens=tokens, n=3, draft_size=3)
    verifier = GreedyVerifier()
    metrics = PlaybackMetrics()
    playback = SpeculativePlayback(
        tokenizer=tokenizer, drafter=drafter, verifier=verifier, metrics=metrics
    )
    assert playback.run_playback(text, use_drafter=True) == text
    summary = metrics.get_summary()
    assert summary["accepted_tokens"] > 0
    assert summary["speculative_steps"] < summary["normal_steps"]
    assert summary["speedup_ratio"] > 1.0
    assert summary["drafter_calls"] > 0
    assert summary["drafter_wall_time_ms"] >= 0.0
    assert summary["playback_runs"] == 1
    assert summary["playback_wall_time_ms"] >= 0.0


def test_end_to_end_playback_without_drafter():
    tokenizer = MockTokenizer()
    text = "simple run"
    drafter = NGramDrafter(corpus_tokens=[], n=3, draft_size=3)
    verifier = GreedyVerifier()
    metrics = PlaybackMetrics()
    playback = SpeculativePlayback(
        tokenizer=tokenizer, drafter=drafter, verifier=verifier, metrics=metrics
    )
    assert playback.run_playback(text, use_drafter=False) == text
    assert metrics.get_summary()["accepted_tokens"] == 0
    assert metrics.get_summary()["speedup_ratio"] == 1.0


def test_playback_empty_input():
    tokenizer = MockTokenizer()
    drafter = NGramDrafter(corpus_tokens=[], n=3, draft_size=3)
    verifier = GreedyVerifier()
    playback = SpeculativePlayback(tokenizer=tokenizer, drafter=drafter, verifier=verifier)
    assert playback.run_playback("", use_drafter=True) == ""


def test_playback_single_token_input():
    tokenizer = MockTokenizer()
    drafter = NGramDrafter(corpus_tokens=[], n=3, draft_size=3)
    verifier = GreedyVerifier()
    playback = SpeculativePlayback(tokenizer=tokenizer, drafter=drafter, verifier=verifier)
    assert playback.run_playback("A") == "A"


def test_playback_drafter_returns_nothing_still_completes():
    tokenizer = MockTokenizer()
    drafter = NGramDrafter(corpus_tokens=[], n=3, draft_size=3)
    verifier = GreedyVerifier()
    playback = SpeculativePlayback(tokenizer=tokenizer, drafter=drafter, verifier=verifier)
    assert playback.run_playback("hello", use_drafter=True) == "hello"


def test_playback_output_identical_with_and_without_drafter():
    tokenizer = MockTokenizer()
    text = "hello world"
    tokens = tokenizer.encode(text)
    result_spec = SpeculativePlayback(
        tokenizer=tokenizer,
        drafter=NGramDrafter(corpus_tokens=tokens, n=3, draft_size=3),
        verifier=GreedyVerifier(),
    ).run_playback(text, use_drafter=True)
    result_norm = SpeculativePlayback(
        tokenizer=tokenizer,
        drafter=NGramDrafter(corpus_tokens=tokens, n=3, draft_size=3),
        verifier=GreedyVerifier(),
    ).run_playback(text, use_drafter=False)
    assert result_spec == result_norm == text


# =============================================================================
# Tensor end-to-end
# =============================================================================


def test_depth_then_verify_roundtrip():
    corpus = [1, 2, 3, 4, 5, 6, 7, 8]
    drafter = TensorNGramDrafter(corpus, n=3, num_sequences=1, draft_depth=3)
    verifier = TensorGreedyVerifier()
    prefix = [1, 2, 3]
    draft = drafter.generate_draft(prefix)
    result = verifier.verify(draft, prefix, corpus)
    assert draft.shape == (1, 3)
    assert result["accepted_count"] == 3
    assert result["accepted_tokens"] == [4, 5, 6, 7]


def test_width_then_verify_selects_correct_branch():
    corpus = [9, 1, 0, 9, 2, 8]
    drafter = TensorNGramDrafter(corpus, n=2, num_sequences=2, draft_depth=2)
    verifier = TensorGreedyVerifier()
    complete = [9, 2, 8]
    prefix = [9]
    draft = drafter.generate_draft(prefix)
    result = verifier.verify(draft, prefix, complete)
    assert draft.shape[0] == 2
    assert result["accepted_count"] == 2
    assert result["accepted_tokens"][:2] == [2, 8]
