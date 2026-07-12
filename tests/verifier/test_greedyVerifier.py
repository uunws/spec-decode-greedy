import torch

from specdecode.simulator.drafter.tensorNGramDrafter import PAD_ID
from specdecode.simulator.verifier.greedyVerifier import GreedyVerifier
from specdecode.simulator.verifier.tensorGreedyVerifier import TensorGreedyVerifier

# =============================================================================
# GreedyVerifier
# =============================================================================


def test_greedy_verifier_all_accepted():
    verifier = GreedyVerifier()
    result = verifier.verify([4, 5, 6], [1, 2, 3], [1, 2, 3, 4, 5, 6, 7])
    assert result["accepted_count"] == 3
    assert result["rejected_count"] == 0
    assert result["accepted_tokens"] == [4, 5, 6, 7]


def test_greedy_verifier_partial_accepted():
    verifier = GreedyVerifier()
    result = verifier.verify([4, 5, 99, 100], [1, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8])
    assert result["accepted_count"] == 2
    assert result["rejected_count"] == 2
    assert result["accepted_tokens"] == [4, 5, 6]


def test_greedy_verifier_all_rejected():
    verifier = GreedyVerifier()
    result = verifier.verify([99, 100], [1, 2, 3], [1, 2, 3, 4, 5, 6])
    assert result["accepted_count"] == 0
    assert result["rejected_count"] == 2
    assert result["accepted_tokens"] == [4]


def test_greedy_verifier_empty_draft():
    verifier = GreedyVerifier()
    result = verifier.verify([], [1, 2, 3], [1, 2, 3, 4, 5])
    assert result["accepted_count"] == 0
    assert result["rejected_count"] == 0
    assert result["accepted_tokens"] == [4]


def test_greedy_verifier_draft_exceeds_sequence_end():
    verifier = GreedyVerifier()
    result = verifier.verify([4, 5, 6, 99, 100], [1, 2, 3], [1, 2, 3, 4, 5, 6])
    assert result["accepted_count"] == 3
    assert result["accepted_tokens"] == [4, 5, 6]


# =============================================================================
# TensorGreedyVerifier
# =============================================================================


def test_tensor_verifier_all_accepted_single_row():
    verifier = TensorGreedyVerifier()
    draft = torch.tensor([[4, 5, 6]], dtype=torch.long)
    result = verifier.verify(draft, [1, 2, 3], [1, 2, 3, 4, 5, 6, 7])
    assert result["accepted_count"] == 3
    assert result["rejected_count"] == 0
    assert result["accepted_tokens"] == [4, 5, 6, 7]
    assert result["chosen_sequence"] == 0


def test_tensor_verifier_partial_accept():
    verifier = TensorGreedyVerifier()
    draft = torch.tensor([[4, 5, 99, 100]], dtype=torch.long)
    result = verifier.verify(draft, [1, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8])
    assert result["accepted_count"] == 2
    assert result["rejected_count"] == 2
    assert result["accepted_tokens"] == [4, 5, 6]
    assert result["chosen_sequence"] == 0


def test_tensor_verifier_full_reject_first_row():
    verifier = TensorGreedyVerifier()
    draft = torch.tensor([[99, 100], [98, 97]], dtype=torch.long)
    result = verifier.verify(draft, [1, 2, 3], [1, 2, 3, 4, 5, 6])
    assert result["accepted_count"] == 0
    assert result["rejected_count"] == 2
    assert result["accepted_tokens"] == [4]
    assert result["chosen_sequence"] == 0


def test_tensor_verifier_picks_longest_matching_candidate():
    verifier = TensorGreedyVerifier()
    draft = torch.tensor([[4, 99, 0], [4, 5, 6], [4, 5, 99]], dtype=torch.long)
    result = verifier.verify(draft, [1, 2, 3], [1, 2, 3, 4, 5, 6, 7])
    assert result["chosen_sequence"] == 1
    assert result["accepted_count"] == 3
    assert result["accepted_tokens"] == [4, 5, 6, 7]


def test_tensor_verifier_tie_breaks_lowest_index():
    verifier = TensorGreedyVerifier()
    draft = torch.tensor([[4, 5, 99], [4, 5, 98]], dtype=torch.long)
    result = verifier.verify(draft, [1, 2, 3], [1, 2, 3, 4, 5, 6, 7])
    assert result["chosen_sequence"] == 0
    assert result["accepted_count"] == 2


def test_tensor_verifier_ignores_pad_tokens():
    verifier = TensorGreedyVerifier()
    draft = torch.tensor([[4, 5, PAD_ID]], dtype=torch.long)
    result = verifier.verify(draft, [1, 2, 3], [1, 2, 3, 4, 5, 6])
    assert result["accepted_count"] == 2
    assert result["rejected_count"] == 0
    assert result["accepted_tokens"] == [4, 5, 6]


def test_tensor_verifier_empty_draft_returns_recovery_only():
    verifier = TensorGreedyVerifier()
    draft = torch.empty((0, 0), dtype=torch.long)
    result = verifier.verify(draft, [1, 2, 3], [1, 2, 3, 4, 5])
    assert result["accepted_count"] == 0
    assert result["rejected_count"] == 0
    assert result["accepted_tokens"] == [4]
    assert result["chosen_sequence"] == -1


def test_tensor_verifier_at_end_of_ground_truth():
    verifier = TensorGreedyVerifier()
    draft = torch.tensor([[4, 5]], dtype=torch.long)
    result = verifier.verify(draft, [1, 2, 3], [1, 2, 3, 4, 5])
    assert result["accepted_count"] == 2
    assert result["accepted_tokens"] == [4, 5]
