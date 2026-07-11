"""Vectorized tensor-based n-gram drafter."""

from typing import List

import torch

from specdecode.simulator.drafter.tensorNGramDrafter import PAD_ID, TensorNGramDrafter


class VectorizeTensorNGramDrafter(TensorNGramDrafter):
    """Tensor n-gram drafter using vectorized corpus search.

    It preserves ``TensorNGramDrafter`` semantics while replacing the Python
    corpus scan with tensor-based window matching.
    """

    def __init__(
        self,
        corpus_tokens: List[int],
        n: int = 3,
        num_sequences: int = 1,
        draft_depth: int = 3,
    ) -> None:
        super().__init__(
            corpus_tokens=corpus_tokens,
            n=n,
            num_sequences=num_sequences,
            draft_depth=draft_depth,
        )
        # This is only a representation change: no n-gram index is built.
        self.corpus_tensor = torch.as_tensor(corpus_tokens, dtype=torch.long).clone()

    def generate_draft(self, prompt: List[int]) -> torch.Tensor:
        """Find n-gram matches with tensor operations and return padded candidates."""
        self.last_n_used = 0
        self.last_match_corpus_idx = -1

        if not prompt or self.corpus_tensor.numel() == 0 or self.draft_depth <= 0:
            return torch.empty((0, 0), dtype=torch.long)

        corpus_len = int(self.corpus_tensor.numel())
        for current_n in range(self.n - 1, 0, -1):
            if len(prompt) < current_n:
                continue

            # A match must have at least one token after its prefix.
            num_starts = corpus_len - current_n
            if num_starts <= 0:
                continue

            query = torch.as_tensor(prompt[-current_n:], dtype=torch.long)
            windows = self.corpus_tensor.unfold(0, current_n, 1)[:num_starts]
            match_mask = torch.all(windows == query, dim=1)
            match_positions = torch.nonzero(match_mask, as_tuple=False).flatten()
            if match_positions.numel() == 0:
                continue

            if self.num_sequences <= 1:
                selected_positions = match_positions[:1]
            else:
                # Keep the earliest match for each distinct first continuation token,
                # then restore corpus order before taking the requested width.
                continuation_positions = match_positions + current_n
                first_tokens = self.corpus_tensor[continuation_positions]
                sorted_first, sorted_order = torch.sort(first_tokens, stable=True)
                first_of_kind = torch.ones(sorted_first.shape, dtype=torch.bool)
                if sorted_first.numel() > 1:
                    first_of_kind[1:] = sorted_first[1:] != sorted_first[:-1]
                unique_match_indices = sorted_order[first_of_kind]
                unique_positions = match_positions[unique_match_indices]
                selected_positions = torch.sort(unique_positions).values[: self.num_sequences]

            starts = selected_positions + current_n
            offsets = torch.arange(self.draft_depth, dtype=torch.long)
            indices = starts[:, None] + offsets[None, :]
            valid = indices < corpus_len
            safe_indices = indices.clamp(max=corpus_len - 1)
            candidates = self.corpus_tensor[safe_indices]
            candidates = torch.where(valid, candidates, torch.full_like(candidates, PAD_ID))

            self.last_n_used = current_n
            self.last_match_corpus_idx = int(selected_positions[0])
            return candidates

        return torch.empty((0, 0), dtype=torch.long)
