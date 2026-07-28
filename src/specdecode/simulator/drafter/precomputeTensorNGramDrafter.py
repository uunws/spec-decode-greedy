"""Precomputed tensor-based n-gram drafter."""

from typing import List, Optional

import torch

from specdecode.simulator.drafter.tensorNGramDrafter import NGramIndex, PAD_ID
from specdecode.simulator.drafter.vectorizeTensorNGramDrafter import (
    VectorizeTensorNGramDrafter,
)


class PrecomputeTensorNGramDrafter(VectorizeTensorNGramDrafter):
    """Tensor n-gram drafter backed by a precomputed prefix-position index.

    The corpus is indexed once during construction. Each generation call then
    performs a dictionary lookup for the matching prefix and uses tensor gathers
    to build and pad the candidate rows.
    """

    def __init__(
        self,
        corpus_tokens: List[int],
        n: int = 3,
        num_sequences: int = 1,
        draft_depth: int = 3,
        cap_positions: int = 256,
        size_limit: Optional[int] = None,
    ) -> None:
        super().__init__(
            corpus_tokens=corpus_tokens,
            n=n,
            num_sequences=num_sequences,
            draft_depth=draft_depth,
        )
        self.cap_positions = cap_positions
        self.index = NGramIndex(
            corpus_tokens=corpus_tokens,
            max_k=max(0, n - 1),
            cap_positions=cap_positions,
        )
        requested_limit = len(corpus_tokens) if size_limit is None else size_limit
        self.size_limit = max(0, min(requested_limit, len(corpus_tokens)))

    def generate_draft(self, prompt: List[int]) -> torch.Tensor:
        """Generate candidates from the precomputed index without corpus scanning."""
        self.last_n_used = 0
        self.last_match_corpus_idx = -1

        if not prompt or self.size_limit == 0 or self.draft_depth <= 0:
            return torch.empty((0, 0), dtype=torch.long)

        for current_n in range(self.n - 1, 0, -1):
            if len(prompt) < current_n or current_n > self.index.max_k:
                continue

            key = tuple(prompt[-current_n:])
            indexed_positions = self.index.tables[current_n].get(key)
            if not indexed_positions:
                continue

            positions = torch.as_tensor(indexed_positions, dtype=torch.long)
            positions = positions[positions < self.size_limit]
            if positions.numel() == 0:
                continue

            if self.num_sequences <= 1:
                selected_positions = positions[:1]
            else:
                # Keep the earliest indexed position for each distinct first token,
                # then restore corpus order before taking the requested width.
                continuation_positions = positions + current_n
                first_tokens = self.corpus_tensor[continuation_positions]
                sorted_first, sorted_order = torch.sort(first_tokens, stable=True)
                first_of_kind = torch.ones(sorted_first.shape, dtype=torch.bool)
                if sorted_first.numel() > 1:
                    first_of_kind[1:] = sorted_first[1:] != sorted_first[:-1]
                unique_match_indices = sorted_order[first_of_kind]
                unique_positions = positions[unique_match_indices]
                selected_positions = torch.sort(unique_positions).values[: self.num_sequences]

            starts = selected_positions + current_n
            offsets = torch.arange(self.draft_depth, dtype=torch.long)
            indices = starts[:, None] + offsets[None, :]
            valid = indices < self.size_limit
            safe_indices = indices.clamp(max=self.size_limit - 1)
            candidates = self.corpus_tensor[safe_indices]
            candidates = torch.where(valid, candidates, torch.full_like(candidates, PAD_ID))

            self.last_n_used = current_n
            self.last_match_corpus_idx = int(selected_positions[0])
            return candidates

        return torch.empty((0, 0), dtype=torch.long)
