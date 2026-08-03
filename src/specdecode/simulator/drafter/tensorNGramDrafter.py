"""Tensor-based n-gram drafter implementations for speculative decoding."""

from typing import Dict, List, Optional, Tuple

import torch

from specdecode.interface.abstractTensorDrafter import AbstractTensorDrafter

# Sentinel used to right-pad candidate rows that are shorter than the draft depth.
PAD_ID = -1


def clamp_size_limit(size_limit: Optional[int], corpus_len: int) -> int:
    """Resolve a corpus-size limit to a valid token count.

    ``None`` means the whole corpus. Every drafter shares this rule so that the
    scanning and index-backed arms see exactly the same corpus slice -- their
    retrieval cost may differ, their drafts must not.
    """
    requested = corpus_len if size_limit is None else size_limit
    return max(0, min(requested, corpus_len))


class NGramIndex:
    """
    Pre-built ``k-gram -> [corpus positions]`` index over a token corpus, for every
    k in ``1..max_k``. Lets a drafter look up, in O(1), all positions whose preceding
    k tokens equal a query k-gram — so n-gram drafting scales to million-token corpora
    instead of the O(corpus) linear scan in NGramDrafter / TensorNGramDrafter.

    Positions for each key are stored in ascending (corpus) order, so a consumer can
    iterate them and stop early (depth needs only the first; width needs the first few).
    ``cap_positions`` bounds memory/iteration for very frequent grams by keeping only
    the earliest occurrences. That is lossless for *depth* drafting, which only ever
    needs the first match, but **lossy for width drafting**: distinct continuations
    occurring beyond the cap become invisible, so an index-backed drafter can find
    fewer branches than a scanning one. Width experiments must set
    ``cap_positions >= len(corpus_tokens)``. A ``size_limit`` at query time restricts
    matches to the first N corpus tokens (for corpus-size sweeps).
    """

    def __init__(
        self,
        corpus_tokens: List[int],
        max_k: int = 2,
        cap_positions: int = 256,
    ) -> None:
        self.corpus_tokens = corpus_tokens
        self.max_k = max_k
        self.cap_positions = cap_positions
        self.tables: Dict[int, Dict[Tuple[int, ...], List[int]]] = {}
        n_corpus = len(corpus_tokens)
        for k in range(1, max_k + 1):
            table: Dict[Tuple[int, ...], List[int]] = {}
            ct = corpus_tokens
            cap = cap_positions
            for i in range(n_corpus - k):
                key = tuple(ct[i : i + k])
                bucket = table.get(key)
                if bucket is None:
                    table[key] = [i]
                elif len(bucket) < cap:
                    bucket.append(i)
            self.tables[k] = table


class TensorNGramDrafter(AbstractTensorDrafter):
    """
    Tensor-emitting n-gram drafter that supports both depth- and width-drafting.

    Like NGramDrafter it matches the last (n-1) tokens of the prompt against a corpus
    (with backoff down to a 1-gram), but instead of returning a single Python list it
    returns a 2-D ``torch.long`` tensor of shape ``[S, T]``:

    - depth-draft:  num_sequences=1       -> one long bet, shape [1, draft_depth]
    - width-draft:  num_sequences=S (S>1) -> S short bets,  shape [S, draft_depth]

    Width-drafting collects several *distinct* continuations of the same matched prefix
    and prefers candidates with different first tokens, which hedges against an uncertain
    branch point. Rows shorter than ``draft_depth`` are right-padded with PAD_ID.
    """

    def __init__(
        self,
        corpus_tokens: List[int],
        n: int = 3,
        num_sequences: int = 1,
        draft_depth: int = 3,
        size_limit: Optional[int] = None,
    ) -> None:
        self.corpus_tokens = corpus_tokens
        self.n = n
        self.num_sequences = num_sequences  # S — number of candidate sequences
        self.draft_depth = draft_depth  # T — token depth per candidate
        # Restrict matching to the first N corpus tokens. Mirrors
        # IndexedTensorNGramDrafter.size_limit so the scanning and index-backed
        # drafters stay draft-equivalent under a corpus-size sweep.
        self.size_limit = clamp_size_limit(size_limit, len(corpus_tokens))
        self.last_n_used: int = 0  # which n-gram size was used in last call
        self.last_match_corpus_idx: int = -1  # where in corpus the first match was found

    def generate_draft(self, prompt: List[int]) -> torch.Tensor:
        self.last_n_used = 0
        self.last_match_corpus_idx = -1

        if not prompt or not self.corpus_tokens:
            return torch.empty((0, 0), dtype=torch.long)

        candidates = self._collect_candidates(prompt)
        if not candidates:
            return torch.empty((0, 0), dtype=torch.long)

        # Right-pad each candidate to draft_depth and stack into a [S, T] tensor.
        padded = [cand + [PAD_ID] * (self.draft_depth - len(cand)) for cand in candidates]
        return torch.tensor(padded, dtype=torch.long)

    def _collect_candidates(self, prompt: List[int]) -> List[List[int]]:  # noqa: C901
        """
        Gather up to ``num_sequences`` distinct continuations (each up to ``draft_depth``
        tokens) for the longest matching prefix, backing off from (n-1)-gram to 1-gram.

        Within the first n-size that yields any match we keep collecting, preferring
        continuations whose first token has not been seen yet.
        """
        for current_n in range(self.n - 1, 0, -1):
            if len(prompt) < current_n:
                continue

            search_prefix = prompt[-current_n:]
            prefix_len = len(search_prefix)

            candidates: List[List[int]] = []
            seen_continuations: set = set()  # dedupe identical continuations
            seen_first_tokens: set = set()  # encourage diverse first tokens (branching)
            lim = self.size_limit

            for i in range(min(lim, len(self.corpus_tokens) - prefix_len)):
                if self.corpus_tokens[i : i + prefix_len] != search_prefix:
                    continue

                start = i + prefix_len
                draft = self.corpus_tokens[start : min(start + self.draft_depth, lim)]
                if not draft:
                    continue

                key = tuple(draft)
                if key in seen_continuations:
                    continue
                # For width-drafting, skip continuations whose first token we already
                # have, so the candidate set hedges across different branches first.
                if self.num_sequences > 1 and draft[0] in seen_first_tokens:
                    continue

                if self.last_match_corpus_idx == -1:
                    self.last_n_used = current_n
                    self.last_match_corpus_idx = i

                seen_continuations.add(key)
                seen_first_tokens.add(draft[0])
                candidates.append(draft)

                if len(candidates) >= self.num_sequences:
                    return candidates

            if candidates:
                return candidates

        return []


class IndexedTensorNGramDrafter(AbstractTensorDrafter):
    """
    Index-backed equivalent of TensorNGramDrafter (depth + width drafting) that scales
    to large corpora by consulting an NGramIndex instead of scanning the corpus.

    Semantics mirror TensorNGramDrafter:
    - backoff from the (n-1)-gram down to a 1-gram,
    - within the first n-size that yields a match, collect up to ``num_sequences``
      distinct continuations (depth: S=1; width: S>1 preferring different first tokens),
    - right-pad short rows with PAD_ID, return a ``[S, T]`` long tensor
      (``[0, 0]`` if none).

    ``size_limit`` caps matches/continuations to the first N corpus tokens, so the same
    index can drive a corpus-size sweep (0.25M, 0.5M, ... full) without rebuilding.
    """

    def __init__(
        self,
        index: NGramIndex,
        n: int = 3,
        num_sequences: int = 1,
        draft_depth: int = 3,
        size_limit: Optional[int] = None,
    ) -> None:
        self.index = index
        self.corpus_tokens = index.corpus_tokens
        self.n = n
        self.num_sequences = num_sequences
        self.draft_depth = draft_depth
        self.size_limit = clamp_size_limit(size_limit, len(self.corpus_tokens))
        self.last_n_used: int = 0
        self.last_match_corpus_idx: int = -1

    def generate_draft(self, prompt: List[int]) -> torch.Tensor:
        self.last_n_used = 0
        self.last_match_corpus_idx = -1

        if not prompt:
            return torch.empty((0, 0), dtype=torch.long)

        candidates = self._collect_candidates(prompt)
        if not candidates:
            return torch.empty((0, 0), dtype=torch.long)

        padded = [cand + [PAD_ID] * (self.draft_depth - len(cand)) for cand in candidates]
        return torch.tensor(padded, dtype=torch.long)

    def _collect_candidates(self, prompt: List[int]) -> List[List[int]]:  # noqa: C901
        ct = self.corpus_tokens
        depth = self.draft_depth
        lim = self.size_limit

        for current_n in range(self.n - 1, 0, -1):
            if len(prompt) < current_n or current_n > self.index.max_k:
                continue

            key = tuple(prompt[-current_n:])
            positions = self.index.tables[current_n].get(key)
            if not positions:
                continue

            candidates: List[List[int]] = []
            seen_continuations: set = set()
            seen_first_tokens: set = set()

            for i in positions:
                if i >= lim:
                    break  # positions are ascending; the rest are outside the corpus slice
                start = i + current_n
                end = min(start + depth, lim)
                draft = ct[start:end]
                if not draft:
                    continue

                key_cont = tuple(draft)
                if key_cont in seen_continuations:
                    continue
                if self.num_sequences > 1 and draft[0] in seen_first_tokens:
                    continue

                if self.last_match_corpus_idx == -1:
                    self.last_n_used = current_n
                    self.last_match_corpus_idx = i

                seen_continuations.add(key_cont)
                seen_first_tokens.add(draft[0])
                candidates.append(list(draft))

                if len(candidates) >= self.num_sequences:
                    return candidates

            if candidates:
                return candidates

        return []
