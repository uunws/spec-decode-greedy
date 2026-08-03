"""Exact gram statistics over a datastore.

The two RQ1 metrics need counts the drafters throw away:

* **historical support** -- how many times the current suffix occurs in the
  datastore. ``NGramIndex`` holds positions for this, but truncates them at
  ``cap_positions``, so its counts are censored exactly where support is highest.
* **prefix branching entropy** -- how the continuations of a matched suffix are
  distributed. The drafters compute a distinct-first-token set and discard it.

So this builds a counting index instead of a position index: memory scales with
the number of *distinct* grams rather than with corpus length times ``max_k``,
and nothing is capped.
"""

import math
from typing import Dict, Mapping, Sequence, Tuple

Gram = Tuple[int, ...]


class SuffixStatsIndex:
    """Uncensored ``gram -> occurrences`` and ``gram -> {next_token: count}``."""

    def __init__(self, tokens: Sequence[int], max_k: int = 3) -> None:
        if max_k < 1:
            raise ValueError("max_k must be >= 1")

        self.max_k = max_k
        self.n_tokens = len(tokens)
        self.gram_count: Dict[int, Dict[Gram, int]] = {}
        self.continuation: Dict[int, Dict[Gram, Dict[int, int]]] = {}

        for k in range(1, max_k + 1):
            counts: Dict[Gram, int] = {}
            conts: Dict[Gram, Dict[int, int]] = {}
            for i in range(len(tokens) - k):
                gram = tuple(tokens[i : i + k])
                counts[gram] = counts.get(gram, 0) + 1
                bucket = conts.setdefault(gram, {})
                nxt = tokens[i + k]
                bucket[nxt] = bucket.get(nxt, 0) + 1
            self.gram_count[k] = counts
            self.continuation[k] = conts

    def support(self, gram: Gram) -> int:
        """How many times this gram occurs, with a token following it."""
        table = self.gram_count.get(len(gram))
        return 0 if table is None else table.get(gram, 0)

    def continuations(self, gram: Gram) -> Mapping[int, int]:
        table = self.continuation.get(len(gram))
        return {} if table is None else table.get(gram, {})

    def branch_factor(self, gram: Gram) -> int:
        """How many *distinct* tokens follow this gram."""
        return len(self.continuations(gram))

    def branch_entropy(self, gram: Gram) -> float:
        """Shannon entropy of the continuation distribution, in bits.

        0 bits means the continuation is certain; 1 bit means a coin flip between
        two options; higher means the drafter is guessing among more branches.
        """
        counts = self.continuations(gram)
        total = sum(counts.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy

    def normalized_branch_entropy(self, gram: Gram) -> float:
        """Entropy divided by its maximum for this branch factor, in [0, 1].

        Scale-free, so workloads with different vocabularies and tokenizers stay
        comparable -- 2 equally likely branches and 8 equally likely branches both
        score 1.0, because in both cases history tells you nothing about which.
        """
        factor = self.branch_factor(gram)
        if factor <= 1:
            return 0.0
        return self.branch_entropy(gram) / math.log2(factor)

    def top1_continuation(self, gram: Gram) -> Tuple[int, float]:
        """Most frequent continuation and its share, or ``(-1, 0.0)`` if unseen."""
        counts = self.continuations(gram)
        if not counts:
            return -1, 0.0
        token = max(counts, key=lambda t: (counts[t], -t))
        return token, counts[token] / sum(counts.values())
