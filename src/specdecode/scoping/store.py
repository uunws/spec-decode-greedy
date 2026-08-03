"""Index cache keyed by scope, which is where the amortization factor R comes from.

A narrow scope is cheap to search but is rebuilt often; a wide scope is expensive to
search but is built once and reused. The RQ2 cost model already expresses that as
``t_build / R``, and this cache supplies ``R`` by counting rather than assuming it:
requests that resolve to the same scope key share one built index, so
``mean_reuse`` is literally requests served per build.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from specdecode.experiments.timing import timed_build
from specdecode.simulator.drafter.tensorNGramDrafter import NGramIndex


@dataclass
class CacheStats:
    builds: int
    requests: int
    build_ns_total: int
    build_bytes_total: int

    @property
    def mean_reuse(self) -> float:
        """R: requests served per index build."""
        return self.requests / self.builds if self.builds else 0.0

    @property
    def mean_build_ns(self) -> float:
        return self.build_ns_total / self.builds if self.builds else 0.0

    @property
    def mean_build_bytes(self) -> float:
        return self.build_bytes_total / self.builds if self.builds else 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "builds": self.builds,
            "requests": self.requests,
            "mean_reuse": self.mean_reuse,
            "mean_build_ns": self.mean_build_ns,
            "mean_build_bytes": self.mean_build_bytes,
            "build_ns_total": self.build_ns_total,
        }


class ScopedIndexCache:
    """Build one ``NGramIndex`` per distinct scope key and reuse it.

    ``capacity`` bounds memory for the wide scopes; it is an LRU of one entry by
    default because the runner processes requests grouped by scope key, so a single
    live index is enough and the global scope never has to coexist with a thousand
    request-local ones.
    """

    def __init__(
        self,
        max_k: int,
        *,
        capacity: int = 1,
        cap_positions: Optional[int] = None,
    ) -> None:
        self.max_k = max_k
        self.capacity = max(1, capacity)
        self.cap_positions = cap_positions
        self._entries: Dict[str, NGramIndex] = {}
        self._recent: List[str] = []
        self.builds = 0
        self.requests = 0
        self.build_ns_total = 0
        self.build_bytes_total = 0
        self._per_key_build: Dict[str, Tuple[int, int]] = {}

    def record_requests(self, n: int) -> None:
        """Count how many requests a built index actually served.

        The runner resolves a scope once per bucket rather than once per request,
        so the lookup count is not the reuse factor. ``R`` is reported from this,
        which is why it has to be told rather than inferred.
        """
        self.requests += n

    def get(self, key: str, tokens: List[int]) -> Tuple[NGramIndex, bool]:
        """Return the index for ``key``, building it if absent.

        The boolean says whether a build happened, so the caller can attribute
        build cost to the first request of a scope rather than to all of them.
        """
        if key in self._entries:
            self._touch(key)
            return self._entries[key], False

        cap = self.cap_positions if self.cap_positions is not None else len(tokens)
        factory: Callable[[], NGramIndex] = lambda: NGramIndex(  # noqa: E731
            corpus_tokens=list(tokens), max_k=self.max_k, cap_positions=cap
        )
        index, record = timed_build(f"scope:{key}", factory)

        self.builds += 1
        self.build_ns_total += record.elapsed_ns
        self.build_bytes_total += record.approx_bytes
        self._per_key_build[key] = (record.elapsed_ns, record.approx_bytes)

        self._entries[key] = index
        self._touch(key)
        self._evict()
        return index, True

    def build_cost(self, key: str) -> Tuple[int, int]:
        """(ns, bytes) paid the first time ``key`` was built."""
        return self._per_key_build.get(key, (0, 0))

    def _touch(self, key: str) -> None:
        if key in self._recent:
            self._recent.remove(key)
        self._recent.append(key)

    def _evict(self) -> None:
        while len(self._recent) > self.capacity:
            oldest = self._recent.pop(0)
            self._entries.pop(oldest, None)

    def stats(self) -> CacheStats:
        return CacheStats(
            builds=self.builds,
            requests=self.requests,
            build_ns_total=self.build_ns_total,
            build_bytes_total=self.build_bytes_total,
        )
