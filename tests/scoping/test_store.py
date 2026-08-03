"""The cache supplies R, the amortization factor, by counting rather than assuming it."""

from specdecode.scoping import ScopedIndexCache


def corpus(n: int = 200) -> list:
    return [i % 17 for i in range(n)]


def test_first_get_builds_and_second_reuses():
    cache = ScopedIndexCache(max_k=2)
    _, built_first = cache.get("k", corpus())
    _, built_second = cache.get("k", corpus())
    assert built_first is True
    assert built_second is False
    assert cache.stats().builds == 1


def test_reuse_counts_requests_served_not_lookups():
    """The runner resolves a scope once per bucket, so lookups are not requests."""
    cache = ScopedIndexCache(max_k=2)
    cache.get("shared", corpus())
    cache.record_requests(40)
    assert cache.stats().builds == 1
    assert cache.stats().requests == 40
    assert cache.stats().mean_reuse == 40.0


def test_mean_reuse_is_requests_per_build():
    cache = ScopedIndexCache(max_k=2)
    cache.get("shared", corpus())
    cache.record_requests(6)
    for i in range(3):
        cache.get(f"private{i}", corpus())
        cache.record_requests(1)
    stats = cache.stats()
    assert stats.builds == 4
    assert stats.requests == 9
    assert stats.mean_reuse == 9 / 4


def test_request_local_pattern_gives_reuse_of_one():
    cache = ScopedIndexCache(max_k=2)
    for i in range(5):
        cache.get(f"local|{i}", corpus())
        cache.record_requests(1)
    assert cache.stats().mean_reuse == 1.0


def test_build_cost_is_recorded_per_key():
    cache = ScopedIndexCache(max_k=2)
    cache.get("k", corpus(400))
    ns, size = cache.build_cost("k")
    assert ns > 0
    assert size > 0
    assert cache.build_cost("missing") == (0, 0)


def test_capacity_evicts_but_keeps_the_counters():
    cache = ScopedIndexCache(max_k=2, capacity=1)
    cache.get("a", corpus())
    cache.get("b", corpus())
    _, rebuilt = cache.get("a", corpus())
    assert rebuilt is True, "evicted entries are rebuilt"
    assert cache.stats().builds == 3


def test_index_is_usable_for_lookup():
    tokens = [1, 2, 3, 1, 2, 4]
    cache = ScopedIndexCache(max_k=2)
    index, _ = cache.get("k", tokens)
    assert index.corpus_tokens == tokens
    assert index.tables[2][(1, 2)] == [0, 3]
