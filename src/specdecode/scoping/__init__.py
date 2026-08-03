"""specdecode.scoping — history scoping as a data transformation (RQ3)."""

from specdecode.scoping.policy import (
    Document,
    GlobalScope,
    GroupScoped,
    NarrowScoped,
    Request,
    RequestLocal,
    Scope,
    ScopePolicy,
    SizeMatchedControl,
    assert_causal,
    assert_key_determines_tokens,
    group_sizes,
    scope_sizes,
    visible_history,
)
from specdecode.scoping.store import CacheStats, ScopedIndexCache

__all__ = [
    "CacheStats",
    "Document",
    "GroupScoped",
    "GlobalScope",
    "NarrowScoped",
    "Request",
    "RequestLocal",
    "Scope",
    "ScopePolicy",
    "ScopedIndexCache",
    "SizeMatchedControl",
    "assert_causal",
    "assert_key_determines_tokens",
    "group_sizes",
    "scope_sizes",
    "visible_history",
]
